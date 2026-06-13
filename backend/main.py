"""
main.py  –  FastAPI backend for rPPG Heart Rate Analyser
=========================================================

Endpoints
---------
POST /analyse          –  Upload a video file → full analysis JSON
GET  /health           –  Health check

Stack: FastAPI + OpenCV + NumPy + SciPy  (pure classical DSP, no ML)
"""

import io
import os
import uuid
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import resample, find_peaks
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="rPPG Heart Rate API",
    description="Classical signal processing heart rate estimation from face video.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Physiological constants ───────────────────────────────────────────────────
FREQ_MIN   = 0.67   # Hz  →  40 BPM
FREQ_MAX   = 3.0    # Hz  → 180 BPM
EMA_ALPHA  = 0.25
WINDOW_S   = 10.0   # seconds per sliding window
OVERLAP    = 0.5

# Haar Cascade loaded once at startup
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade  = cv2.CascadeClassifier(_CASCADE_PATH)

MAX_FILE_MB   = 100
ALLOWED_TYPES = {"video/mp4", "video/webm", "video/quicktime", "video/x-msvideo"}

# ── DSP helpers ───────────────────────────────────────────────────────────────

def _normalise(sig: np.ndarray) -> np.ndarray:
    mu, sigma = np.mean(sig), np.std(sig)
    return (sig - mu) / sigma if sigma > 1e-8 else sig - mu


def _fft_bandpass(sig: np.ndarray, dt: float):
    """Hann-windowed FFT bandpass. Returns (sigfilt, freq, PSD_orig, PSD_filt)."""
    n      = len(sig)
    window = np.hanning(n)
    fcomp  = np.fft.fft(sig * window, n)
    PSD_o  = (fcomp * np.conj(fcomp)).real / n

    freq  = (1.0 / (dt * n)) * np.arange(n)
    half  = n // 2
    pmask = (freq[:half] >= FREQ_MIN) & (freq[:half] <= FREQ_MAX)
    mask  = np.zeros(n, dtype=bool)
    mask[1:half]          = pmask[1:]
    mask[n - half + 1:]   = pmask[1:][::-1]

    ffilt       = fcomp.copy()
    ffilt[~mask] = 0.0
    PSD_f       = (ffilt * np.conj(ffilt)).real / n
    sigfilt     = np.fft.ifft(ffilt).real
    return sigfilt, freq, PSD_o, PSD_f


def _snr(sig: np.ndarray, sigfilt: np.ndarray) -> float:
    p_o = np.dot(sig, sig) / max(len(sig), 1)
    p_f = np.dot(sigfilt, sigfilt) / max(len(sigfilt), 1)
    if p_o < 1e-12:
        return 0.0
    return float(10.0 * np.log10((p_f + 1e-12) / p_o))


def _estimate_bpm(sigfilt: np.ndarray, t: np.ndarray, duration: float):
    fps_eff  = len(sigfilt) / duration
    min_dist = max(int(fps_eff * 0.40), 1)

    peaks, props = find_peaks(sigfilt, height=0.3, distance=min_dist)
    minima, _    = find_peaks(-sigfilt, distance=min_dist)

    nbp  = (len(peaks) + len(minima)) / 2.0
    bpm  = int(np.around(60.0 * nbp / duration)) if duration > 0 else 0

    hrv_ms = 0.0
    if len(peaks) >= 3:
        rr     = np.diff(t[peaks]) * 1000.0
        hrv_ms = float(np.sqrt(np.mean(np.diff(rr) ** 2)))

    peak_times  = t[peaks].tolist()  if len(peaks)  else []
    trough_times = t[minima].tolist() if len(minima) else []
    peak_vals   = props["peak_heights"].tolist() if len(peaks) else []
    trough_vals = (-sigfilt[minima]).tolist()    if len(minima) else []

    return bpm, hrv_ms, peak_times, trough_times, peak_vals, trough_vals


# ── Video processing ──────────────────────────────────────────────────────────

def _extract_green_forehead(frame: np.ndarray):
    """Return mean forehead green-channel value, or None if no face found."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    roi = frame[y : y + int(h * 0.40), x + int(w * 0.20) : x + int(w * 0.80)]
    if roi.size == 0:
        return None
    _, g, _ = cv2.split(roi)
    return float(np.mean(g))


def process_video(path: str) -> dict:
    """
    Extract forehead green signal from video, run full DSP pipeline,
    return structured analysis result.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration     = total_frames / fps

    if duration < 5.0:
        cap.release()
        raise ValueError("Video is too short. Please provide at least 5 seconds of footage.")
    if duration > 120.0:
        cap.release()
        raise ValueError("Video is too long. Maximum supported duration is 120 seconds.")

    raw_green = []
    frame_c   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        g = _extract_green_forehead(frame)
        if g is None and raw_green:
            g = raw_green[-1]          # hold-last interpolation
        elif g is None:
            g = 0.0
        raw_green.append(g)
        frame_c += 1

    cap.release()

    if len(raw_green) < int(fps * 5):
        raise ValueError("Not enough valid face frames. Ensure face is clearly visible.")

    sig = _normalise(np.asarray(raw_green, dtype=float))

    # Resample ×2.1 for finer freq resolution
    n_resamp = (len(sig) * 2) + int(0.1 * len(sig))
    sig_r    = resample(sig, n_resamp)
    n        = len(sig_r)
    dt       = duration / n
    t        = np.linspace(0, duration, n)

    sigfilt, freq, PSD_o, PSD_f = _fft_bandpass(sig_r, dt)
    bpm, hrv_ms, peak_t, trough_t, peak_v, trough_v = _estimate_bpm(sigfilt, t, duration)
    snr_db = abs(_snr(sig_r, sigfilt))

    # Sliding-window BPM trace
    fps_r        = n / duration
    win_samples  = int(WINDOW_S * fps_r)
    step_samples = max(int(win_samples * (1.0 - OVERLAP)), 1)
    bpm_trace    = []
    time_pts     = []
    smoothed_bpm = None
    start        = 0

    while start + win_samples <= n:
        chunk   = _normalise(sig_r[start : start + win_samples])
        sfilt, _, _, _ = _fft_bandpass(chunk, WINDOW_S / len(chunk))
        t_c     = np.linspace(0, WINDOW_S, len(sfilt))
        b, _, _, _, _, _ = _estimate_bpm(sfilt, t_c, WINDOW_S)
        smoothed_bpm = (EMA_ALPHA * b + (1 - EMA_ALPHA) * smoothed_bpm
                        if smoothed_bpm is not None else float(b))
        bpm_trace.append(round(smoothed_bpm, 1))
        time_pts.append(round((start + win_samples / 2) / fps_r, 2))
        start += step_samples

    # Downsample signal traces for JSON transport (max 500 pts each)
    def _ds(arr, max_pts=500):
        a = np.asarray(arr)
        if len(a) <= max_pts:
            return a.tolist()
        idx = np.round(np.linspace(0, len(a) - 1, max_pts)).astype(int)
        return a[idx].tolist()

    half  = n // 2
    L     = np.arange(1, half // 2, dtype=int)
    f_ds  = _ds(freq[L])
    po_ds = _ds(PSD_o[L])
    pf_ds = _ds(PSD_f[L])

    t_ds  = _ds(t)
    sf_ds = _ds(sigfilt)

    # BPM interpretation
    if bpm < 50:
        interp = "Bradycardic (low)"
        interp_color = "#3B82F6"
    elif bpm <= 100:
        interp = "Normal sinus rhythm"
        interp_color = "#22C55E"
    elif bpm <= 120:
        interp = "Slightly elevated"
        interp_color = "#F59E0B"
    else:
        interp = "Tachycardic (high)"
        interp_color = "#EF4444"

    return {
        "summary": {
            "bpm":            bpm,
            "hrv_ms":         round(hrv_ms, 1),
            "snr_db":         round(snr_db, 3),
            "duration_s":     round(duration, 2),
            "frames":         frame_c,
            "fps":            round(fps, 2),
            "interpretation": interp,
            "interp_color":   interp_color,
        },
        "waveform": {
            "time":    t_ds,
            "signal":  sf_ds,
            "peaks":   {"times": peak_t,   "values": peak_v},
            "troughs": {"times": trough_t, "values": trough_v},
        },
        "psd": {
            "freq":        f_ds,
            "psd_orig":    po_ds,
            "psd_filt":    pf_ds,
            "freq_min":    FREQ_MIN,
            "freq_max":    FREQ_MAX,
        },
        "bpm_trace": {
            "times":  time_pts,
            "values": bpm_trace,
            "mean":   round(float(np.mean(bpm_trace)), 1) if bpm_trace else bpm,
        },
        "raw_signal": _ds(raw_green, 500),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "cascade_loaded": not face_cascade.empty()}


@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    # Validate content type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Accepted: mp4, webm, mov, avi."
        )

    # Read and size-check
    data = await file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_MB} MB."
        )

    # Write to temp file (OpenCV needs a path)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = process_video(tmp_path)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal processing error.")
    finally:
        os.unlink(tmp_path)
