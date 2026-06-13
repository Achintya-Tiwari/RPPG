# RPPG — Remote Heart Rate Estimation from Video

Estimate heart rate (BPM) from facial or fingertip video using classical signal processing — no machine learning, no wearable sensors.

The pipeline detects a face via Haar Cascade, extracts the green-channel intensity frame by frame, applies an FFT bandpass filter (0.67–3 Hz → 40–180 BPM), and finds the dominant pulse frequency.

![rPPG pipeline flowchart] : refer the rppg_pipeline_flowchart.png file

---

## Repository Structure

```
RPPG/
├── app.py                  ← Streamlit web app (quickest way to try it)
├── sigMain.py              ← Signal processing module (FFT filter, peak detection, HRV)
├── SNR.py                  ← Signal-to-noise ratio helper
├── requirements.txt        ← Python dependencies for the Streamlit app
│
├── rppg-app/               ← Full-stack web app (FastAPI + vanilla JS frontend)
│   ├── docker-compose.yml
│   ├── backend/
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── frontend/
│       └── index.html
│
├── Face rppg.ipynb         ← Jupyter notebook (face-based rPPG, original research)
├── Fingertip rppg.py       ← Fingertip HSV V-channel script (original research)
├── Samples/                ← Place your test videos here (not tracked by git)
├── flowchart.svg           ← Pipeline flowchart (shown in README)
├── face rppg flowchart.png ← Original hand-drawn flowchart
├── results.png
└── LICENSE
```

---

## Quick Start — Streamlit App

The fastest way to run the project. Works on Windows, macOS, and Linux.

### Prerequisites

- Python 3.9 or higher
- pip
- A webcam (optional, for live mode)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/Achintya-Tiwari/RPPG.git
cd RPPG

# 2. Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload a video of a face (or use your webcam) and click **Analyze**.

---

## Full-Stack App — Docker (FastAPI + Frontend)

A more polished version with a custom frontend, REST API, and Docker deployment.

### Prerequisites

- Docker and Docker Compose

### Steps

```bash
# 1. Clone the repo (skip if already done)
git clone https://github.com/Achintya-Tiwari/RPPG.git
cd RPPG/rppg-app

# 2. Build and start both services
docker compose up --build

# 3. Open in browser
#    Frontend  →  http://localhost:3000
#    API docs  →  http://localhost:8000/docs
```

To stop: `Ctrl+C` or `docker compose down`.

### Running Without Docker

```bash
cd rppg-app/backend
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `rppg-app/frontend/index.html` in a browser. If CORS errors occur, serve the frontend with:

```bash
cd rppg-app/frontend
python -m http.server 3000
```

---

## Running the Original Research Scripts

These are the standalone scripts from the initial research phase.

### Face-based (Jupyter Notebook)

```bash
pip install opencv-python scipy matplotlib numpy
jupyter notebook "Face rppg.ipynb"
```

Place your test video in a `Samples/` folder and update the file path inside the notebook.

### Fingertip-based

```bash
pip install opencv-python scipy matplotlib numpy
python "Fingertip rppg.py"
```

Expects a video at `Samples/82.mp4` — update the path in the script to point to your own video.

---

## Video Tips for Best Results

| Factor | Recommendation |
|---|---|
| Duration | 10–60 seconds (longer = more accurate) |
| Format | MP4, AVI, MOV, or WebM |
| Lighting | Even, frontal light — avoid backlighting or flickering |
| Face | Clearly visible, minimal head movement |
| Fingertip | Press fingertip lightly against the camera lens with flash on |

---

## How It Works

1. **Face detection** — Haar Cascade locates the face (or forehead ROI) in each frame
2. **Signal extraction** — Mean green-channel intensity (face) or HSV V-channel (fingertip) per frame
3. **Resampling** — Signal is upsampled 2.1× for finer frequency resolution
4. **FFT bandpass** — Frequencies outside 0.67–3.0 Hz are zeroed out
5. **Peak detection** — `scipy.signal.find_peaks` identifies heartbeat peaks
6. **BPM** — Dominant frequency × 60 gives beats per minute
7. **HRV** — RMSSD computed from successive RR-interval differences

---

## Limitations

- Requires a clearly visible face throughout the video
- Performance degrades with strong motion, poor lighting, or occlusion
- Results are **indicative only** — this is not a medical device
- Single-face detection only

---

## License

MIT — see [LICENSE](LICENSE) for details.