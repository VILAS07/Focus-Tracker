# 🧠 Focus Tracke

> **Real-time attention monitoring using facial landmark detection — no distractions allowed.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

Focus Tracker is a lightweight, webcam-powered productivity tool that watches your face in real time and tells you — honestly — how focused you actually are. It uses **Google MediaPipe's 468-point face mesh** to measure eye aspect ratio and head pose, then visualises your attention span on a live analytics dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| 👁️ **EAR Detection** | Detects eye closure using Eye Aspect Ratio to catch drowsiness or zoning out |
| 🔄 **Head Pose Estimation** | Tracks yaw & pitch angles to detect when you look away from the screen |
| 📊 **Live Dashboard** | Real-time charts showing 60-second focus history, session summary, and per-minute trends |
| 📈 **Trend Line** | Linear regression overlay on per-minute focus to show improvement over time |
| 💾 **Auto-Save** | Dashboard snapshots saved automatically on exit; manual saves anytime with `D` |
| ⏱️ **Session Timer** | Tracks elapsed session time with pause/resume support |
| 🎨 **Dark UI** | Clean, dark-themed overlays directly on the camera feed |

---

## 🖥️ Demo

```
State:      FOCUSED | DISTRACTED | AWAY
Live Focus: [████████████░░░░░░░] 65%
Session:    00:04:32

Yaw: +3   Pitch: -12   EAR: 0.28
```

The dashboard renders as a separate window with three panels:
- **Line chart** — smoothed focus level over the last 60 seconds
- **Bar chart** — percentage of time spent Focused / Distracted / Away
- **Trend chart** — focus score per minute with a regression line

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- A working webcam
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/focus-tracker.git
cd focus-tracker

# Install dependencies
pip install opencv-python mediapipe numpy matplotlib
```

### Run

```bash
python focus_tracker.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `S` | Start / Pause the session |
| `R` | Reset all session data |
| `D` | Save current dashboard snapshot manually |
| `Q` / `ESC` | Quit (auto-saves final dashboard) |

---

## ⚙️ Configuration

All thresholds are defined at the top of `focus_tracker.py` and can be tuned to suit your face/environment:

```python
EAR_THRESH   = 0.20   # Eye closure threshold (lower = stricter)
YAW_THRESH   = 30     # Head turn angle in degrees
PITCH_THRESH = 35     # Head tilt angle in degrees

SAMPLE_RATE   = 5     # Samples per second
HISTORY_SEC   = 60    # Rolling window length (seconds)
CHART_REFRESH = 1.5   # Dashboard update interval (seconds)
```

Snapshots are saved to the `focus_snapshots/` directory by default. Change `SAVE_DIR` to use a different path.

---

## 🔍 How It Works

```
Webcam Frame
     │
     ▼
MediaPipe Face Mesh  ──►  468 3D landmarks
     │
     ├──► Eye Aspect Ratio (EAR)
     │         vertical distance / horizontal distance
     │         EAR < 0.20  →  eyes closed / drowsy
     │
     └──► Head Pose (Yaw / Pitch)
               nose vs. ear midpoint geometry
               |Yaw| > 30°  →  looking away

State: FOCUSED | DISTRACTED | AWAY
     │
     ▼
Sampled 5×/sec  →  rolling deque (60 s)
     │
     ▼
Matplotlib dashboard rendered off-screen (Agg backend)
→ converted to BGR → displayed via OpenCV window
```

### State Definitions

| State | Condition |
|---|---|
| `FOCUSED` | Face detected, EAR ≥ 0.20, |Yaw| ≤ 30° |
| `DISTRACTED` | Face detected but eyes closed **or** head turned |
| `AWAY` | No face detected in frame |

---

## 📁 Project Structure

```
focus-tracker/
├── focus_tracker.py       # Main application
├── focus_snapshots/       # Auto-created; stores dashboard PNGs
│   ├── dashboard_manual_20240101_120000.png
│   └── dashboard_final_20240101_123045.png
└── README.md
```

---

## 🛠️ Dependencies

| Library | Purpose |
|---|---|
| `opencv-python` | Webcam capture, frame rendering, window management |
| `mediapipe` | Face mesh landmark detection (468 points) |
| `numpy` | Vector math for EAR and pose calculations |
| `matplotlib` | Off-screen dashboard chart rendering (Agg backend) |

---

## 🔮 Roadmap

- [ ] Blink rate tracking for fatigue detection
- [ ] CSV / JSON export of session data
- [ ] Desktop notification when focus drops below threshold
- [ ] GUI settings panel (adjust thresholds without editing code)
- [ ] Multi-session comparison view
- [ ] Optional audio alerts

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push and open a PR

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Google MediaPipe](https://mediapipe.dev/) for the incredible face mesh model
- [OpenCV](https://opencv.org/) for real-time computer vision primitives
- Inspired by academic work on Eye Aspect Ratio (EAR) by Soukupová & Čech (2016)

---

<p align="center">Built with 👁️ and ☕ — because your focus deserves better than a Pomodoro timer.</p>
