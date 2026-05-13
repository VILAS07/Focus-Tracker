import cv2
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import collections
import os
from datetime import datetime

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

NOSE      = 1
CHIN      = 152
L_EYE_TOP = 159; L_EYE_BOT = 145
R_EYE_TOP = 386; R_EYE_BOT = 374
L_EYE_HOR = (33, 133); R_EYE_HOR = (362, 263)

EAR_THRESH   = 0.20
YAW_THRESH   = 30
PITCH_THRESH = 35

# ── Save directory ──────────────────────────────────────────────────────────
SAVE_DIR = "focus_snapshots"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_dashboard(img, label="manual"):
    """Save dashboard image to focus_snapshots/ with a timestamp filename."""
    if img is None:
        print("[Save] No dashboard to save yet.")
        return
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"dashboard_{label}_{ts}.png")
    cv2.imwrite(path, img)
    print(f"[Save] Dashboard saved → {path}")
    return path
# ────────────────────────────────────────────────────────────────────────────

def landmark_arr(lm, idx, w, h):
    p = lm[idx]
    return np.array([p.x * w, p.y * h, p.z * w])

def eye_aspect_ratio(lm, top, bot, hor, w, h):
    t = landmark_arr(lm, top, w, h)
    b = landmark_arr(lm, bot, w, h)
    l = landmark_arr(lm, hor[0], w, h)
    r = landmark_arr(lm, hor[1], w, h)
    return np.linalg.norm(t - b) / (np.linalg.norm(l - r) + 1e-6)

def head_pose_angles(lm, w, h):
    nose    = landmark_arr(lm, NOSE, w, h)
    chin    = landmark_arr(lm, CHIN, w, h)
    l_ear   = landmark_arr(lm, 234, w, h)
    r_ear   = landmark_arr(lm, 454, w, h)
    mid_ear = (l_ear + r_ear) / 2
    yaw   = np.degrees(np.arctan2(nose[0] - mid_ear[0], 200))
    pitch = np.degrees(np.arctan2(nose[1] - chin[1], 200))
    return yaw, pitch

SAMPLE_RATE   = 5
HISTORY_SEC   = 60
CHART_REFRESH = 1.5

running       = False
session_start = None

focus_history          = collections.deque(maxlen=HISTORY_SEC * SAMPLE_RATE)
per_minute_focus       = []
current_minute_samples = []
current_minute_start   = None

session_focused    = 0
session_distracted = 0
session_away       = 0

last_sample_time = 0
last_chart_time  = 0
dashboard_img    = None

# Feedback overlay: (message, expiry_time)
save_feedback = ("", 0)

DASH_W, DASH_H = 900, 620

def build_dashboard():
    fig = plt.figure(figsize=(DASH_W/100, DASH_H/100), dpi=100, facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.45, wspace=0.35,
                            left=0.08, right=0.97, top=0.90, bottom=0.10)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    fig.suptitle("Focus Tracker Dashboard", color="#e0e0ff",
                 fontsize=13, fontweight="bold", y=0.97)

    ax1.set_title("Focus Level — Last 60 s", color="#aaaacc", fontsize=9, pad=6)
    ax1.set_ylim(-5, 105)
    ax1.set_ylabel("Focus %", color="#aaaacc", fontsize=8)
    ax1.axhline(80, color="#44ff88", linewidth=0.6, linestyle="--", alpha=0.5)
    ax1.text(0.01, 82, "good", transform=ax1.get_yaxis_transform(),
             color="#44ff88", fontsize=7, alpha=0.7)

    if len(focus_history) > 2:
        ys = np.array(list(focus_history), dtype=float) * 100
        xs = np.linspace(0, min(len(ys)/SAMPLE_RATE, HISTORY_SEC), len(ys))
        kernel = np.ones(SAMPLE_RATE) / SAMPLE_RATE
        ys_sm  = np.convolve(ys, kernel, mode="same")[:len(xs)]
        ax1.fill_between(xs, ys_sm, alpha=0.25, color="#5588ff")
        ax1.plot(xs, ys_sm, color="#7799ff", linewidth=1.8)
        ax1.set_xlim(0, max(xs[-1], 10))
    else:
        ax1.text(0.5, 0.5, "Session not started", transform=ax1.transAxes,
                 ha="center", va="center", color="#555577", fontsize=10)
    ax1.set_xlabel("seconds ago  ←", color="#aaaacc", fontsize=7)

    ax2.set_title("Session Summary", color="#aaaacc", fontsize=9, pad=6)
    total = session_focused + session_distracted + session_away or 1
    vals  = [session_focused/total*100, session_distracted/total*100, session_away/total*100]
    bars  = ax2.bar(["Focused", "Distracted", "Away"], vals,
                    color=["#44cc88", "#ff6655", "#ffaa33"], width=0.5, zorder=3)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("%", color="#aaaacc", fontsize=8)
    ax2.yaxis.grid(True, color="#222244", linewidth=0.5, zorder=0)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val:.0f}%", ha="center", color="#dddddd", fontsize=8)

    ax3.set_title("Focus per Minute  (improvement trend)", color="#aaaacc", fontsize=9, pad=6)
    if per_minute_focus:
        labels = [p[0] for p in per_minute_focus]
        scores = [p[1] for p in per_minute_focus]
        xs3    = range(len(scores))
        ax3.bar(xs3, scores, color="#5577ff", alpha=0.7, width=0.6, zorder=3)
        ax3.yaxis.grid(True, color="#222244", linewidth=0.5, zorder=0)
        if len(scores) >= 2:
            z  = np.polyfit(list(xs3), scores, 1)
            p_ = np.poly1d(z)
            ax3.plot(list(xs3), p_(list(xs3)), color="#ffdd55", linewidth=1.5, linestyle="--")
        ax3.set_xticks(list(xs3))
        ax3.set_xticklabels(labels, fontsize=7, color="#aaaacc")
        ax3.set_ylim(0, 110)
        ax3.set_ylabel("Focus %", color="#aaaacc", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Data after 1 min", transform=ax3.transAxes,
                 ha="center", va="center", color="#555577", fontsize=10)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

print("Focus Tracker ready.  S: start/pause  R: reset  D: save dashboard  Q/ESC: quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    now = time.time()

    state = "away"

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        ear_l = eye_aspect_ratio(lm, L_EYE_TOP, L_EYE_BOT, L_EYE_HOR, w, h)
        ear_r = eye_aspect_ratio(lm, R_EYE_TOP, R_EYE_BOT, R_EYE_HOR, w, h)
        ear   = (ear_l + ear_r) / 2

        yaw, pitch = head_pose_angles(lm, w, h)

        if ear < EAR_THRESH or abs(yaw) > YAW_THRESH:
            state = "distracted"
        else:
            state = "focused"

        for idx in [NOSE, CHIN, L_EYE_TOP, R_EYE_TOP]:
            p = lm[idx]
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (100, 180, 255), -1)

        cv2.putText(frame, f"Yaw:{yaw:+.0f}  Pitch:{pitch:+.0f}  EAR:{ear:.2f}",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (120, 120, 180), 1, cv2.LINE_AA)

    if running and now - last_sample_time >= 1.0 / SAMPLE_RATE:
        last_sample_time = now
        is_focused = 1 if state == "focused" else 0
        focus_history.append(is_focused)
        current_minute_samples.append(is_focused)

        if state == "focused":        session_focused    += 1
        elif state == "distracted":   session_distracted += 1
        else:                         session_away       += 1

        if current_minute_start and now - current_minute_start >= 60:
            if current_minute_samples:
                pct = sum(current_minute_samples) / len(current_minute_samples) * 100
                per_minute_focus.append((f"m{len(per_minute_focus)+1}", pct))
            current_minute_samples = []
            current_minute_start   = now

    if now - last_chart_time >= CHART_REFRESH:
        dashboard_img   = build_dashboard()
        last_chart_time = now

    pill_color = {"focused": (40,180,80), "distracted": (50,60,220), "away": (30,140,200)}[state]
    pill_label = {"focused": "FOCUSED", "distracted": "DISTRACTED", "away": "AWAY"}[state]
    cv2.rectangle(frame, (10, 10), (200, 50), pill_color, -1)
    cv2.rectangle(frame, (10, 10), (200, 50), (255,255,255), 1)
    cv2.putText(frame, pill_label, (20, 37),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

    if running and session_start:
        elapsed = int(now - session_start)
        mins, secs = divmod(elapsed, 60)
        cv2.putText(frame, f"Session  {mins:02d}:{secs:02d}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1, cv2.LINE_AA)

    recent   = list(focus_history)[-SAMPLE_RATE*10:] if focus_history else []
    live_pct = int(sum(recent)/len(recent)*100) if recent else 0
    bar_w    = int(live_pct / 100 * 200)
    cv2.rectangle(frame, (10, 90), (210, 108), (40,40,60), -1)
    cv2.rectangle(frame, (10, 90), (10 + bar_w, 108),
                  (40,180,80) if live_pct >= 70 else (50,60,220), -1)
    cv2.putText(frame, f"Live focus: {live_pct}%", (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

    cv2.putText(frame, f"{'S: pause' if running else 'S: start'}  R: reset  D: save  Q: quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (100,100,100), 1, cv2.LINE_AA)

    if not running:
        cv2.putText(frame, "PAUSED — press S to begin",
                    (w//2 - 160, h//2), cv2.FONT_HERSHEY_DUPLEX,
                    0.85, (80, 160, 255), 2, cv2.LINE_AA)

    # ── Save feedback overlay on camera window ───────────────────────────────
    msg, expiry = save_feedback
    if now < expiry:
        cv2.rectangle(frame, (w//2 - 220, h//2 - 25), (w//2 + 220, h//2 + 25), (20, 20, 20), -1)
        cv2.rectangle(frame, (w//2 - 220, h//2 - 25), (w//2 + 220, h//2 + 25), (80, 200, 100), 2)
        cv2.putText(frame, msg, (w//2 - 210, h//2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 120), 2, cv2.LINE_AA)
    # ─────────────────────────────────────────────────────────────────────────

    cv2.imshow("Focus Tracker — Camera", frame)
    if dashboard_img is not None:
        cv2.imshow("Focus Tracker — Dashboard", dashboard_img)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        # Auto-save final dashboard on quit
        saved_path = save_dashboard(dashboard_img, label="final")
        if saved_path:
            print(f"[Save] Final dashboard auto-saved on exit.")
        break
    elif key == ord('s'):
        running = not running
        if running and session_start is None:
            session_start        = now
            current_minute_start = now
            last_sample_time     = now
    elif key == ord('d'):
        # Manual save
        saved_path = save_dashboard(dashboard_img, label="manual")
        if saved_path:
            save_feedback = (f"Saved: {os.path.basename(saved_path)}", now + 2.5)
        else:
            save_feedback = ("Nothing to save yet!", now + 2.0)
    elif key == ord('r'):
        running = False
        session_start = current_minute_start = None
        focus_history.clear()
        per_minute_focus.clear()
        current_minute_samples = []
        session_focused = session_distracted = session_away = 0
        dashboard_img   = None

cap.release()
cv2.destroyAllWindows()
face_mesh.close()