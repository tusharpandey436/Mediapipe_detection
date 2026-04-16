"""
Advanced AI Eye Tracker  v4
================================
Fixes over v3:
  1. EAR now uses NORMALIZED landmarks (0-1 range) → correct 0.15–0.35 values
     Previously used pixel coords → EAR was 0.89/0.96 → threshold 0.20 never triggered
  2. Blink detection: EAR drop below threshold on NORMALIZED values works correctly
  3. Sustained-direction windows: eye/head must hold a direction for ~3 s before
     it counts as a real distraction (natural movement during conversation ignored)
  4. Head sensitivity reduced: larger angle dead-zone (±18° yaw, ±15° pitch)
  5. Eye sensitivity reduced: wider center zone (0.35–0.65 x, 0.35–0.65 y)
  6. Attention model: only penalizes SUSTAINED off-center, not momentary glances
  7. Blink rate tracking: blinks/min displayed (normal = 12–20/min)
  8. Auto-calibration runs silently for first 60 frames at startup
  9. Microsleep: uses correct normalized EAR, triggering at real eye-close events
 10. Focus% now reflects actual attention time correctly

Keyboard:
  ESC   – quit
  C     – re-run EAR calibration (look straight, keep eyes natural)
  H     – toggle heatmap window
  R     – reset all counters
  S     – save session JSON now
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
# EAR (Eye Aspect Ratio) — normalized landmark range: open ~0.25–0.35, closed <0.20
EAR_CLOSED_FIXED     = 0.20   # fallback before calibration
EAR_CLOSED_RATIO     = 0.70   # calibrated threshold = open_baseline × this
CALIB_FRAMES_AUTO    = 60     # silent auto-calib at startup
CALIB_FRAMES_MANUAL  = 90     # manual C-key calib

# Blink
BLINK_CONSEC_FRAMES  = 2      # EAR must be below threshold for this many frames

# Microsleep
MICROSLEEP_MS        = 500    # eye closure longer than this = microsleep

# Sustained-direction gate (seconds before direction is "confirmed" distraction)
EYE_SUSTAIN_S        = 2.5    # eye must point same way for 2.5 s
HEAD_SUSTAIN_S       = 3.0    # head must point same way for 3.0 s

# Eye classification dead-zones (wider = less sensitive)
EYE_X_LEFT   = 0.35
EYE_X_RIGHT  = 0.65
EYE_Y_UP     = 0.35
EYE_Y_DOWN   = 0.65

# Head pose thresholds (degrees) — wider dead-zone than v3
YAW_THRESH   = 18
PITCH_THRESH = 15

# Attention
ATTN_DECAY_HEAD   = 0.97   # per-frame penalty for sustained head distraction
ATTN_DECAY_EYE    = 0.99   # per-frame penalty for sustained eye distraction
ATTN_RECOVER      = 1.025  # per-frame recovery when focused
ATTN_FAST_DROP    = 0.90   # when no face detected

# Smoothing buffers
EYE_SMOOTH   = 7
HEAD_SMOOTH  = 9

# PERCLOS
PERCLOS_WINDOW = 90   # ~3 s at 30 fps

# Heatmap
HEATMAP_DECAY    = 0.997
HEATMAP_WIN_SIZE = (320, 240)

# Alerts
ALERT_COOLDOWN_S = 6.0

SESSION_FILE = "session_log.json"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ══════════════════════════════════════════════
# LANDMARK INDICES
# ══════════════════════════════════════════════
L_IRIS, R_IRIS = 468, 473

# EAR: 6 points per eye (standard Soukupová ordering)
# p0=left corner, p1=top-left, p2=top-right, p3=right corner, p4=bot-right, p5=bot-left
L_EAR_PTS = [33, 160, 158, 133, 153, 144]
R_EAR_PTS = [362, 385, 387, 263, 373, 380]

# Iris bounding points
L_EYE_LEFT,  L_EYE_RIGHT = 33, 133
L_EYE_TOP,   L_EYE_BOT   = 159, 145
R_EYE_LEFT,  R_EYE_RIGHT = 362, 263
R_EYE_TOP,   R_EYE_BOT   = 386, 374

# Head pose
NOSE_TIP  = 1
CHIN      = 152
L_TEMPLE  = 234
R_TEMPLE  = 454
L_CORNER  = 33
R_CORNER  = 263

# ══════════════════════════════════════════════
# GEOMETRY — all in NORMALIZED coords (0-1)
# ══════════════════════════════════════════════
def calc_ear_norm(lm, pts):
    """
    EAR using normalized landmark coordinates.
    Result is in the correct 0.15–0.35 range (not pixel-scale).
    """
    p = np.array([[lm[i].x, lm[i].y] for i in pts])
    # Vertical distances
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    # Horizontal distance
    hz = np.linalg.norm(p[0] - p[3])
    return (v1 + v2) / (2.0 * hz + 1e-6)


def calc_head_pose(lm, h, w):
    """solvePnP head pose. Returns (yaw, pitch, roll) in degrees."""
    img_pts = np.array([
        [lm[NOSE_TIP].x * w,  lm[NOSE_TIP].y * h],
        [lm[CHIN].x * w,      lm[CHIN].y * h],
        [lm[L_CORNER].x * w,  lm[L_CORNER].y * h],
        [lm[R_CORNER].x * w,  lm[R_CORNER].y * h],
        [lm[L_TEMPLE].x * w,  lm[L_TEMPLE].y * h],
        [lm[R_TEMPLE].x * w,  lm[R_TEMPLE].y * h],
    ], dtype=np.float64)

    mdl_pts = np.array([
        [  0.0,    0.0,   0.0],
        [  0.0,  -63.6, -12.5],
        [-43.3,   32.7, -26.0],
        [ 43.3,   32.7, -26.0],
        [-57.5,  -10.0, -56.0],
        [ 57.5,  -10.0, -56.0],
    ], dtype=np.float64)

    f   = w
    cam = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(mdl_pts, img_pts, cam, np.zeros((4,1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    pitch = np.degrees(np.arcsin(np.clip(-rmat[2,0], -1, 1)))
    yaw   = np.degrees(np.arctan2(rmat[2,1], rmat[2,2]))
    roll  = np.degrees(np.arctan2(rmat[1,0], rmat[0,0]))
    return yaw, pitch, roll


def iris_ratios(lm, iris_id, lx, rx, ty, by):
    xr = (lm[iris_id].x - lm[lx].x) / (lm[rx].x - lm[lx].x + 1e-6)
    yr = (lm[iris_id].y - lm[ty].y) / (lm[by].y  - lm[ty].y + 1e-6)
    return xr, yr


def classify_eye(xr, yr):
    if xr < EYE_X_LEFT:  return "Left"
    if xr > EYE_X_RIGHT: return "Right"
    if yr < EYE_Y_UP:    return "Up"
    if yr > EYE_Y_DOWN:  return "Down"
    return "Center"


def classify_head(yaw, pitch):
    if   yaw   < -YAW_THRESH:   return "Left"
    elif yaw   >  YAW_THRESH:   return "Right"
    elif pitch < -PITCH_THRESH: return "Up"
    elif pitch >  PITCH_THRESH: return "Down"
    return "Center"


def buf_mode(buf):
    return max(set(buf), key=buf.count) if buf else "Center"

# ══════════════════════════════════════════════
# SUSTAINED DIRECTION TRACKER
# ══════════════════════════════════════════════
class SustainedDir:
    """
    Only emits a direction change after it has been stable for `sustain_s` seconds.
    Momentary glances / natural head movement during conversation are ignored.
    """
    def __init__(self, sustain_s):
        self.sustain_s   = sustain_s
        self.confirmed   = "Center"
        self._candidate  = "Center"
        self._since      = time.time()

    def update(self, raw):
        now = time.time()
        if raw == self._candidate:
            if (now - self._since) >= self.sustain_s:
                self.confirmed = self._candidate
        else:
            self._candidate = raw
            self._since     = now
        return self.confirmed

# ══════════════════════════════════════════════
# HUD HELPERS
# ══════════════════════════════════════════════
def put(frame, txt, x, y, color=(255,255,255), scale=0.52, thick=1):
    cv2.putText(frame, txt, (x,y), FONT, scale, color, thick, cv2.LINE_AA)

def draw_bar(frame, x, y, bw, val, max_val, color, label):
    filled = int(bw * min(max(val, 0), max_val) / max_val)
    cv2.rectangle(frame, (x, y-13), (x+bw, y+1), (30,30,30), -1)
    if filled > 0:
        cv2.rectangle(frame, (x, y-13), (x+filled, y+1), color, -1)
    cv2.rectangle(frame, (x, y-13), (x+bw, y+1), (70,70,70), 1)
    put(frame, label, x, y-17, (160,160,160), 0.38)
    put(frame, f"{val:.1f}", x+bw+5, y-1, color, 0.42)

FACE_COLORS = [(0,255,80),(255,160,0),(80,180,255),(255,80,200)]

# ══════════════════════════════════════════════
# SESSION
# ══════════════════════════════════════════════
session_start  = time.time()
session_events = []
_export        = {}

def log_event(kind, detail=""):
    session_events.append({
        "t": round(time.time()-session_start, 2),
        "kind": kind, "detail": detail
    })

def save_session():
    with open(SESSION_FILE, "w") as f:
        json.dump({"duration_s": round(time.time()-session_start,1),
                   "events": session_events, **_export}, f, indent=2)
    print(f"[Session saved → {SESSION_FILE}]")

# ══════════════════════════════════════════════
# RUNTIME STATE
# ══════════════════════════════════════════════
# Counts
blink_count = microsleep_count = 0
eye_l = eye_r = eye_u = eye_d = 0
head_l = head_r = head_u = head_d = 0

# Blink state machine
blink_consec   = 0       # consecutive frames below EAR threshold
blink_flag     = False   # True = currently in a blink
eyes_closed_t  = None    # time eyes first closed (for microsleep)

# EAR calibration
ear_open_base  = None    # set during calib
ear_thresh     = EAR_CLOSED_FIXED
calib_buf      = []
calib_mode     = False
auto_calibrated = False
calib_msg      = ""
calib_msg_end  = 0.0

# Sustained trackers
eye_sustain  = SustainedDir(EYE_SUSTAIN_S)
head_sustain = SustainedDir(HEAD_SUSTAIN_S)

# Smoothing buffers (for raw classification before sustain gate)
eye_raw_buf  = deque(maxlen=EYE_SMOOTH)
head_raw_buf = deque(maxlen=HEAD_SMOOTH)

# PERCLOS
perclos_w = deque(maxlen=PERCLOS_WINDOW)

# Attention
attention   = 70.0
total_frames = focus_frames = 0

# Heatmap
heatmap_acc  = None
show_heatmap = True

# Blink rate
blink_rate_buf = deque(maxlen=300)   # 1-per-frame: 1 if blink just detected else 0
last_blink_ts  = []                  # timestamps of last blinks for rate calc

# Alert state
alert_last = {}
def maybe_alert(key):
    now = time.time()
    if now - alert_last.get(key, 0) > ALERT_COOLDOWN_S:
        alert_last[key] = now
        return True
    return False

# Safe EAR defaults (for HUD before face detected)
ear_l_val = ear_r_val = EAR_CLOSED_FIXED + 0.05
yaw = pitch = roll = 0.0
eye_dir_confirmed = head_dir_confirmed = "Center"
perclos = 0.0

frame_times = deque(maxlen=30)

# ══════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════
cap = cv2.VideoCapture(0)

with mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
) as face_mesh:

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        if heatmap_acc is None:
            heatmap_acc = np.zeros((h, w), dtype=np.float32)

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        total_frames  += 1
        face_detected  = False
        alert_msgs     = []
        heatmap_acc   *= HEATMAP_DECAY

        if results.multi_face_landmarks:
            face_detected = True

            # ── Multi-face bounding boxes ──
            for fi, fl in enumerate(results.multi_face_landmarks):
                lm  = fl.landmark
                col = FACE_COLORS[fi % len(FACE_COLORS)]
                xs  = [int(p.x*w) for p in lm]
                ys  = [int(p.y*h) for p in lm]
                cv2.rectangle(frame, (min(xs),min(ys)), (max(xs),max(ys)), col, 1)
                put(frame, f"Face {fi+1}", min(xs), min(ys)-5, col, 0.42)

            # ── Primary face ──
            lm = results.multi_face_landmarks[0].landmark

            # ── EAR (normalized coords) ──
            ear_l_val = calc_ear_norm(lm, L_EAR_PTS)
            ear_r_val = calc_ear_norm(lm, R_EAR_PTS)
            avg_ear   = (ear_l_val + ear_r_val) / 2.0

            # ── Auto-calibration (first CALIB_FRAMES_AUTO frames) ──
            if not auto_calibrated and not calib_mode:
                calib_buf.append(avg_ear)
                if len(calib_buf) >= CALIB_FRAMES_AUTO:
                    ear_open_base  = float(np.percentile(calib_buf, 75))
                    ear_thresh     = ear_open_base * EAR_CLOSED_RATIO
                    auto_calibrated = True
                    calib_buf.clear()
                    calib_msg      = f"Auto-calib done: open={ear_open_base:.3f} thr={ear_thresh:.3f}"
                    calib_msg_end  = time.time() + 4.0
                    log_event("auto_calib", f"open={ear_open_base:.3f} thr={ear_thresh:.3f}")

            # ── Manual calibration ──
            if calib_mode:
                calib_buf.append(avg_ear)
                if len(calib_buf) >= CALIB_FRAMES_MANUAL:
                    ear_open_base  = float(np.percentile(calib_buf, 75))
                    ear_thresh     = ear_open_base * EAR_CLOSED_RATIO
                    calib_mode     = False
                    calib_buf.clear()
                    calib_msg      = f"Calibrated: open={ear_open_base:.3f} thr={ear_thresh:.3f}"
                    calib_msg_end  = time.time() + 5.0
                    log_event("manual_calib", f"open={ear_open_base:.3f} thr={ear_thresh:.3f}")

            eyes_closed = avg_ear < ear_thresh
            perclos_w.append(1 if eyes_closed else 0)
            perclos = (sum(perclos_w) / len(perclos_w)) * 100.0

            # ── Blink state machine ──
            if eyes_closed:
                blink_consec += 1
                if eyes_closed_t is None:
                    eyes_closed_t = time.time()
                # Count blink when threshold first crossed for BLINK_CONSEC_FRAMES
                if blink_consec == BLINK_CONSEC_FRAMES and not blink_flag:
                    blink_count += 1
                    blink_flag   = True
                    last_blink_ts.append(time.time())
                    log_event("blink", f"ear={avg_ear:.3f}")
                # Microsleep check
                if eyes_closed_t and (time.time()-eyes_closed_t)*1000 > MICROSLEEP_MS:
                    if maybe_alert("microsleep"):
                        microsleep_count += 1
                        log_event("microsleep",
                                  f"{int((time.time()-eyes_closed_t)*1000)}ms")
            else:
                blink_consec  = 0
                blink_flag    = False
                eyes_closed_t = None

            # ── Blink rate (blinks/min) ──
            now = time.time()
            last_blink_ts = [t for t in last_blink_ts if now - t < 60.0]
            blink_rate = len(last_blink_ts)  # blinks in last 60 s

            # ── Iris direction ──
            xr_l, yr_l = iris_ratios(lm, L_IRIS, L_EYE_LEFT, L_EYE_RIGHT,
                                      L_EYE_TOP, L_EYE_BOT)
            xr_r, yr_r = iris_ratios(lm, R_IRIS, R_EYE_LEFT, R_EYE_RIGHT,
                                      R_EYE_TOP, R_EYE_BOT)
            xr_avg = (xr_l + xr_r) / 2.0
            yr_avg = (yr_l + yr_r) / 2.0

            # Raw classification → smoothing buffer → sustained gate
            raw_eye = classify_eye(xr_avg, yr_avg)
            eye_raw_buf.append(raw_eye)
            smooth_eye = buf_mode(list(eye_raw_buf))
            prev_eye   = eye_dir_confirmed
            eye_dir_confirmed = eye_sustain.update(smooth_eye)

            if eye_dir_confirmed != prev_eye:
                if eye_dir_confirmed == "Left":  eye_l += 1
                elif eye_dir_confirmed == "Right": eye_r += 1
                elif eye_dir_confirmed == "Up":    eye_u += 1
                elif eye_dir_confirmed == "Down":  eye_d += 1
                log_event("eye", eye_dir_confirmed)

            # ── Iris dots ──
            for iid in [L_IRIS, R_IRIS]:
                cv2.circle(frame, (int(lm[iid].x*w), int(lm[iid].y*h)),
                           4, (0,255,255), -1)

            # ── Heatmap ──
            gx = np.clip(int(lm[L_IRIS].x*w), 0, w-1)
            gy = np.clip(int(lm[L_IRIS].y*h), 0, h-1)
            cv2.circle(heatmap_acc, (gx,gy), 18, 1.0, -1)

            # ── Head pose ──
            try:
                yaw, pitch, roll = calc_head_pose(lm, h, w)
            except Exception:
                yaw = pitch = roll = 0.0

            raw_head = classify_head(yaw, pitch)
            head_raw_buf.append(raw_head)
            smooth_head = buf_mode(list(head_raw_buf))
            prev_head   = head_dir_confirmed
            head_dir_confirmed = head_sustain.update(smooth_head)

            if head_dir_confirmed != prev_head:
                if head_dir_confirmed == "Left":  head_l += 1
                elif head_dir_confirmed == "Right": head_r += 1
                elif head_dir_confirmed == "Up":    head_u += 1
                elif head_dir_confirmed == "Down":  head_d += 1
                log_event("head", head_dir_confirmed)

            # ── Head pose arrows on nose ──
            nx, ny = int(lm[NOSE_TIP].x*w), int(lm[NOSE_TIP].y*h)
            cv2.circle(frame, (nx,ny), 3, (255,80,0), -1)
            cv2.arrowedLine(frame, (nx,ny),
                            (int(nx + yaw*1.2), ny), (255,80,0), 1, tipLength=0.3)
            cv2.arrowedLine(frame, (nx,ny),
                            (nx, int(ny + pitch*1.2)), (80,200,255), 1, tipLength=0.3)

        else:
            perclos   = 0.0
            blink_rate = len([t for t in last_blink_ts
                               if time.time()-t < 60.0])

        # ── Attention (only penalizes CONFIRMED sustained distraction) ──
        if not face_detected:
            attention *= ATTN_FAST_DROP
        elif head_dir_confirmed != "Center":
            attention *= ATTN_DECAY_HEAD
        elif eye_dir_confirmed != "Center":
            attention *= ATTN_DECAY_EYE
        else:
            attention = min(100.0, attention * ATTN_RECOVER)
        attention = max(0.0, attention)

        if face_detected and head_dir_confirmed == "Center" and eye_dir_confirmed == "Center":
            focus_frames += 1

        # ── Alerts ──
        if blink_rate < 8 and total_frames > 300 and maybe_alert("low_blink"):
            alert_msgs.append(("Low blink rate — eye strain?", (0,120,255)))
            log_event("alert","low_blink_rate")
        if blink_count > 30 and maybe_alert("fatigue"):
            alert_msgs.append(("Fatigue detected", (0,60,255)))
            log_event("alert","fatigue")
        if perclos > 15 and maybe_alert("drowsy"):
            alert_msgs.append((f"Drowsy  PERCLOS {perclos:.0f}%", (0,40,200)))
            log_event("alert","drowsy")
        if microsleep_count >= 1 and maybe_alert("microsleep_w"):
            alert_msgs.append(("Microsleep detected!", (0,20,180)))
            log_event("alert","microsleep_warn")
        if attention < 35 and maybe_alert("low_attn"):
            alert_msgs.append(("Low attention", (30,80,255)))
            log_event("alert","low_attention")
        if not face_detected and maybe_alert("no_face"):
            alert_msgs.append(("No face", (120,120,255)))

        # ── Gaze heatmap window ──
        if show_heatmap:
            hm_u8   = cv2.normalize(heatmap_acc, None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
            hm_col  = cv2.applyColorMap(hm_u8, cv2.COLORMAP_TURBO)
            hm_sm   = cv2.resize(hm_col, HEATMAP_WIN_SIZE)
            put(hm_sm, "Gaze Heatmap", 6, 20, (220,220,220), 0.5)
            cv2.imshow("Gaze Heatmap", hm_sm)

        # ── HUD ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (4,4), (295,330), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (4,4), (295,330), (55,55,55), 1)

        # FPS + clock
        frame_times.append(time.perf_counter()-t0)
        fps = 1.0 / (sum(frame_times)/len(frame_times) + 1e-9)
        put(frame, f"FPS {fps:.0f}   {time.strftime('%H:%M:%S')}",
            10, 22, (130,130,130), 0.44)

        # Blinks & microsleeps
        put(frame, f"Blinks: {blink_count}  Rate: {blink_rate}/min  uSleep: {microsleep_count}",
            10, 44, (255,255,255), 0.47)

        # EAR values
        ear_col = (0,60,255) if (ear_l_val < ear_thresh or ear_r_val < ear_thresh) \
                  else (200,200,0)
        put(frame, f"EAR  L:{ear_l_val:.3f}  R:{ear_r_val:.3f}  thr:{ear_thresh:.3f}",
            10, 64, ear_col, 0.45)

        # PERCLOS bar
        pc_col = (0,60,255) if perclos>20 else (0,180,255) if perclos>10 else (0,210,160)
        draw_bar(frame, 10, 95, 195, perclos, 40, pc_col, "PERCLOS %")

        # Eye
        eye_col = (0,255,255) if eye_dir_confirmed=="Center" else (0,160,255)
        pending = "" if eye_sustain.confirmed == eye_sustain._candidate \
                  else f"→{eye_sustain._candidate}"
        put(frame, f"Eye: {eye_dir_confirmed:<8}{pending}  Y:{yaw:+.0f}°P:{pitch:+.0f}°",
            10, 120, eye_col, 0.45)
        put(frame, f"  L:{eye_l} R:{eye_r} U:{eye_u} D:{eye_d}",
            10, 138, (0,180,180), 0.43)

        # Head
        head_col = (100,255,100) if head_dir_confirmed=="Center" else (80,130,255)
        ph = "" if head_sustain.confirmed == head_sustain._candidate \
             else f"→{head_sustain._candidate}"
        put(frame, f"Head: {head_dir_confirmed:<8}{ph}",
            10, 162, head_col, 0.47)
        put(frame, f"  L:{head_l} R:{head_r} U:{head_u} D:{head_d}",
            10, 180, (60,200,80), 0.43)

        # Attention bar
        a_col = (0,255,80) if attention>=70 else (0,200,255) if attention>=45 else (0,80,255)
        draw_bar(frame, 10, 215, 195, attention, 100, a_col, "Attention %")

        # Focus %
        fp = focus_frames / max(total_frames,1) * 100
        draw_bar(frame, 10, 245, 195, fp, 100, (120,255,160), "Focus %")

        # Calib feedback
        if calib_mode:
            rem = (CALIB_FRAMES_MANUAL - len(calib_buf)) / max(fps,1)
            put(frame, f"CALIBRATING {rem:.1f}s — look straight",
                10, 270, (255,220,0), 0.48)
        elif time.time() < calib_msg_end:
            put(frame, calib_msg[:38], 10, 270, (100,255,100), 0.40)

        put(frame, "C=calib  H=heatmap  R=reset  S=save  ESC",
            10, 322, (70,70,70), 0.37)

        # ── Alerts bottom-right ──
        for i, (msg, col) in enumerate(alert_msgs):
            ay = h - 18 - i*26
            (tw, _), _ = cv2.getTextSize(msg, FONT, 0.56, 1)
            ov2 = frame.copy()
            cv2.rectangle(ov2, (w-tw-18, ay-18), (w-4, ay+6), (0,0,0), -1)
            cv2.addWeighted(ov2, 0.6, frame, 0.4, 0, frame)
            put(frame, msg, w-tw-12, ay, col, 0.56, 1)

        cv2.imshow("Advanced AI Tracker  v4  [ESC=quit]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('c'), ord('C')):
            calib_mode = True
            calib_buf.clear()
            print("[Manual calibration — look straight, keep eyes natural]")
        elif key in (ord('h'), ord('H')):
            show_heatmap = not show_heatmap
            if not show_heatmap:
                cv2.destroyWindow("Gaze Heatmap")
        elif key in (ord('r'), ord('R')):
            blink_count=microsleep_count=eye_l=eye_r=eye_u=eye_d=0
            head_l=head_r=head_u=head_d=0
            attention=70.0; focus_frames=total_frames=0
            perclos_w.clear(); session_events.clear()
            last_blink_ts.clear()
            heatmap_acc[:]=0
            print("[Reset]")
        elif key in (ord('s'), ord('S')):
            save_session()

# ══════════════════════════════════════════════
# CLEANUP + SUMMARY
# ══════════════════════════════════════════════
cap.release()
cv2.destroyAllWindows()

_export.update({
    "total_frames": total_frames,  "focus_frames": focus_frames,
    "attention_end": round(attention,1),
    "blinks": blink_count,         "microsleeps": microsleep_count,
    "blink_rate_last_min": len([t for t in last_blink_ts if time.time()-t<60]),
    "ear_open_baseline": round(ear_open_base or EAR_CLOSED_FIXED,4),
    "ear_closed_thresh": round(ear_thresh,4),
    "eye": {"l":eye_l,"r":eye_r,"u":eye_u,"d":eye_d},
    "head":{"l":head_l,"r":head_r,"u":head_u,"d":head_d},
})
save_session()

def bar(v, mx=100, bw=22):
    f=int(bw*min(max(v,0),mx)/mx)
    return "█"*f+"░"*(bw-f)

fp = focus_frames / max(total_frames,1) * 100
dur= round(time.time()-session_start,1)

print("\n╔══════════════════════════════════╗")
print("║    SESSION SUMMARY  v4           ║")
print("╠══════════════════════════════════╣")
print(f"║ Duration      {dur:>8.1f} s          ║")
print(f"║ Blinks        {blink_count:>8}            ║")
print(f"║ Microsleeps   {microsleep_count:>8}            ║")
print(f"║ EAR open      {(ear_open_base or 0):>8.3f}            ║")
print(f"║ EAR threshold {ear_thresh:>8.3f}            ║")
print(f"║ Focus time    {fp:>7.1f}%            ║")
print(f"║ Attention end {attention:>7.1f}%            ║")
print(f"║ Eye  L{eye_l:>3} R{eye_r:>3} U{eye_u:>3} D{eye_d:>3}      ║")
print(f"║ Head L{head_l:>3} R{head_r:>3} U{head_u:>3} D{head_d:>3}      ║")
print(f"╠══════════════════════════════════╣")
print(f"║ Attn  {bar(attention)}  ║")
print(f"║ Focus {bar(fp)}  ║")
print("╚══════════════════════════════════╝")