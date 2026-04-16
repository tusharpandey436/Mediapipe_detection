import mediapipe as mp
import numpy as np
from collections import deque
import time
import cv2
import json
import os

# EAR constants                                 
EAR_CLOSED_FIXED    = 0.20
EAR_CLOSED_RATIO    = 0.70
CALIB_FRAMES_AUTO   = 60
POSE_CALIB_FRAMES   = 20
BLINK_CONSEC_FRAMES = 2
MICROSLEEP_MS       = 500

# Sustained-direction gates (seconds before a direction counts as real)
EYE_SUSTAIN_S  = 0.25
HEAD_SUSTAIN_S = 0.35

# Eye / head dead-zones (Tighter = More sensitive)
EYE_X_THRESH    = 0.05
EYE_Y_THRESH    = 0.07
YAW_THRESH      = 8
PITCH_UP_THRESH = 8
PITCH_DOWN_THRESH = 10

# --- Attention model (target-based, not multiplicative) ---
# Attention moves toward the "target" at a fixed rate per update call.
# This means focus always causes recovery, distraction causes controlled decay.
ATTN_RATE_RECOVER   = 2.0
ATTN_RATE_EYE       = 1.7
ATTN_RATE_HEAD      = 2.2
ATTN_RATE_COMBINED  = 2.7
ATTN_RATE_DROWSY    = 2.8
ATTN_RATE_NO_FACE   = 3.3
ATTN_DROWSY_PERCLOS = 18.0
ATTN_MIN, ATTN_MAX  = 0.0, 100.0

# Smoothing buffers
EYE_SMOOTH   = 7
HEAD_SMOOTH  = 9
PERCLOS_WINDOW = 90

# Session output directory
SESSION_DIR = os.path.join(os.path.dirname(__file__), "sessions")

# ── Landmark indices ─────────────────────────────────────────────────
L_IRIS, R_IRIS = 468, 473
L_EAR_PTS = [33, 160, 158, 133, 153, 144]
R_EAR_PTS = [362, 385, 387, 263, 373, 380]
L_EYE_LEFT, L_EYE_RIGHT = 33, 133
L_EYE_TOP,  L_EYE_BOT   = 159, 145
R_EYE_LEFT, R_EYE_RIGHT = 362, 263
R_EYE_TOP,  R_EYE_BOT   = 386, 374
NOSE_TIP, CHIN           = 1, 152
L_TEMPLE, R_TEMPLE       = 234, 454
L_CORNER, R_CORNER       = 33, 263


# ── Geometry helpers ─────────────────────────────────────────────────
def calc_ear_norm(lm, pts):
    p = np.array([[lm[i].x, lm[i].y] for i in pts])
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    hz = np.linalg.norm(p[0] - p[3])
    return float((v1 + v2) / (2.0 * hz + 1e-6))


def calc_head_pose(lm, h, w):
    # Image points from MediaPipe
    img_pts = np.array([
        [lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h],
        [lm[CHIN].x * w,     lm[CHIN].y * h],
        [lm[L_CORNER].x * w, lm[L_CORNER].y * h],
        [lm[R_CORNER].x * w, lm[R_CORNER].y * h],
        [lm[L_TEMPLE].x * w, lm[L_TEMPLE].y * h],
        [lm[R_TEMPLE].x * w, lm[R_TEMPLE].y * h],
    ], dtype=np.float64)

    # Corrected 3D model points (OpenCV convention: X-right, Y-down, Z-forward)
    mdl_pts = np.array([
        [  0.0,    0.0,   0.0],    # NOSE_TIP
        [  0.0,   63.6, -12.5],    # CHIN (Down is +Y)
        [-43.3,  -32.7, -26.0],    # L_CORNER (Up is -Y)
        [ 43.3,  -32.7, -26.0],    # R_CORNER
        [-57.5,  -10.0, -56.0],    # L_TEMPLE
        [ 57.5,  -10.0, -56.0],    # R_TEMPLE
    ], dtype=np.float64)

    f = w
    cam = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(mdl_pts, img_pts, cam, np.zeros((4, 1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)
    
    # Decompose into Euler angles using RQDecomp (returns [pitch, yaw, roll])
    # Note: RQDecomp returns values in degrees after basic normalization
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    pitch = angles[0]
    yaw   = angles[1]
    roll  = angles[2]
    
    # Flip yaw if necessary for "Left is Left" behavior
    # In standard setup, turning right yields positive yaw
    return float(yaw), float(pitch), float(roll)


def iris_ratios(lm, iris_id, lx, rx, ty, by):
    xr = (lm[iris_id].x - lm[lx].x) / (lm[rx].x - lm[lx].x + 1e-6)
    yr = (lm[iris_id].y - lm[ty].y) / (lm[by].y  - lm[ty].y + 1e-6)
    return xr, yr


def classify_eye(xr, yr, neutral_x, neutral_y):
    dx = xr - neutral_x
    dy = yr - neutral_y

    x_score = abs(dx) / EYE_X_THRESH
    y_score = abs(dy) / EYE_Y_THRESH

    if x_score < 1.0 and y_score < 1.0:
        return "Center"

    # Pick the stronger axis so vertical movements are not swallowed by horizontal noise.
    if y_score > x_score:
        if dy <= -EYE_Y_THRESH:
            return "Up"
        if dy >= EYE_Y_THRESH:
            return "Down"
    else:
        if dx <= -EYE_X_THRESH:
            return "Left"
        if dx >= EYE_X_THRESH:
            return "Right"
    return "Center"


def classify_head(yaw, pitch, neutral_yaw, neutral_pitch):
    yaw_delta = yaw - neutral_yaw
    pitch_delta = pitch - neutral_pitch

    yaw_score = abs(yaw_delta) / YAW_THRESH
    up_score = pitch_delta / PITCH_UP_THRESH
    down_score = -pitch_delta / PITCH_DOWN_THRESH

    vertical_score = max(up_score, down_score)
    horizontal_score = yaw_score

    if vertical_score < 1.0 and horizontal_score < 1.0:
        return "Center"

    # In this pose setup, positive pitch generally means head up.
    if vertical_score > horizontal_score:
        if up_score >= 1.0:
            return "Up"
        if down_score >= 1.0:
            return "Down"
    else:
        if yaw_delta < -YAW_THRESH:
            return "Left"
        if yaw_delta > YAW_THRESH:
            return "Right"
    return "Center"


def buf_mode(buf):
    return max(set(buf), key=buf.count) if buf else "Center"


# ── Sustained direction tracker ──────────────────────────────────────
class SustainedDir:
    """Only emits a change after the direction has been held for `sustain_s`."""
    def __init__(self, sustain_s):
        self.sustain_s  = sustain_s
        self.confirmed  = "Center"
        self._candidate = "Center"
        self._since     = time.time()

    def update(self, raw, now=None):
        now = now or time.time()
        if raw == self._candidate:
            if (now - self._since) >= self.sustain_s:
                self.confirmed = self._candidate
        else:
            self._candidate = raw
            self._since     = now
        return self.confirmed


# ── Main session class ───────────────────────────────────────────────
class EyeTrackerSession:
    def __init__(self):
        self.session_start  = time.time()
        self.last_frame_t   = self.session_start

        # Counters
        self.blink_count      = 0
        self.microsleep_count = 0
        self.total_frames     = 0
        self.focus_frames     = 0

        # Attention — starts at 70, moves toward target at a rate×dt approach
        self.attention = 100.0

        # Blink state
        self.blink_consec  = 0
        self.blink_flag    = False
        self.eyes_closed_t = None

        # EAR calibration
        self.calib_buf      = []
        self.auto_calibrated= False
        self.ear_open_base  = None
        self.ear_thresh     = EAR_CLOSED_FIXED
        self.eye_center_x   = 0.5
        self.eye_center_y   = 0.5
        self.neutral_yaw    = 0.0
        self.neutral_pitch  = 0.0
        self._eye_center_samples = []
        self._head_pose_samples  = []

        # Direction trackers
        self.eye_sustain  = SustainedDir(EYE_SUSTAIN_S)
        self.head_sustain = SustainedDir(HEAD_SUSTAIN_S)
        self.eye_raw_buf  = deque(maxlen=EYE_SMOOTH)
        self.head_raw_buf = deque(maxlen=HEAD_SMOOTH)
        self._prev_eye_dir  = "Center"
        self._prev_head_dir = "Center"

        # Eye direction counts
        self.eye_l = self.eye_r = self.eye_u = self.eye_d = 0

        # Head direction counts
        self.head_l = self.head_r = self.head_u = self.head_d = 0

        # PERCLOS
        self.perclos_w = deque(maxlen=PERCLOS_WINDOW)
        self.perclos   = 0.0

        # Blink rate
        self.last_blink_ts = []

        # Session event log
        self.events = []

        # Smoothing for raw angles (EMA)
        self.smooth_yaw = 0.0
        self.smooth_pitch = 0.0
        self.ema_alpha = 0.25  # Lower = smoother but more lag

        print("Initializing MediaPipe FaceMesh...")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def _resolve_now(self, timestamp_s=None):
        if timestamp_s is None:
            return time.time()
        return self.session_start + max(float(timestamp_s), 0.0)

    def _log(self, kind, detail="", now=None):
        now = now or time.time()
        self.events.append({
            "t": round(now - self.session_start, 2),
            "kind": kind,
            "detail": detail,
        })

    def process_frame(self, rgb_image, timestamp_s=None):
        now = self._resolve_now(timestamp_s)
        dt  = now - self.last_frame_t
        dt  = min(dt, 0.5)
        self.last_frame_t = now
        self.total_frames += 1
        frame_events = []

        if self.total_frames % 50 == 0:
            print(f"[Tracker] Processing frame {self.total_frames}...")

        h, w, _ = rgb_image.shape
        results  = self.face_mesh.process(rgb_image)

        face_detected      = False
        alerts             = []
        eye_dir_confirmed  = "Center"
        head_dir_confirmed = "Center"
        av_ear             = 0.0
        yaw = pitch = roll = 0.0
        face_box = None

        if results.multi_face_landmarks:
            face_detected = True
            lm = results.multi_face_landmarks[0].landmark
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            x1 = max(0.0, min(xs))
            y1 = max(0.0, min(ys))
            x2 = min(1.0, max(xs))
            y2 = min(1.0, max(ys))
            face_box = {
                "x": round(x1, 4),
                "y": round(y1, 4),
                "w": round(max(0.0, x2 - x1), 4),
                "h": round(max(0.0, y2 - y1), 4),
            }

            # ── EAR ────────────────────────────────────────────────
            ear_l = calc_ear_norm(lm, L_EAR_PTS)
            ear_r = calc_ear_norm(lm, R_EAR_PTS)
            avg_ear = (ear_l + ear_r) / 2.0
            av_ear  = avg_ear

            # Auto-calibration (first CALIB_FRAMES_AUTO frames with a face)
            if not self.auto_calibrated:
                self.calib_buf.append(avg_ear)
                if len(self.calib_buf) >= CALIB_FRAMES_AUTO:
                    self.ear_open_base = float(np.percentile(self.calib_buf, 75))
                    self.ear_thresh    = self.ear_open_base * EAR_CLOSED_RATIO
                    self.auto_calibrated = True
                    self.calib_buf.clear()
                    self._log("auto_calib",
                              f"open={self.ear_open_base:.3f} thr={self.ear_thresh:.3f}",
                              now=now)
                    frame_events.append({"t": round(now - self.session_start, 2), "kind": "auto_calib", "detail": f"open={self.ear_open_base:.3f} thr={self.ear_thresh:.3f}"})

            eyes_closed = avg_ear < self.ear_thresh
            self.perclos_w.append(1 if eyes_closed else 0)
            self.perclos = (sum(self.perclos_w) / len(self.perclos_w)) * 100.0

            # ── Blink state machine ────────────────────────────────
            if eyes_closed:
                self.blink_consec += 1
                if self.eyes_closed_t is None:
                    self.eyes_closed_t = now
                if self.blink_consec == BLINK_CONSEC_FRAMES and not self.blink_flag:
                    self.blink_count += 1
                    self.blink_flag   = True
                    self.last_blink_ts.append(now)
                    self._log("blink", f"ear={avg_ear:.3f}", now=now)
                    frame_events.append({"t": round(now - self.session_start, 2), "kind": "blink", "detail": f"ear={avg_ear:.3f}"})
                # Microsleep
                if self.eyes_closed_t and (now - self.eyes_closed_t) * 1000 > MICROSLEEP_MS:
                    if self.blink_consec % 15 == 0:   # rate-limit duplicate events
                        self.microsleep_count += 1
                        alerts.append("microsleep")
                        self._log("microsleep", now=now)
                        frame_events.append({"t": round(now - self.session_start, 2), "kind": "microsleep", "detail": ""})
            else:
                self.blink_consec  = 0
                self.blink_flag    = False
                self.eyes_closed_t = None

            # Prune old blink timestamps
            self.last_blink_ts = [t for t in self.last_blink_ts if now - t < 60.0]

            # ── Iris direction ─────────────────────────────────────
            xr_l, yr_l = iris_ratios(lm, L_IRIS, L_EYE_LEFT, L_EYE_RIGHT, L_EYE_TOP, L_EYE_BOT)
            xr_r, yr_r = iris_ratios(lm, R_IRIS, R_EYE_LEFT, R_EYE_RIGHT, R_EYE_TOP, R_EYE_BOT)
            xr_avg = (xr_l + xr_r) / 2.0
            yr_avg = (yr_l + yr_r) / 2.0

            if len(self._eye_center_samples) < POSE_CALIB_FRAMES:
                self._eye_center_samples.append((xr_avg, yr_avg))
                xs, ys = zip(*self._eye_center_samples)
                self.eye_center_x = float(np.median(xs))
                self.eye_center_y = float(np.median(ys))

            raw_eye = classify_eye(xr_avg, yr_avg, self.eye_center_x, self.eye_center_y)
            self.eye_raw_buf.append(raw_eye)
            smooth_eye        = buf_mode(list(self.eye_raw_buf))
            eye_dir_confirmed = self.eye_sustain.update(smooth_eye, now=now)

            # Count transitions into a new confirmed eye direction
            if eye_dir_confirmed != self._prev_eye_dir:
                self._prev_eye_dir = eye_dir_confirmed
                if eye_dir_confirmed == "Left":  self.eye_l += 1
                elif eye_dir_confirmed == "Right": self.eye_r += 1
                elif eye_dir_confirmed == "Up":    self.eye_u += 1
                elif eye_dir_confirmed == "Down":  self.eye_d += 1
                self._log("eye", eye_dir_confirmed, now=now)
                frame_events.append({"t": round(now - self.session_start, 2), "kind": "eye", "detail": eye_dir_confirmed})

            # ── Head pose ─────────────────────────────────────────
            raw_yaw = raw_pitch = 0.0
            try:
                raw_yaw, raw_pitch, roll = calc_head_pose(lm, h, w)
                # Apply Exponential Moving Average smoothing
                self.smooth_yaw   = (self.ema_alpha * raw_yaw)   + (1 - self.ema_alpha) * self.smooth_yaw
                self.smooth_pitch = (self.ema_alpha * raw_pitch) + (1 - self.ema_alpha) * self.smooth_pitch
                yaw, pitch = self.smooth_yaw, self.smooth_pitch
            except Exception:
                yaw = pitch = roll = 0.0

            if len(self._head_pose_samples) < POSE_CALIB_FRAMES:
                self._head_pose_samples.append((raw_yaw, raw_pitch))
                yaws, pitches = zip(*self._head_pose_samples)
                self.neutral_yaw = float(np.median(yaws))
                self.neutral_pitch = float(np.median(pitches))

            raw_head = classify_head(yaw, pitch, self.neutral_yaw, self.neutral_pitch)
            self.head_raw_buf.append(raw_head)
            smooth_head        = buf_mode(list(self.head_raw_buf))
            head_dir_confirmed = self.head_sustain.update(smooth_head, now=now)

            # Count transitions into a new confirmed head direction
            if head_dir_confirmed != self._prev_head_dir:
                self._prev_head_dir = head_dir_confirmed
                if head_dir_confirmed == "Left":  self.head_l += 1
                elif head_dir_confirmed == "Right": self.head_r += 1
                elif head_dir_confirmed == "Up":    self.head_u += 1
                elif head_dir_confirmed == "Down":  self.head_d += 1
                self._log("head", head_dir_confirmed, now=now)
                frame_events.append({"t": round(now - self.session_start, 2), "kind": "head", "detail": head_dir_confirmed})

        else:
            self.perclos = 0.0

        # ── Attention update (rate × dt, not multiplicative) ───────
        attention_delta = 0.0
        if not face_detected:
            attention_delta = -ATTN_RATE_NO_FACE * dt
            alerts.append("no_face")
        else:
            distracted_eye = eye_dir_confirmed != "Center"
            distracted_head = head_dir_confirmed != "Center"
            drowsy = self.perclos >= ATTN_DROWSY_PERCLOS or self.microsleep_count > 0

            if drowsy:
                attention_delta = -ATTN_RATE_DROWSY * dt
            elif distracted_eye and distracted_head:
                attention_delta = -ATTN_RATE_COMBINED * dt
            elif distracted_head:
                attention_delta = -ATTN_RATE_HEAD * dt
            elif distracted_eye:
                attention_delta = -ATTN_RATE_EYE * dt
            else:
                attention_delta = ATTN_RATE_RECOVER * dt

        self.attention += attention_delta

        self.attention = max(ATTN_MIN, min(ATTN_MAX, self.attention))

        # ── Focus frame counter ────────────────────────────────────
        if face_detected and head_dir_confirmed == "Center" and eye_dir_confirmed == "Center":
            self.focus_frames += 1

        # ── Alerts ────────────────────────────────────────────────
        if self.attention < 35:
            alerts.append("low_attention")
        if self.perclos > 15:
            alerts.append("drowsy")

        blink_rate_val = len(self.last_blink_ts)   # blinks in last 60 s

        return {
            "face_detected": face_detected,
            "attention":     round(self.attention, 1),
            "focus_pct":     round(self.focus_frames / max(self.total_frames, 1) * 100, 1),
            "blinks":        self.blink_count,
            "blink_rate":    blink_rate_val,
            "microsleeps":   self.microsleep_count,
            "perclos":       round(self.perclos, 1),
            "eye_dir":       eye_dir_confirmed,
            "head_dir":      head_dir_confirmed,
            "ear":           round(av_ear, 3),
            "alerts":        list(set(alerts)),
            "eye_counts":    {"L": self.eye_l,  "R": self.eye_r,  "U": self.eye_u,  "D": self.eye_d},
            "head_counts":   {"L": self.head_l, "R": self.head_r, "U": self.head_u, "D": self.head_d},
            "pose":          {
                "yaw": round(yaw, 1),
                "pitch": round(pitch, 1),
                "roll": round(roll, 1),
                "neutral_yaw": round(self.neutral_yaw, 1),
                "neutral_pitch": round(self.neutral_pitch, 1),
            },
            "session_t": round(now - self.session_start, 2),
            "frame_events": frame_events,
            "face_box": face_box,
        }

    def build_summary(self, duration_s=None):
        actual_duration = duration_s
        if actual_duration is None:
            actual_duration = max(self.last_frame_t - self.session_start, 0.0)

        return {
            "session_start":     time.strftime('%Y-%m-%d %H:%M:%S',
                                               time.localtime(self.session_start)),
            "duration_s":        round(actual_duration, 1),
            "total_frames":      self.total_frames,
            "focus_frames":      self.focus_frames,
            "focus_pct":         round(self.focus_frames / max(self.total_frames, 1) * 100, 1),
            "attention_final":   round(self.attention, 1),
            "blinks":            self.blink_count,
            "microsleeps":       self.microsleep_count,
            "ear_open_baseline": round(self.ear_open_base or EAR_CLOSED_FIXED, 4),
            "ear_closed_thresh": round(self.ear_thresh, 4),
            "eye_center":        {"x": round(self.eye_center_x, 4), "y": round(self.eye_center_y, 4)},
            "head_neutral":      {"yaw": round(self.neutral_yaw, 2), "pitch": round(self.neutral_pitch, 2)},
            "eye_counts":        {"Left": self.eye_l,  "Right": self.eye_r,
                                   "Up":   self.eye_u,  "Down":  self.eye_d},
            "head_counts":       {"Left": self.head_l, "Right": self.head_r,
                                   "Up":   self.head_u, "Down":  self.head_d},
            "events":            self.events,
        }

    def save_session(self, duration_s=None) -> str:
        """Save session log to one rolling JSON file. Returns the file path."""
        os.makedirs(SESSION_DIR, exist_ok=True)
        filename = "latest_session.json"
        filepath = os.path.join(SESSION_DIR, filename)

        summary = self.build_summary(duration_s=duration_s)

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[Session saved → {filepath}]")
        return filepath

    def close(self):
        self.face_mesh.close()
