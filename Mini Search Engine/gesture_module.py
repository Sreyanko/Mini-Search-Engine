import cv2
import mediapipe as mp
import numpy as np
import math
import time


# ─────────────────────────────────────────────
# Simple 2-D Kalman Filter for finger-tip tracking
# ─────────────────────────────────────────────
class KalmanFilter2D:
    """Tracks (x, y) position with constant-velocity model."""

    def __init__(self, process_noise=1e-2, measurement_noise=5.0):
        self.state = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1000

        self.F = np.eye(4, dtype=np.float64)
        self.H = np.zeros((2, 4), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        self.Q_base = np.eye(4, dtype=np.float64) * process_noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

        self.last_time = None
        self.initialized = False

    def reset(self):
        self.state = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1000
        self.last_time = None
        self.initialized = False

    def predict_and_update(self, mx, my):
        now = time.time()
        if not self.initialized:
            self.state[:2] = [mx, my]
            self.last_time = now
            self.initialized = True
            return int(mx), int(my)

        dt = max(now - self.last_time, 1e-3)
        self.last_time = now

        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q_base * dt

        z = np.array([mx, my], dtype=np.float64)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return int(self.state[0]), int(self.state[1])


# ─────────────────────────────────────────────
# Gesture Processor
# ─────────────────────────────────────────────
class GestureProcessor:
    def __init__(self, width=640, height=480):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.width = width
        self.height = height

        # Canvas: neon strokes on black
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Kalman filter
        self.kalman = KalmanFilter2D(process_noise=1e-2, measurement_noise=4.0)

        self.prev_x, self.prev_y = 0, 0
        self.is_drawing = False

        # Dead-zone
        self.dead_zone = 3

        # Point buffer for Bézier
        self.point_buffer = []

        # Thresholds
        self.swipe_threshold = 100
        self.history_x = []

        # Neon glow colour
        self.neon_core = (255, 255, 0)  # cyan in BGR

        # HUD
        self.frame_counter = 0

        # Cooldown for "done" gestures (thumbs-up / fist after drawing)
        self.last_done_time = 0
        self.done_cooldown = 2.0

    def reset_canvas(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.kalman.reset()
        self.point_buffer = []

    def calculate_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # ── Finger state helpers ──
    def _is_finger_extended(self, lm, tip, pip):
        """Check if a finger is extended (tip above PIP joint)."""
        return lm[tip].y < lm[pip].y

    def _get_finger_states(self, lm):
        """Returns (index_up, middle_up, ring_up, pinky_up, thumb_up)."""
        index_up  = self._is_finger_extended(lm, 8, 6)
        middle_up = self._is_finger_extended(lm, 12, 10)
        ring_up   = self._is_finger_extended(lm, 16, 14)
        pinky_up  = self._is_finger_extended(lm, 20, 18)
        # Thumb: tip above IP joint AND IP above MCP
        thumb_up  = lm[4].y < lm[3].y < lm[2].y
        return index_up, middle_up, ring_up, pinky_up, thumb_up

    def _is_pointing(self, lm):
        """Index finger pointing up, all other fingers curled.
        This is the DRAW gesture — fast and natural."""
        idx, mid, ring, pinky, _ = self._get_finger_states(lm)
        return idx and not mid and not ring and not pinky

    def _is_open_palm(self, lm):
        """All four fingers extended."""
        idx, mid, ring, pinky, _ = self._get_finger_states(lm)
        return idx and mid and ring and pinky

    def _is_thumbs_up(self, lm):
        """Thumb up, all fingers curled."""
        idx, mid, ring, pinky, thumb = self._get_finger_states(lm)
        return thumb and not idx and not mid and not ring and not pinky

    def _is_fist(self, lm):
        """All fingers curled including thumb (closed fist)."""
        idx, mid, ring, pinky, _ = self._get_finger_states(lm)
        thumb_curled = lm[4].y > lm[3].y  # thumb tip below IP
        return not idx and not mid and not ring and not pinky and thumb_curled

    def _is_peace(self, lm):
        """Index + middle up, ring + pinky curled — STOP/PAUSE gesture."""
        idx, mid, ring, pinky, _ = self._get_finger_states(lm)
        return idx and mid and not ring and not pinky

    # ── Bézier curve drawing ──
    def _draw_bezier(self, pts, color, thickness):
        if len(pts) < 3:
            if len(pts) == 2:
                cv2.line(self.canvas, pts[0], pts[1], color, thickness, cv2.LINE_AA)
            return

        p0 = np.array(pts[-3], dtype=np.float64)
        p1 = np.array(pts[-2], dtype=np.float64)
        p2 = np.array(pts[-1], dtype=np.float64)

        num_steps = 8
        prev = tuple(p0.astype(int))
        for i in range(1, num_steps + 1):
            t = i / num_steps
            point = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
            curr = (int(point[0]), int(point[1]))
            cv2.line(self.canvas, prev, curr, color, thickness, cv2.LINE_AA)
            prev = curr

    # ── Neon glow compositing ──
    def _apply_neon_glow(self, frame):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

        if cv2.countNonZero(mask) == 0:
            return frame

        glow = cv2.GaussianBlur(self.canvas, (21, 21), 10)
        glow2 = cv2.GaussianBlur(self.canvas, (41, 41), 20)

        glow_mask_gray = cv2.cvtColor(glow2, cv2.COLOR_BGR2GRAY)
        _, glow_mask = cv2.threshold(glow_mask_gray, 2, 255, cv2.THRESH_BINARY)

        result = frame.copy()

        glow_region = cv2.bitwise_and(glow2, glow2, mask=glow_mask)
        result = cv2.add(result, (glow_region * 0.4).astype(np.uint8))

        glow_region_inner = cv2.bitwise_and(glow, glow, mask=glow_mask)
        result = cv2.add(result, (glow_region_inner * 0.3).astype(np.uint8))

        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(result, result, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        result = cv2.add(frame_bg, canvas_fg)

        return result

    # ── Pulsing cursor ──
    def _draw_cursor(self, frame, pos, drawing):
        pulse = math.sin(time.time() * 8) * 0.3 + 0.7

        if drawing:
            r1 = int(18 * pulse)
            r2 = int(12 * pulse)
            cv2.circle(frame, pos, r1, (200, 150, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, pos, r2, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, pos, 4, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            r1 = int(14 * pulse)
            cv2.circle(frame, pos, r1, (0, 0, 200), 1, cv2.LINE_AA)
            cv2.circle(frame, pos, 5, (0, 0, 255), -1, cv2.LINE_AA)

    # ── Futuristic HUD overlay ──
    def _draw_hud(self, frame, gesture_name="TRACKING"):
        h, w = frame.shape[:2]
        hud_color = (200, 150, 0)
        dim_color = (100, 80, 0)

        bracket_len = 30
        thickness = 2

        corners = [
            ((5, 5), (5 + bracket_len, 5), (5, 5 + bracket_len)),
            ((w - 5, 5), (w - 5 - bracket_len, 5), (w - 5, 5 + bracket_len)),
            ((5, h - 5), (5 + bracket_len, h - 5), (5, h - 5 - bracket_len)),
            ((w - 5, h - 5), (w - 5 - bracket_len, h - 5), (w - 5, h - 5 - bracket_len)),
        ]
        for corner, h_end, v_end in corners:
            cv2.line(frame, corner, h_end, hud_color, thickness, cv2.LINE_AA)
            cv2.line(frame, corner, v_end, hud_color, thickness, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Gesture-specific colors
        color_map = {
            "DRAWING":    (0, 255, 100),
            "THUMBS UP":  (0, 255, 0),
            "OPEN PALM":  (0, 200, 255),
            "PEACE":      (255, 200, 0),
            "FIST":       (0, 150, 255),
            "TRACKING":   hud_color,
        }
        s_color = color_map.get(gesture_name, hud_color)

        cv2.putText(frame, f"[ {gesture_name} ]", (15, h - 15), font, 0.5, s_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "AIR WRITE v3.0", (w - 160, h - 15), font, 0.5, dim_color, 1, cv2.LINE_AA)

        self.frame_counter += 1
        cv2.putText(frame, f"FRM {self.frame_counter:05d}", (w - 130, 25), font, 0.4, dim_color, 1, cv2.LINE_AA)

        # Subtle scan-line effect
        if self.frame_counter % 2 == 0:
            overlay = frame.copy()
            for y_line in range(0, h, 4):
                cv2.line(overlay, (0, y_line), (w, y_line), (0, 0, 0), 1)
            cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)

        return frame

    # ── Main frame processing ──
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        action = None
        gesture_name = "TRACKING"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand skeleton (subdued)
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(50, 50, 50), thickness=1, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(100, 100, 100), thickness=1)
                )

                h, w, c = frame.shape
                lm = hand_landmarks.landmark

                # Index fingertip position
                index_tip = (int(lm[8].x * w), int(lm[8].y * h))

                # Detect gestures
                is_pointing  = self._is_pointing(lm)
                is_open_palm = self._is_open_palm(lm)
                is_thumbs_up = self._is_thumbs_up(lm)
                is_peace     = self._is_peace(lm)

                # ═══════════════════════════════════════
                # GESTURE PRIORITY (highest → lowest)
                # ═══════════════════════════════════════

                # 1. OPEN PALM → swipe to erase, or just stop drawing
                if is_open_palm:
                    gesture_name = "OPEN PALM"
                    self.history_x.append(lm[0].x * w)
                    if len(self.history_x) > 10:
                        self.history_x.pop(0)
                        if self.history_x[0] - self.history_x[-1] > self.swipe_threshold:
                            action = "swipe_left"
                            self.history_x = []
                        elif self.history_x[-1] - self.history_x[0] > self.swipe_threshold:
                            action = "swipe_right"
                            self.history_x = []

                    action = action or "cancel"
                    self.is_drawing = False
                    self.prev_x, self.prev_y = 0, 0
                    self.kalman.reset()
                    self.point_buffer = []

                # 2. THUMBS UP → "done" (OCR + search)
                elif is_thumbs_up:
                    gesture_name = "THUMBS UP"
                    now = time.time()
                    if now - self.last_done_time > self.done_cooldown:
                        action = "done"
                        self.last_done_time = now
                        print("Gesture: Thumbs-up → DONE")
                    self.is_drawing = False
                    self.prev_x, self.prev_y = 0, 0
                    self.kalman.reset()
                    self.point_buffer = []

                # 3. PEACE SIGN (V) → stop drawing without any action
                elif is_peace:
                    gesture_name = "PEACE"
                    self.is_drawing = False
                    self.prev_x, self.prev_y = 0, 0
                    self.kalman.reset()
                    self.point_buffer = []

                # 4. INDEX FINGER POINTING → DRAW
                elif is_pointing:
                    gesture_name = "DRAWING"
                    raw_x, raw_y = index_tip
                    kx, ky = self.kalman.predict_and_update(raw_x, raw_y)

                    if not self.is_drawing:
                        self.is_drawing = True
                        self.prev_x, self.prev_y = kx, ky
                        self.point_buffer = [(kx, ky)]
                    else:
                        dx = abs(kx - self.prev_x)
                        dy = abs(ky - self.prev_y)
                        if dx > self.dead_zone or dy > self.dead_zone:
                            self.point_buffer.append((kx, ky))

                            if len(self.point_buffer) >= 3:
                                self._draw_bezier(self.point_buffer, self.neon_core, 6)
                            else:
                                cv2.line(self.canvas, (self.prev_x, self.prev_y), (kx, ky),
                                         self.neon_core, 6, cv2.LINE_AA)

                            self.prev_x, self.prev_y = kx, ky

                            if len(self.point_buffer) > 50:
                                self.point_buffer = self.point_buffer[-10:]

                    self._draw_cursor(frame, (kx, ky) if self.is_drawing else index_tip, True)

                # 5. Anything else → idle
                else:
                    gesture_name = "TRACKING"
                    if self.is_drawing:
                        self.is_drawing = False
                        self.prev_x, self.prev_y = 0, 0
                        self.kalman.reset()
                        self.point_buffer = []
                    self.history_x = []
                    self._draw_cursor(frame, index_tip, False)

        # Compose neon canvas + HUD
        combined_frame = self._apply_neon_glow(frame)
        combined_frame = self._draw_hud(combined_frame, gesture_name)

        return combined_frame, action

    def get_canvas(self):
        return self.canvas
