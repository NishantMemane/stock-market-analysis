import cv2
import numpy as np
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Colors (BGR)
C_UP_MAJOR   = (0, 255, 0)       # Bright Green
C_UP_WEAK    = (180, 255, 180)   # Pale Green
C_DOWN_MAJOR = (0, 0, 255)       # Bright Red
C_DOWN_WEAK  = (180, 180, 255)   # Pale Red
C_SIDE       = (230, 230, 230)   # Light Grey
C_TEXT       = (0, 0, 0)

# Thresholds (Pixel Movement)
# How many pixels of vertical movement constitutes a trend?
# Adjust based on image resolution. For 1080p, 20 pixels is a good base.
THRESH_MAJOR_PIXELS = 40
THRESH_WEAK_PIXELS  = 10

# ==========================================
# 2. DATA MODELS
# ==========================================

@dataclass
class SwingPoint:
    type: str; x: int; y: int

@dataclass
class TrendSection:
    id: int
    start_x: int
    end_x: int
    trend_label: str
    direction: str
    color: tuple
    start_y: float # Interpolated Pixel Y
    end_y: float   # Interpolated Pixel Y
    delta_y: float

@dataclass
class ShiftZone:
    type: str; top_y: int; bottom_y: int; start_x: int; end_x: int

# ==========================================
# 3. ANALYZER ENGINE
# ==========================================

class MarketStructureAnalyzer:
    def __init__(self, image_path: str, json_path: str):
        if not os.path.exists(image_path): raise FileNotFoundError("Image not found")
        self.img = cv2.imread(image_path)
        self.height, self.width, _ = self.img.shape
        self.points = self._load_points(json_path)
        self.sections: List[TrendSection] = []
        self.shift_zones: List[ShiftZone] = []

    def _load_points(self, path):
        with open(path, 'r') as f: data = json.load(f)
        # We rely on X and Y pixels now, not "val/price", to avoid coordinate confusion
        pts = [SwingPoint(p["label"], p["x"], p["y"]) for p in data]
        return sorted(pts, key=lambda p: p.x)

    # ---------------------------------------------------------
    # CORE LOGIC: PIXEL INTERPOLATION + INVERTED Y CHECK
    # ---------------------------------------------------------

    def _get_interpolated_y(self, target_x: int) -> float:
        """
        Finds the Y-Pixel value at a specific X on the continuous line.
        """
        if not self.points: return 0.0

        # Boundary checks
        if target_x <= self.points[0].x: return float(self.points[0].y)
        if target_x >= self.points[-1].x: return float(self.points[-1].y)

        # Find segment
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i+1]

            if p1.x <= target_x <= p2.x:
                if p2.x == p1.x: return float(p1.y)

                # Linear Interpolation
                slope = (p2.y - p1.y) / (p2.x - p1.x)
                interp_y = p1.y + (slope * (target_x - p1.x))
                return interp_y

        return 0.0

    def segment_chart(self):
        num_sections = 6
        sec_w = self.width // num_sections

        for i in range(num_sections):
            sx = i * sec_w
            ex = self.width if i == num_sections - 1 else (i + 1) * sec_w

            # Get Y-Coordinates (Pixels)
            # REMEMBER: Y=0 is Top, Y=High is Bottom
            y_start = self._get_interpolated_y(sx)
            y_end   = self._get_interpolated_y(ex)

            # Calculate Delta
            # If y_end (200) < y_start (800) -> Delta is NEGATIVE (-600) -> Visual UP
            # If y_end (800) > y_start (200) -> Delta is POSITIVE (+600) -> Visual DOWN
            delta_y = y_end - y_start

            # --- THE LOGIC FIX ---
            # Negative Delta = UP (Green)
            # Positive Delta = DOWN (Red)

            if delta_y < -THRESH_MAJOR_PIXELS:
                label, direction, color = "Major Uptrend", "Bullish", C_UP_MAJOR
            elif delta_y < -THRESH_WEAK_PIXELS:
                label, direction, color = "Weak Uptrend", "Bullish", C_UP_WEAK
            elif delta_y > THRESH_MAJOR_PIXELS:
                label, direction, color = "Major Downtrend", "Bearish", C_DOWN_MAJOR
            elif delta_y > THRESH_WEAK_PIXELS:
                label, direction, color = "Weak Downtrend", "Bearish", C_DOWN_WEAK
            else:
                label, direction, color = "Sideways", "Neutral", C_SIDE

            self.sections.append(TrendSection(
                id=i + 1,
                start_x=sx,
                end_x=ex,
                trend_label=label,
                direction=direction,
                color=color,
                start_y=y_start,
                end_y=y_end,
                delta_y=delta_y
            ))

    # ---------------------------------------------------------
    # SHIFT ZONES (UNCHANGED)
    # ---------------------------------------------------------
    def detect_shift_zones(self):
        bias = "Neutral"
        last_high = None; last_low = None

        for p in self.points:
            # Note: For Y-pixels, Lower Value = Higher Price
            if p.type == "HH": last_high = p; bias = "Bullish"
            if p.type == "LL": last_low = p; bias = "Bearish"

            # Bearish BOS: Price breaks BELOW previous Low (Y gets bigger)
            if bias == "Bullish" and p.type == "LL" and last_low:
                if p.y > last_low.y: # Y check flipped for pixels
                    self.shift_zones.append(ShiftZone("Bearish", last_high.y, last_low.y, p.x, self.width))

            # Bullish BOS: Price breaks ABOVE previous High (Y gets smaller)
            if bias == "Bearish" and p.type == "HH" and last_high:
                if p.y < last_high.y: # Y check flipped for pixels
                    self.shift_zones.append(ShiftZone("Bullish", last_low.y, last_high.y, p.x, self.width))

    # ---------------------------------------------------------
    # FINAL VERDICT
    # ---------------------------------------------------------
    def final_verdict(self):
        if len(self.sections) < 2: return "Insufficient Data"

        curr = self.sections[-1]
        prev = self.sections[-2]

        if "Major" in curr.trend_label:
            return f"Strong {curr.direction} Momentum"

        if curr.direction == prev.direction and curr.direction != "Neutral":
            return f"{curr.direction} Continuation"

        if prev.direction != "Neutral" and curr.direction != "Neutral" and prev.direction != curr.direction:
            return f"Reversal to {curr.direction}"

        if curr.direction == "Neutral":
            return "Consolidation / Range"

        return "Mixed Market Conditions"

    # ---------------------------------------------------------
    # OUTPUT GENERATION
    # ---------------------------------------------------------
    def generate_outputs(self, out_img, out_json):
        overlay = self.img.copy()

        # 1. Sections
        for s in self.sections:
            cv2.rectangle(overlay, (s.start_x, 0), (s.end_x, self.height), s.color, -1)
            cv2.line(self.img, (s.end_x, 0), (s.end_x, self.height), (50,50,50), 1)

            cx = (s.start_x + s.end_x) // 2
            cv2.putText(self.img, s.trend_label, (cx - 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT, 2, cv2.LINE_AA)

            # Debug: Show Delta Y
            # cv2.putText(self.img, f"dY: {int(s.delta_y)}", (cx - 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)

        cv2.addWeighted(overlay, 0.2, self.img, 0.8, 0, self.img)

        # 2. Shift Zones
        zone_overlay = self.img.copy()
        for z in self.shift_zones:
            c = (100,100,255) if z.type == "Bearish" else (100,255,100)
            cv2.rectangle(zone_overlay, (z.start_x, z.top_y), (z.end_x, z.bottom_y), c, -1)
        cv2.addWeighted(zone_overlay, 0.3, self.img, 0.7, 0, self.img)

        # 3. Verdict
        verdict = self.final_verdict()
        cv2.rectangle(self.img, (0, 0), (450, 100), (30,30,30), -1)

        v_color = (0,255,0) if "Bullish" in verdict or "Up" in verdict else (0,0,255)
        if "Consolidation" in verdict: v_color = (200,200,200)

        cv2.putText(self.img, "MARKET STATUS:", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        cv2.putText(self.img, verdict, (15, 70), cv2.FONT_HERSHEY_DUPLEX, 0.9, v_color, 2, cv2.LINE_AA)

        # 4. Draw Invisible Line (To visualize the math)
        if len(self.points) > 1:
            pts_array = np.array([[p.x, p.y] for p in self.points], dtype=np.int32)
            pts_array = pts_array.reshape((-1, 1, 2))
            cv2.polylines(self.img, [pts_array], isClosed=False, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite(out_img, self.img)

        with open(out_json, "w") as f:
            json.dump({
                "verdict": verdict,
                "sections": [{"id": s.id, "trend": s.trend_label} for s in self.sections]
            }, f, indent=4)

if __name__ == "__main__":
    img_in = input("Chart image path: ").strip()
    json_in = input("Points JSON path: ").strip()

    try:
        analyzer = MarketStructureAnalyzer(img_in, json_in)
        analyzer.segment_chart()
        analyzer.detect_shift_zones()
        analyzer.generate_outputs("final_chart.png", "final_data.json")
    except Exception as e:
        print(f"Error: {e}")
