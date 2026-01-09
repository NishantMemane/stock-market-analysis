import cv2
import numpy as np
import json
import os
from dataclasses import dataclass
from typing import List, Optional

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Colors (BGR)
C_BULL_ZONE   = (220, 255, 220)  # Light Green Background
C_BEAR_ZONE   = (220, 220, 255)  # Light Red Background
C_NEUTRAL_ZONE= (240, 240, 240)  # Grey Background

C_TEXT_BULL   = (0, 150, 0)
C_TEXT_BEAR   = (0, 0, 150)
C_TEXT_NEUT   = (100, 100, 100)

C_BOS_LINE    = (0, 0, 0)        # Break of Structure Line

# ==========================================
# 2. DATA MODELS
# ==========================================

@dataclass
class SwingPoint:
    type: str  # HH, HL, LH, LL
    x: int
    y: int     # Pixel Y (Lower Value = Higher Price)
    price: float

@dataclass
class MarketZone:
    start_x: int
    end_x: int
    trend: str      # "Uptrend", "Downtrend", "Choppy"
    bias: str       # "Bullish", "Bearish"
    color: tuple
    label: str

@dataclass
class StructureBreak:
    x: int
    y: int
    type: str       # "BOS Bullish", "BOS Bearish"

# ==========================================
# 3. ANALYZER ENGINE
# ==========================================

class MarketStructureAnalyzer:
    def __init__(self, image_path: str, json_path: str):
        if not os.path.exists(image_path): raise FileNotFoundError("Image not found")
        self.img = cv2.imread(image_path)
        self.height, self.width, _ = self.img.shape
        self.points = self._load_points(json_path)

        self.zones: List[MarketZone] = []
        self.structure_breaks: List[StructureBreak] = []
        self.final_bias = "Neutral"

    def _load_points(self, path):
        with open(path, 'r') as f: data = json.load(f)
        # Sort strictly by X (Time)
        pts = [SwingPoint(p["label"], p["x"], p["y"], p["val"]) for p in data]
        return sorted(pts, key=lambda p: p.x)

    # ---------------------------------------------------------
    # CORE LOGIC: PURE DOW THEORY (SEQUENTIAL ANALYSIS)
    # ---------------------------------------------------------
    def analyze_structure(self):
        if not self.points: return

        # State Variables
        current_trend = "Neutral" # Uptrend, Downtrend
        last_major_high = None    # The price level to beat for Bullish BOS
        last_major_low = None     # The price level to beat for Bearish BOS

        zone_start_x = 0

        # We iterate through points to find "Events" that change the trend
        for i, p in enumerate(self.points):

            # 1. INITIALIZE (First few points set the baseline)
            if current_trend == "Neutral":
                if p.type in ["HH", "HL"]:
                    current_trend = "Uptrend"
                    last_major_high = p
                    if p.type == "HL": last_major_low = p # Use HL as low anchor
                elif p.type in ["LL", "LH"]:
                    current_trend = "Downtrend"
                    last_major_low = p
                    if p.type == "LH": last_major_high = p # Use LH as high anchor
                continue

            # 2. BULLISH LOGIC
            if current_trend == "Uptrend":
                # Update Highs/Lows
                if p.type == "HH": last_major_high = p
                if p.type == "HL": last_major_low = p

                # CHECK FOR BEARISH REVERSAL (CHOCH/BOS)
                # Valid Reversal: Price makes an LL that is LOWER than the previous HL
                # Note: Y-Pixel: Higher Value = Lower Price
                if p.type == "LL" and last_major_low:
                    if p.y > last_major_low.y:
                        # !!! TREND FLIP DETECTED !!!
                        self._create_zone(zone_start_x, p.x, "Uptrend")
                        self.structure_breaks.append(StructureBreak(p.x, last_major_low.y, "BOS Bearish"))

                        # Reset State
                        current_trend = "Downtrend"
                        zone_start_x = p.x
                        last_major_low = p

            # 3. BEARISH LOGIC
            elif current_trend == "Downtrend":
                # Update Highs/Lows
                if p.type == "LL": last_major_low = p
                if p.type == "LH": last_major_high = p

                # CHECK FOR BULLISH REVERSAL (CHOCH/BOS)
                # Valid Reversal: Price makes an HH that is HIGHER than the previous LH
                # Note: Y-Pixel: Lower Value = Higher Price
                if p.type == "HH" and last_major_high:
                    if p.y < last_major_high.y:
                        # !!! TREND FLIP DETECTED !!!
                        self._create_zone(zone_start_x, p.x, "Downtrend")
                        self.structure_breaks.append(StructureBreak(p.x, last_major_high.y, "BOS Bullish"))

                        # Reset State
                        current_trend = "Uptrend"
                        zone_start_x = p.x
                        last_major_high = p

        # 4. CLOSE FINAL ZONE
        self._create_zone(zone_start_x, self.width, current_trend)
        self.final_bias = current_trend

    def _create_zone(self, sx, ex, trend):
        if trend == "Uptrend":
            color = C_BULL_ZONE
            label = "Structural Uptrend"
            bias = "Bullish"
        elif trend == "Downtrend":
            color = C_BEAR_ZONE
            label = "Structural Downtrend"
            bias = "Bearish"
        else:
            color = C_NEUTRAL_ZONE
            label = "Consolidation"
            bias = "Neutral"

        self.zones.append(MarketZone(sx, ex, trend, bias, color, label))

    # ---------------------------------------------------------
    # FINAL VERDICT
    # ---------------------------------------------------------
    def final_verdict(self):
        if not self.points: return "No Data"

        last_zone = self.zones[-1]
        last_pts = self.points[-3:] # Look at very recent structure

        # Check momentum within the structure
        structure_labels = [p.type for p in last_pts]

        if last_zone.bias == "Bullish":
            if "HH" in structure_labels[-1]: return "Strong Bullish Momentum"
            if "HL" in structure_labels[-1]: return "Bullish Continuation (Pullback)"
            return "Bullish Structure"

        elif last_zone.bias == "Bearish":
            if "LL" in structure_labels[-1]: return "Strong Bearish Momentum"
            if "LH" in structure_labels[-1]: return "Bearish Continuation (Pullback)"
            return "Bearish Structure"

        return "Mixed / Ranging"

    # ---------------------------------------------------------
    # OUTPUT GENERATION
    # ---------------------------------------------------------
    def generate_outputs(self, out_img, out_json):
        overlay = self.img.copy()

        # 1. Draw Variable Width Zones
        for z in self.zones:
            cv2.rectangle(overlay, (z.start_x, 0), (z.end_x, self.height), z.color, -1)

            # Draw Divider Line
            cv2.line(self.img, (z.end_x, 0), (z.end_x, self.height), (100,100,100), 1, cv2.LINE_AA)

            # Label the Zone
            cx = (z.start_x + z.end_x) // 2
            txt_color = C_TEXT_BULL if z.bias == "Bullish" else C_TEXT_BEAR
            cv2.putText(self.img, z.label, (cx - 60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)

        cv2.addWeighted(overlay, 0.3, self.img, 0.7, 0, self.img)

        # 2. Draw BOS Lines (Shift Zones)
        for b in self.structure_breaks:
            # Draw horizontal line extending back slightly
            start_line = max(0, b.x - 50)
            c = (0, 200, 0) if "Bullish" in b.type else (0, 0, 200)

            cv2.line(self.img, (start_line, b.y), (b.x + 50, b.y), c, 2)
            cv2.putText(self.img, b.type, (b.x + 5, b.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        # 3. Verdict Box
        verdict = self.final_verdict()
        cv2.rectangle(self.img, (0, 0), (450, 100), (20,20,20), -1)

        v_color = (0,255,0) if "Bullish" in verdict else (0,0,255)
        if "Mixed" in verdict: v_color = (200,200,200)

        cv2.putText(self.img, "MARKET STATUS:", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        cv2.putText(self.img, verdict, (15, 70), cv2.FONT_HERSHEY_DUPLEX, 0.9, v_color, 2, cv2.LINE_AA)

        cv2.imwrite(out_img, self.img)

        with open(out_json, "w") as f:
            json.dump({
                "verdict": verdict,
                "zones": [{"start": z.start_x, "end": z.end_x, "trend": z.trend} for z in self.zones]
            }, f, indent=4)

if __name__ == "__main__":
    img_in = input("Chart image path: ").strip()
    json_in = input("Points JSON path: ").strip()

    try:
        analyzer = MarketStructureAnalyzer(img_in, json_in)
        analyzer.analyze_structure()
        analyzer.generate_outputs("final_chart.png", "final_data.json")
    except Exception as e:
        print(f"Error: {e}")
