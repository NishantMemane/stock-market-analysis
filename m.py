import cv2
import numpy as np
import sys
import os

class StructureFlowAnalyzer:
    def __init__(self, image, mask):
        self.original_image = image
        self.candlestick_mask = mask
        self.price_data = []
        self.pivots = []
        self.zones = []
        if mask is not None:
            self.h, self.w = mask.shape
        else:
            self.h, self.w = (0, 0)

    def extract_price_data(self):
        if self.candlestick_mask is None: return
        raw_data = [None] * self.w
        for x in range(self.w):
            column = self.candlestick_mask[:, x]
            pixels = np.where(column > 0)[0]
            if len(pixels) > 0:
                high_y = np.min(pixels)
                low_y = np.max(pixels)
                if (low_y - high_y) > 1:
                    raw_data[x] = {'x': x, 'high': high_y, 'low': low_y}

        self.price_data = [None] * self.w
        last_valid = raw_data[0]
        for x in range(self.w):
            if raw_data[x] is not None:
                last_valid = raw_data[x]
            if last_valid:
                self.price_data[x] = last_valid
        print("Price data extracted.")

    def calculate_major_structure(self, deviation_factor=5.0):
        """
        Finds ONLY major turning points.
        Auto-calculates threshold based on average candle size.
        """
        # 1. Calculate Average Candle Size to determine Scale
        candle_sizes = []
        for p in self.price_data:
             if p: candle_sizes.append(abs(p['high'] - p['low']))

        avg_candle = np.mean(candle_sizes)

        # A major swing must be X times larger than an average candle
        deviation_px = avg_candle * deviation_factor
        print(f"Avg Candle: {int(avg_candle)}px. Structure Minimum Move: {int(deviation_px)}px")

        self.pivots = []

        trend = 0
        last_high = self.price_data[0]
        last_low = self.price_data[0]

        for i, bar in enumerate(self.price_data):
            if not bar: continue

            if trend == 0:
                if bar['high'] < (last_low['low'] - deviation_px):
                    trend = 1
                    last_high = bar
                elif bar['low'] > (last_high['high'] + deviation_px):
                    trend = -1
                    last_low = bar

            elif trend == 1: # Uptrend
                if bar['high'] < last_high['high']:
                    last_high = bar

                if bar['low'] > (last_high['high'] + deviation_px):
                    self.pivots.append({'x': last_high['x'], 'y': last_high['high'], 'type': 'Resistance'})
                    trend = -1
                    last_low = bar

            elif trend == -1: # Downtrend
                if bar['low'] > last_low['low']:
                    last_low = bar

                if bar['high'] < (last_low['low'] - deviation_px):
                    self.pivots.append({'x': last_low['x'], 'y': last_low['low'], 'type': 'Support'})
                    trend = 1
                    last_high = bar

        print(f"Identified {len(self.pivots)} Major Structure Points.")

    def generate_flow_zones(self, merge_proximity=20):
        """
        The 'Scanner' Logic.
        Walks forward. If a zone is active, check if new pivots align with it.
        If yes, extend the zone. If price breaks, kill the zone.
        """
        self.zones = []
        active_zones = []

        # We need to process pixel by pixel to handle breaks correctly
        # But we only care about pivots for zone creation/extension

        pivot_dict = {p['x']: p for p in self.pivots}

        for x in range(self.w):
            bar = self.price_data[x]
            if not bar: continue

            # 1. CHECK ACTIVE ZONES FOR BREAKS
            for z in active_zones:
                if not z['active']: continue

                # Check for break
                tolerance = 6
                if z['type'] == 'Resistance':
                    if bar['high'] < (z['y'] - tolerance):
                        z['active'] = False
                        z['end_x'] = x
                else:
                    if bar['low'] > (z['y'] + tolerance):
                        z['active'] = False
                        z['end_x'] = x

            # 2. CHECK FOR NEW PIVOTS & MERGE
            if x in pivot_dict:
                p = pivot_dict[x]

                merged = False

                # Try to merge into an existing active zone
                for z in active_zones:
                    if z['active'] and z['type'] == p['type']:
                        # Is it at the same height?
                        if abs(z['y'] - p['y']) < merge_proximity:
                            # Merge!
                            # We adjust the Y to be the average of touches to keep it centered
                            z['y'] = int((z['y'] * z['touches'] + p['y']) / (z['touches'] + 1))
                            z['touches'] += 1
                            merged = True
                            break

                # If not merged, create new zone
                if not merged:
                    active_zones.append({
                        'type': p['type'],
                        'start_x': x,
                        'end_x': self.w, # Assume valid until proven otherwise
                        'y': p['y'],
                        'active': True,
                        'touches': 1
                    })

        # Cleanup: Copy result
        # Filter: Min length of 50px (removes tiny noise zones)
        self.zones = [z for z in active_zones if (z['end_x'] - z['start_x']) > 50]
        print(f"Generated {len(self.zones)} verified flow zones.")

    def visualize(self, output_path):
        if self.original_image is None: return

        final_img = self.original_image.copy()
        overlay = self.original_image.copy()

        # CONSTANT HEIGHT BOXES
        BOX_H = 8 # +/- 8px = 16px total

        for z in self.zones:
            x1, x2 = z['start_x'], z['end_x']
            y = z['y']

            pt1 = (x1, y - BOX_H)
            pt2 = (x2, y + BOX_H)

            if z['type'] == 'Resistance':
                color = (0, 0, 255) # Red
                border = (0, 0, 200)
            else:
                color = (0, 255, 0) # Green
                border = (0, 200, 0)

            cv2.rectangle(overlay, pt1, pt2, color, -1)
            cv2.rectangle(final_img, pt1, pt2, border, 1)
            cv2.line(final_img, (x1, y), (x2, y), border, 1)

        cv2.addWeighted(overlay, 0.3, final_img, 0.7, 0, final_img)
        cv2.imwrite(output_path, final_img)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    from preprocessing import ChartPreprocessor

    # --- SETTINGS ---
    # 1. DEVIATION_FACTOR:
    #    The most important setting.
    #    High (6.0) = Only massive swings.
    #    Low (3.0) = Minor swings included.
    #    Try 6.0 to fix your "complete wrong" issue.
    DEVIATION_FACTOR = 6.0

    # 2. MERGE_PROXIMITY:
    #    20px means if a new corner is within 20px of the old line,
    #    it pulls the line to it instead of starting a new one.
    MERGE_PROXIMITY = 20
    # ----------------

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        print("--- Structure Flow Analyzer ---")
        img_path = input("Image Path: ").strip()
        if img_path.startswith('"') and img_path.endswith('"'):
            img_path = img_path[1:-1]

    if os.path.exists(img_path):
        prep = ChartPreprocessor(img_path)
        try:
            prep.load_image()
            prep.select_roi()
            mask, crop = prep.process_image()

            analyzer = StructureFlowAnalyzer(crop, mask)
            analyzer.extract_price_data()

            # 1. Identify Main Corners (ZigZag)
            analyzer.calculate_major_structure(deviation_factor=DEVIATION_FACTOR)

            # 2. Connect them into Flow Zones
            analyzer.generate_flow_zones(merge_proximity=MERGE_PROXIMITY)

            output = f"flow_{os.path.basename(img_path)}"
            analyzer.visualize(output)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("File not found.")
