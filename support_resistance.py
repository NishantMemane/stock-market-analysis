import cv2
import numpy as np
import sys
import os

class SniperZoneAnalyzer:
    def __init__(self, image, mask):
        self.original_image = image
        self.candlestick_mask = mask
        self.price_series = []
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
                    raw_data[x] = {'high': high_y, 'low': low_y}

        self.price_series = [None] * self.w
        last_valid = raw_data[0]
        for x in range(self.w):
            if raw_data[x] is not None:
                last_valid = raw_data[x]
            if last_valid:
                self.price_series[x] = last_valid
        print("Price data extracted.")

    def find_and_cluster_swings(self, window=20, cluster_threshold=15):
        """
        Phase 1: Find Pivots and immediately Cluster them using Median Y.
        This prevents creating wide/fat boxes from wicks.
        """
        # 1. Identify Raw Pivots
        pivots = []
        for x in range(window, self.w - window):
            if self.price_series[x] is None: continue
            curr_high = self.price_series[x]['high']
            curr_low = self.price_series[x]['low']

            is_res = True
            is_sup = True
            for i in range(-window, window + 1):
                p = self.price_series[x+i]
                if not p: continue
                if p['high'] < curr_high: is_res = False
                if p['low'] > curr_low: is_sup = False

            if is_res: pivots.append({'type': 'Resistance', 'x': x, 'y': curr_high})
            if is_sup: pivots.append({'type': 'Support', 'x': x, 'y': curr_low})

        # 2. Cluster by Median Y
        self.zones = []

        # Process Resistance and Support separately
        for z_type in ['Resistance', 'Support']:
            subset = [p for p in pivots if p['type'] == z_type]
            subset.sort(key=lambda p: p['y']) # Sort by price

            used = [False] * len(subset)

            for i in range(len(subset)):
                if used[i]: continue

                base = subset[i]
                cluster = [base]
                used[i] = True

                # Find all neighbors within threshold
                for j in range(i+1, len(subset)):
                    if used[j]: continue
                    target = subset[j]

                    if abs(target['y'] - base['y']) <= cluster_threshold:
                        cluster.append(target)
                        used[j] = True

                # If this is a valid cluster (or even a strong single point)
                # Calculate Center Line using Median (Resistant to outliers)
                if len(cluster) > 0:
                    y_values = [p['y'] for p in cluster]
                    median_y = int(np.median(y_values))

                    x_coords = [p['x'] for p in cluster]
                    start_x = min(x_coords)

                    # Initialize Zone -> We will extend logic next
                    self.zones.append({
                        'type': z_type,
                        'x1': start_x,
                        'x2': start_x, # Placeholder
                        'y': median_y, # FIXED CENTER
                        'initial_cluster': cluster
                    })

    def extend_and_validate(self, break_tolerance=6):
        """
        Phase 2: Extend the FIXED center line forward until it breaks.
        """
        valid_zones = []

        for z in self.zones:
            level_y = z['y']
            start_x = z['x1']
            type_z = z['type']

            end_x = self.w

            # Scan forward from start
            for x in range(start_x + 10, self.w):
                bar = self.price_series[x]
                if not bar: continue

                if type_z == 'Resistance':
                    # Break: Price closes ABOVE the center line + tolerance
                    if bar['high'] < (level_y - break_tolerance):
                        end_x = x
                        break
                else:
                    # Break: Price closes BELOW the center line + tolerance
                    if bar['low'] > (level_y + break_tolerance):
                        end_x = x
                        break

            length = end_x - start_x
            # Filter noise: Must last at least 40px
            if length > 40:
                z['x2'] = end_x
                valid_zones.append(z)

        self.zones = valid_zones
        print(f"Generated {len(self.zones)} Fixed-Height Zones.")

    def bridge_gaps(self, gap_threshold=40):
        """
        Phase 3: The "Ghost" Logic.
        Connects broken zones if they align perfectly and gap is small.
        """
        if not self.zones: return

        # Sort by X start
        self.zones.sort(key=lambda z: z['x1'])

        merged = []
        while len(self.zones) > 0:
            current = self.zones.pop(0)

            merged_happen = True
            while merged_happen:
                merged_happen = False
                # Look for a partner in the remaining list
                for i, candidate in enumerate(self.zones):

                    # Must be same type and same height (Tight vertical tolerance)
                    if current['type'] == candidate['type']:
                        if abs(current['y'] - candidate['y']) < 8:

                            # Check Gap
                            gap = candidate['x1'] - current['x2']

                            # Valid Gap Bridge?
                            # Allow gap if < threshold OR overlapping
                            if gap < gap_threshold:
                                # MERGE
                                current['x2'] = max(current['x2'], candidate['x2'])
                                # Recalculate Y center for better accuracy
                                current['y'] = int((current['y'] + candidate['y']) / 2)

                                self.zones.pop(i)
                                merged_happen = True
                                break

            merged.append(current)

        self.zones = merged
        print(f"Bridged gaps, final count: {len(self.zones)}")

    def visualize(self, output_path):
        if self.original_image is None: return

        final_img = self.original_image.copy()
        overlay = self.original_image.copy()

        # CONSTANT HEIGHT for clean look
        # This prevents the "Fat Block" issue
        FIXED_HEIGHT = 6 # +/- 6 pixels = 12px total height

        for z in self.zones:
            x1, x2 = z['x1'], z['x2']
            y = z['y']

            pt1 = (x1, y - FIXED_HEIGHT)
            pt2 = (x2, y + FIXED_HEIGHT)

            if z['type'] == 'Resistance':
                color = (0, 0, 255) # Red
                border = (0, 0, 200)
            else:
                color = (0, 255, 0) # Green
                border = (0, 200, 0)

            # Draw
            cv2.rectangle(overlay, pt1, pt2, color, -1)
            cv2.rectangle(final_img, pt1, pt2, border, 1)

            # Center Line (The exact level)
            cv2.line(final_img, (x1, y), (x2, y), border, 1)

        cv2.addWeighted(overlay, 0.3, final_img, 0.7, 0, final_img)
        cv2.imwrite(output_path, final_img)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    from preprocessing import ChartPreprocessor

    # --- SETTINGS ---
    CLUSTER_DIST = 15   # Pivot grouping distance
    GAP_BRIDGE_PX = 40  # Bridge breaks smaller than this
    # ----------------

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        print("--- Sniper (Fixed-Height) Analyzer ---")
        img_path = input("Image Path: ").strip()
        if img_path.startswith('"') and img_path.endswith('"'):
            img_path = img_path[1:-1]

    if os.path.exists(img_path):
        prep = ChartPreprocessor(img_path)
        try:
            prep.load_image()
            prep.select_roi()
            mask, crop = prep.process_image()

            analyzer = SniperZoneAnalyzer(crop, mask)
            analyzer.extract_price_data()

            # 1. Cluster pivots by Median height
            analyzer.find_and_cluster_swings(cluster_threshold=CLUSTER_DIST)

            # 2. Extend zones forward
            analyzer.extend_and_validate()

            # 3. Bridge Gaps
            analyzer.bridge_gaps(gap_threshold=GAP_BRIDGE_PX)

            output = f"sniper_{os.path.basename(img_path)}"
            analyzer.visualize(output)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("File not found.")
