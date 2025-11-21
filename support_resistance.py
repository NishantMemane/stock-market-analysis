import cv2
import numpy as np
import sys
import os
from scipy.signal import find_peaks

class StockChartAnalyzer:
    def __init__(self, image, mask):
        self.original_image = image
        self.candlestick_mask = mask
        self.price_series = []
        self.pivots = []
        self.sd_zones = []
        if mask is not None:
            self.image_height, self.image_width = mask.shape
        else:
            self.image_height, self.image_width = (0, 0)

    def extract_price_data(self):
        if self.candlestick_mask is None: return

        raw_data = []
        # Initialize price map for fast lookups: index is x-coordinate
        self.price_map = [None] * self.image_width

        for x in range(self.image_width):
            column = self.candlestick_mask[:, x]
            pixels = np.where(column > 0)[0]
            if len(pixels) > 0:
                high_y = np.min(pixels)
                low_y = np.max(pixels)
                if (low_y - high_y) < 2: continue
                entry = {'x': x, 'high': high_y, 'low': low_y}
                raw_data.append(entry)
                self.price_map[x] = entry

        if not raw_data: return

        # Fill gaps
        self.price_series = []
        for i in range(len(raw_data) - 1):
            curr = raw_data[i]
            next_p = raw_data[i+1]
            self.price_series.append(curr)
            if next_p['x'] - curr['x'] > 1 and next_p['x'] - curr['x'] < 5:
                for gap_x in range(curr['x'] + 1, next_p['x']):
                    filled = {'x': gap_x, 'high': curr['high'], 'low': curr['low']}
                    self.price_series.append(filled)
                    self.price_map[gap_x] = filled
        self.price_series.append(raw_data[-1])
        print(f"Data points: {len(self.price_series)}")

    def identify_fractals(self, window=20):
        """
        Increased window to 20 to ignore minor bumps.
        """
        if not self.price_series: return

        highs = [p['high'] for p in self.price_series]
        lows = [p['low'] for p in self.price_series]
        xs = [p['x'] for p in self.price_series]
        n = len(self.price_series)

        self.pivots = []

        for i in range(window, n - window):
            current_high = highs[i]
            current_low = lows[i]

            # Check Highs
            local_min_y = np.min(highs[i-window:i+window+1])
            if current_high == local_min_y:
                 self.pivots.append({'type': 'Resistance', 'x': xs[i], 'y': current_high})

            # Check Lows
            local_max_y = np.max(lows[i-window:i+window+1])
            if current_low == local_max_y:
                self.pivots.append({'type': 'Support', 'x': xs[i], 'y': current_low})

        print(f"Found {len(self.pivots)} raw pivots.")

    def generate_clustered_zones(self, y_tolerance=15, max_gap=200):
        """
        NEW LOGIC: Discontinuous Zones with Break Detection.
        Groups pivots into chains.
        - Checks X-gap (max_gap)
        - Checks if price BROKE the level in between (break detection)
        """
        if not self.pivots: return

        # Sort pivots by X (Time)
        self.pivots.sort(key=lambda p: p['x'])

        self.sd_zones = []
        active_chains = [] # List of dicts: {'type', 'y_sum', 'count', 'start_x', 'last_x', 'pivots'}

        for p in self.pivots:

            # 1. Try to fit this pivot into an existing active chain
            best_chain = None
            min_dist = float('inf')

            # Check all active chains
            for chain in active_chains:
                # STRICT TYPE CHECK: Only connect Support to Support, Resistance to Resistance
                if chain['type'] != p['type']:
                    continue

                # Calculate average Y of the chain
                avg_y = chain['y_sum'] / chain['count']

                # Check Y proximity
                if abs(p['y'] - avg_y) <= y_tolerance:
                    # Check X Gap
                    if (p['x'] - chain['last_x']) <= max_gap:

                        # --- BREAK DETECTION ---
                        # Check if price violated the level between chain['last_x'] and p['x']
                        is_broken = False
                        start_check = chain['last_x'] + 1
                        end_check = p['x']

                        # We only check if there is data in between
                        if start_check < end_check:
                            check_type = chain['type']

                            for x_k in range(start_check, end_check):
                                if x_k >= len(self.price_map) or self.price_map[x_k] is None:
                                    continue

                                price_bar = self.price_map[x_k]

                                if check_type == 'Resistance':
                                    # Break if High is significantly ABOVE the level
                                    if price_bar['high'] < (avg_y - 5):
                                        is_broken = True
                                        break
                                elif check_type == 'Support':
                                    # Break if Low is significantly BELOW the level
                                    if price_bar['low'] > (avg_y + 5):
                                        is_broken = True
                                        break

                        if not is_broken:
                            # It fits! Check if it's the *best* fit
                            dist = abs(p['y'] - avg_y)
                            if dist < min_dist:
                                min_dist = dist
                                best_chain = chain
            if best_chain:
                # Add to existing
                best_chain['y_sum'] += p['y']
                best_chain['count'] += 1
                best_chain['last_x'] = p['x']
                best_chain['pivots'].append(p)
            else:
                # Start new chain
                active_chains.append({
                    'type': p['type'], # Initial type, might mix later
                    'y_sum': p['y'],
                    'count': 1,
                    'start_x': p['x'],
                    'last_x': p['x'],
                    'pivots': [p]
                })

        # Convert chains to zones
        for chain in active_chains:
            # RULE: Must have at least 2 points AND some width
            if chain['count'] < 2: continue

            width = chain['last_x'] - chain['start_x']
            if width < 20: continue # Filter out tiny "dots"

            avg_y = int(chain['y_sum'] / chain['count'])

            # Determine dominant type
            res_c = sum(1 for p in chain['pivots'] if p['type'] == 'Resistance')
            sup_c = sum(1 for p in chain['pivots'] if p['type'] == 'Support')
            z_type = 'Resistance' if res_c >= sup_c else 'Support'

            self.sd_zones.append({
                'type': z_type,
                'y_top': avg_y - 10,
                'y_bottom': avg_y + 10,
                'start_x': chain['start_x'],
                'end_x': chain['last_x'], # Finite end!
                'strength': chain['count']
            })

        print(f"Generated {len(self.sd_zones)} discontinuous zones.")

    def visualize(self, output_path):
        if self.original_image is None: return

        overlay = self.original_image.copy()
        final_img = self.original_image.copy()

        for zone in self.sd_zones:
            x1 = zone['start_x']
            # STRICT ENDING: No extension
            x2 = zone['end_x']

            y1 = zone['y_top']
            y2 = zone['y_bottom']

            if zone['type'] == 'Resistance':
                color = (0, 0, 255)
                border = (0, 0, 180)
            else:
                color = (0, 255, 0)
                border = (0, 180, 0)

            # Draw Zone
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.line(final_img, (x1, y1), (x2, y1), border, 1)
            cv2.line(final_img, (x1, y2), (x2, y2), border, 1)

            # End Cap
            cv2.line(final_img, (x2, y1), (x2, y2), border, 1)

        cv2.addWeighted(overlay, 0.3, final_img, 0.7, 0, final_img)
        cv2.imwrite(output_path, final_img)
        print(f"Saved clean chart to {output_path}")

if __name__ == "__main__":
    from preprocessing import ChartPreprocessor

    # --- SETTINGS FOR CLEAN LOOK ---
    FRACTAL_WINDOW = 25      # Higher = Finds only major swings (reduces noise)
    # -------------------------------

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        print("--- Support & Resistance Cleaner ---")
        img_path = input("Drag and drop image path: ").strip()
        if img_path.startswith('"') and img_path.endswith('"'):
            img_path = img_path[1:-1]

    if os.path.exists(img_path):
        prep = ChartPreprocessor(img_path)
        try:
            prep.load_image()
            prep.select_roi() # Ensure you select only the candle area!
            mask, crop = prep.process_image()

            analyzer = StockChartAnalyzer(crop, mask)
            analyzer.extract_price_data()

            # 1. Find Pivots
            analyzer.identify_fractals(window=FRACTAL_WINDOW)

            # 2. Generate Discontinuous Zones
            analyzer.generate_clustered_zones(y_tolerance=15, max_gap=300)

            # 3. Visualize

            analyzer.visualize(f"clean_{os.path.basename(img_path)}")

            # Auto-open the result
            output_file = f"clean_{os.path.basename(img_path)}"
            if os.path.exists(output_file):
                if sys.platform == "win32":
                    os.startfile(output_file)
                else:
                    import subprocess
                    opener = "open" if sys.platform == "darwin" else "xdg-open"
                    subprocess.call([opener, output_file])

        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
    else:
        print("File not found.")
