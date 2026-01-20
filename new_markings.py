import cv2
import numpy as np
import sys
import os
import json
from scipy.signal import find_peaks, savgol_filter

# ==========================================
# CONFIGURATION
# ==========================================
# Minimum pixel movement required to confirm a swing
# Smaller = faster detection, more noise
# Larger = slower detection, high confidence
CONFIRM_PX = 25

class ChartPreprocessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = None
        self.cropped_image = None
        self.mask = None
        self.crop_x = 0
        self.crop_y = 0
        self.crop_w = 0
        self.crop_h = 0

    def load_image(self):
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {self.image_path}")

    def set_manual_crop(self, x, y, w, h):
        if self.original_image is None: return
        self.crop_x, self.crop_y, self.crop_w, self.crop_h = x, y, w, h
        h_img, w_img = self.original_image.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, w_img - x), min(h, h_img - y)
        self.cropped_image = self.original_image[y:y+h, x:x+w]
        print(f"Applied previous crop: {x}, {y}, {w}, {h}")

    def select_roi(self):
        if self.original_image is None: return
        print("Please select the chart area and press SPACE/ENTER.")
        try:
            r = cv2.selectROI("Select Chart Area", self.original_image, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Select Chart Area")
            if r[2] == 0 or r[3] == 0:
                print("No area selected. Using full image.")
                self.cropped_image = self.original_image.copy()
                self.crop_w, self.crop_h = self.original_image.shape[1], self.original_image.shape[0]
            else:
                x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
                self.crop_x, self.crop_y, self.crop_w, self.crop_h = x, y, w, h
                self.cropped_image = self.original_image[y:y+h, x:x+w]
                print(f"Image cropped to: {x}, {y}, {w}, {h}")
        except Exception as e:
            print(f"Selection error: {e}. Using full image.")
            self.cropped_image = self.original_image.copy()

    def process_image(self):
        if self.cropped_image is None: self.cropped_image = self.original_image.copy()
        gray = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        background_intensity = int(np.argmax(hist))

        diff = cv2.absdiff(gray, background_intensity)
        _, self.mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_horizontal = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, horizontal_kernel)
        height, width = self.mask.shape
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(height * 0.5)))
        detected_vertical = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, vertical_kernel)
        grid_mask = cv2.bitwise_or(detected_horizontal, detected_vertical)
        self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(grid_mask))

        kernel_clean = np.ones((3,3), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel_clean)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel_clean)
        return self.mask, self.cropped_image

class StockChartAnalyzer:
    def __init__(self, full_image, cropped_image, mask, crop_offset):
        self.full_image = full_image
        self.cropped_image = cropped_image
        self.candlestick_mask = mask
        self.crop_x, self.crop_y = crop_offset
        self.price_series = []
        self.pivots = [] # This will hold CONFIRMED pivots

    def extract_price_data(self):
        if self.candlestick_mask is None: return
        height, width = self.candlestick_mask.shape
        raw_data = []

        for x in range(width):
            column = self.candlestick_mask[:, x]
            pixels = np.where(column > 0)[0]
            if len(pixels) > 0:
                high_y, low_y = np.min(pixels), np.max(pixels)
                if (low_y - high_y) < 3: continue
                if (low_y - high_y) < height * 0.9:
                    raw_data.append({'x': x, 'high_y': high_y, 'low_y': low_y})

        if not raw_data: return

        filtered_data = []
        x_coords = [p['x'] for p in raw_data]
        x_set = set(x_coords)
        for p in raw_data:
            neighbors = 0
            for offset in range(-5, 6):
                if (p['x'] + offset) in x_set: neighbors += 1
            if neighbors >= 5: filtered_data.append(p)
        if not filtered_data: return

        clusters = []
        current_cluster = [filtered_data[0]]
        for i in range(1, len(filtered_data)):
            if filtered_data[i]['x'] - filtered_data[i-1]['x'] < 50:
                current_cluster.append(filtered_data[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [filtered_data[i]]
        clusters.append(current_cluster)
        largest_cluster = max(clusters, key=len)

        y_values = [p['high_y'] for p in largest_cluster]
        median_y, std_dev = np.median(y_values), np.std(y_values)
        self.price_series = []
        for p in largest_cluster:
            if abs(p['high_y'] - median_y) < 4 * std_dev:
                self.price_series.append(p)

    def analyze_structure(self, distance=20, prominence=10):
        if not self.price_series: return

        # 1. Prepare Data
        x_data = np.array([p['x'] for p in self.price_series])
        high_y_data = np.array([p['high_y'] for p in self.price_series])
        low_y_data = np.array([p['low_y'] for p in self.price_series])

        # Smooth for detection
        window_length = 15
        smooth_high = savgol_filter(high_y_data, window_length, 3) if len(high_y_data) > window_length else high_y_data
        smooth_low = savgol_filter(low_y_data, window_length, 3) if len(low_y_data) > window_length else low_y_data

        # 2. Detect Candidates (Raw Swings)
        high_signal = -smooth_high
        low_signal = smooth_low

        peaks, _ = find_peaks(high_signal, distance=distance, prominence=prominence)
        troughs, _ = find_peaks(low_signal, distance=distance, prominence=prominence)

        candidates = []
        for i in peaks:
            candidates.append({
                'type': 'High',
                'x': x_data[i],
                'y': int(high_y_data[i]), # Use actual Y, not smoothed
                'val': -high_y_data[i],   # Logical value (negated Y)
                'data_idx': i
            })
        for i in troughs:
            candidates.append({
                'type': 'Low',
                'x': x_data[i],
                'y': int(low_y_data[i]),
                'val': -low_y_data[i],
                'data_idx': i
            })

        # Sort candidates by time (x)
        candidates.sort(key=lambda k: k['x'])

        # 3. Confirmation Logic & Labeling
        confirmed_swings = []
        last_confirmed_high = None
        last_confirmed_low = None
        swing_counter = 1

        for cand in candidates:
            # --- Step A: Check Confirmation ---
            is_confirmed = False
            confirmation_dist = 0
            confirm_dir = ""

            # Start looking from the point AFTER the candidate
            start_search = cand['data_idx'] + 1

            # Look ahead in the price series
            for j in range(start_search, len(self.price_series)):
                current_bar = self.price_series[j]

                if cand['type'] == 'High':
                    # To confirm a High, price must go DOWN (Increase in Y)
                    # We check the Low of subsequent candles
                    dist = current_bar['low_y'] - cand['y']
                    if dist >= CONFIRM_PX:
                        is_confirmed = True
                        confirmation_dist = dist
                        confirm_dir = "DOWN"
                        break

                elif cand['type'] == 'Low':
                    # To confirm a Low, price must go UP (Decrease in Y)
                    # We check the High of subsequent candles
                    dist = cand['y'] - current_bar['high_y']
                    if dist >= CONFIRM_PX:
                        is_confirmed = True
                        confirmation_dist = dist
                        confirm_dir = "UP"
                        break

            # If not confirmed, we skip this candidate entirely (it's noise/unconfirmed)
            if not is_confirmed:
                continue

            # --- Step B: Classify (Label) ---
            # Only label if confirmed
            label = ""
            if cand['type'] == 'High':
                if last_confirmed_high is None:
                    label = "H"
                else:
                    if cand['val'] > last_confirmed_high['val']: label = "HH"
                    elif cand['val'] < last_confirmed_high['val']: label = "LH"
                    else: label = "EH" # Equal High
                last_confirmed_high = cand

            elif cand['type'] == 'Low':
                if last_confirmed_low is None:
                    label = "L"
                else:
                    if cand['val'] > last_confirmed_low['val']: label = "HL"
                    elif cand['val'] < last_confirmed_low['val']: label = "LL"
                    else: label = "EL" # Equal Low
                last_confirmed_low = cand

            # --- Step C: Store Final Object ---
            cand['label'] = label
            cand['confirmed'] = True
            cand['confirm_direction'] = confirm_dir
            cand['confirm_distance_px'] = confirmation_dist
            cand['swing_index'] = swing_counter

            confirmed_swings.append(cand)
            swing_counter += 1

        self.pivots = confirmed_swings

    def visualize_results(self, output_path='output.png', draw_on_original=False):
        if self.full_image is None: return None

        if draw_on_original:
            result_img = self.full_image.copy()
            offset_x, offset_y = self.crop_x, self.crop_y
        else:
            result_img = self.cropped_image.copy()
            offset_x, offset_y = 0, 0

        for pivot in self.pivots:
            # Draw only CONFIRMED points
            px = pivot['x'] + offset_x
            py = pivot['y'] + offset_y

            color = (0, 255, 0) if pivot['type'] == 'High' else (0, 0, 255)

            # --- VISUALIZATION UPDATE START ---
            # Made circle smaller (Radius 3 instead of 5)
            cv2.circle(result_img, (px, py), 3, color, -1)

            if 'label' in pivot:
                # Adjusted offset for smaller text
                label_y = py - 10 if pivot['type'] == 'High' else py + 20
                label_text = f"{pivot['label']}"

                # Made text thinner and smaller
                # fontScale: 0.45 (was 0.6)
                # thickness: 1 (was 2)
                cv2.putText(result_img, label_text, (px - 10, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            # --- VISUALIZATION UPDATE END ---

        cv2.imwrite(output_path, result_img)
        print(f"Image saved: {output_path}")
        return result_img

    def save_markings_to_json(self, output_path):
        data = []
        for i, pivot in enumerate(self.pivots):
            # Building the detailed JSON object
            item = {
                "id": i + 1,
                "swing_index": pivot.get('swing_index', 0),
                "type": pivot['type'],
                "label": pivot.get('label', ''),
                "x": int(pivot['x']),
                "y": int(pivot['y']),
                "val": float(pivot['val']),
                "confirmed": pivot.get('confirmed', False),
                "confirm_direction": pivot.get('confirm_direction', ''),
                "confirm_distance_px": int(pivot.get('confirm_distance_px', 0))
            }
            data.append(item)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved: {output_path}")

def run_markings_logic(img_path, scale, output_prefix="output", manual_crop_rect=None, output_dir=None):
    scale = max(1, min(10, scale))
    distance = 20 + (scale - 1) * 10
    prominence = 10 + (scale - 1) * 5

    preprocessor = ChartPreprocessor(img_path)
    preprocessor.load_image()

    if manual_crop_rect:
        preprocessor.set_manual_crop(*manual_crop_rect)
    else:
        preprocessor.select_roi()

    mask, cropped_image = preprocessor.process_image()

    crop_offset = (preprocessor.crop_x, preprocessor.crop_y)

    analyzer = StockChartAnalyzer(preprocessor.original_image, cropped_image, mask, crop_offset)
    analyzer.extract_price_data()

    # Run analysis (Confirmation logic is inside here now)
    analyzer.analyze_structure(distance=distance, prominence=prominence)

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_img_name = f"{output_prefix}_{base_name}.png"
    out_json_name = f"{output_prefix}_{base_name}.json"

    if output_dir:
        out_img_name = os.path.join(output_dir, out_img_name)
        out_json_name = os.path.join(output_dir, out_json_name)

    analyzer.visualize_results(output_path=out_img_name, draw_on_original=True)
    analyzer.save_markings_to_json(out_json_name)

    crop_rect = (preprocessor.crop_x, preprocessor.crop_y, preprocessor.crop_w, preprocessor.crop_h)

    return crop_rect, out_json_name, out_img_name

if __name__ == "__main__":
    path = input("Image Path: ").strip().strip('"') if len(sys.argv) < 2 else sys.argv[1]
    if os.path.exists(path):
        try: s = int(input("Scale (1-10): "))
        except: s = 5
        run_markings_logic(path, s, "analyzed")
