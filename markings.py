import cv2
import numpy as np
import sys
import os
from scipy.signal import find_peaks, savgol_filter

class ChartPreprocessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = None
        self.cropped_image = None
        self.mask = None

    def load_image(self):
        """Loads the image from the specified path."""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        print(f"Image loaded: {self.image_path} with shape {self.original_image.shape}")

    def select_roi(self):
        """
        Opens a window to let the user select the chart area.
        """
        if self.original_image is None:
            return

        print("Please select the chart area in the popup window and press SPACE or ENTER. Press 'c' to cancel.")

        # cv2.selectROI allows user to draw a box.
        # Returns (x, y, w, h)
        try:
            r = cv2.selectROI("Select Chart Area", self.original_image, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Select Chart Area")

            # If width or height is 0, user cancelled or selected nothing
            if r[2] == 0 or r[3] == 0:
                print("No area selected. Using full image.")
                self.cropped_image = self.original_image.copy()
            else:
                # Crop the image
                x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
                self.cropped_image = self.original_image[y:y+h, x:x+w]
                print(f"Image cropped to: {x}, {y}, {w}, {h}")

        except Exception as e:
            print(f"Error during selection: {e}. Using full image.")
            self.cropped_image = self.original_image.copy()

    def process_image(self):
        """
        Robustly preprocesses the CROPPED image to extract the chart structure.
        """
        if self.cropped_image is None:
            self.cropped_image = self.original_image.copy()

        # 1. Convert to Grayscale
        gray = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2GRAY)

        # 2. Detect Background Color (Dominant Pixel Intensity)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        background_intensity = int(np.argmax(hist))
        print(f"Detected Background Intensity: {background_intensity}")

        # 3. Create Mask (Difference from Background)
        diff = cv2.absdiff(gray, background_intensity)
        _, self.mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        # 4. Remove Grid Lines and Crosshairs
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_horizontal = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, horizontal_kernel)

        height, width = self.mask.shape
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(height * 0.5)))
        detected_vertical = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, vertical_kernel)

        grid_mask = cv2.bitwise_or(detected_horizontal, detected_vertical)
        self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(grid_mask))

        # 5. Clean up Noise
        kernel_clean = np.ones((3,3), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel_clean)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel_clean)

        return self.mask, self.cropped_image

class StockChartAnalyzer:
    def __init__(self, image, mask):
        self.original_image = image
        self.candlestick_mask = mask
        self.price_series = [] # List of (x, high, low)
        self.pivots = []
        self.sd_zones = []

    def extract_price_data(self):
        """
        Scans the mask to find High and Low pixels for each X coordinate.
        Includes NOISE FILTERING to ignore UI elements, text, and sidebars.
        """
        if self.candlestick_mask is None:
            raise ValueError("Image not preprocessed yet.")

        height, width = self.candlestick_mask.shape
        raw_data = []

        # 1. Extract Raw Data
        for x in range(width):
            column = self.candlestick_mask[:, x]
            pixels = np.where(column > 0)[0]

            if len(pixels) > 0:
                high_y = np.min(pixels)
                low_y = np.max(pixels)

                # Filter: Ignore tiny specks (noise)
                if (low_y - high_y) < 3:
                    continue

                # Basic sanity check: A candle shouldn't be the entire height of the image
                if (low_y - high_y) < height * 0.9:
                    raw_data.append({'x': x, 'high_y': high_y, 'low_y': low_y})

        if not raw_data:
            print("No data found.")
            self.price_series = []
            return

        # 2. Filter: Remove Sparse Noise (Text/Icons)
        # We keep points only if they have neighbors close by.
        filtered_data = []
        x_coords = [p['x'] for p in raw_data]
        x_set = set(x_coords)

        for p in raw_data:
            # Check for neighbors within 5 pixels
            neighbors = 0
            for offset in range(-5, 6):
                if (p['x'] + offset) in x_set:
                    neighbors += 1

            # A real chart line is dense. Text is sparse.
            # We require at least 5 neighbors in a 10px window.
            if neighbors >= 5:
                filtered_data.append(p)

        if not filtered_data:
            print("Data filtered out as noise.")
            self.price_series = []
            return

        # 3. Filter: Keep Main Chart Area (Largest Cluster)
        # We group data by X proximity. If there's a big gap (e.g. sidebar), we split.
        clusters = []
        current_cluster = [filtered_data[0]]

        for i in range(1, len(filtered_data)):
            if filtered_data[i]['x'] - filtered_data[i-1]['x'] < 50: # 50px gap tolerance
                current_cluster.append(filtered_data[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [filtered_data[i]]
        clusters.append(current_cluster)

        # Keep the largest cluster (the chart)
        largest_cluster = max(clusters, key=len)

        # 4. Filter: Remove Y-Outliers (Top/Bottom UI bars)
        # Calculate median Y of the cluster
        y_values = [p['high_y'] for p in largest_cluster]
        median_y = np.median(y_values)
        std_dev = np.std(y_values)

        # Keep points within 3 standard deviations (covers most of the chart)
        # Relaxed to 4 std_dev to allow for trends
        self.price_series = []
        for p in largest_cluster:
            if abs(p['high_y'] - median_y) < 4 * std_dev:
                self.price_series.append(p)

        print(f"Extracted {len(self.price_series)} clean data points (filtered from {len(raw_data)}).")

    def analyze_structure(self, distance=20, prominence=10):
        """
        Identifies Major Swing Points using Peak Detection (SMC style).
        Includes Right Edge Logic.
        """
        if not self.price_series:
            return

        x_data = np.array([p['x'] for p in self.price_series])
        high_y_data = np.array([p['high_y'] for p in self.price_series])
        low_y_data = np.array([p['low_y'] for p in self.price_series])

        # Smooth the data to reduce noise (Gaussian-like smoothing)
        window_length = 15
        if len(high_y_data) > window_length:
            high_y_data = savgol_filter(high_y_data, window_length, 3)
            low_y_data = savgol_filter(low_y_data, window_length, 3)

        # Invert for signal processing
        high_signal = -high_y_data
        low_signal = low_y_data

        # Print parameters for user transparency
        print(f"--- Analysis Parameters ---")
        print(f"Min Distance: {distance} pixels")
        print(f"Min Prominence: {prominence} pixels")

        # Find Major Peaks
        peaks, _ = find_peaks(high_signal, distance=distance, prominence=prominence)
        troughs, _ = find_peaks(low_signal, distance=distance, prominence=prominence)

        self.pivots = []

        for i in peaks:
            self.pivots.append({
                'type': 'High',
                'x': x_data[i],
                'y': int(high_y_data[i]),
                'val': high_signal[i]
            })

        for i in troughs:
            self.pivots.append({
                'type': 'Low',
                'x': x_data[i],
                'y': int(low_y_data[i]),
                'val': -low_y_data[i]
            })

        # Calculate average distance
        if len(self.pivots) > 1:
            sorted_x = sorted([p['x'] for p in self.pivots])
            diffs = np.diff(sorted_x)
            avg_dist = np.mean(diffs)
            print(f"Average distance between found pivots: {int(avg_dist)} pixels")
        print(f"---------------------------")

        # --- Right Edge Logic ---
        if len(x_data) > 0:
            last_idx = -1
            last_x = x_data[last_idx]
            last_high_y = high_y_data[last_idx]
            last_low_y = low_y_data[last_idx]

            lookback = min(len(high_y_data), distance)
            recent_highs = high_y_data[-lookback:]

            # Check if last point is the highest (min Y) in recent history
            if last_high_y <= np.min(recent_highs) + 2: # Tolerance of 2px
                 self.pivots.append({
                    'type': 'High',
                    'x': last_x,
                    'y': int(last_high_y),
                    'val': -last_high_y,
                    'is_working': True
                })

            recent_lows = low_y_data[-lookback:]
            # Check if last point is the lowest (max Y) in recent history
            if last_low_y >= np.max(recent_lows) - 2:
                 self.pivots.append({
                    'type': 'Low',
                    'x': last_x,
                    'y': int(last_low_y),
                    'val': -last_low_y,
                    'is_working': True
                })

        self.pivots.sort(key=lambda k: k['x'])

        # Classify HH/HL/LH/LL
        last_high = None
        last_low = None

        for pivot in self.pivots:
            if pivot['type'] == 'High':
                if last_high is None:
                    pivot['label'] = 'H'
                else:
                    if pivot['val'] > last_high['val']:
                        pivot['label'] = 'HH'
                    elif pivot['val'] < last_high['val']:
                        pivot['label'] = 'LH'
                    else:
                        pivot['label'] = 'EH'
                last_high = pivot

            elif pivot['type'] == 'Low':
                if last_low is None:
                    pivot['label'] = 'L'
                else:
                    if pivot['val'] > last_low['val']:
                        pivot['label'] = 'HL'
                    elif pivot['val'] < last_low['val']:
                        pivot['label'] = 'LL'
                    else:
                        pivot['label'] = 'EL'
                last_low = pivot

    def visualize_results(self, output_path='output.png'):
        """
        Draws the analysis on the image.
        """
        if self.original_image is None:
            return

        result_img = self.original_image.copy()

        # Draw Pivots and Labels
        for pivot in self.pivots:
            color = (0, 255, 0) if pivot['type'] == 'High' else (0, 0, 255)
            cv2.circle(result_img, (pivot['x'], pivot['y']), 4, color, -1)

            # Draw Label (HH, HL, LH, LL)
            if 'label' in pivot:
                label_y = pivot['y'] - 10 if pivot['type'] == 'High' else pivot['y'] + 20
                cv2.putText(result_img, pivot['label'], (pivot['x'] - 10, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.imwrite(output_path, result_img)
        print(f"Result saved to {output_path}")
        return result_img

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input("Enter the path to the chart image: ").strip()
        if img_path.startswith('"') and img_path.endswith('"'):
            img_path = img_path[1:-1]

    if os.path.exists(img_path):
        # Get Scale of Details from User
        try:
            scale_input = input("Enter scale of details (1-10, 1=Detailed, 10=Major Swings): ").strip()
            scale = int(scale_input)
            scale = max(1, min(10, scale)) # Clamp between 1 and 10
        except ValueError:
            print("Invalid input. Defaulting to scale 5.")
            scale = 5

        # Map scale to distance and prominence
        # Scale 1: distance=20, prominence=10
        # Scale 10: distance=110, prominence=55
        distance = 20 + (scale - 1) * 10
        prominence = 10 + (scale - 1) * 5

        print(f"Using Scale {scale}: Distance={distance}, Prominence={prominence}")

        # 1. Preprocessing Phase
        preprocessor = ChartPreprocessor(img_path)
        try:
            preprocessor.load_image()
            preprocessor.select_roi() # Interactive selection
            mask, cropped_image = preprocessor.process_image()

            # 2. Analysis Phase
            analyzer = StockChartAnalyzer(cropped_image, mask)
            analyzer.extract_price_data()
            analyzer.analyze_structure(distance=distance, prominence=prominence)
            # analyzer.find_support_resistance() # Removed as per request

            output_filename = f"analyzed_{os.path.basename(img_path)}"
            analyzer.visualize_results(output_path=output_filename)
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("File not found. Please check the path.")
