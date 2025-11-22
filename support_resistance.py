import cv2
import numpy as np
import json
import os
import sys

class ZoneOnlyDrawer:
    def __init__(self):
        self.image = None
        self.points = []
        self.candle_mask = None
        self.img_h = 0
        self.img_w = 0

    def load_data(self, image_path, json_path):
        # 1. Load Image
        self.image = cv2.imread(image_path)
        if self.image is None:
            print("Error: Could not load image.")
            return False

        self.img_h, self.img_w, _ = self.image.shape

        # 2. Load JSON
        try:
            with open(json_path, 'r') as f:
                self.points = json.load(f)
            self.points.sort(key=lambda k: k['x'])
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return False

        return True

    def create_obstruction_mask(self):
        """
        Creates a mask of obstacles (candles).
        """
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Detect Black/Gray (Candles/Text/Grid)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 120])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # Clean up the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.candle_mask = np.zeros_like(mask)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            # Filter: Only keep things that look like candles
            if area > 40 or (h > 10 and w < 30):
                cv2.drawContours(self.candle_mask, [cnt], -1, 255, -1)

        # Remove the Pivot Dots from the mask
        for p in self.points:
            cv2.circle(self.candle_mask, (p['x'], p['y']), 12, 0, -1)

    def is_blocked(self, p1, p2):
        """
        Checks if candles exist between p1 and p2.
        """
        x1, x2 = min(p1['x'], p2['x']), max(p1['x'], p2['x'])

        # Check vertical center
        y_center = int((p1['y'] + p2['y']) / 2)

        # Check narrow strip
        y_top = y_center - 2
        y_bottom = y_center + 2

        # Buffer x1 and x2
        roi = self.candle_mask[y_top:y_bottom, x1+10 : x2-10]

        if roi.size == 0: return False

        obstruction_count = cv2.countNonZero(roi)

        # RELAXED TOLERANCE
        if obstruction_count > 15:
            return True
        return False

    def draw_chart(self, output_filename):
        if self.image is None: return

        result = self.image.copy()
        overlay = result.copy()

        # [REMOVED] Dynamic Background Grid
        # [REMOVED] Zig Zag Lines

        # --- Zone Logic ---
        highs = [p for p in self.points if p['type'] == 'High']
        lows = [p for p in self.points if p['type'] == 'Low']

        # DYNAMIC TOLERANCE: 5% of image height
        tolerance = self.img_h * 0.05

        def process_zones(subset, color):
            for i in range(len(subset)):
                p1 = subset[i]

                # Check all future points
                for j in range(i + 1, len(subset)):
                    p2 = subset[j]

                    # 1. Vertical Alignment Check
                    if abs(p1['y'] - p2['y']) < tolerance:

                        # 2. Obstruction Check
                        if not self.is_blocked(p1, p2):
                            # DRAW ZONE
                            top = min(p1['y'], p2['y']) - 8
                            bottom = max(p1['y'], p2['y']) + 8

                            cv2.rectangle(overlay, (p1['x'], top), (p2['x'], bottom), color, -1)
                            cv2.rectangle(result, (p1['x'], top), (p2['x'], bottom), color, 1)

                            # Break after finding nearest connection
                            break
                        else:
                            # If obstructed, stop checking this p1
                            break

        # Draw Resistance (Red)
        process_zones(highs, (0, 0, 255))
        # Draw Support (Green)
        process_zones(lows, (0, 255, 0))

        # Blend Transparency
        cv2.addWeighted(overlay, 0.4, result, 0.6, 0, result)

        # --- Dots & Labels ---
        for p in self.points:
            color = (0, 255, 0) if p['type'] == 'High' else (0, 0, 255)

            # Dot
            cv2.circle(result, (p['x'], p['y']), 5, color, -1)
            cv2.circle(result, (p['x'], p['y']), 6, (0,0,0), 1)

            # Label
            label = p.get('label', '')
            if label:
                label_y = p['y'] - 15 if p['type'] == 'High' else p['y'] + 25
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                # White background for text
                cv2.rectangle(result, (p['x']-10, label_y-th), (p['x']-10+tw, label_y+3), (255,255,255), -1)
                cv2.putText(result, label, (p['x']-10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        cv2.imwrite(output_filename, result)
        print(f"Saved to {output_filename}")
        cv2.imshow("Zones Only Analysis", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("--- Zone Only Drawer (No Grid/ZigZag) ---")
    img_input = input("Enter Chart Image Path: ").strip().strip('"')
    json_input = input("Enter JSON File Path: ").strip().strip('"')

    if os.path.exists(img_input) and os.path.exists(json_input):
        app = ZoneOnlyDrawer()
        if app.load_data(img_input, json_input):
            app.create_obstruction_mask()

            out_file = "zones_" + os.path.basename(img_input)
            app.draw_chart(out_file)
    else:
        print("Files not found.")
