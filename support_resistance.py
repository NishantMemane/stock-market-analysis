import cv2
import numpy as np
import json
import os

class ZoneOnlyDrawer:
    def __init__(self):
        self.image = None
        self.points = []
        self.candle_mask = None
        self.crop_x = 0
        self.crop_y = 0
        self.crop_w = 0
        self.crop_h = 0

    def load_data(self, image_path, json_path, crop_rect):
        self.image = cv2.imread(image_path)
        if self.image is None:
            print("Error: Could not load image.")
            return False

        # Store crop info
        self.crop_x, self.crop_y, self.crop_w, self.crop_h = crop_rect

        try:
            with open(json_path, 'r') as f:
                self.points = json.load(f)
            self.points.sort(key=lambda k: k['x'])
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return False
        return True

    def create_obstruction_mask(self):
        # We work on the ROI (Region of Interest) for obstruction detection
        # This matches the logic used in markings.py's internal processing
        roi = self.image[self.crop_y : self.crop_y+self.crop_h, self.crop_x : self.crop_x+self.crop_w]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 120])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.candle_mask = np.zeros_like(mask)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            _, _, w, h = cv2.boundingRect(cnt)
            if area > 40 or (h > 10 and w < 30):
                cv2.drawContours(self.candle_mask, [cnt], -1, 255, -1)

        for p in self.points:
            # Points in JSON are relative to crop, so they match ROI coords directly
            cv2.circle(self.candle_mask, (p['x'], p['y']), 12, 0, -1)

    def is_blocked(self, p1, p2):
        # Checks obstruction within the ROI mask
        x1, x2 = min(p1['x'], p2['x']), max(p1['x'], p2['x'])
        y_center = int((p1['y'] + p2['y']) / 2)

        # Boundary check to prevent crashes
        y_start = max(0, y_center-2)
        y_end = min(self.candle_mask.shape[0], y_center+2)
        x_start = max(0, x1+10)
        x_end = min(self.candle_mask.shape[1], x2-10)

        roi_strip = self.candle_mask[y_start:y_end, x_start:x_end]

        if roi_strip.size == 0: return False
        if cv2.countNonZero(roi_strip) > 15: return True
        return False

    def draw_chart(self, output_filename, draw_labels=True):
        if self.image is None: return

        # Result is the FULL original image
        result = self.image.copy()
        overlay = result.copy()

        highs = [p for p in self.points if p['type'] == 'High']
        lows = [p for p in self.points if p['type'] == 'Low']

        # Tolerance based on the Crop Height, not Full Image Height
        tolerance = self.crop_h * 0.05

        def process_zones(subset, color):
            for i in range(len(subset)):
                p1 = subset[i]
                for j in range(i + 1, len(subset)):
                    p2 = subset[j]
                    if abs(p1['y'] - p2['y']) < tolerance:
                        if not self.is_blocked(p1, p2):
                            # Draw Zone - Convert to GLOBAL coordinates
                            top, bottom = min(p1['y'], p2['y']) - 8, max(p1['y'], p2['y']) + 8

                            pt1 = (p1['x'] + self.crop_x, top + self.crop_y)
                            pt2 = (p2['x'] + self.crop_x, bottom + self.crop_y)

                            cv2.rectangle(overlay, pt1, pt2, color, -1)
                            cv2.rectangle(result, pt1, pt2, color, 1)
                            break
                        else:
                            break

        process_zones(highs, (0, 0, 255))
        process_zones(lows, (0, 255, 0))

        cv2.addWeighted(overlay, 0.4, result, 0.6, 0, result)

        if draw_labels:
            for p in self.points:
                px = p['x'] + self.crop_x
                py = p['y'] + self.crop_y
                color = (0, 255, 0) if p['type'] == 'High' else (0, 0, 255)
                cv2.circle(result, (px, py), 5, color, -1)
                cv2.circle(result, (px, py), 6, (0,0,0), 1)
                label = p.get('label', '')
                if label:
                    label_y = py - 15 if p['type'] == 'High' else py + 25
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(result, (px-10, label_y-th), (px-10+tw, label_y+3), (255,255,255), -1)
                    cv2.putText(result, label, (px-10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        cv2.imwrite(output_filename, result)
        print(f"Final output saved: {output_filename}")
        return output_filename

def run_support_resistance_logic(img_path, json_path, crop_rect, output_prefix="zones", draw_labels=True, output_dir=None):
    app = ZoneOnlyDrawer()
    if app.load_data(img_path, json_path, crop_rect):
        app.create_obstruction_mask()
        base_name = os.path.basename(img_path)
        out_file = f"{output_prefix}_{base_name}"
        if output_dir:
            out_file = os.path.join(output_dir, out_file)
        return app.draw_chart(out_file, draw_labels=draw_labels)
    return None

if __name__ == "__main__":
    print("This script is now designed to run via pipeline.py to handle crop coordinates.")
