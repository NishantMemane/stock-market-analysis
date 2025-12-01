import os
import sys
import datetime
import markings
import support_resistance

def create_output_dir(img_path):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("outputs", base_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main():
    print("=========================================")
    print("      AUTOMATED CHART PIPELINE")
    print("=========================================")

    # 1. Get User Input
    img_path = input("Enter chart image path: ").strip()
    if img_path.startswith('"') and img_path.endswith('"'):
        img_path = img_path[1:-1]

    if not os.path.exists(img_path):
        print(f"Error: File not found at {img_path}")
        return

    try:
        scale_input = input("Enter scale for visual markings (Step 1): ").strip()
        user_scale = int(scale_input)
    except ValueError:
        user_scale = 1

    # Create Output Directory
    output_dir = create_output_dir(img_path)
    print(f"Output directory created: {output_dir}")

    print("\n--- STEP 1: Detailed Visuals (Interactive) ---")
    print("Select crop area...")

    # This returns the Full Image with Scale 1 markings (on full size)
    crop_rect, _, img_step1 = markings.run_markings_logic(
        img_path=img_path,
        scale=user_scale,
        output_prefix="step1_visuals",
        manual_crop_rect=None,
        output_dir=output_dir
    )

    print(f"\n--- STEP 2: Structural Data (Auto - Scale 5) ---")

    # This generates the JSON for structure (coords are relative to crop)
    _, json_step2, _ = markings.run_markings_logic(
        img_path=img_path,
        scale=3,
        output_prefix="step2_data",
        manual_crop_rect=crop_rect,
        output_dir=output_dir
    )

    print("\n--- STEP 3: Merging Zones ---")

    # We pass the image from Step 1 (Full size + Visuals)
    # We pass the JSON from Step 2 (Structure data)
    # We pass the Crop Rect so SupportResistance knows where the zones belong
    final_output = support_resistance.run_support_resistance_logic(
        img_path=img_step1,
        json_path=json_step2,
        crop_rect=crop_rect,
        output_prefix="FINAL",
        draw_labels=False, # Do not draw Scale 5 labels over Scale 1 visuals
        output_dir=output_dir
    )

    if final_output:
        print("\n=========================================")
        print(f"DONE. Result saved as: {final_output}")
        print("=========================================")

        try:
            import platform, subprocess
            abs_output_path = os.path.abspath(final_output)
            print(f"Opening file: {abs_output_path}")
            if platform.system() == 'Windows': os.startfile(abs_output_path)
            elif platform.system() == 'Darwin': subprocess.call(('open', abs_output_path))
            else: subprocess.call(('xdg-open', abs_output_path))
        except Exception as e:
            print(f"Error opening file: {e}")

if __name__ == "__main__":
    main()
