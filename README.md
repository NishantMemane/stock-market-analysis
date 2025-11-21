# Stock Market Analysis Tool

A powerful Python-based toolset for analyzing stock market charts using computer vision and algorithmic analysis. This project provides automated identification of Support & Resistance zones, Market Structure (HH/LL), and ZigZag patterns directly from chart images.

## Features

### 1. Support & Resistance Analysis (`support_resistance.py`)
- **Automated Zone Detection**: Identifies key support and resistance levels based on pivot points.
- **Discontinuous Zones**: Draws zones that respect price action, terminating when the level is invalidated ("broken").
- **Break Detection**: Smart logic to prevent zones from cutting through price candles.
- **Clustering Algorithm**: Groups nearby pivots to form strong, significant zones rather than single lines.

### 2. Market Structure Analysis (`markings.py`)
- **Swing Point Identification**: Automatically detects Highs and Lows using signal processing (`scipy.signal.find_peaks`).
- **Structure Labeling**: Classifies points as:
  - **HH** (Higher High)
  - **HL** (Higher Low)
  - **LH** (Lower High)
  - **LL** (Lower Low)
- **Interactive ROI Selection**: Allows users to select the specific chart area to analyze.
- **Noise Filtering**: robust preprocessing to remove gridlines, text, and background noise.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-market-analysis.git
   cd stock-market-analysis
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python numpy scipy
   ```

## Usage

### Support & Resistance
Run the script and provide the path to your chart image:
```bash
python support_resistance.py "path/to/your/chart.png"
```

### Market Structure (HH/LL)
Run the markings script:
```bash
python markings.py "path/to/your/chart.png"
```

## How It Works

1. **Image Preprocessing**: The tool converts the chart image to a binary mask, removing background noise and gridlines.
2. **Data Extraction**: It scans the mask to extract high/low price data for every column of pixels.
3. **Algorithmic Analysis**:
   - **S/R**: Uses a clustering algorithm to find horizontal levels with multiple touches.
   - **Structure**: Uses peak detection to find local maxima/minima and compares them to identify trends.
4. **Visualization**: Overlays the results (zones, labels) onto the original image and saves the output.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- SciPy

## License
MIT License
