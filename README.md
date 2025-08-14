# GroundingDINO + SAM + CLIP Detection Workflow

This document explains the complete workflow for an intelligent object detection and segmentation system that combines GroundingDINO, Segment Anything Model (SAM), and CLIP for optimal threshold selection.

## Overview

The system performs multi-threshold object detection using natural language prompts, automatically selects the best detection parameters using CLIP similarity scoring, and generates precise segmentation masks.

### Key Features
- **Adaptive threshold selection** based on prompt analysis
- **Multi-model integration** (GroundingDINO + SAM + CLIP)
- **Automated parameter optimization** using CLIP scoring
- **CPU-only operation** for accessibility
- **DBSCAN clustering** for box merging

## Installation

### 1. System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (optional, runs on CPU)
- At least 8GB RAM recommended

### 2. Clone Required Repositories

```bash
# Clone GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..

# Clone Segment Anything
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
cd ..
```

### 3. Install Python Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python
pip install scikit-learn
pip install pillow
pip install open-clip-torch
pip install numpy
```

### 4. Download Model Weights

Create a `weights/` directory and download the required model files:

```bash
mkdir weights
cd weights

# Download GroundingDINO weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Download SAM weights (ViT-H model)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

cd ..
```

### 5. Project Structure

Ensure your project directory looks like this:

```
project/
├── main.py                              # Your detection script
├── test.png                            # Input image
├── weights/
│   ├── groundingdino_swint_ogc.pth     # GroundingDINO weights
│   └── sam_vit_h_4b8939.pth            # SAM weights
├── GroundingDINO/                      # Cloned repository
├── segment-anything/                   # Cloned repository
└── threshold_runs/                     # Output directory (created automatically)
```

## Workflow Explanation

### Phase 1: Prompt Analysis & Threshold Selection

The system analyzes your text prompt to automatically select appropriate detection thresholds:

```python
THRESHOLD_PRESETS = {
    "low":    [(0.05, 0.25), (0.10, 0.28), (0.20, 0.30)],  # Many objects
    "medium": [(0.15, 0.32), (0.15, 0.35), (0.18, 0.38)],  # Average prompts  
    "high":   [(0.30, 0.40), (0.35, 0.45), (0.25, 0.50)]   # Specific objects
}
```

**Logic:**
- **Low thresholds**: For prompts with 3+ objects (e.g., "cats, dogs, and birds")
- **Medium thresholds**: For prompts with 2 objects (e.g., "cars and trucks")
- **High thresholds**: For single, specific objects (e.g., "red sports car")

### Phase 2: Multi-Threshold Detection

For each threshold pair, the system:

1. **Runs GroundingDINO detection** with current box_threshold and text_threshold
2. **Merges detected boxes** using DBSCAN clustering into a single bounding box
3. **Generates annotations** with bounding box visualization
4. **Creates segmentation masks** using SAM
5. **Calculates CLIP similarity score** between annotated image and prompt

### Phase 3: Optimal Result Selection

The system:
- Compares CLIP similarity scores across all threshold runs
- Selects the configuration with the highest CLIP score
- Copies the best results for easy access
- Overwrites the original image with the final annotated version

## Usage

### Basic Usage

1. Place your image as `test.png` in the project directory
2. Modify the `TEXT_PROMPT` variable in the script
3. Run the detection:

```bash
python main.py
```

### Customization Options

#### Change Input Image
```python
IMAGE_PATH = "your_image.jpg"
```

#### Modify Detection Prompt
```python
TEXT_PROMPT = "person wearing red shirt"
```

#### Adjust Device (GPU/CPU)
```python
DEVICE = "cuda"  # or "cpu"
```

#### Custom Threshold Presets
```python
THRESHOLD_PRESETS = {
    "custom": [(0.20, 0.35), (0.25, 0.40)]
}
```

## Output Files

The system generates several outputs:

### Automatic Outputs
- `threshold_runs/run_X_bY.YY_tZ.ZZ_annot.png` - Annotated images for each threshold
- `threshold_runs/run_X_bY.YY_tZ.ZZ_mask.png` - Segmentation masks for each threshold
- `best_annotated.png` - Best detection result
- `best_mask.png` - Best segmentation mask
- `test.png` - Original image overwritten with final annotation

### Console Output
```
[Prompt Analysis] Category: medium, Thresholds: [(0.15, 0.32), (0.15, 0.35), (0.18, 0.38)]
[run_0_b0.15_t0.32] CLIP score = 0.3245
[run_1_b0.15_t0.35] CLIP score = 0.3891
[run_2_b0.18_t0.38] CLIP score = 0.3654

======================
Best thresholds: box_th=0.15, text_th=0.35
Best CLIP score: 0.3891
Annotated: threshold_runs/run_1_b0.15_t0.35_annot.png
Mask: threshold_runs/run_1_b0.15_t0.35_mask.png
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure GroundingDINO is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/GroundingDINO"
```

**2. CUDA Out of Memory**
```python
# Switch to CPU mode
DEVICE = "cpu"
```

**3. Model Weight Files Missing**
- Verify all `.pth` files are downloaded to `weights/` directory
- Check file permissions and sizes

**4. No Detections Found**
- Try lower threshold values
- Verify your prompt matches objects in the image
- Check image quality and resolution

### Performance Optimization

**For Faster Processing:**
- Use GPU if available (`DEVICE = "cuda"`)
- Reduce number of threshold pairs to test
- Use smaller input images

**For Better Accuracy:**
- Add more threshold pairs to test
- Use higher resolution images
- Refine your text prompts to be more specific

## Advanced Configuration

### Custom DBSCAN Parameters
```python
def dbscan_merge_to_single_box(boxes_xyxy_np, eps_frac=0.1):  # Increase clustering radius
```

### Different SAM Models
```python
# Use smaller SAM model for faster processing
sam_model = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
```

### Alternative CLIP Models
```python
# Use different CLIP model
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
```

## Technical Details

### Model Architecture
- **GroundingDINO**: Text-conditioned object detection using transformer architecture
- **SAM**: Universal image segmentation using vision transformer
- **CLIP**: Vision-language model for similarity scoring

### Key Algorithms
- **DBSCAN Clustering**: Groups nearby detections for box merging
- **Cosine Similarity**: CLIP scoring mechanism for result ranking
- **Multi-threshold Search**: Exhaustive parameter exploration

This workflow provides a robust, automated approach to object detection and segmentation with minimal manual parameter tuning.