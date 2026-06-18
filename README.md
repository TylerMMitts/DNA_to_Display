
## Quick Setup

### 1. Prepare Your Data

Place your raw dataset folder in the root directory:
```
DNA_to_Display/
├── your_dataset/          # Your raw image data folder
├── models/
├── results/
└── ...
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Organize Your Data

Run the organize-data command inside of claude to reorganize your dataset:

```
/organize-data
```

This will:
- Analyze your dataset structure and naming conventions
- Map images to genotypes (automatically detects metadata)
- Handle arbitrary folder structures and naming formats
- Extract pixel scale information
- Create standardized project structure in `dataset/` folder
- Generate required metadata CSV files

### 4. Crop Root Images

Once your data is organized, run:

```bash
python code/crop_root_model.py
```

This will:
- Load the pre-trained YOLOv8 root detection model
- Process all images in `dataset/images/`
- Extract root regions and save to `results/cropped_images/`
- Generate detection metadata

## Project Structure

See [claude.md](claude.md) for complete project structure and specifications.

## Output

Cropped root images are saved to:
```
results/cropped_images/
```

