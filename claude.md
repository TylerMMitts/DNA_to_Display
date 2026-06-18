## Complete Project Structure

```
project_root/
├── code/                          # All analysis and processing scripts
│   └── crop_root_model.py        # YOLOv8-based root detection
├── dataset/                       # All data files (primary working data)
│   ├── images/                   # All LAT images (FLAT - no subfolders)
│   │   └── [image].JPG
│   └── metadata/                 # All metadata CSV files
│       ├── image_metadata.csv    # Image info + genotype mapping
│       ├── kinship_matrix.csv    # SNP-based genetic relatedness
│       └── genotype_list.csv     # Unique genotype list
├── models/                        # Pre-trained models
│   └── root_detection.pt         # YOLOv8 model for root detection
├── results/                       # All output files and results
│   ├── figures/                  # Generated plots and visualizations
│   ├── analysis/                 # Analysis results and statistics
│   └── cropped_images/           # Output cropped root images
├── README.md                      # Pipeline documentation
├── claude.md                      # This file - project structure spec
└── [other scripts].py            # Utility scripts in root
```

## Data Organization Specifications

### Dataset Folder Structure

**`dataset/images/`** - All LAT Images (Flat Directory)
- All images in a single directory (NO subfolders)
- Strict naming convention: `{genotype}_{rootnode}_{replication}_{rootnumber}.JPG`
- Components:
  - `genotype`: Genotype identifier (e.g., MEMA001)
  - `rootnode`: Root type (primary, seminal, crown, etc.)
  - `replication`: Plant replicate number (1 or 2)
  - `rootnumber`: Root ID within plant (1 or 2)
- Example: `MEMA001_primary_1_1.JPG`

**`dataset/metadata/image_metadata.csv`** - Image Information & Genotype Mapping
- CSV with image metadata for all images in dataset/images/
- Required columns (11 total, in this order):
  - `image_id`: Unique numeric identifier (auto-generated)
  - `new_filename`: Standardized JPG filename
  - `original_filename`: Original filename from raw data
  - `genotype`: Genotype ID (MEMA001, etc.)
  - `rootnode`: Root type classification (primary, etc.)
  - `replication`: Plant replicate (1 or 2)
  - `rootnumber`: Root number (1 or 2)
  - `pixel_size_um`: Resolution (micrometers per pixel)
  - `image_width_px`: Image width in pixels
  - `image_height_px`: Image height in pixels
  - `quality_notes`: Quality assessment/notes
- One row per image

**`dataset/metadata/kinship_matrix.csv`** - Genetic Relatedness Data
- Square matrix with genotype IDs as row and column labels
- Values are SNP-based correlations (0-1 scale, higher = more related)
- Dimensions: N x N (where N = number of unique genotypes)
- Example: 158 x 158 for MEMA dataset (158 genotypes)

**`dataset/metadata/genotype_list.csv`** - Unique Genotypes
- Simple list of all unique genotypes in the dataset
- Single column: `genotype_id`
- Example: MEMA001, MEMA002, MEMA003, etc.
- One row per unique genotype

### Models Folder

**`models/`** - Pre-trained Models Directory
- `root_detection.pt`: YOLOv8 trained model for detecting root regions
  - Input: Full LAT image
  - Output: Bounding boxes around detected roots
  - Used by: `code/crop_root_model.py`
  - Format: PyTorch .pt file

### Results Folder

**`results/`** - All Output Files
- `figures/`: Generated plots, charts, visualizations
- `models/`: Trained or fine-tuned model outputs
- `analysis/`: Analysis results, statistics, reports
- `cropped_images/`: Root-cropped images from YOLOv8 detection
  - Naming: Same as source with `_crop_#` suffix
  - Example: `MEMA001_primary_1_1_crop_1.JPG`


## Command: `/organize-data`

**How to use:** Type this directly into the LLM chat:
/organize-data /path/to/your/data

text
The AI will guide you through organizing your raw LAT data into the standardized project structure. Just follow the conversation.

**What the AI does:**
1. Explores your data (finds images, metadata, detects naming patterns)
2. Asks clarifying questions only when needed
3. Copies images to `dataset/images/` with standardized names: `{genotype}_{rootnode}_{replication}_{rootnumber}.JPG`
4. Generates `dataset/metadata/image_metadata.csv` (image_id, new_filename, original_filename, genotype, rootnode, replication, rootnumber, pixel_size_um, image_width_px, image_height_px, quality_notes)
5. Generates `dataset/metadata/genotype_list.csv`
6. Handles kinship matrix if found
7. Provides summary and next steps


### Agent Task: Analyze and Organize User's Dataset

**Trigger:** User provides a new dataset or runs `/organize-data` command

**Your Job:**
1. **Discover and Analyze the Dataset**
   - Examine the user's raw image data folder structure
   - Find all image files (JPG, PNG, TIFF, etc.)
   - Identify metadata files (CSV, Excel, text files with genotype/image mappings)
   - Detect any pixel scale information (from filenames, scale images, documentation, metadata)
   - Analyze naming conventions and folder organization patterns

2. **Understand the Data Structure**
   - Parse image filenames to extract: genotype, root type, replication, root number, date, etc.
   - Map image files to genotypes using any available metadata
   - Determine pixel scales (um/pixel) from:
     - Embedded in filenames
     - Scale calibration files/folders
     - Metadata CSV/Excel files
     - Documentation in the dataset
     - If unavailable, ask user or estimate from scale reference images

3. **Organize Into Project Structure**
   - Create/use `dataset/images/` directory (flat, no subfolders)
   - Rename images to standard format: `{genotype}_{rootnode}_{replication}_{rootnumber}.JPG`
   - Create `dataset/metadata/` directory
   - Generate `image_metadata.csv` with columns:
     - `image_id`, `new_filename`, `original_filename`, `genotype`, `rootnode`, `replication`, `rootnumber`, `pixel_size_um`, `image_width_px`, `image_height_px`, `quality_notes`
   - Generate `genotype_list.csv` with unique genotypes
   - Handle kinship/genetic data if available, else note it's missing

4. **Handle Edge Cases**
   - Unknown/ambiguous naming conventions → ask user or make reasonable assumptions
   - Multiple possible pixel scales → ask user which applies or use best estimate
   - Missing metadata → extract from filenames or folder structure
   - Duplicate filenames → append counter suffix
   - Non-standard image formats → convert to JPG if needed

5. **Provide Summary and Confirmation**
   - Report number of images found and organized
   - List unique genotypes detected
   - Confirm pixel scales used
   - Point user to next step: `python code/crop_root_model.py`

### Key Principles

- **Flexible, Not Rigid**: Handle ANY dataset format, don't assume specific structure
- **Best-Effort Detection**: Use all available clues (filenames, folders, metadata, scale images) to understand the data
- **User Guidance**: Ask clarifying questions only when truly ambiguous
- **No Hardcoded Paths**: Work with whatever the user provides
- **Complete Automation**: End result should be a fully organized, ready-to-process dataset
