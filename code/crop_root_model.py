from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path

def get_pixel_size_from_metadata(image_filename, metadata_csv="dataset/metadata/image_metadata.csv"):

    if not os.path.exists(metadata_csv):
        print(f"Metadata CSV not found at {metadata_csv}")
        return None, None
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_csv)
    
    # Look up the image by new_filename
    match = metadata_df[metadata_df['new_filename'] == image_filename]
    
    if match.empty:
        # Try matching by original filename or partial match
        match = metadata_df[metadata_df['original_filename'] == image_filename]
    
    if match.empty:
        print(f"No metadata found for {image_filename}")
        return None, None
    
    pixel_size_um = match.iloc[0].get('pixel_size_um', None)
    
    # Handle various ways pixel_size_um might be stored
    if pixel_size_um is not None:
        try:
            # Convert to float if it's a string
            if isinstance(pixel_size_um, str):
                pixel_size_um = float(pixel_size_um)
            # Check if it's valid
            if pd.isna(pixel_size_um) or pixel_size_um == 0:
                print(f"Invalid pixel_size_um ({pixel_size_um}) for {image_filename}")
                return None, match.iloc[0]
            return float(pixel_size_um), match.iloc[0]
        except (ValueError, TypeError):
            print(f"Could not convert pixel_size_um '{pixel_size_um}' to float for {image_filename}")
            return None, match.iloc[0]
    
    print(f"No pixel_size_um found for {image_filename}")
    return None, match.iloc[0]

def process_root_detection(
    weights_path="models/root_detection.pt",
    image_folder="dataset/images",
    metadata_csv="dataset/metadata/image_metadata.csv",
    confidence=0.70,
    output_dir="results/cropped_images"
):
    # Check if files exist
    if not os.path.exists(weights_path):
        print(f"Model weights not found: {weights_path}")
        print("Please place the root detection model at this path.")
        return None
    
    if not os.path.exists(image_folder):
        print(f"Image folder not found: {image_folder}")
        return None
    
    if not os.path.exists(metadata_csv):
        print(f"Metadata CSV not found: {metadata_csv}")
        print("Please run /organize-data first to create the metadata.")
        return None
    
    # Load the model
    model = YOLO(weights_path)
    
    # Load the image metadata
    image_metadata = pd.read_csv(metadata_csv)
    print(f"Loaded metadata for {len(image_metadata)} images")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]
    
    # Prepare for crop metadata
    crop_records = []
    
    for img_file in images:
        img_path = os.path.join(image_folder, img_file)
        
        # Look up pixel_size_um for this image
        pixel_size_um, metadata_row = get_pixel_size_from_metadata(img_file, metadata_csv)
        
        if pixel_size_um is None:
            print(f"Skipping {img_file}: No valid pixel_size_um found")
            continue
        
        # Run detection
        results = model(img_path, conf=confidence)
        
        # Load image for cropping
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image {img_path}")
            continue
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Find the detection with the highest confidence
            best_idx = 0
            best_conf = 0.0
            detections_data = []
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections_data.append((i, x1, y1, x2, y2, conf))
                
                if conf > best_conf:
                    best_conf = conf
                    best_idx = i
            
            # Only process the best detection
            best_i, x1, y1, x2, y2, best_conf = detections_data[best_idx]
            
            # Crop the root region
            root_crop = img[y1:y2, x1:x2]
            
            # Keep the same filename as the original
            crop_path = os.path.join(output_dir, img_file)
            
            # Save cropped image with the same filename
            cv2.imwrite(crop_path, root_crop)
            
            # Get cropped dimensions
            crop_height_px, crop_width_px = root_crop.shape[:2]
            
            # Extract other metadata from original row
            image_id = metadata_row.get('image_id', '')
            genotype = str(metadata_row.get('genotype', 'unknown'))
            rootnode = str(metadata_row.get('rootnode', 'unknown'))
            replication = str(metadata_row.get('replication', 'unknown'))
            rootnumber = str(metadata_row.get('rootnumber', 'unknown'))
            quality_notes = str(metadata_row.get('quality_notes', ''))
            
            # Clean up the values
            rootnode_clean = rootnode.replace('.0', '')
            replication_clean = replication.replace('.0', '')
            rootnumber_clean = rootnumber.replace('.0', '')
            
            # Add record for the summary CSV
            crop_records.append({
                'image_id': str(image_id),
                'new_filename': str(img_file), 
                'original_filename': str(img_file),  
                'genotype': str(genotype),
                'rootnode': str(rootnode_clean),
                'replication': str(replication_clean),
                'rootnumber': str(rootnumber_clean),
                'pixel_size_um': float(pixel_size_um), 
                'image_width_px': int(crop_width_px),   # Updated to cropped width
                'image_height_px': int(crop_height_px), # Updated to cropped height
                'quality_notes': str(quality_notes)
            })
            
            print(f"Saved crop: {img_file} ({crop_width_px}x{crop_height_px} px, confidence: {best_conf:.3f})")
        else:
            print(f"No detections found for {img_file}")
    
    # Save the summary CSV with updated dimensions
    if crop_records:
        summary_df = pd.DataFrame(crop_records)
        summary_path = os.path.join(output_dir, 'cropped_image_metadata.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nCropped image metadata saved to: {summary_path}")
        
        return summary_df
    else:
        print("\nNo crops were saved. Check your image folder and metadata.")
        return None

def batch_process_roots(
    weights_path="models/root_detection.pt",
    image_folder="dataset/images",
    metadata_csv="dataset/metadata/image_metadata.csv",
    confidence=0.70,
    output_dir="results/cropped_images"
):

    return process_root_detection(
        weights_path=weights_path,
        image_folder=image_folder,
        metadata_csv=metadata_csv,
        confidence=confidence,
        output_dir=output_dir
    )

if __name__ == "__main__":
    # Run batch processing
    summary = batch_process_roots(
        weights_path="models/root_detection.pt",
        image_folder="dataset/images",
        metadata_csv="dataset/metadata/image_metadata.csv",
        confidence=0.70,
        output_dir="results/cropped_images"
    )
    