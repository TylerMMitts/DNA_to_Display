# Pairs each root image with its genotype's SNP vector for training.

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import pandas as pd

class MaizeRootDataset(Dataset):
    def __init__(self,
                 image_dir,
                 snp_file,
                 transform=None,
                 snp_transform=None):
        
        self.image_dir = image_dir
        self.transform = transform
        
        # Load SNP data
        self.load_snp_data(snp_file)
        
        # Get list of images
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def load_snp_data(self, snp_file):
        self.snp_df = pd.read_csv(snp_file)
        
        # Extract SNP columns
        self.snp_columns = self.snp_df.columns[1:].tolist()
        self.num_snps = len(self.snp_columns)
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load SNP data
        snp_row = self.snp_df[self.snp_df['image_name'] == image_name]
        snp_data = snp_row[self.snp_columns].values[0].astype(np.float32)
        snp_tensor = torch.tensor(snp_data, dtype=torch.float32)
        
        return {
            'image': image,
            'snp': snp_tensor,
            'image_name': image_name
        }