from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
import os
import torch
import numpy as np

class LineartColorDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.lineart_path = config["lineart_path"]
        self.color_ref_path = config["color_ref_path"]
        self.mapping_csv = config["mapping_csv"]
        
        # Load the mapping CSV file
        self.mapping_df = pd.read_csv(self.mapping_csv)
        
        # Image transformations
        self.transform = T.Compose([
            T.Resize((config["target_size"], config["target_size"])),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        
        self.condition_transform = T.Compose([
            T.Resize((config["condition_size"], config["condition_size"])),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.mapping_df)

    def __getitem__(self, idx):
        # Get image paths from CSV
        row = self.mapping_df.iloc[idx]
        img_dir = row["dir"]
        color_name = os.path.join(img_dir.split("/")[-1], row["target"])
        ref_name = os.path.join(img_dir.split("/")[-1], row["reference"])

        # Load images
        lineart_path = os.path.join(self.lineart_path, color_name)
        color_path = os.path.join(self.color_ref_path, color_name)
        ref_path = os.path.join(self.color_ref_path, ref_name)
        
        lineart_img = Image.open(lineart_path).convert("RGB")
        color_img = Image.open(color_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")
        
        # Apply transformations
        lineart_tensor = self.condition_transform(lineart_img)
        ref_tensor = self.condition_transform(ref_img)
        color_tensor = self.transform(color_img)
        
        return {
            "image": color_tensor,  # Target image (color)
            "condition_1": lineart_tensor,  # lineart as condition
            "condition_2": ref_tensor,  # reference as condition
            "condition_type_1": "lineart",
            "condition_type_2": "reference",
            "description": "",  # Empty description as we don't have text
            "position_delta": np.array([0, 0])  # Use numpy.array with two values
        }
