import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
from PIL import Image
from torchvision import transforms


class RSNADataset(Dataset):

    def __init__(self, csv_file, split, transform=None):

        self.df = pd.read_csv(csv_file)

        # Select requested split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.transform = transform

        print(f"{split} images: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # Read DICOM
        dicom = pydicom.dcmread(row["dicom_path"])

        # Extract pixel data
        image = dicom.pixel_array.astype(np.float32)

        # Normalize pixel values to 0-255
        image -= image.min()

        if image.max() > 0:
            image /= image.max()

        image *= 255.0

        image = image.astype(np.uint8)

        # Convert grayscale to PIL image
        image = Image.fromarray(image).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Binary label
        label = torch.tensor(
            row["pneumonia"],
            dtype=torch.long
        )

        return image, label


def get_transforms(train=True):

    if train:

        return transforms.Compose([
            transforms.Resize((224, 224)),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomRotation(degrees=5),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    else:

        return transforms.Compose([
            transforms.Resize((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])