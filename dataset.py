import os
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio


class StrokeImageDataset(Dataset):
    """
    PyTorch Dataset for stroke outcome prediction from MRI or CT scans.

    - Supports consistent padding/cropping to unify spatial dimensions.
    - Includes light augmentation during training (flip, affine, noise, blur).
    - Assumes a CSV file with columns: ['patient', 'mrs_90d', 'split'].
      * 'patient' — case identifier (without file extension)
      * 'mrs_90d' — 90-day modified Rankin Scale (mRS)
      * 'split'   — dataset split label ('train', 'val', or 'test')
    """

    def __init__(
        self,
        csv_path,
        image_dir,
        split: str = 'train',
        target_shape=(30, 256, 256),
        return_meta: bool = False
    ):
        """
        Args:
            csv_path (str): Path to CSV file containing patient info and labels.
            image_dir (str): Directory containing preprocessed NIfTI images.
            split (str): Dataset split to load ('train', 'val', 'test').
            target_shape (tuple): Desired (Z, Y, X) image shape.
            return_meta (bool): If True, returns patient ID along with data.
        """
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.image_dir = image_dir
        self.split = split
        self.return_meta = return_meta

        # Ensure consistent image shape via padding or cropping
        self.resizing_transform = tio.CropOrPad(target_shape)

        # Define augmentations for training only
        if self.split == 'train':
            self.augment_transform = tio.Compose([
                tio.RandomFlip(axes=('LR',), flip_probability=0.5),
                tio.RandomAffine(
                    scales=(0.9, 1.2),
                    degrees=10,
                    isotropic=False,
                    default_pad_value='minimum'
                ),
                tio.RandomNoise(p=0.25),
                tio.RandomBlur(p=0.25),
            ])
        else:
            self.augment_transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row['patient']).strip()
        filename = patient_id + ".nii.gz"
        filepath = os.path.join(self.image_dir, filename)

        # Binary outcome: 1 = poor (mRS >= 3), 0 = favorable (mRS < 3)
        label = 1 if row['mrs_90d'] >= 3 else 0

        # Load and normalize the NIfTI image
        img = nib.load(filepath).get_fdata()
        img = np.nan_to_num(img)
        img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Convert to tensor and reorder dimensions: (H, W, D) → (C, Z, Y, X)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        img_tensor = img_tensor.permute(0, 3, 2, 1)
        img_tensor = self.resizing_transform(img_tensor)

        # Apply augmentation for training
        if self.augment_transform is not None:
            img_tensor = self.augment_transform(img_tensor)

        label_tensor = torch.tensor([label], dtype=torch.float32)

        if self.return_meta:
            return img_tensor, label_tensor, patient_id
        else:
            return img_tensor, label_tensor
