from pathlib import Path

import cv2
import pandas as pd
import torch

from torch.utils.data import Dataset


class FingerprintPairDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_dir,
        transform=None
    ):

        self.df = pd.read_csv(
            csv_file
        )

        self.image_dir = Path(
            image_dir
        )

        self.transform = transform

    def __len__(self):

        return len(
            self.df
        )

    def load_image(
        self,
        filename
    ):

        img_path = (
            self.image_dir /
            filename
        )

        img = cv2.imread(
            str(img_path),
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:

            raise FileNotFoundError(
                f"Cannot load: {img_path}"
            )

        return img

    def __getitem__(
        self,
        idx
    ):

        row = self.df.iloc[idx]

        img1 = self.load_image(
            row["img1"]
        )

        img2 = self.load_image(
            row["img2"]
        )

        label = row["label"]

        if self.transform:

            img1 = self.transform(
                img1
            )

            img2 = self.transform(
                img2
            )

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return (
            img1,
            img2,
            label
        )