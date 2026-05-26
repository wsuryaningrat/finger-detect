from dataset import (
    FingerprintPairDataset
)

from utils import (
    get_transform
)

dataset = FingerprintPairDataset(
    csv_file="data/pairs_train.csv",
    image_dir="data/processed/Real",
    transform=get_transform()
)

print(
    "Dataset Size:",
    len(dataset)
)

img1, img2, label = dataset[0]

print(
    "Image 1:",
    img1.shape
)

print(
    "Image 2:",
    img2.shape
)

print(
    "Label:",
    label
)