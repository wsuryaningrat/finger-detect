import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

MINUTIAE_DIR = (
    ROOT /
    "external" /
    "fingerprint-minutiae-extraction"
)

sys.path.append(
    str(MINUTIAE_DIR)
)

from normalization import normalize
from segmentation import create_segmented_and_variance_images
from orientation import calculate_angles
from frequency import ridge_freq
from gabor_filter import gabor_filter
from skeletonize import skeletonize
from crossing_number import calculate_minutiaes

IMG_PATH = (
    ROOT /
    "data" /
    "processed" /
    "Real" /
    "298__M_Right_thumb_finger.BMP"
)

img = cv2.imread(
    str(IMG_PATH),
    0
)

print(
    "Image Shape:",
    img.shape
)