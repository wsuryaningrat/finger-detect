import sys
from pathlib import Path

import cv2
import numpy as np
from functools import lru_cache

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
from frequency import ridge_freq
from gabor_filter import gabor_filter
from skeletonize import skeletonize
from crossing_number import calculate_minutiaes
import orientation


def extract_minutiae(img):

    block_size = 16

    normalized_img = normalize(
        img.copy(),
        100.0,
        100.0
    )

    segmented_img, normim, mask = (
        create_segmented_and_variance_images(
            normalized_img,
            block_size,
            0.2
        )
    )

    angles = orientation.calculate_angles(
        normalized_img,
        W=block_size,
        smoth=False
    )

    freq = ridge_freq(
        normim,
        mask,
        angles,
        block_size,
        kernel_size=5,
        minWaveLength=5,
        maxWaveLength=15
    )

    gabor_img = gabor_filter(
        normim,
        angles,
        freq
    )

    skeleton = skeletonize(
        gabor_img
    )

    minutiae_result = calculate_minutiaes(
        skeleton
    )

    return {

    "normalized": normalized_img,

    "segmented": segmented_img,

    "gabor": gabor_img,

    "skeleton": skeleton,

    "overlay":
        minutiae_result[
            "overlay"
        ],

    "endings":
        minutiae_result[
            "endings"
        ],

    "bifurcations":
        minutiae_result[
            "bifurcations"
        ]
}