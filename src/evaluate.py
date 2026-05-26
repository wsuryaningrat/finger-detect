from pathlib import Path

import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.metrics import roc_curve
from model import SiameseNetwork

from dataset import FingerprintPairDataset
from utils import get_valid_transform

import torch.nn.functional as F

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "models/best_model.pth"

BATCH_SIZE = 32

# =====================================================
# DEVICE
# =====================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using Device: {DEVICE}")

# =====================================================
# DATASET
# =====================================================

test_dataset = FingerprintPairDataset(
    csv_file="data/pairs_test.csv",
    image_dir="data/processed",
    transform=get_valid_transform()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(
    f"Test Pairs: {len(test_dataset):,}"
)

# =====================================================
# MODEL
# =====================================================

model = SiameseNetwork()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint
)

model = model.to(DEVICE)

model.eval()

# =====================================================
# INFERENCE
# =====================================================

all_labels = []
 
all_distances = []
with torch.no_grad():

    for img1, img2, labels in test_loader:

        img1 = img1.to(DEVICE)

        img2 = img2.to(DEVICE)

        emb1, emb2 = model(
            img1,
            img2
        )

        distance = F.pairwise_distance(
            emb1,
            emb2
        )

        all_distances.extend(
            distance.cpu().numpy()
        )

        labels = labels.numpy()
 

        all_labels.extend(
            labels
        )

all_distances = np.array(
    all_distances
)

all_labels = np.array(
    all_labels
)

# =====================================================
# THRESHOLD
# =====================================================

fpr, tpr, thresholds = roc_curve(
    all_labels,
    -all_distances
)

best_idx = (
    tpr - fpr
).argmax()

best_threshold = thresholds[
    best_idx
]

print(
    f"Best Threshold: "
    f"{best_threshold:.4f}"
)

predictions = (
    -all_distances >
    best_threshold
).astype(int)

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(
    all_labels,
    predictions
)

precision = precision_score(
    all_labels,
    predictions
)

recall = recall_score(
    all_labels,
    predictions
)

f1 = f1_score(
    all_labels,
    predictions
)


roc_auc = roc_auc_score(
    all_labels,
    -all_distances
)
# =====================================================
# OUTPUT
# =====================================================

print("\n========== RESULTS ==========")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    all_labels,
    predictions
)

print("\nConfusion Matrix")

print(cm)