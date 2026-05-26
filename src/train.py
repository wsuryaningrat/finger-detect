from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import FingerprintPairDataset
  
from model import SiameseNetwork
from loss import ContrastiveLoss

from utils import (
    get_train_transform,
    get_valid_transform
)

# =====================================================
# CONFIG
# =====================================================

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-4

PATIENCE = 5

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    exist_ok=True
)

# =====================================================
# DEVICE
# =====================================================

if torch.backends.mps.is_available():

    DEVICE = "mps"

elif torch.cuda.is_available():

    DEVICE = "cuda"

else:

    DEVICE = "cpu"

print(f"\nUsing Device: {DEVICE}")

# =====================================================
# DATASET
# =====================================================

transform = get_train_transform()

train_dataset = FingerprintPairDataset(
    csv_file="data/pairs_train.csv",
    image_dir="data/processed",
    transform=get_train_transform()
)
 

valid_dataset = FingerprintPairDataset(
    csv_file="data/pairs_valid.csv",
    image_dir="data/processed",
    transform=get_valid_transform()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    f"Train Pairs: {len(train_dataset):,}"
)

print(
    f"Valid Pairs: {len(valid_dataset):,}"
)

# =====================================================
# MODEL
# =====================================================
model = SiameseNetwork().to(
    DEVICE
)

criterion = ContrastiveLoss(
    margin=1.0
)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

# =====================================================
# TRAIN STEP
# =====================================================

def train_one_epoch():

    model.train()

    running_loss = 0

    for img1, img2, labels in tqdm(
        train_loader,
        leave=False
    ):

        img1 = img1.to(
            DEVICE
        )

        img2 = img2.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        emb1, emb2 = model(
            img1,
            img2
        )

        labels = labels.float()


        
        loss = criterion(
            emb1,
            emb2,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
        )

    return (
        running_loss /
        len(train_loader)
    )

# =====================================================
# VALIDATION STEP
# =====================================================

@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0

    for img1, img2, labels in valid_loader:

        img1 = img1.to(
            DEVICE
        )

        img2 = img2.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        emb1, emb2 = model(
        img1,
        img2
        )

        labels = labels.float()

        loss = criterion(
            emb1,
            emb2,
            labels
        )

        running_loss += (
            loss.item()
        )

    return (
        running_loss /
        len(valid_loader)
    )

# =====================================================
# MAIN TRAIN LOOP
# =====================================================

best_valid_loss = float(
    "inf"
)

patience_counter = 0

history = []

for epoch in range(EPOCHS):

    train_loss = train_one_epoch()

    valid_loss = validate()

    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss
        }
    )

    print(
        f"\nEpoch [{epoch+1}/{EPOCHS}]"
    )

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Valid Loss : {valid_loss:.4f}"
    )

    # --------------------------------

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        patience_counter = 0

        torch.save(
            model.state_dict(),
            "models/best_model.pth"
        )

        print(
            "✓ Best model saved"
        )

    else:

        patience_counter += 1

        print(
            f"Patience: "
            f"{patience_counter}/{PATIENCE}"
        )

    # --------------------------------

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping triggered"
        )

        break

print("\nTraining Finished")