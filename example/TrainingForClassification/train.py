import torch
from torch.utils.data import DataLoader
import pandas as pd
from rna_dataset import RNADensityDataset
from unet_classifier import UNetClassifier
import torch.nn as nn
import torch.optim as optim
import os

train_df = pd.read_csv('rna_labels.csv')

val_df = pd.read_csv('rna_validation_labels.csv')

train_ds = RNADensityDataset(train_df, base_path='dataset/')
val_ds = RNADensityDataset(val_df, base_path='validation')

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetClassifier(pdb_feat_dim=900, num_classes=len(train_df['label'].unique())).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

os.makedirs("saved_models", exist_ok=True)

for epoch in range(10):
    model.train()
    total_loss = 0

    for vol, pdb_feat, label in train_loader:
        vol, pdb_feat, label = vol.to(device), pdb_feat.to(device), label.to(device)

        out = model(vol.float(), pdb_feat.float())
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Training Loss: {total_loss:.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for vol, pdb_feat, label in val_loader:
            vol, pdb_feat, label = vol.to(device), pdb_feat.to(device), label.to(device)

            preds = model(vol.float(), pdb_feat.float()).argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1} - Validation Accuracy: {val_accuracy:.2f}%")

    torch.save(model.state_dict(), f"saved_models/unet_epoch{epoch+1}.pt")
    print(f"Saved model checkpoint: unet_epoch{epoch+1}.pt\n")
