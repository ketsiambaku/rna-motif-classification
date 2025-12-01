import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import time

from rna_dataset import RNADensityDataset
from unet_classifier_fixed import UNetClassifier

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    for batch_idx, (density, pdb_features, labels) in enumerate(train_loader):
        batch_start = time.time()
        
        density, pdb_features, labels = density.to(device), pdb_features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(density, pdb_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        batch_time = time.time() - batch_start
        print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f} - Time: {batch_time:.2f}s", flush=True)
    
    epoch_time = time.time() - start_time
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss, accuracy, epoch_time

def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (density, pdb_features, labels) in enumerate(val_loader):
            density, pdb_features, labels = density.to(device), pdb_features.to(device), labels.to(device)
            outputs = model(density, pdb_features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 5 == 0:
                print(f"  Val Batch {batch_idx+1}/{len(val_loader)}", flush=True)
    
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = running_loss / len(val_loader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, macro_f1, all_preds, all_labels

def main():
    # Configuration
    base_dir = Path('/Users/ketsiambaku/Repositories/rna-motif-classification')
    dataset_dir = base_dir / 'dataset2'
    train_csv = 'rna_train_dataset2_small.csv'
    val_csv = 'rna_val_dataset2_small.csv'
    
    batch_size = 8  # Smaller batch for faster feedback
    num_epochs = 10
    learning_rate = 0.001
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("✗ MPS not available, using CPU")
    
    print(f"Device: {device}", flush=True)
    
    # Load datasets
    print("\n=== Loading Datasets ===", flush=True)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    train_dataset = RNADensityDataset(train_df, str(dataset_dir))
    val_dataset = RNADensityDataset(val_df, str(dataset_dir))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}", flush=True)
    
    # Create data loaders (num_workers=0 for MPS)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}", flush=True)
    
    # Compute class weights
    print("\n=== Computing Class Weights ===", flush=True)
    class_counts = train_df['label'].value_counts().sort_index()
    total_samples = len(train_df)
    num_classes = len(class_counts)
    class_weights = total_samples / (num_classes * class_counts.values)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    for i, (label, count) in enumerate(class_counts.items()):
        print(f"  Class {i} ({label}): {count} samples, weight={class_weights[i]:.3f}")
    
    # Initialize model
    print("\n=== Initializing Model ===", flush=True)
    pdb_feat_dim = 900  # 30x30 phosphate distance matrix
    model = UNetClassifier(pdb_feat_dim=pdb_feat_dim, num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print("Model moved to device", flush=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\n{'='*50}")
    print(f"Starting Training - {num_epochs} epochs")
    print(f"{'='*50}", flush=True)
    
    best_macro_f1 = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]", flush=True)
        
        # Train
        train_loss, train_acc, train_time = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"  Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}% | Time: {train_time:.1f}s", flush=True)
        
        # Validate
        val_loss, val_acc, macro_f1, val_preds, val_labels = validate(model, val_loader, criterion, device, epoch)
        print(f"  Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}% | Macro-F1: {macro_f1:.4f}", flush=True)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            print(f"  ✓ New best Macro-F1: {best_macro_f1:.4f}", flush=True)
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Macro-F1: {best_macro_f1:.4f}")
    print(f"{'='*50}", flush=True)

if __name__ == '__main__':
    main()
