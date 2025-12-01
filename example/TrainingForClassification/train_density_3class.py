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

from dataset_coarse import CoarseDataset
from unet_density_only import DensityOnlyUNet

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    for batch_idx, (density, labels) in enumerate(train_loader):
        density, labels = density.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(density)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}", flush=True)
    
    epoch_time = time.time() - start_time
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss, accuracy, epoch_time

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for density, labels in val_loader:
            density, labels = density.to(device), labels.to(device)
            outputs = model(density)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = running_loss / len(val_loader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, macro_f1, all_preds, all_labels

def main():
    # Configuration
    base_dir = Path('/Users/ketsiambaku/Repositories/rna-motif-classification')
    dataset_dir = base_dir / 'dataset2'
    train_csv = 'rna_train_3class_small.csv'
    val_csv = 'rna_val_3class_small.csv'
    
    batch_size = 8
    num_epochs = 15
    learning_rate = 0.001
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    print(f"\n{'='*70}")
    print("DENSITY-ONLY 3-CLASS COARSE CLASSIFICATION")
    print(f"{'='*70}")
    print(f"Task: Classify bulge vs hairpin vs internal (topology learning)")
    print(f"Expected accuracy: 60-80% (much more learnable than 14-class)")
    print(f"{'='*70}\n")
    
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # Load original fine-grained labels for folder mapping
    train_df_fine = pd.read_csv('rna_train_dataset2_small.csv')
    val_df_fine = pd.read_csv('rna_val_dataset2_small.csv')
    
    # Rename column for compatibility
    train_df = train_df.rename(columns={'coarse_label': 'label'})
    val_df = val_df.rename(columns={'coarse_label': 'label'})
    
    train_dataset = CoarseDataset(train_df, train_df_fine, str(dataset_dir))
    val_dataset = CoarseDataset(val_df, val_df_fine, str(dataset_dir))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    class_names = sorted(train_df['label'].unique())
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}\n")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Compute class weights
    print("Computing class weights...")
    class_counts = train_df['label'].value_counts().sort_index()
    total_samples = len(train_df)
    class_weights = total_samples / (num_classes * class_counts.values)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print("\nClass distribution:")
    for i, (label, count) in enumerate(class_counts.items()):
        print(f"  {label}: {count} samples (weight={class_weights[i]:.3f})")
    
    # Initialize model
    print("\nInitializing Density-Only U-Net (3-class)...")
    model = DensityOnlyUNet(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting Training - {num_epochs} epochs")
    print(f"{'='*70}\n")
    
    best_macro_f1 = 0.0
    best_accuracy = 0.0
    patience_counter = 0
    patience_limit = 5
    
    results = []
    
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch+1}/{num_epochs}]")
        
        # Train
        train_loss, train_acc, train_time = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"  Train - Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}% | Time: {train_time:.1f}s")
        
        # Validate
        val_loss, val_acc, macro_f1, val_preds, val_labels = validate(model, val_loader, criterion, device)
        print(f"  Val   - Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}% | Macro-F1: {macro_f1:.4f}")
        
        # Learning rate scheduling
        scheduler.step(macro_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track best model
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_accuracy = val_acc
            patience_counter = 0
            print(f"  ✓ New best Macro-F1: {best_macro_f1:.4f} (Acc: {best_accuracy:.2f}%)")
            torch.save(model.state_dict(), 'best_density_3class_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\n  Early stopping triggered (no improvement for {patience_limit} epochs)")
                break
        
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'macro_f1': macro_f1
        })
        
        print()
    
    print(f"{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Best Macro-F1: {best_macro_f1:.4f}")
    print(f"{'='*70}\n")
    
    # Generate final predictions
    print("Generating final evaluation...")
    model.load_state_dict(torch.load('best_density_3class_model.pth'))
    val_loss, val_acc, macro_f1, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    # Print classification report
    print("\nPer-Class Performance:")
    print(classification_report(val_labels, val_preds, target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    print("\nConfusion Matrix:")
    print(f"         {' '.join(f'{c:>8}' for c in class_names)}")
    for i, row_label in enumerate(class_names):
        print(f"{row_label:>8} {' '.join(f'{cm[i,j]:>8}' for j in range(len(class_names)))}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('density_3class_training_results.csv', index=False)
    print("\n✓ Saved: density_3class_training_results.csv")
    
    # Final comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON: Fine-Grained vs Coarse Classification")
    print(f"{'='*70}")
    print(f"{'Task':<35} {'Accuracy':<15} {'Macro-F1':<15}")
    print("-" * 70)
    print(f"{'14-class (PDB-only, leakage)':<35} {'98.8%':<15} {'~1.000':<15}")
    print(f"{'14-class (density-only)':<35} {'21.99%':<15} {'0.0621':<15}")
    print(f"{'3-class (density-only)':<35} {f'{val_acc:.2f}%':<15} {f'{macro_f1:.4f}':<15}")
    print(f"{'='*70}")
    
    if val_acc > 50:
        print("\n✓ 3-class accuracy > 50% shows model CAN learn topology!")
        print("  The model successfully distinguishes bulge vs hairpin vs internal")
        print("  Fine-grained size classification is just much harder from density alone")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
