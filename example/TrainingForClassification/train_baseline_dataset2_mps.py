import torch
from torch.utils.data import DataLoader
import pandas as pd
from rna_dataset import RNADensityDataset
from unet_classifier_fixed import UNetClassifier
import torch.nn as nn
import torch.optim as optim
import os
import time
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import numpy as np

print("="*80)
print("DATASET2 TRAINING - OPTIMIZED FOR M1 Mac (MPS)")
print("="*80)

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\n✓ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("\n✓ Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("\n⚠️  Using CPU (slow)")

# Load data
train_df = pd.read_csv('rna_train_dataset2_clean.csv')
val_df = pd.read_csv('rna_val_dataset2_clean.csv')

print(f"\nTraining samples: {len(train_df):,}")
print(f"Validation samples: {len(val_df):,}")
print(f"Number of classes: {train_df['label'].nunique()}")

print(f"\nClass distribution (train):")
class_counts = train_df['label'].value_counts().sort_index()
for label, count in class_counts.items():
    pct = 100 * count / len(train_df)
    print(f"  {label:12s}: {count:5,d} ({pct:5.2f}%)")

# Compute class weights
total_samples = len(train_df)
n_classes = train_df['label'].nunique()
class_weights_dict = {}
for label in sorted(train_df['label'].unique()):
    count = (train_df['label'] == label).sum()
    weight = total_samples / (n_classes * count)
    class_weights_dict[label] = weight

# Create datasets
train_ds = RNADensityDataset(train_df, base_path='../../dataset2/')
val_ds = RNADensityDataset(val_df, base_path='../../dataset2/')

# OPTIMIZED: Larger batch size + parallel data loading
BATCH_SIZE = 16  # Increased from 4
NUM_WORKERS = 4  # Parallel loading

train_loader = DataLoader(
    train_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0  # 0 for MPS due to multiprocessing issues
)
val_loader = DataLoader(
    val_ds, 
    batch_size=BATCH_SIZE, 
    num_workers=0
)

print(f"\nOptimizations enabled:")
print(f"  Batch size: {BATCH_SIZE} (was 4)")
print(f"  GPU acceleration: M1 Metal (MPS)")

# Initialize model
num_classes = train_df['label'].nunique()
model = UNetClassifier(pdb_feat_dim=900, num_classes=num_classes)

# OPTIMIZED: Model compilation (PyTorch 2.0+)
try:
    model = torch.compile(model, mode='default')
    print(f"  Model compilation: enabled")
except:
    print(f"  Model compilation: not available (PyTorch <2.0)")

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")

# Class weights
label_to_idx = {label: i for i, label in enumerate(sorted(train_df['label'].unique()))}
class_weights_tensor = torch.tensor([class_weights_dict[label] 
                                     for label in sorted(train_df['label'].unique())],
                                    dtype=torch.float32).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# OPTIMIZED: Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2
)

# Create output directories
os.makedirs("saved_models_dataset2_mps", exist_ok=True)
os.makedirs("results_dataset2_mps", exist_ok=True)

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
    'val_macro_f1': [],
    'val_balanced_accuracy': [],
    'epoch_time': []
}

# OPTIMIZED: Early stopping
num_epochs = 20
best_val_f1 = 0.0
patience = 3
no_improve_count = 0

print("\n" + "="*80)
print("Starting training with early stopping (patience=3)...")
print("="*80)

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    # Training phase
    model.train()
    train_loss = 0.0
    train_batches = 0
    
    for batch_idx, (vol, pdb_feat, label) in enumerate(train_loader):
        vol, pdb_feat, label = vol.to(device), pdb_feat.to(device), label.to(device)
        
        # Forward pass
        out = model(vol.float(), pdb_feat.float())
        loss = criterion(out, label)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_batches += 1
        
        # Less frequent printing for speed
        if (batch_idx + 1) % 200 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
    avg_train_loss = train_loss / train_batches
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for vol, pdb_feat, label in val_loader:
            vol, pdb_feat, label = vol.to(device), pdb_feat.to(device), label.to(device)
            
            out = model(vol.float(), pdb_feat.float())
            loss = criterion(out, label)
            val_loss += loss.item()
            
            preds = out.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    
    # Compute metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    balanced_acc = 100 * np.mean([
        np.sum((np.array(all_preds) == c) & (np.array(all_labels) == c)) / np.sum(np.array(all_labels) == c)
        for c in range(num_classes) if np.sum(np.array(all_labels) == c) > 0
    ])
    
    epoch_time = time.time() - epoch_start
    
    # Store history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_accuracy'].append(val_accuracy)
    history['val_macro_f1'].append(macro_f1)
    history['val_balanced_accuracy'].append(balanced_acc)
    history['epoch_time'].append(epoch_time)
    
    # Print epoch summary
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss:        {avg_train_loss:.4f}")
    print(f"  Val Loss:          {avg_val_loss:.4f}")
    print(f"  Val Accuracy:      {val_accuracy:.2f}%")
    print(f"  Val Macro-F1:      {macro_f1:.4f}")
    print(f"  Val Balanced Acc:  {balanced_acc:.2f}%")
    print(f"  Time:              {epoch_time:.2f}s")
    
    # Learning rate scheduling
    scheduler.step(macro_f1)
    
    # Save best model
    if macro_f1 > best_val_f1:
        best_val_f1 = macro_f1
        no_improve_count = 0
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'val_macro_f1': macro_f1,
            'val_balanced_accuracy': balanced_acc,
        }, "saved_models_dataset2_mps/best_model.pt")
        print(f"  ✓ Saved best model (macro-F1: {macro_f1:.4f})")
    else:
        no_improve_count += 1
        print(f"  No improvement ({no_improve_count}/{patience})")
    
    # Early stopping
    if no_improve_count >= patience:
        print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
        print(f"  Best validation macro-F1: {best_val_f1:.4f}")
        break
    
    print()

# Final evaluation
print("\n" + "="*80)
print("Training Complete!")
print("="*80)
print(f"Best validation macro-F1: {best_val_f1:.4f}")
print(f"Average epoch time: {np.mean(history['epoch_time']):.2f}s")
print(f"Total training time: {sum(history['epoch_time'])/60:.2f} minutes")

# Load best model
checkpoint = torch.load("saved_models_dataset2_mps/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Final predictions
all_preds = []
all_labels = []
with torch.no_grad():
    for vol, pdb_feat, label in val_loader:
        vol, pdb_feat, label = vol.to(device), pdb_feat.to(device), label.to(device)
        out = model(vol.float(), pdb_feat.float())
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

# Convert to label names
label_names = sorted(train_df['label'].unique())
idx_to_label = {i: label for i, label in enumerate(label_names)}
pred_labels = [idx_to_label[p] for p in all_preds]
true_labels = [idx_to_label[l] for l in all_labels]

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=label_names)
print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nDetailed Classification Report:")
print(classification_report(true_labels, pred_labels, labels=label_names, zero_division=0))

# Per-class performance
print("\nPer-Class Performance:")
print(f"{'Class':<12} {'Samples':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-"*70)
for i, label in enumerate(label_names):
    n_samples = np.sum(np.array(all_labels) == i)
    if n_samples > 0:
        accuracy = 100 * cm[i, i] / np.sum(cm[i, :])
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
        fn = np.sum(cm[i, :]) - tp
        recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{label:<12} {n_samples:8d} {accuracy:9.2f}% {precision:9.2f}% {recall:9.2f}% {f1:9.2f}%")

# Save results
np.save('results_dataset2_mps/training_history.npy', history)
np.save('results_dataset2_mps/confusion_matrix.npy', cm)
np.save('results_dataset2_mps/predictions.npy', {
    'preds': pred_labels, 
    'labels': true_labels,
    'label_names': label_names
})

print("\n" + "="*80)
print("Results saved to results_dataset2_mps/ directory")
print("Models saved to saved_models_dataset2_mps/ directory")
print("="*80)
