import torch
from torch.utils.data import DataLoader
import pandas as pd
from rna_dataset import RNADensityDataset
from unet_classifier_fixed import UNetClassifier
import torch.nn as nn
import torch.optim as optim
import os
import time
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Load data
train_df = pd.read_csv('rna_train.csv')
val_df = pd.read_csv('rna_validation_labels.csv')

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Class distribution (train):\n{train_df['label'].value_counts()}\n")

# Create datasets
train_ds = RNADensityDataset(train_df, base_path='../../dataset/')
val_ds = RNADensityDataset(val_df, base_path='../../dataset/')

# Create dataloaders
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=4, num_workers=0)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Initialize model
num_classes = len(train_df['label'].unique())
model = UNetClassifier(pdb_feat_dim=900, num_classes=num_classes).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}\n")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Create output directory
os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
    'epoch_time': []
}

# Training loop
num_epochs = 10
best_val_acc = 0.0

print("Starting training...\n")
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
        
        if (batch_idx + 1) % 100 == 0:
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
    epoch_time = time.time() - epoch_start
    
    # Store history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_accuracy'].append(val_accuracy)
    history['epoch_time'].append(epoch_time)
    
    # Print epoch summary
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Val Accuracy: {val_accuracy:.2f}%")
    print(f"  Time: {epoch_time:.2f}s")
    
    # Save best model
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
        }, "saved_models/best_model.pt")
        print(f"  ✓ Saved best model (accuracy: {val_accuracy:.2f}%)")
    
    # Save checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
    }, f"saved_models/checkpoint_epoch{epoch+1}.pt")
    
    print()

# Final evaluation
print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Average epoch time: {np.mean(history['epoch_time']):.2f}s")

# Load best model for final evaluation
checkpoint = torch.load("saved_models/best_model.pt")
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

# Confusion matrix
label_names = sorted(train_df['label'].unique())
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(f"Labels: {label_names}")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_names))

# Save results
np.save('results/training_history.npy', history)
np.save('results/confusion_matrix.npy', cm)
np.save('results/predictions.npy', {'preds': all_preds, 'labels': all_labels})

print("\nResults saved to results/ directory")
print("Models saved to saved_models/ directory")
