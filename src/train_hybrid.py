"""
Training script for Hybrid U-Net on RNA motif classification.

Phase 2.1: Density + Sequence features (24 size-invariant features)
Target: Establish 15-class baseline and improve with sequence features
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.hybrid_dataset import HybridDataset
from models.hybrid_unet import HybridUNet


class Trainer:
    """Trainer for Hybrid U-Net model."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_dir,
        n_epochs=50,
        early_stopping_patience=10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.n_epochs} [Train]')
        
        for batch in pbar:
            density = batch['density'].to(self.device)
            sequence = batch['sequence'].to(self.device)
            pairing = batch['pairing'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(density, sequence, pairing)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * density.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate on validation set."""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        all_probs = []  # Store probabilities for ROC curves
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.n_epochs} [Val]  ')
            
            for batch in pbar:
                density = batch['density'].to(self.device)
                sequence = batch['sequence'].to(self.device)
                pairing = batch['pairing'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(density, sequence, pairing)
                loss = self.criterion(logits, labels)
                
                # Get probabilities for ROC curves
                probs = torch.softmax(logits, dim=1)
                
                # Statistics
                running_loss += loss.item() * density.size(0)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for confusion matrix and ROC curves
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def train(self):
        """Full training loop."""
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.n_epochs}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Save directory: {self.save_dir}")
        print("="*80 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.n_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels, val_probs = self.validate(epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.n_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                
                checkpoint_path = self.save_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'history': self.history
                }, checkpoint_path)
                
                print(f"  ✓ New best model saved! Val Acc: {val_acc:.2f}%")
                
                # Save confusion matrix and probabilities for best model
                np.save(self.save_dir / 'best_val_preds.npy', val_preds)
                np.save(self.save_dir / 'best_val_labels.npy', val_labels)
                np.save(self.save_dir / 'best_val_probs.npy', val_probs)
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.early_stopping_patience})")
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'history': self.history
                }, checkpoint_path)
            
            print("-" * 80)
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("Training Complete!")
        print(f"{'='*80}")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})")
        print(f"Total Time: {elapsed_time/3600:.2f} hours")
        print(f"{'='*80}\n")
        
        # Save training history
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def compute_class_metrics(preds, labels, class_names):
    """Compute per-class precision, recall, F1, and accuracy."""
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
    
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Per-class accuracy (diagonal of normalized confusion matrix)
    per_class_accuracy = np.zeros(len(class_names))
    for i in range(len(class_names)):
        if i < len(cm) and support[i] > 0:
            per_class_accuracy[i] = cm[i, i] / support[i]
    
    # Overall accuracy
    overall_acc = accuracy_score(labels, preds)
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 78)
    
    for i, class_name in enumerate(class_names):
        if i < len(precision):
            print(f"{class_name:<18} {per_class_accuracy[i]:>10.4f} {precision[i]:>10.4f} "
                  f"{recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")
    
    # Macro averages
    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f1 = f1.mean()
    macro_acc = per_class_accuracy.mean()
    
    print("-" * 78)
    print(f"{'Macro Avg':<18} {macro_acc:>10.4f} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f}")
    print(f"{'Overall Accuracy':<18} {overall_acc:>10.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f"{'':>18}", end='')
    for name in class_names:
        print(f"{name[:8]:>9}", end='')
    print()
    print("-" * (18 + 9 * len(class_names)))
    
    for i, true_class in enumerate(class_names):
        if i < len(cm):
            print(f"{true_class:<18}", end='')
            for j in range(len(class_names)):
                if j < len(cm[i]):
                    print(f"{cm[i][j]:>9}", end='')
                else:
                    print(f"{'0':>9}", end='')
            print()
    
    return {
        'accuracy': per_class_accuracy.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'support': support.tolist(),
        'macro_accuracy': float(macro_acc),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'overall_accuracy': float(overall_acc),
        'confusion_matrix': cm.tolist()
    }


def main():
    """Main training function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Hybrid U-Net for RNA motif classification')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--use-subset', type=float, default=0.1, help='Fraction of dataset to use (0.0-1.0)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/mps/cpu/auto)')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--consolidate', action='store_true', help='Use 6-class consolidation instead of 15 classes')
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device_str = 'cuda'
        elif torch.backends.mps.is_available():
            device_str = 'mps'
        else:
            device_str = 'cpu'
    else:
        device_str = args.device
    
    # Configuration
    config = {
        'dataset_root': 'dataset2',
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'n_epochs': 100,  # Increased for full dataset training
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'early_stopping_patience': 7,  # Reduced from 10 to prevent overfitting
        'use_subset': args.use_subset,
        'random_seed': 42,
        'device': device_str,
        'consolidate': args.consolidate
    }
    
    # Set random seeds
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = '_6class' if config['consolidate'] else '_15class'
    save_dir = Path('experiments') / f'hybrid_phase2_1{mode_suffix}_{timestamp}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("="*80)
    print("Hybrid U-Net Training - Phase 2.1")
    print("="*80)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create datasets
    print("\n" + "="*80)
    print("Loading Datasets")
    print("="*80)
    
    train_dataset = HybridDataset(
        root_dir=config['dataset_root'],
        split='train',
        use_subset=config['use_subset'],
        random_seed=config['random_seed'],
        consolidate=config['consolidate']
    )
    
    val_dataset = HybridDataset(
        root_dir=config['dataset_root'],
        split='val',
        use_subset=config['use_subset'],
        random_seed=config['random_seed'],
        consolidate=config['consolidate']
    )
    
    test_dataset = HybridDataset(
        root_dir=config['dataset_root'],
        split='test',
        use_subset=config['use_subset'],
        random_seed=config['random_seed'],
        consolidate=config['consolidate']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Get class weights
    class_weights = train_dataset.get_class_weights()
    print(f"\nClass weights computed (imbalance ratio: {class_weights.max()/class_weights.min():.2f}x)")
    
    # Create model
    print("\n" + "="*80)
    print("Creating Model")
    print("="*80)
    
    device = torch.device(config['device'])
    n_classes = 6 if config['consolidate'] else 15
    model = HybridUNet(n_classes=n_classes).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: HybridUNet")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        n_epochs=config['n_epochs'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    # Train
    history = trainer.train()
    
    # Load best model and evaluate
    print("\n" + "="*80)
    print("Final Evaluation on Validation Set (Best Model)")
    print("="*80)
    
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_preds = np.load(save_dir / 'best_val_preds.npy')
    val_labels = np.load(save_dir / 'best_val_labels.npy')
    
    # Compute detailed metrics
    class_names = train_dataset.CONSOLIDATED_CLASS_NAMES if config['consolidate'] else train_dataset.CLASS_NAMES
    val_metrics = compute_class_metrics(val_preds, val_labels, class_names)
    
    # Save validation metrics
    with open(save_dir / 'final_val_metrics.json', 'w') as f:
        json.dump(val_metrics, f, indent=2)
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("Final Evaluation on Test Set (Best Model)")
    print("="*80)
    
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []  # Store probabilities for ROC curves
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            density = batch['density'].to(device)
            sequence = batch['sequence'].to(device)
            pairing = batch['pairing'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(density, sequence, pairing)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)
    
    # Compute test metrics
    test_metrics = compute_class_metrics(test_preds, test_labels, class_names)
    
    # Save test predictions, probabilities, and metrics
    np.save(save_dir / 'test_preds.npy', test_preds)
    np.save(save_dir / 'test_labels.npy', test_labels)
    np.save(save_dir / 'test_probs.npy', test_probs)
    with open(save_dir / 'final_test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Calculate test accuracy
    test_acc = 100.0 * np.sum(test_preds == test_labels) / len(test_labels)
    
    print(f"\n✓ Training complete! Results saved to: {save_dir}")
    print(f"✓ Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"✓ Final test accuracy: {test_acc:.2f}%")


if __name__ == '__main__':
    main()
