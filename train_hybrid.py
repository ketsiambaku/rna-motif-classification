"""
Training script for Hybrid U-Net on RNA motif classification.

Phase 2.1: Density + Sequence features (24 size-invariant features)
Target: Establish 15-class baseline and improve with sequence features
"""

import os
import sys
import time
import json
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
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(density, sequence)
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
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.n_epochs} [Val]  ')
            
            for batch in pbar:
                density = batch['density'].to(self.device)
                sequence = batch['sequence'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(density, sequence)
                loss = self.criterion(logits, labels)
                
                # Statistics
                running_loss += loss.item() * density.size(0)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)
    
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
            val_loss, val_acc, val_preds, val_labels = self.validate(epoch)
            
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
                
                # Save confusion matrix for best model
                np.save(self.save_dir / 'best_val_preds.npy', val_preds)
                np.save(self.save_dir / 'best_val_labels.npy', val_labels)
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
    """Compute per-class precision, recall, F1."""
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        if i < len(precision):
            print(f"{class_name:<12} {precision[i]:>10.4f} {recall[i]:>10.4f} "
                  f"{f1[i]:>10.4f} {support[i]:>10}")
    
    # Macro averages
    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f1 = f1.mean()
    
    print("-" * 60)
    print(f"{'Macro Avg':<12} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    
    return {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'support': support.tolist(),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'confusion_matrix': cm.tolist()
    }


def main():
    """Main training function."""
    # Configuration
    config = {
        'dataset_root': 'dataset2',
        'batch_size': 8,
        'num_workers': 0,  # Use 0 for M1 Mac compatibility
        'n_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
        'use_subset': 0.1,  # 10% subset for Phase 2.1
        'random_seed': 42,
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
    }
    
    # Set random seeds
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path('experiments') / f'hybrid_phase2_1_{timestamp}'
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
        random_seed=config['random_seed']
    )
    
    val_dataset = HybridDataset(
        root_dir=config['dataset_root'],
        split='val',
        use_subset=config['use_subset'],
        random_seed=config['random_seed']
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
    
    # Get class weights
    class_weights = train_dataset.get_class_weights()
    print(f"\nClass weights computed (imbalance ratio: {class_weights.max()/class_weights.min():.2f}x)")
    
    # Create model
    print("\n" + "="*80)
    print("Creating Model")
    print("="*80)
    
    device = torch.device(config['device'])
    model = HybridUNet(n_classes=15).to(device)
    
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
    print("Final Evaluation on Best Model")
    print("="*80)
    
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_preds = np.load(save_dir / 'best_val_preds.npy')
    val_labels = np.load(save_dir / 'best_val_labels.npy')
    
    # Compute detailed metrics
    metrics = compute_class_metrics(val_preds, val_labels, train_dataset.CLASS_NAMES)
    
    # Save metrics
    with open(save_dir / 'final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Training complete! Results saved to: {save_dir}")
    print(f"✓ Best validation accuracy: {trainer.best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
