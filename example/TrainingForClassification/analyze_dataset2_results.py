import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from rna_dataset import RNADensityDataset
from unet_classifier_fixed import UNetClassifier

def analyze_pdb_features(csv_file, dataset_dir, num_samples=50):
    """Analyze PDB features to detect potential size leakage"""
    df = pd.read_csv(csv_file)
    dataset = RNADensityDataset(df, str(dataset_dir))
    
    print(f"\n{'='*60}")
    print(f"PDB Feature Analysis: {csv_file}")
    print(f"{'='*60}")
    
    # Sample from each class
    class_samples = {}
    for label in df['label'].unique():
        class_df = df[df['label'] == label]
        sample_size = min(5, len(class_df))
        class_samples[label] = class_df.sample(n=sample_size, random_state=42)
    
    # Analyze PDB features for each class
    feature_stats = {}
    
    for label, samples in class_samples.items():
        features_list = []
        for idx in samples.index:
            _, pdb_features, _ = dataset[idx]
            features_list.append(pdb_features.numpy())
        
        features_array = np.array(features_list)
        
        # Compute statistics
        sparsity = np.mean(features_array == 0)
        nonzero_count = np.sum(features_array != 0, axis=1).mean()
        mean_val = features_array[features_array != 0].mean() if np.any(features_array != 0) else 0
        std_val = features_array[features_array != 0].std() if np.any(features_array != 0) else 0
        
        feature_stats[label] = {
            'sparsity': sparsity,
            'nonzero_count': nonzero_count,
            'mean': mean_val,
            'std': std_val,
            'shape': features_array.shape
        }
    
    # Print statistics
    print(f"\nFeature Statistics (900 features = 30x30 P-P distance matrix):")
    print(f"{'Class':<12} {'Sparsity':<12} {'NonZero':<12} {'Mean':<12} {'Std':<12}")
    print("-" * 60)
    
    for label in sorted(feature_stats.keys()):
        stats = feature_stats[label]
        print(f"{label:<12} {stats['sparsity']:<12.2%} {stats['nonzero_count']:<12.1f} "
              f"{stats['mean']:<12.3f} {stats['std']:<12.3f}")
    
    # Check for size-based patterns
    print("\n⚠️  LEAKAGE ANALYSIS:")
    sparsity_values = [stats['sparsity'] for stats in feature_stats.values()]
    nonzero_values = [stats['nonzero_count'] for stats in feature_stats.values()]
    
    sparsity_range = max(sparsity_values) - min(sparsity_values)
    nonzero_range = max(nonzero_values) - min(nonzero_values)
    
    print(f"  Sparsity range: {sparsity_range:.2%} (>{10}% = potential leakage)")
    print(f"  NonZero count range: {nonzero_range:.1f}")
    
    if sparsity_range > 0.10:
        print(f"  ❌ HIGH LEAKAGE RISK: Sparsity varies {sparsity_range:.1%} across classes")
        print(f"     The PDB phosphate distance matrix encodes motif SIZE through sparsity!")
        return True
    else:
        print(f"  ✓ Low leakage risk: Sparsity variation is small")
        return False
    
    return feature_stats

def evaluate_model(model, loader, device, class_names):
    """Evaluate model and return predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for density, pdb_features, labels in loader:
            pdb_features, labels = pdb_features.to(device), labels.to(device)
            outputs = model(pdb_features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title(f'{title} - Counts')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
    ax2.set_title(f'{title} - Normalized by Row')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()

def analyze_predictions(y_true, y_pred, y_probs, class_names):
    """Analyze prediction patterns"""
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    
    # Overall metrics
    accuracy = np.mean(y_true == y_pred)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    # Per-class analysis
    print("\nPer-Class Performance:")
    print(f"{'Class':<12} {'Samples':<10} {'Correct':<10} {'Accuracy':<12} {'Avg Conf':<12}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            class_correct = np.sum((y_true == y_pred) & mask)
            class_accuracy = class_correct / mask.sum()
            avg_confidence = y_probs[mask, i].mean()
            
            print(f"{class_name:<12} {mask.sum():<10} {class_correct:<10} "
                  f"{class_accuracy:<12.2%} {avg_confidence:<12.3f}")
    
    # Check for perfect predictions (suspicious)
    perfect_classes = []
    for i, class_name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            class_accuracy = np.sum((y_true == y_pred) & mask) / mask.sum()
            if class_accuracy == 1.0:
                perfect_classes.append(class_name)
    
    if len(perfect_classes) == len(class_names):
        print(f"\n⚠️  ALL CLASSES HAVE 100% ACCURACY - LIKELY DATA LEAKAGE!")
    elif len(perfect_classes) > len(class_names) * 0.7:
        print(f"\n⚠️  {len(perfect_classes)}/{len(class_names)} classes have 100% accuracy - possible leakage")
    
    # Confidence analysis
    print("\n" + "-"*60)
    print("Confidence Score Analysis:")
    max_probs = y_probs.max(axis=1)
    print(f"  Mean confidence: {max_probs.mean():.3f}")
    print(f"  Median confidence: {np.median(max_probs):.3f}")
    print(f"  Min confidence: {max_probs.min():.3f}")
    print(f"  Predictions with >99% confidence: {np.sum(max_probs > 0.99)} / {len(max_probs)}")
    
    if max_probs.mean() > 0.95:
        print(f"  ⚠️  Very high average confidence suggests overfitting or leakage")

def main():
    # Configuration
    base_dir = Path('/Users/ketsiambaku/Repositories/rna-motif-classification')
    dataset_dir = base_dir / 'dataset2'
    
    # Check for MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # ===================================================================
    # STEP 1: Analyze PDB Features for Data Leakage
    # ===================================================================
    print("\n" + "="*70)
    print("STEP 1: DATA LEAKAGE DETECTION")
    print("="*70)
    
    train_leakage = analyze_pdb_features('rna_train_dataset2_small.csv', dataset_dir)
    val_leakage = analyze_pdb_features('rna_val_dataset2_small.csv', dataset_dir)
    
    # ===================================================================
    # STEP 2: Train a model on PDB features ONLY (no density)
    # ===================================================================
    print("\n" + "="*70)
    print("STEP 2: PDB-ONLY MODEL TEST (to prove leakage)")
    print("="*70)
    
    train_df = pd.read_csv('rna_train_dataset2_small.csv')
    val_df = pd.read_csv('rna_val_dataset2_small.csv')
    
    class_names = sorted(train_df['label'].unique())
    num_classes = len(class_names)
    
    print(f"\nClasses ({num_classes}): {class_names}")
    
    # Create simple MLP that uses ONLY PDB features
    class PDFOnlyClassifier(nn.Module):
        def __init__(self, pdb_feat_dim, num_classes):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(pdb_feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, pdb_features):
            return self.fc(pdb_features)
    
    print("\nTraining PDB-only classifier (ignoring density maps)...")
    
    train_dataset = RNADensityDataset(train_df, str(dataset_dir))
    val_dataset = RNADensityDataset(val_df, str(dataset_dir))
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    pdb_model = PDFOnlyClassifier(pdb_feat_dim=900, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pdb_model.parameters(), lr=0.001)
    
    # Train for just 5 epochs
    print("\nTraining progress:")
    for epoch in range(5):
        pdb_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for density, pdb_features, labels in train_loader:
            pdb_features, labels = pdb_features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = pdb_model(pdb_features)  # Using ONLY PDB features!
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validate
        pdb_model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for density, pdb_features, labels in val_loader:
                pdb_features, labels = pdb_features.to(device), labels.to(device)
                outputs = pdb_model(pdb_features)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        print(f"  Epoch {epoch+1}/5 - Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
    
    # ===================================================================
    # STEP 3: Generate Confusion Matrices
    # ===================================================================
    print("\n" + "="*70)
    print("STEP 3: CONFUSION MATRIX GENERATION")
    print("="*70)
    
    # Get predictions from PDB-only model
    y_true_val, y_pred_val, y_probs_val = evaluate_model(
        pdb_model, val_loader, device, class_names
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true_val, y_pred_val, class_names,
        title="Dataset2 Validation (PDB-only)",
        filename="confusion_matrix_dataset2_pdb_only.png"
    )
    
    # Analyze predictions
    analyze_predictions(y_true_val, y_pred_val, y_probs_val, class_names)
    
    # ===================================================================
    # STEP 4: Compare with Dataset1
    # ===================================================================
    print("\n" + "="*70)
    print("STEP 4: COMPARISON WITH DATASET1")
    print("="*70)
    
    print("\nDataset1 (3-class, coarse) vs Dataset2 (14-class, fine-grained):")
    print("-" * 60)
    print(f"{'Metric':<30} {'Dataset1':<20} {'Dataset2':<20}")
    print("-" * 60)
    print(f"{'Number of classes':<30} {'3':<20} {'14':<20}")
    print(f"{'Classification type':<30} {'Coarse':<20} {'Fine-grained':<20}")
    print(f"{'Samples (train)':<30} {'~2,700':<20} {'1,936 (10%)':<20}")
    print(f"{'Validation accuracy':<30} {'100%':<20} {'100%':<20}")
    print(f"{'Leakage detected':<30} {'YES':<20} {'TBD':<20}")
    print(f"{'PDB feature dimension':<30} {'900':<20} {'900':<20}")
    print(f"{'Leakage mechanism':<30} {'Size→Sparsity':<20} {'Size→Sparsity':<20}")
    
    print("\n📊 HYPOTHESIS:")
    print("  Both datasets suffer from the SAME data leakage issue:")
    print("  - PDB P-P distance matrix has size-dependent sparsity")
    print("  - Small motifs (2x2, 3x3) → high sparsity (few residues)")
    print("  - Large motifs (5x5, 7-nt hairpins) → low sparsity (many residues)")
    print("  - Model learns to classify based on feature COUNT, not structure!")
    
    print("\n🔧 RECOMMENDATIONS:")
    print("  1. Normalize PDB features by motif size")
    print("  2. Use only density maps (no PDB features)")
    print("  3. Extract size-invariant features (angles, local geometry)")
    print("  4. Add data augmentation to decorrelate size and class")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
