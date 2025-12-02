#!/usr/bin/env python3
"""
Analyze training results and generate comprehensive performance reports.

Usage:
    python analyze_results.py <experiment_dir>
    python analyze_results.py experiments/hybrid_phase2_1_6class_20251202_000623
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_metrics(exp_dir):
    """Load all metrics from experiment directory."""
    exp_dir = Path(exp_dir)
    
    metrics = {}
    
    # Load config
    if (exp_dir / 'config.json').exists():
        with open(exp_dir / 'config.json') as f:
            metrics['config'] = json.load(f)
    
    # Load training history
    if (exp_dir / 'training_history.json').exists():
        with open(exp_dir / 'training_history.json') as f:
            metrics['history'] = json.load(f)
    
    # Load validation metrics (old format)
    if (exp_dir / 'final_metrics.json').exists():
        with open(exp_dir / 'final_metrics.json') as f:
            metrics['val'] = json.load(f)
    
    # Load new format metrics
    if (exp_dir / 'final_val_metrics.json').exists():
        with open(exp_dir / 'final_val_metrics.json') as f:
            metrics['val'] = json.load(f)
    
    if (exp_dir / 'final_test_metrics.json').exists():
        with open(exp_dir / 'final_test_metrics.json') as f:
            metrics['test'] = json.load(f)
    
    return metrics


def print_summary(metrics):
    """Print comprehensive summary of results."""
    config = metrics.get('config', {})
    
    print("="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    print("\nConfiguration:")
    print(f"  Dataset: {config.get('dataset_root', 'N/A')}")
    print(f"  Classes: {'6 (consolidated)' if config.get('consolidate') else '15 (original)'}")
    print(f"  Subset: {config.get('use_subset', 1.0)*100:.1f}%")
    print(f"  Batch size: {config.get('batch_size', 'N/A')}")
    print(f"  Device: {config.get('device', 'N/A')}")
    print(f"  Max epochs: {config.get('n_epochs', 'N/A')}")
    print(f"  Early stopping: {config.get('early_stopping_patience', 'N/A')}")
    
    # Class names
    class_names = [
        'small_internal', 'large_internal', 'small_bulge',
        'large_bulge', 'small_hairpin', 'large_hairpin'
    ] if config.get('consolidate') else [
        '1x1', '2x2', '3x3', '4x4', '5x5',
        'bulge1', 'bulge2', 'bulge3', 'bulge4', 'bulge5',
        'hairpin3', 'hairpin4', 'hairpin5', 'hairpin6', 'hairpin7'
    ]
    
    # Validation results
    if 'val' in metrics:
        print("\n" + "="*80)
        print("VALIDATION SET RESULTS (Best Model)")
        print("="*80)
        print_detailed_metrics(metrics['val'], class_names)
    
    # Test results
    if 'test' in metrics:
        print("\n" + "="*80)
        print("TEST SET RESULTS (Final Unbiased Estimate)")
        print("="*80)
        print_detailed_metrics(metrics['test'], class_names)


def print_detailed_metrics(metrics, class_names):
    """Print detailed per-class metrics."""
    
    # Overall accuracy
    if 'overall_accuracy' in metrics:
        print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
    
    # Macro averages
    print("\nMacro-Averaged Metrics:")
    if 'macro_accuracy' in metrics:
        print(f"  Accuracy:  {metrics['macro_accuracy']:.4f}")
    print(f"  Precision: {metrics['macro_precision']:.4f}")
    print(f"  Recall:    {metrics['macro_recall']:.4f}")
    print(f"  F1 Score:  {metrics['macro_f1']:.4f}")
    
    # Per-class metrics
    print("\nPer-Class Performance:")
    print(f"{'Class':<18} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>10}")
    print("-" * 78)
    
    for i, class_name in enumerate(class_names):
        if i < len(metrics.get('precision', [])):
            acc = metrics.get('accuracy', [0]*len(class_names))[i]
            prec = metrics['precision'][i]
            rec = metrics['recall'][i]
            f1 = metrics['f1'][i]
            supp = metrics['support'][i]
            
            print(f"{class_name:<18} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {supp:>10}")
    
    # Confusion matrix
    if 'confusion_matrix' in metrics:
        cm = np.array(metrics['confusion_matrix'])
        print("\nConfusion Matrix:")
        print_confusion_matrix(cm, class_names)
        
        # Analyze common confusions
        print("\nMost Common Misclassifications:")
        analyze_confusions(cm, class_names)


def print_confusion_matrix(cm, class_names):
    """Print formatted confusion matrix."""
    n_classes = min(len(cm), len(class_names))
    
    # Header
    print(f"{'True \\ Pred':<18}", end='')
    for i in range(n_classes):
        print(f"{class_names[i][:8]:>9}", end='')
    print()
    print("-" * (18 + 9 * n_classes))
    
    # Rows
    for i in range(n_classes):
        print(f"{class_names[i]:<18}", end='')
        for j in range(n_classes):
            val = cm[i][j] if i < len(cm) and j < len(cm[i]) else 0
            # Highlight diagonal (correct predictions)
            if i == j:
                print(f"\033[92m{val:>9}\033[0m", end='')  # Green
            elif val > 0:
                print(f"\033[91m{val:>9}\033[0m", end='')  # Red
            else:
                print(f"{val:>9}", end='')
        print()


def analyze_confusions(cm, class_names):
    """Find and report most common misclassifications."""
    confusions = []
    
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            if i != j and cm[i][j] > 0:  # Off-diagonal elements
                confusions.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': cm[i][j],
                    'true_idx': i,
                    'pred_idx': j
                })
    
    # Sort by count
    confusions.sort(key=lambda x: x['count'], reverse=True)
    
    # Print top 10
    for conf in confusions[:10]:
        true_total = sum(cm[conf['true_idx']])
        pct = 100 * conf['count'] / true_total if true_total > 0 else 0
        print(f"  {conf['true']:<18} → {conf['pred']:<18}: {conf['count']:>4} ({pct:>5.1f}% of {conf['true']})")


def plot_training_curves(metrics, save_dir):
    """Plot training and validation curves."""
    if 'history' not in metrics:
        print("No training history found, skipping plots")
        return
    
    history = metrics['history']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: {save_path}")
    plt.close()


def plot_confusion_matrices(metrics, class_names, save_dir):
    """Plot confusion matrices for val and test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, (split, ax) in enumerate([('val', axes[0]), ('test', axes[1])]):
        if split not in metrics:
            ax.text(0.5, 0.5, f'No {split} set results', 
                   ha='center', va='center', fontsize=14)
            ax.set_title(f'{split.capitalize()} Set', fontsize=14, fontweight='bold')
            continue
        
        cm = np.array(metrics[split]['confusion_matrix'])
        
        # Normalize to percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Plot
        sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Percentage (%)'})
        
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        
        acc = metrics[split].get('overall_accuracy', metrics[split]['macro_precision'])
        ax.set_title(f'{split.capitalize()} Set Confusion Matrix (Acc: {acc*100:.2f}%)', 
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'confusion_matrices.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved to: {save_path}")
    plt.close()


def plot_roc_curves(exp_dir, class_names, save_dir):
    """Plot ROC curves for validation and test sets."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    exp_dir = Path(exp_dir)
    n_classes = len(class_names)
    
    # Try to load probabilities
    val_probs_path = exp_dir / 'best_val_probs.npy'
    test_probs_path = exp_dir / 'test_probs.npy'
    
    if not val_probs_path.exists() and not test_probs_path.exists():
        print("⚠ No probability files found, skipping ROC curves")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, (split, ax, probs_path, labels_path) in enumerate([
        ('Validation', axes[0], val_probs_path, exp_dir / 'best_val_labels.npy'),
        ('Test', axes[1], test_probs_path, exp_dir / 'test_labels.npy')
    ]):
        if not probs_path.exists() or not labels_path.exists():
            ax.text(0.5, 0.5, f'No {split.lower()} probabilities', 
                   ha='center', va='center', fontsize=14)
            ax.set_title(f'{split} Set', fontsize=14, fontweight='bold')
            continue
        
        # Load data
        probs = np.load(probs_path)
        labels = np.load(labels_path)
        
        # Binarize labels for multi-class ROC (one-hot encoding)
        labels_bin = np.zeros((len(labels), n_classes))
        for i, label in enumerate(labels):
            labels_bin[i, int(label)] = 1
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(labels_bin.ravel(), probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot ROC curves
        colors = plt.get_cmap('tab10')(np.linspace(0, 1, n_classes))
        
        # Plot individual class curves
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                   label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plot micro-average curve
        ax.plot(fpr["micro"], tpr["micro"],
               color='deeppink', linestyle=':', linewidth=3,
               label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{split} Set ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'roc_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curves saved to: {save_path}")
    plt.close()


def plot_per_class_comparison(metrics, class_names, save_dir):
    """Plot per-class metrics comparison between val and test."""
    if 'test' not in metrics:
        print("No test results for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    x = np.arange(len(class_names))
    width = 0.35
    
    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        val_scores = metrics['val'].get(metric, [0]*len(class_names))[:len(class_names)]
        test_scores = metrics['test'].get(metric, [0]*len(class_names))[:len(class_names)]
        
        ax.bar(x - width/2, val_scores, width, label='Validation', alpha=0.8)
        ax.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'Per-Class {label}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([name[:10] for name in class_names], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'per_class_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-class comparison saved to: {save_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <experiment_dir>")
        print("\nExample:")
        print("  python analyze_results.py experiments/hybrid_phase2_1_6class_20251202_000623")
        sys.exit(1)
    
    exp_dir = Path(sys.argv[1])
    
    if not exp_dir.exists():
        print(f"Error: Directory not found: {exp_dir}")
        sys.exit(1)
    
    print(f"Analyzing results from: {exp_dir}\n")
    
    # Load metrics
    metrics = load_metrics(exp_dir)
    
    # Print summary
    print_summary(metrics)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    config = metrics.get('config', {})
    class_names = [
        'small_internal', 'large_internal', 'small_bulge',
        'large_bulge', 'small_hairpin', 'large_hairpin'
    ] if config.get('consolidate') else [
        '1x1', '2x2', '3x3', '4x4', '5x5',
        'bulge1', 'bulge2', 'bulge3', 'bulge4', 'bulge5',
        'hairpin3', 'hairpin4', 'hairpin5', 'hairpin6', 'hairpin7'
    ]
    
    plot_training_curves(metrics, exp_dir)
    plot_confusion_matrices(metrics, class_names, exp_dir)
    plot_roc_curves(exp_dir, class_names, exp_dir)
    
    if 'test' in metrics:
        plot_per_class_comparison(metrics, class_names, exp_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {exp_dir}")


if __name__ == '__main__':
    main()
