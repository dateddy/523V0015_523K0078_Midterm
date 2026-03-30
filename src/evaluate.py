"""
Evaluation Metrics, Visualizations, and Error Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_auc_score)
from typing import Dict, Tuple, List


class MetricsCalculator:
    """Compute classification metrics."""
    
    @staticmethod
    def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Binary predictions (0 or 1)
            labels: Ground truth labels
            
        Returns:
            Dict with accuracy, precision, recall, f1, auc
        """
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions, zero_division=0)
        rec = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
    
    @staticmethod
    def compute_metrics_proba(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics including AUC from logits.
        
        Args:
            logits: Raw model outputs
            labels: Ground truth labels
            
        Returns:
            Dict with accuracy, precision, recall, f1, auc
        """
        predictions = (logits > 0.5).astype(int)
        metrics = MetricsCalculator.compute_metrics(predictions, labels)
        
        try:
            auc = roc_auc_score(labels, logits)
            metrics['auc'] = auc
        except:
            metrics['auc'] = 0.0
        
        return metrics


class Visualizations:
    """Create visualizations for results."""
    
    @staticmethod
    def plot_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, 
                            title: str = "Confusion Matrix", 
                            save_path: str = None) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(labels, predictions)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
    
    @staticmethod
    def plot_learning_curves(train_losses: List[float], val_losses: List[float],
                            save_path: str = None) -> None:
        """Plot training and validation loss curves."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
        ax.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Learning Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(models_metrics: Dict[str, Dict],
                              save_path: str = None) -> None:
        """
        Compare metrics across multiple models.
        
        Args:
            models_metrics: Dict of model_name -> metrics_dict
        """
        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(metrics_names))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        for i, (model_name, metrics) in enumerate(models_metrics.items()):
            values = [metrics.get(m, 0) for m in metrics_names]
            ax.bar(x + i*width, values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models_metrics) - 1) / 2)
        ax.set_xticklabels(metrics_names)
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()


class ErrorAnalysis:
    """Analyze model errors."""
    
    @staticmethod
    def get_misclassified(predictions: np.ndarray, labels: np.ndarray, 
                         texts: List[str] = None) -> Dict:
        """
        Get misclassified examples.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            texts: Optional text samples
            
        Returns:
            Dict with false positives and false negatives
        """
        misclassified = predictions != labels
        fp_indices = np.where((predictions == 1) & (labels == 0))[0]
        fn_indices = np.where((predictions == 0) & (labels == 1))[0]
        
        result = {
            'false_positives': fp_indices.tolist(),
            'false_negatives': fn_indices.tolist(),
            'total_misclassified': np.sum(misclassified),
            'accuracy': 1 - np.mean(misclassified)
        }
        
        if texts is not None and len(texts) > 0:
            result['fp_examples'] = [texts[i] for i in fp_indices[:3]]
            result['fn_examples'] = [texts[i] for i in fn_indices[:3]]
        
        return result
    
    @staticmethod
    def print_error_report(error_analysis: Dict) -> None:
        """Print error analysis report."""
        print("\n" + "="*70)
        print("ERROR ANALYSIS REPORT")
        print("="*70)
        print(f"Total Misclassified: {error_analysis['total_misclassified']}")
        print(f"Accuracy: {error_analysis['accuracy']:.4f}")
        print(f"False Positives: {len(error_analysis['false_positives'])}")
        print(f"False Negatives: {len(error_analysis['false_negatives'])}")
        
        if 'fp_examples' in error_analysis and error_analysis['fp_examples']:
            print("\nFalse Positive Examples (predicted pos, actually neg):")
            for i, text in enumerate(error_analysis['fp_examples'], 1):
                print(f"  FP{i}: {text[:100]}...")
        
        if 'fn_examples' in error_analysis and error_analysis['fn_examples']:
            print("\nFalse Negative Examples (predicted neg, actually pos):")
            for i, text in enumerate(error_analysis['fn_examples'], 1):
                print(f"  FN{i}: {text[:100]}...")
        
        print("="*70 + "\n")
