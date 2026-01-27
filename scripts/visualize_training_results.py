#!/usr/bin/env python3
"""
Visualization script for embedding fine-tuning results.

Generates comprehensive charts comparing base model vs fine-tuned model performance.

Usage:
    python scripts/visualize_training_results.py \
        --summary models/fine_tuned_embeddings/training_summary.json \
        --output figures/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_summary(summary_path: str) -> Dict[str, Any]:
    """Load training summary JSON."""
    with open(summary_path, 'r') as f:
        return json.load(f)


def extract_metrics(data: Dict[str, Any]) -> tuple:
    """Extract base and fine-tuned metrics."""
    base_metrics = data['metrics_history'][0]['score']  # Base model
    ft_metrics = data['test_results']  # Fine-tuned model
    
    return base_metrics, ft_metrics


def plot_metric_comparison_bar(base_metrics: Dict, ft_metrics: Dict, output_dir: Path):
    """Bar chart comparing key metrics before/after."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract metrics
    metrics_to_plot = {
        'Recall@1': ('qa_retrieval_cosine_recall@1', 'recall@1'),
        'Recall@5': ('qa_retrieval_cosine_recall@5', 'recall@5'),
        'Recall@10': ('qa_retrieval_cosine_recall@10', 'recall@10'),
        'MRR@10': ('qa_retrieval_cosine_mrr@10', 'mrr@10'),
        'NDCG@10': ('qa_retrieval_cosine_ndcg@10', 'ndcg@10'),
        'MAP@10': ('qa_retrieval_cosine_map@10', 'map@10'),
    }
    
    labels = list(metrics_to_plot.keys())
    base_values = [base_metrics.get(metrics_to_plot[k][0], 0) for k in labels]
    ft_values = [ft_metrics.get(metrics_to_plot[k][1], 0) for k in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, base_values, width, label='Base Model', 
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, ft_values, width, label='Fine-tuned Model',
                   color='#2ecc71', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Performance Comparison: Base vs Fine-tuned', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_comparison_bar.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'metric_comparison_bar.png'}")


def plot_improvement_percentage(base_metrics: Dict, ft_metrics: Dict, output_dir: Path):
    """Histogram showing percentage improvement."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_to_plot = {
        'Recall@1': ('qa_retrieval_cosine_recall@1', 'recall@1'),
        'Recall@5': ('qa_retrieval_cosine_recall@5', 'recall@5'),
        'Recall@10': ('qa_retrieval_cosine_recall@10', 'recall@10'),
        'MRR@10': ('qa_retrieval_cosine_mrr@10', 'mrr@10'),
        'NDCG@10': ('qa_retrieval_cosine_ndcg@10', 'ndcg@10'),
        'MAP@10': ('qa_retrieval_cosine_map@10', 'map@10'),
    }
    
    labels = list(metrics_to_plot.keys())
    improvements = []
    
    for k in labels:
        base_key, ft_key = metrics_to_plot[k]
        base_val = base_metrics.get(base_key, 0)
        ft_val = ft_metrics.get(ft_key, 0)
        if base_val > 0:
            improvement = ((ft_val - base_val) / base_val) * 100
        else:
            improvement = 0
        improvements.append(improvement)
    
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]
    bars = ax.barh(labels, improvements, color=colors, alpha=0.8)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax.text(val, i, f'{val:+.1f}%',
               ha='left' if val > 0 else 'right', va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Improvement (%)', fontweight='bold')
    ax.set_title('Performance Improvement After Fine-tuning', fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_percentage.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'improvement_percentage.png'}")


def plot_recall_at_k_comparison(base_metrics: Dict, ft_metrics: Dict, output_dir: Path):
    """Line chart showing Recall@k for different k values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_values = [1, 5, 10]
    base_recall = [
        base_metrics.get('qa_retrieval_cosine_recall@1', 0),
        base_metrics.get('qa_retrieval_cosine_recall@5', 0),
        base_metrics.get('qa_retrieval_cosine_recall@10', 0),
    ]
    ft_recall = [
        ft_metrics.get('recall@1', 0),
        ft_metrics.get('recall@5', 0),
        ft_metrics.get('recall@10', 0),
    ]
    
    ax.plot(k_values, base_recall, marker='o', linewidth=2.5, markersize=8,
           label='Base Model', color='#3498db')
    ax.plot(k_values, ft_recall, marker='s', linewidth=2.5, markersize=8,
           label='Fine-tuned Model', color='#2ecc71')
    
    # Add value annotations
    for k, base, ft in zip(k_values, base_recall, ft_recall):
        ax.annotate(f'{base:.3f}', (k, base), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=9, color='#3498db')
        ax.annotate(f'{ft:.3f}', (k, ft), textcoords="offset points",
                   xytext=(0,-15), ha='center', fontsize=9, color='#2ecc71')
    
    ax.set_xlabel('k (Top-k Results)', fontweight='bold')
    ax.set_ylabel('Recall@k', fontweight='bold')
    ax.set_title('Recall@k Comparison: Base vs Fine-tuned', fontweight='bold', pad=20)
    ax.set_xticks(k_values)
    ax.set_ylim([0, 1.0])
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'recall_at_k_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'recall_at_k_comparison.png'}")


def plot_radar_chart(base_metrics: Dict, ft_metrics: Dict, output_dir: Path):
    """Radar/spider chart comparing multiple metrics."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['Recall@1', 'Recall@5', 'Recall@10', 'MRR@10', 'NDCG@10', 'MAP@10']
    base_values = [
        base_metrics.get('qa_retrieval_cosine_recall@1', 0),
        base_metrics.get('qa_retrieval_cosine_recall@5', 0),
        base_metrics.get('qa_retrieval_cosine_recall@10', 0),
        base_metrics.get('qa_retrieval_cosine_mrr@10', 0),
        base_metrics.get('qa_retrieval_cosine_ndcg@10', 0),
        base_metrics.get('qa_retrieval_cosine_map@10', 0),
    ]
    ft_values = [
        ft_metrics.get('recall@1', 0),
        ft_metrics.get('recall@5', 0),
        ft_metrics.get('recall@10', 0),
        ft_metrics.get('mrr@10', 0),
        ft_metrics.get('ndcg@10', 0),
        ft_metrics.get('map@10', 0),
    ]
    
    # Close the plot
    categories += [categories[0]]
    base_values += [base_values[0]]
    ft_values += [ft_values[0]]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True).tolist()
    
    ax.plot(angles, base_values, 'o-', linewidth=2, label='Base Model', color='#3498db')
    ax.fill(angles, base_values, alpha=0.25, color='#3498db')
    
    ax.plot(angles, ft_values, 'o-', linewidth=2, label='Fine-tuned Model', color='#2ecc71')
    ax.fill(angles, ft_values, alpha=0.25, color='#2ecc71')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1])
    ax.set_ylim([0, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)
    
    ax.set_title('Comprehensive Performance Comparison\n(Base vs Fine-tuned)', 
                fontweight='bold', pad=30, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_chart.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'radar_chart.png'}")


def plot_histogram_distribution(base_metrics: Dict, ft_metrics: Dict, output_dir: Path):
    """Histogram showing distribution of metric values."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics_config = [
        ('Recall@1', 'qa_retrieval_cosine_recall@1', 'recall@1'),
        ('Recall@5', 'qa_retrieval_cosine_recall@5', 'recall@5'),
        ('Recall@10', 'qa_retrieval_cosine_recall@10', 'recall@10'),
        ('MRR@10', 'qa_retrieval_cosine_mrr@10', 'mrr@10'),
        ('NDCG@10', 'qa_retrieval_cosine_ndcg@10', 'ndcg@10'),
        ('MAP@10', 'qa_retrieval_cosine_map@10', 'map@10'),
    ]
    
    for idx, (title, base_key, ft_key) in enumerate(metrics_config):
        ax = axes[idx]
        
        base_val = base_metrics.get(base_key, 0)
        ft_val = ft_metrics.get(ft_key, 0)
        
        bars = ax.bar(['Base Model', 'Fine-tuned Model'], [base_val, ft_val],
                     color=['#3498db', '#2ecc71'], alpha=0.8, width=0.6)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Performance Metrics: Before vs After Fine-tuning', 
                fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'histogram_distribution.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'histogram_distribution.png'}")


def plot_heatmap_comparison(base_metrics: Dict, ft_metrics: Dict, output_dir: Path):
    """Heatmap showing metric values side-by-side."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'MRR@10', 'NDCG@10', 'MAP@10']
    base_keys = ['qa_retrieval_cosine_recall@1', 'qa_retrieval_cosine_recall@5',
                'qa_retrieval_cosine_recall@10', 'qa_retrieval_cosine_mrr@10',
                'qa_retrieval_cosine_ndcg@10', 'qa_retrieval_cosine_map@10']
    ft_keys = ['recall@1', 'recall@5', 'recall@10', 'mrr@10', 'ndcg@10', 'map@10']
    
    base_values = [base_metrics.get(k, 0) for k in base_keys]
    ft_values = [ft_metrics.get(k, 0) for k in ft_keys]
    
    data = np.array([base_values, ft_values])
    
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_yticklabels(['Base Model', 'Fine-tuned Model'])
    
    # Add text annotations
    for i in range(2):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Performance Heatmap: Base vs Fine-tuned Model', 
                fontweight='bold', pad=20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'heatmap_comparison.png'}")


def plot_training_summary(data: Dict, output_dir: Path):
    """Summary statistics visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training duration
    started = data['started_at']
    completed = data['completed_at']
    # Parse timestamps (simplified)
    duration_seconds = (np.datetime64(completed) - np.datetime64(started)) / np.timedelta64(1, 's')
    duration_hours = duration_seconds / 3600
    
    ax1.bar(['Training Duration'], [duration_hours], color='#9b59b6', alpha=0.8)
    ax1.set_ylabel('Hours', fontweight='bold')
    ax1.set_title(f'Training Time: {duration_hours:.2f} hours', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Dataset size
    train_examples = data['train_examples']
    ax2.bar(['Training Examples'], [train_examples / 1e6], color='#e67e22', alpha=0.8)
    ax2.set_ylabel('Millions', fontweight='bold')
    ax2.set_title(f'Dataset Size: {train_examples:,} examples', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Epochs
    epochs = data['num_epochs']
    ax3.bar(['Epochs'], [epochs], color='#16a085', alpha=0.8)
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title(f'Training Epochs: {epochs}', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Test set size
    test_size = data['test_results']['num_queries']
    ax4.bar(['Test Queries'], [test_size], color='#c0392b', alpha=0.8)
    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_title(f'Evaluation Set: {test_size:,} queries', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Training Summary Statistics', fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_summary.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'training_summary.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization charts for embedding fine-tuning results"
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="models/fine_tuned_embeddings/training_summary.json",
        help="Path to training summary JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures",
        help="Output directory for figures"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading training summary from: {args.summary}")
    data = load_summary(args.summary)
    
    # Extract metrics
    base_metrics, ft_metrics = extract_metrics(data)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    print("=" * 60)
    
    # Generate all charts
    plot_metric_comparison_bar(base_metrics, ft_metrics, output_dir)
    plot_improvement_percentage(base_metrics, ft_metrics, output_dir)
    plot_recall_at_k_comparison(base_metrics, ft_metrics, output_dir)
    plot_radar_chart(base_metrics, ft_metrics, output_dir)
    plot_histogram_distribution(base_metrics, ft_metrics, output_dir)
    plot_heatmap_comparison(base_metrics, ft_metrics, output_dir)
    plot_training_summary(data, output_dir)
    
    print("=" * 60)
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print("\nGenerated charts:")
    print("  1. metric_comparison_bar.png - Side-by-side bar chart")
    print("  2. improvement_percentage.png - Percentage improvement histogram")
    print("  3. recall_at_k_comparison.png - Recall@k line chart")
    print("  4. radar_chart.png - Radar/spider chart")
    print("  5. histogram_distribution.png - Individual metric histograms")
    print("  6. heatmap_comparison.png - Heatmap visualization")
    print("  7. training_summary.png - Training statistics")


if __name__ == "__main__":
    main()
