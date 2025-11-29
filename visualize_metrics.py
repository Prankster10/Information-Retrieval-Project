"""
Visual demonstration of evaluation metrics with matplotlib charts
"""

import matplotlib.pyplot as plt
import numpy as np
from test_evaluation import SimpleIREvaluator

def visualize_metrics():
    """Create visual charts for evaluation metrics"""
    
    evaluator = SimpleIREvaluator()
    
    # Test queries
    test_cases = [
        {
            'query': 'machine learning and artificial intelligence',
            'relevant': ['Machine Learning Basics', 'AI Transformation', 'Deep Learning Guide']
        },
        {
            'query': 'deep learning neural networks',
            'relevant': ['Deep Learning Guide', 'Neural Networks', 'Machine Learning Basics']
        },
        {
            'query': 'data science analytics insights',
            'relevant': ['Data Science Overview', 'Big Data Analytics', 'Unsupervised Learning']
        },
        {
            'query': 'natural language processing',
            'relevant': ['NLP Introduction', 'Machine Learning Basics']
        },
        {
            'query': 'computer vision image',
            'relevant': ['Computer Vision Fundamentals', 'Deep Learning Guide']
        }
    ]
    
    # Calculate metrics for all test cases
    results = []
    for test in test_cases:
        metrics = evaluator.calculate_metrics(
            test['query'], 
            test['relevant'], 
            top_k=5
        )
        results.append(metrics)
    
    # Extract data for visualization
    queries = [r['query'][:25] + '...' if len(r['query']) > 25 else r['query'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    maps = [r['map'] for r in results]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Information Retrieval Evaluation Metrics Dashboard', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Precision comparison
    ax1 = axes[0, 0]
    colors_p = ['green' if p >= 0.6 else 'orange' if p >= 0.4 else 'red' for p in precisions]
    bars1 = ax1.barh(queries, precisions, color=colors_p, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Precision Score', fontweight='bold')
    ax1.set_title('Precision by Query', fontweight='bold', fontsize=12)
    ax1.set_xlim(0, 1)
    for i, (bar, p) in enumerate(zip(bars1, precisions)):
        ax1.text(p + 0.02, i, f'{p:.3f}', va='center', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Recall comparison
    ax2 = axes[0, 1]
    colors_r = ['green' if r >= 0.8 else 'orange' if r >= 0.6 else 'red' for r in recalls]
    bars2 = ax2.barh(queries, recalls, color=colors_r, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Recall Score', fontweight='bold')
    ax2.set_title('Recall by Query', fontweight='bold', fontsize=12)
    ax2.set_xlim(0, 1)
    for i, (bar, r) in enumerate(zip(bars2, recalls)):
        ax2.text(r + 0.02, i, f'{r:.3f}', va='center', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: MAP comparison
    ax3 = axes[1, 0]
    colors_map = ['green' if m >= 0.8 else 'orange' if m >= 0.5 else 'red' for m in maps]
    bars3 = ax3.barh(queries, maps, color=colors_map, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('MAP Score', fontweight='bold')
    ax3.set_title('Mean Average Precision by Query', fontweight='bold', fontsize=12)
    ax3.set_xlim(0, 1)
    for i, (bar, m) in enumerate(zip(bars3, maps)):
        ax3.text(m + 0.02, i, f'{m:.3f}', va='center', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Plot 4: Overall metric comparison (radar-style alternative)
    ax4 = axes[1, 1]
    x_pos = np.arange(len(queries))
    width = 0.25
    
    bars_p = ax4.bar(x_pos - width, precisions, width, label='Precision', 
                     color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars_r = ax4.bar(x_pos, recalls, width, label='Recall', 
                     color='#4ECDC4', alpha=0.8, edgecolor='black')
    bars_m = ax4.bar(x_pos + width, maps, width, label='MAP', 
                     color='#45B7D1', alpha=0.8, edgecolor='black')
    
    ax4.set_ylabel('Score', fontweight='bold')
    ax4.set_title('All Metrics Comparison', fontweight='bold', fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'Q{i+1}' for i in range(len(queries))], fontsize=10)
    ax4.set_ylim(0, 1.1)
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    
    # Add value labels on bars
    for bars in [bars_p, bars_r, bars_m]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('evaluation_metrics_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Dashboard saved as 'evaluation_metrics_dashboard.png'")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall   : {np.mean(recalls):.4f}")
    print(f"Average MAP      : {np.mean(maps):.4f}")
    print(f"\nBest Query (by MAP): {queries[np.argmax(maps)]} ({max(maps):.4f})")
    print(f"Worst Query (by MAP): {queries[np.argmin(maps)]} ({min(maps):.4f})")
    print("="*70 + "\n")


if __name__ == "__main__":
    visualize_metrics()
