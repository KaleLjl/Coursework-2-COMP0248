import numpy as np
import matplotlib.pyplot as plt


def plot_metrics_comparison(metrics_dict, title="Performance Metrics",
                            save_path=None, figsize=(10, 6)):
    """
    Plot a bar chart comparing different performance metrics.

    Args:
        metrics_dict (dict): Dictionary containing metric names and values
        title (str): Chart title
        save_path (str): Path to save the figure
        figsize (tuple): Figure size (width, height)
    """
    # Create figure
    plt.figure(figsize=figsize)

    # Get metric names and values
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    # Create bar colors - use a blue gradient
    colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(labels)))

    # Create horizontal bar chart
    bars = plt.barh(labels, values, color=colors)

    # Customize plot
    plt.xlabel('Value')
    plt.ylabel('Metric')
    plt.title(title)
    plt.xlim(0, 1.0)  # Assuming metrics are between 0 and 1

    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.4f}', ha='left', va='center')

    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Tight layout
    plt.tight_layout()

    # Save or display figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
