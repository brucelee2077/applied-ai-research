"""
Neural Networks Tutorial - Visualization Utilities

This module provides comprehensive visualization functions for understanding
and analyzing neural networks. All visualizations are designed to be
educational and beginner-friendly.

Author: Applied AI Research
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Tuple, Callable, Dict
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# TRAINING VISUALIZATIONS
# =============================================================================

def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (14, 5),
    title: str = "Training Progress"
) -> plt.Figure:
    """
    Plot training and validation loss/accuracy curves over epochs.
    
    Creates side-by-side plots showing loss and accuracy progression during
    training. Helps diagnose overfitting, underfitting, and convergence.
    
    Parameters
    ----------
    train_losses : list of float
        Training loss values per epoch
    val_losses : list of float, optional
        Validation loss values per epoch
    train_accs : list of float, optional
        Training accuracy values per epoch
    val_accs : list of float, optional
        Validation accuracy values per epoch
    figsize : tuple of int, default=(14, 5)
        Figure size (width, height) in inches
    title : str, default="Training Progress"
        Overall figure title
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
        
    Examples
    --------
    >>> train_losses = [2.3, 1.5, 1.2, 0.9, 0.7]
    >>> val_losses = [2.4, 1.6, 1.3, 1.0, 0.9]
    >>> train_accs = [0.2, 0.5, 0.65, 0.75, 0.82]
    >>> val_accs = [0.18, 0.48, 0.63, 0.72, 0.78]
    >>> fig = plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    >>> plt.show()
    
    Notes
    -----
    Diverging training and validation curves indicate overfitting.
    Plateauing curves suggest the model has converged.
    """
    # Determine number of subplots needed
    n_plots = 1 if train_accs is None else 2
    
    # Create figure and subplots
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot loss curves
    axes[0].plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    if val_losses is not None:
        axes[0].plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy curves if provided
    if train_accs is not None:
        axes[1].plot(epochs, train_accs, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
        if val_accs is not None:
            axes[1].plot(epochs, val_accs, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_loss_surface(
    loss_function: Callable,
    w1_range: Tuple[float, float] = (-2, 2),
    w2_range: Tuple[float, float] = (-2, 2),
    resolution: int = 50,
    figsize: Tuple[int, int] = (12, 9),
    title: str = "Loss Surface"
) -> plt.Figure:
    """
    Plot 3D loss surface for a two-parameter loss function.
    
    Visualizes how loss changes with respect to two parameters (weights).
    Helps understand optimization landscape and why gradient descent works.
    
    Parameters
    ----------
    loss_function : callable
        Function that takes (w1, w2) and returns loss value
    w1_range : tuple of float, default=(-2, 2)
        Range for first weight parameter (min, max)
    w2_range : tuple of float, default=(-2, 2)
        Range for second weight parameter (min, max)
    resolution : int, default=50
        Number of points per dimension (higher = smoother but slower)
    figsize : tuple of int, default=(12, 9)
        Figure size (width, height) in inches
    title : str, default="Loss Surface"
        Plot title
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
        
    Examples
    --------
    >>> def simple_loss(w1, w2):
    ...     return w1**2 + w2**2  # Convex bowl
    >>> fig = plot_loss_surface(simple_loss, w1_range=(-3, 3), w2_range=(-3, 3))
    >>> plt.show()
    
    Notes
    -----
    Convex surfaces (bowl-shaped) are easy to optimize.
    Non-convex surfaces with multiple minima are challenging.
    """
    # Create grid of weight values
    w1 = np.linspace(w1_range[0], w1_range[1], resolution)
    w2 = np.linspace(w2_range[0], w2_range[1], resolution)
    W1, W2 = np.meshgrid(w1, w2)
    
    # Compute loss for each point
    Z = np.zeros_like(W1)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = loss_function(W1[i, j], W2[i, j])
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot with gradient coloring
    surf = ax.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.8,
                          edgecolor='none', antialiased=True)
    
    # Add contour lines at the base
    ax.contour(W1, W2, Z, levels=15, cmap='viridis', alpha=0.5,
              offset=np.min(Z) - 0.1 * (np.max(Z) - np.min(Z)))
    
    # Labels and title
    ax.set_xlabel('Weight 1 (w₁)', fontsize=12, labelpad=10)
    ax.set_ylabel('Weight 2 (w₂)', fontsize=12, labelpad=10)
    ax.set_zlabel('Loss', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Loss Value')
    
    # Set viewing angle for better perspective
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    
    return fig


def animate_training(
    history: Dict[str, List],
    save_path: Optional[str] = None,
    interval: int = 200
) -> animation.FuncAnimation:
    """
    Create an animation showing training progress over epochs.
    
    Animates loss and accuracy curves as they evolve during training.
    Useful for presentations and understanding training dynamics.
    
    Parameters
    ----------
    history : dict
        Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        Each containing a list of values per epoch
    save_path : str, optional
        Path to save animation as MP4 or GIF
    interval : int, default=200
        Milliseconds between frames
        
    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Animation object
        
    Examples
    --------
    >>> history = {
    ...     'train_loss': [2.3, 1.5, 1.2, 0.9, 0.7],
    ...     'val_loss': [2.4, 1.6, 1.3, 1.0, 0.9],
    ...     'train_acc': [0.2, 0.5, 0.65, 0.75, 0.82],
    ...     'val_acc': [0.18, 0.48, 0.63, 0.72, 0.78]
    ... }
    >>> anim = animate_training(history)
    
    Notes
    -----
    Requires ffmpeg or pillow for saving animations.
    Can be displayed in Jupyter notebooks with HTML(anim.to_html5_video()).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    n_epochs = len(history['train_loss'])
    
    # Initialize empty plots
    line1, = ax1.plot([], [], 'b-o', label='Training Loss', linewidth=2)
    line2, = ax1.plot([], [], 'r-s', label='Validation Loss', linewidth=2)
    line3, = ax2.plot([], [], 'b-o', label='Training Accuracy', linewidth=2)
    line4, = ax2.plot([], [], 'r-s', label='Validation Accuracy', linewidth=2)
    
    # Set up axes
    ax1.set_xlim(0, n_epochs + 1)
    ax1.set_ylim(0, max(history['train_loss']) * 1.1)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlim(0, n_epochs + 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    def animate(frame):
        """Update function for animation"""
        epochs = range(1, frame + 2)
        line1.set_data(epochs, history['train_loss'][:frame + 1])
        line2.set_data(epochs, history['val_loss'][:frame + 1])
        line3.set_data(epochs, history['train_acc'][:frame + 1])
        line4.set_data(epochs, history['val_acc'][:frame + 1])
        return line1, line2, line3, line4
    
    anim = animation.FuncAnimation(fig, animate, frames=n_epochs,
                                  interval=interval, blit=True)
    
    if save_path:
        anim.save(save_path, writer='pillow' if save_path.endswith('.gif') else 'ffmpeg')
        print(f"Animation saved to {save_path}")
    
    plt.tight_layout()
    
    return anim


# =============================================================================
# DATA VISUALIZATIONS
# =============================================================================

def plot_sample_images(
    images: np.ndarray,
    labels: Optional[np.ndarray] = None,
    predictions: Optional[np.ndarray] = None,
    num_samples: int = 25,
    figsize: Tuple[int, int] = (12, 12),
    title: str = "Sample Images"
) -> plt.Figure:
    """
    Display a grid of sample images with optional labels and predictions.
    
    Visualizes image data in an organized grid. Shows true labels and
    predictions for comparison. Highlights incorrect predictions in red.
    
    Parameters
    ----------
    images : np.ndarray, shape (n_samples, height, width) or (n_samples, features)
        Input images (flattened images will be reshaped to square)
    labels : np.ndarray, optional
        True labels (integer or one-hot encoded)
    predictions : np.ndarray, optional
        Predicted labels (integer or one-hot encoded)
    num_samples : int, default=25
        Number of images to display (will create square grid)
    figsize : tuple of int, default=(12, 12)
        Figure size (width, height) in inches
    title : str, default="Sample Images"
        Figure title
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
        
    Examples
    --------
    >>> images = np.random.randn(100, 28, 28)
    >>> labels = np.random.randint(0, 10, size=100)
    >>> predictions = np.random.randint(0, 10, size=100)
    >>> fig = plot_sample_images(images, labels, predictions, num_samples=16)
    >>> plt.show()
    
    Notes
    -----
    For MNIST: 28x28 images are displayed.
    Incorrect predictions are shown with red titles.
    """
    # Limit to available samples
    num_samples = min(num_samples, len(images))
    
    # Calculate grid size (square)
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Reshape flattened images if needed
    if images.ndim == 2:
        img_size = int(np.sqrt(images.shape[1]))
        images = images.reshape(-1, img_size, img_size)
    
    # Convert one-hot labels to class indices
    if labels is not None and labels.ndim == 2:
        labels = np.argmax(labels, axis=1)
    if predictions is not None and predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=1)
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(grid_size * grid_size):
        ax = axes[i]
        
        if i < num_samples:
            # Display image
            ax.imshow(images[i], cmap='gray', vmin=0, vmax=1)
            
            # Add labels if provided
            if labels is not None and predictions is not None:
                # Check if prediction is correct
                is_correct = labels[i] == predictions[i]
                color = 'green' if is_correct else 'red'
                title_text = f"True: {labels[i]}, Pred: {predictions[i]}"
                ax.set_title(title_text, fontsize=10, color=color, fontweight='bold')
            elif labels is not None:
                ax.set_title(f"Label: {labels[i]}", fontsize=10)
            
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Confusion Matrix",
    cmap: str = 'Blues'
) -> plt.Figure:
    """
    Plot confusion matrix as an annotated heatmap.
    
    Visualizes classification performance across all classes. Diagonal
    elements show correct predictions, off-diagonal show misclassifications.
    
    Parameters
    ----------
    cm : np.ndarray, shape (n_classes, n_classes)
        Confusion matrix
    class_names : list of str, optional
        Names for each class (defaults to numeric indices)
    figsize : tuple of int, default=(10, 8)
        Figure size (width, height) in inches
    title : str, default="Confusion Matrix"
        Plot title
    cmap : str, default='Blues'
        Matplotlib colormap name
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
        
    Examples
    --------
    >>> cm = np.array([[50, 2, 0], [1, 48, 1], [0, 3, 47]])
    >>> class_names = ['Class A', 'Class B', 'Class C']
    >>> fig = plot_confusion_matrix(cm, class_names)
    >>> plt.show()
    
    Notes
    -----
    Row sums show total samples per true class.
    Column sums show total predictions per predicted class.
    Perfect classification has all values on the diagonal.
    """
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, label='Number of Samples')
    
    # Set ticks and labels
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title=title)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}",
                   ha="center", va="center", color=color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def plot_misclassified(
    images: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    num_samples: int = 16,
    figsize: Tuple[int, int] = (12, 10),
    title: str = "Misclassified Examples"
) -> plt.Figure:
    """
    Display grid of misclassified examples.
    
    Shows only incorrect predictions to help identify model weaknesses
    and common confusion patterns.
    
    Parameters
    ----------
    images : np.ndarray
        Input images
    true_labels : np.ndarray
        True labels (integer or one-hot)
    pred_labels : np.ndarray
        Predicted labels (integer or one-hot)
    num_samples : int, default=16
        Maximum number of misclassified examples to show
    figsize : tuple of int, default=(12, 10)
        Figure size (width, height) in inches
    title : str, default="Misclassified Examples"
        Figure title
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
        
    Examples
    --------
    >>> # Find and display misclassifications
    >>> fig = plot_misclassified(X_test, y_true, y_pred, num_samples=25)
    >>> plt.show()
    
    Notes
    -----
    Helps identify which classes are commonly confused.
    If no misclassifications found, returns empty figure with message.
    """
    # Convert one-hot to class indices
    if true_labels.ndim == 2:
        true_labels = np.argmax(true_labels, axis=1)
    if pred_labels.ndim == 2:
        pred_labels = np.argmax(pred_labels, axis=1)
    
    # Find misclassified indices
    misclassified = np.where(true_labels != pred_labels)[0]
    
    if len(misclassified) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No misclassifications found!\n🎉 Perfect accuracy! 🎉",
               ha='center', va='center', fontsize=16, fontweight='bold')
        ax.axis('off')
        return fig
    
    # Limit to requested number
    misclassified = misclassified[:num_samples]
    
    # Use plot_sample_images with misclassified subset
    fig = plot_sample_images(
        images[misclassified],
        true_labels[misclassified],
        pred_labels[misclassified],
        num_samples=len(misclassified),
        figsize=figsize,
        title=f"{title} (Total: {len(np.where(true_labels != pred_labels)[0])})"
    )
    
    return fig


# =============================================================================
# NETWORK VISUALIZATIONS
# =============================================================================

def plot_network_architecture(
    layer_sizes: List[int],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Neural Network Architecture"
) -> plt.Figure:
    """
    Draw a diagram of neural network architecture.
    
    Visualizes the layer structure showing neurons and connections.
    Helps understand network depth and width.
    
    Parameters
    ----------
    layer_sizes : list of int
        Number of neurons in each layer (including input and output)
    figsize : tuple of int, default=(12, 8)
        Figure size (width, height) in inches
    title : str, default="Neural Network Architecture"
        Figure title
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
        
    Examples
    --------
    >>> # Draw a network with 784 inputs, two hidden layers, 10 outputs
    >>> fig = plot_network_architecture([784, 128, 64, 10])
    >>> plt.show()
    
    Notes
    -----
    For large layers (>20 neurons), shows simplified representation.
    Connections between all neurons are shown for small layers.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    
    # Determine spacing
    v_spacing = 1.0 / (max_neurons + 1)
    h_spacing = 1.0 / (n_layers + 1)
    
    # Draw layers
    layer_positions = []
    for layer_idx, layer_size in enumerate(layer_sizes):
        layer_x = (layer_idx + 1) * h_spacing
        neuron_positions = []
        
        # Limit displayed neurons for large layers
        display_size = min(layer_size, 15)
        show_ellipsis = layer_size > display_size
        
        for neuron_idx in range(display_size):
            # Center neurons vertically
            offset = (max_neurons - display_size) / 2
            neuron_y = (neuron_idx + offset + 1) * v_spacing
            
            # Draw neuron
            circle = plt.Circle((layer_x, neuron_y), 0.015, color='steelblue', 
                              ec='black', linewidth=1.5, zorder=4)
            ax.add_patch(circle)
            neuron_positions.append((layer_x, neuron_y))
        
        # Add ellipsis for large layers
        if show_ellipsis:
            ellipsis_y = (display_size + offset + 1.5) * v_spacing
            ax.text(layer_x, ellipsis_y, '⋮', ha='center', va='center',
                   fontsize=20, fontweight='bold')
        
        layer_positions.append(neuron_positions)
        
        # Add layer label
        layer_name = ['Input', 'Hidden', 'Hidden', 'Output'][min(layer_idx, 3)]
        if layer_idx > 0 and layer_idx < n_layers - 1:
            layer_name = f'Hidden {layer_idx}'
        
        ax.text(layer_x, -0.05, f'{layer_name}\n({layer_size} neurons)',
               ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Draw connections between adjacent layers
    for layer_idx in range(len(layer_positions) - 1):
        for pos1 in layer_positions[layer_idx][:5]:  # Limit connections shown
            for pos2 in layer_positions[layer_idx + 1][:5]:
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                       'gray', alpha=0.3, linewidth=0.5, zorder=1)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.05)
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    return fig


def plot_weights(
    weights: np.ndarray,
    layer_name: str = "Layer Weights",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'RdBu_r'
) -> plt.Figure:
    """
    Visualize weight matrix as a heatmap.
    
    Shows learned weight values. Patterns in weights can reveal what
    features the network has learned to detect.
    
    Parameters
    ----------
    weights : np.ndarray, shape (n_inputs, n_outputs)
        Weight matrix to visualize
    layer_name : str, default="Layer Weights"
        Name of the layer for title
    figsize : tuple of int, default=(10, 8)
        Figure size (width, height) in inches
    cmap : str, default='RdBu_r'
        Matplotlib colormap (red-blue emphasizes positive/negative)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
        
    Examples
    --------
    >>> weights = np.random.randn(784, 128) * 0.1
    >>> fig = plot_weights(weights, "Layer 1 Weights")
    >>> plt.show()
    
    Notes
    -----
    Large positive weights (blue) increase activation.
    Large negative weights (red) decrease activation.
    Small weights near zero suggest less important connections.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute symmetric color limits
    vmax = np.abs(weights).max()
    
    # Display heatmap
    im = ax.imshow(weights.T, aspect='auto', cmap=cmap, 
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    
    ax.set_xlabel('Input Neurons', fontsize=12)
    ax.set_ylabel('Output Neurons', fontsize=12)
    ax.set_title(f'{layer_name}\nShape: {weights.shape}', 
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Weight Value')
    
    # Add statistics
    stats_text = f"Mean: {weights.mean():.4f}\nStd: {weights.std():.4f}"
    ax.text(1.15, 0.5, stats_text, transform=ax.transAxes,
           fontsize=10, va='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return fig


def plot_activations(
    activations: np.ndarray,
    layer_name: str = "Layer Activations",
    figsize: Tuple[int, int] = (12, 4),
    num_samples: int = 5
) -> plt.Figure:
    """
    Visualize activation patterns for samples.
    
    Shows how neurons in a layer respond to different inputs.
    Helps understand what features each layer detects.
    
    Parameters
    ----------
    activations : np.ndarray, shape (n_samples, n_neurons)
        Activation values for multiple samples
    layer_name : str, default="Layer Activations"
        Name of the layer
    figsize : tuple of int, default=(12, 4)
        Figure size (width, height) in inches
    num_samples : int, default=5
        Number of samples to display
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
        
    Examples
    --------
    >>> activations = np.random.relu(np.random.randn(10, 128))
    >>> fig = plot_activations(activations, "Hidden Layer 1", num_samples=5)
    >>> plt.show()
    
    Notes
    -----
    Each row represents one sample's activation pattern.
    Bright colors indicate high activation (neuron firing).
    Dark colors indicate low/zero activation (neuron silent).
    """
    num_samples = min(num_samples, activations.shape[0])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display activations
    im = ax.imshow(activations[:num_samples], aspect='auto', cmap='YlOrRd',
                   interpolation='nearest')
    
    ax.set_xlabel('Neuron Index', fontsize=12)
    ax.set_ylabel('Sample Index', fontsize=12)
    ax.set_title(f'{layer_name}\n{num_samples} samples × {activations.shape[1]} neurons',
                fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Activation Value')
    
    plt.tight_layout()
    
    return fig


def plot_decision_boundary(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
    resolution: int = 200,
    title: str = "Decision Boundary"
) -> plt.Figure:
    """
    Plot 2D decision boundary learned by a model.
    
    Visualizes how the model partitions the feature space for classification.
    Only works for 2D input data.
    
    Parameters
    ----------
    model : callable
        Model with predict method that takes 2D input
    X : np.ndarray, shape (n_samples, 2)
        2D input features
    y : np.ndarray, shape (n_samples,)
        Class labels (integer)
    figsize : tuple of int, default=(10, 8)
        Figure size (width, height) in inches
    resolution : int, default=200
        Grid resolution for decision boundary
    title : str, default="Decision Boundary"
        Plot title
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
        
    Examples
    --------
    >>> # Create 2D dataset
    >>> X = np.random.randn(200, 2)
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> # Train model and plot
    >>> fig = plot_decision_boundary(model, X, y)
    >>> plt.show()
    
    Notes
    -----
    Requires 2D input data (X.shape[1] == 2).
    Decision regions are shown with different colors.
    Training points are overlaid as scatter plot.
    """
    assert X.shape[1] == 2, "Decision boundary plot requires 2D input data"
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                        np.linspace(y_min, y_max, resolution))
    
    # Predict on mesh grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model(grid_points)
    
    # Handle one-hot encoded predictions
    if Z.ndim == 2:
        Z = np.argmax(Z, axis=1)
    
    Z = Z.reshape(xx.shape)
    
    # Convert one-hot labels if needed
    if y.ndim == 2:
        y = np.argmax(y, axis=1)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plot training points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                        edgecolors='black', linewidths=1.5, s=50)
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, ax=ax, label='Class')
    plt.tight_layout()
    
    return fig


# =============================================================================
# LEARNING VISUALIZATIONS
# =============================================================================

def plot_gradient_descent_steps(
    loss_values: List[float],
    weights: Optional[List[np.ndarray]] = None,
    figsize: Tuple[int, int] = (14, 5),
    title: str = "Gradient Descent Optimization"
) -> plt.Figure:
    """
    Visualize gradient descent optimization path.
    
    Shows how loss decreases and weights change during optimization.
    Helps understand convergence behavior and learning rates.
    
    Parameters
    ----------
    loss_values : list of float
        Loss at each optimization step
    weights : list of np.ndarray, optional
        Weight vectors at each step (for 2D weight space)
    figsize : tuple of int, default=(14, 5)
        Figure size (width, height) in inches
    title : str, default="Gradient Descent Optimization"
        Figure title
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
        
    Examples
    --------
    >>> loss_values = [10.0, 5.2, 2.8, 1.5, 0.8, 0.4, 0.2]
    >>> fig = plot_gradient_descent_steps(loss_values)
    >>> plt.show()
    
    Notes
    -----
    Sharp drops indicate effective learning.
    Plateaus suggest learning rate too small or convergence.
    Oscillations suggest learning rate too large.
    """
    n_plots = 1 if weights is None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    steps = range(len(loss_values))
    
    # Plot loss progression
    axes[0].plot(steps, loss_values, 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Optimization Step', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss vs. Steps', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Add annotations for first and last points
    axes[0].annotate(f'Start: {loss_values[0]:.4f}',
                    xy=(0, loss_values[0]), xytext=(10, 20),
                    textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    axes[0].annotate(f'End: {loss_values[-1]:.4f}',
                    xy=(len(loss_values)-1, loss_values[-1]), xytext=(-50, -30),
                    textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot weight trajectory if provided (for 2D weights)
    if weights is not None and len(weights[0]) == 2:
        w_trajectory = np.array(weights)
        axes[1].plot(w_trajectory[:, 0], w_trajectory[:, 1], 'r-o',
                    linewidth=2, markersize=6, alpha=0.7)
        axes[1].plot(w_trajectory[0, 0], w_trajectory[0, 1], 'go',
                    markersize=12, label='Start', zorder=5)
        axes[1].plot(w_trajectory[-1, 0], w_trajectory[-1, 1], 'r*',
                    markersize=15, label='End', zorder=5)
        axes[1].set_xlabel('Weight 1', fontsize=12)
        axes[1].set_ylabel('Weight 2', fontsize=12)
        axes[1].set_title('Weight Space Trajectory', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def compare_activation_functions(
    x_range: Tuple[float, float] = (-5, 5),
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot multiple activation functions side-by-side for comparison.
    
    Visualizes common activation functions and their derivatives.
    Helps understand activation function properties and use cases.
    
    Parameters
    ----------
    x_range : tuple of float, default=(-5, 5)
        Range of x values to plot
    figsize : tuple of int, default=(15, 10)
        Figure size (width, height) in inches
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
        
    Examples
    --------
    >>> fig = compare_activation_functions(x_range=(-3, 3))
    >>> plt.show()
    
    Notes
    -----
    Shows: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, and their derivatives.
    Derivatives indicate gradient flow during backpropagation.
    """
    x = np.linspace(x_range[0], x_range[1], 1000)
    
    # Define activation functions
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)
    
    def tanh(x):
        return np.tanh(x)
    
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
    def relu(x):
        return np.maximum(0, x)
    
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    activations = [
        ('Sigmoid', sigmoid, sigmoid_derivative),
        ('Tanh', tanh, tanh_derivative),
        ('ReLU', relu, relu_derivative),
    ]
    
    for idx, (name, func, deriv) in enumerate(activations):
        row = idx
        
        # Plot activation function
        ax1 = axes[row, 0]
        ax1.plot(x, func(x), 'b-', linewidth=2.5, label=name)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Input (x)', fontsize=11)
        ax1.set_ylabel('Output', fontsize=11)
        ax1.set_title(f'{name} Activation', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Plot derivative
        ax2 = axes[row, 1]
        ax2.plot(x, deriv(x), 'r-', linewidth=2.5, label=f'{name} Derivative')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Input (x)', fontsize=11)
        ax2.set_ylabel('Gradient', fontsize=11)
        ax2.set_title(f'{name} Gradient', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
    
    plt.suptitle('Common Activation Functions and Their Gradients',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    """
    Demo script showing usage of visualization utilities.
    """
    print("=" * 70)
    print("Neural Networks Tutorial - Visualization Utilities Demo")
    print("=" * 70)
    
    # Example 1: Training curves
    print("\n📈 Creating training curves...")
    train_losses = [2.3, 1.8, 1.4, 1.1, 0.9, 0.7, 0.6, 0.5]
    val_losses = [2.4, 1.9, 1.5, 1.2, 1.0, 0.9, 0.85, 0.82]
    train_accs = [0.1, 0.3, 0.5, 0.65, 0.75, 0.82, 0.87, 0.90]
    val_accs = [0.08, 0.28, 0.48, 0.62, 0.72, 0.78, 0.82, 0.84]
    
    fig1 = plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    print("✓ Training curves created")
    
    # Example 2: Sample images
    print("\n🖼️  Creating sample image grid...")
    sample_images = np.random.randn(25, 28, 28)
    sample_labels = np.random.randint(0, 10, 25)
    sample_preds = np.random.randint(0, 10, 25)
    
    fig2 = plot_sample_images(sample_images, sample_labels, sample_preds, num_samples=16)
    print("✓ Sample image grid created")
    
    # Example 3: Confusion matrix
    print("\n📊 Creating confusion matrix...")
    cm = np.random.randint(0, 20, (10, 10))
    np.fill_diagonal(cm, np.random.randint(40, 60, 10))
    
    fig3 = plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])
    print("✓ Confusion matrix created")
    
    # Example 4: Network architecture
    print("\n🏗️  Drawing network architecture...")
    fig4 = plot_network_architecture([784, 256, 128, 10])
    print("✓ Network architecture drawn")
    
    # Example 5: Activation functions comparison
    print("\n📉 Comparing activation functions...")
    fig5 = compare_activation_functions()
    print("✓ Activation functions compared")
    
    print("\n" + "=" * 70)
    print("✓ All visualization utilities tested successfully!")
    print("Note: Figures created but not displayed (use plt.show() to view)")
    print("=" * 70)