"""
Neural Networks Tutorial - Utility Functions

This module provides comprehensive utility functions for the neural networks
beginner tutorial. All functions are designed to be beginner-friendly with
extensive documentation and error handling.

Author: Applied AI Research
License: MIT
"""

import numpy as np
from typing import Tuple, Optional, List
import urllib.request
import gzip
import os


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_mnist(
    data_dir: str = './data',
    normalize: bool = True,
    flatten: bool = True,
    one_hot: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load the MNIST dataset from online source or local cache.
    
    Downloads and caches the MNIST dataset if not already present. Returns
    training and test sets with optional preprocessing applied.
    
    Parameters
    ----------
    data_dir : str, default='./data'
        Directory to store/load MNIST data files
    normalize : bool, default=True
        Whether to normalize pixel values to [0, 1] range
    flatten : bool, default=True
        Whether to flatten images from (28, 28) to (784,)
    one_hot : bool, default=True
        Whether to convert labels to one-hot encoding
        
    Returns
    -------
    train_data : tuple of (np.ndarray, np.ndarray)
        Training images and labels
    test_data : tuple of (np.ndarray, np.ndarray)
        Test images and labels
        
    Examples
    --------
    >>> (X_train, y_train), (X_test, y_test) = load_mnist()
    >>> print(f"Training samples: {X_train.shape[0]}")
    Training samples: 60000
    >>> print(f"Feature dimension: {X_train.shape[1]}")
    Feature dimension: 784
    
    Notes
    -----
    MNIST dataset contains 60,000 training and 10,000 test grayscale images
    of handwritten digits (0-9), each 28x28 pixels.
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # URLs for MNIST dataset files
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    # Download files if not present
    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
    
    # Load training images
    with gzip.open(os.path.join(data_dir, files['train_images']), 'rb') as f:
        # Skip header (16 bytes)
        X_train = np.frombuffer(f.read(), np.uint8, offset=16)
        X_train = X_train.reshape(-1, 28, 28)
    
    # Load training labels
    with gzip.open(os.path.join(data_dir, files['train_labels']), 'rb') as f:
        # Skip header (8 bytes)
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    
    # Load test images
    with gzip.open(os.path.join(data_dir, files['test_images']), 'rb') as f:
        X_test = np.frombuffer(f.read(), np.uint8, offset=16)
        X_test = X_test.reshape(-1, 28, 28)
    
    # Load test labels
    with gzip.open(os.path.join(data_dir, files['test_labels']), 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)
    
    # Apply preprocessing
    if normalize:
        X_train = normalize_images(X_train)
        X_test = normalize_images(X_test)
    
    if flatten:
        X_train = flatten_images(X_train)
        X_test = flatten_images(X_test)
    
    if one_hot:
        y_train = one_hot_encode(y_train, num_classes=10)
        y_test = one_hot_encode(y_test, num_classes=10)
    
    return (X_train, y_train), (X_test, y_test)


def create_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create mini-batches from input data for batch gradient descent.
    
    Splits data into batches of specified size. Optionally shuffles data
    before batching to improve training dynamics.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input features
    y : np.ndarray, shape (n_samples,) or (n_samples, n_classes)
        Target labels
    batch_size : int, default=32
        Number of samples per batch
    shuffle : bool, default=True
        Whether to shuffle data before creating batches
        
    Returns
    -------
    batches : list of tuples
        List of (X_batch, y_batch) tuples
        
    Examples
    --------
    >>> X = np.random.randn(1000, 784)
    >>> y = np.random.randint(0, 10, size=(1000, 10))
    >>> batches = create_batches(X, y, batch_size=32)
    >>> len(batches)
    32
    >>> batches[0][0].shape
    (32, 784)
    
    Notes
    -----
    The last batch may be smaller than batch_size if n_samples is not
    evenly divisible by batch_size.
    """
    # Validate inputs
    assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
    
    n_samples = X.shape[0]
    
    # Shuffle if requested
    if shuffle:
        X, y = shuffle_data(X, y)
    
    # Create batches
    batches = []
    for i in range(0, n_samples, batch_size):
        # Handle last batch (may be smaller)
        end_idx = min(i + batch_size, n_samples)
        X_batch = X[i:end_idx]
        y_batch = y[i:end_idx]
        batches.append((X_batch, y_batch))
    
    return batches


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    shuffle: bool = True,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training, validation, and test sets.
    
    Divides dataset into three mutually exclusive subsets for training,
    hyperparameter tuning (validation), and final evaluation (test).
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input features
    y : np.ndarray, shape (n_samples,) or (n_samples, n_classes)
        Target labels
    train_ratio : float, default=0.7
        Proportion of data for training (0 < train_ratio < 1)
    val_ratio : float, default=0.15
        Proportion of data for validation (0 < val_ratio < 1)
    shuffle : bool, default=True
        Whether to shuffle data before splitting
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : np.ndarray
        Training, validation, and test splits
        
    Examples
    --------
    >>> X = np.random.randn(1000, 784)
    >>> y = np.random.randint(0, 10, size=(1000,))
    >>> splits = train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15)
    >>> X_train, X_val, X_test, y_train, y_val, y_test = splits
    >>> X_train.shape[0], X_val.shape[0], X_test.shape[0]
    (700, 150, 150)
    
    Notes
    -----
    Test ratio is automatically computed as (1 - train_ratio - val_ratio).
    Ratios must sum to <= 1.0.
    """
    # Validate ratios
    assert 0 < train_ratio < 1, "train_ratio must be between 0 and 1"
    assert 0 < val_ratio < 1, "val_ratio must be between 0 and 1"
    assert train_ratio + val_ratio < 1, "train_ratio + val_ratio must be < 1"
    assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
    
    # Set random seed if provided
    if random_seed is not None:
        set_random_seed(random_seed)
    
    n_samples = X.shape[0]
    
    # Shuffle if requested
    if shuffle:
        X, y = shuffle_data(X, y)
    
    # Calculate split indices
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # Split data
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def normalize_images(images: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range.
    
    Converts pixel values from [0, 255] integer range to [0, 1] float range.
    This normalization improves neural network training stability.
    
    Parameters
    ----------
    images : np.ndarray
        Input images with pixel values in [0, 255]
        
    Returns
    -------
    normalized : np.ndarray
        Images with pixel values in [0, 1]
        
    Examples
    --------
    >>> images = np.array([[[0, 128, 255]]])  # Shape: (1, 1, 3)
    >>> normalized = normalize_images(images)
    >>> normalized
    array([[[0.   , 0.502, 1.   ]]])
    
    Notes
    -----
    Input is converted to float32 to prevent integer division issues.
    """
    # Convert to float32 and normalize
    return images.astype(np.float32) / 255.0


def flatten_images(images: np.ndarray) -> np.ndarray:
    """
    Flatten 2D images into 1D vectors.
    
    Reshapes images from (n_samples, height, width) to (n_samples, height*width).
    This format is required for fully connected neural networks.
    
    Parameters
    ----------
    images : np.ndarray, shape (n_samples, height, width)
        Input images
        
    Returns
    -------
    flattened : np.ndarray, shape (n_samples, height*width)
        Flattened images
        
    Examples
    --------
    >>> images = np.random.randn(100, 28, 28)
    >>> flattened = flatten_images(images)
    >>> flattened.shape
    (100, 784)
    
    Notes
    -----
    For MNIST: 28x28 images become 784-dimensional vectors.
    """
    # Flatten all dimensions except the first (sample dimension)
    return images.reshape(images.shape[0], -1)


def one_hot_encode(
    labels: np.ndarray,
    num_classes: Optional[int] = None
) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded vectors.
    
    Transforms class labels from integers to binary vectors with a single 1
    at the class index. Used for multi-class classification with softmax.
    
    Parameters
    ----------
    labels : np.ndarray, shape (n_samples,)
        Integer class labels (0 to num_classes-1)
    num_classes : int, optional
        Number of classes. If None, inferred as max(labels) + 1
        
    Returns
    -------
    one_hot : np.ndarray, shape (n_samples, num_classes)
        One-hot encoded labels
        
    Examples
    --------
    >>> labels = np.array([0, 2, 1, 0])
    >>> one_hot_encode(labels, num_classes=3)
    array([[1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.],
           [1., 0., 0.]])
    
    Notes
    -----
    For MNIST with 10 classes:
    - Label 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    """
    # Infer number of classes if not provided
    if num_classes is None:
        num_classes = int(np.max(labels)) + 1
    
    # Validate labels
    assert np.min(labels) >= 0, "Labels must be non-negative"
    assert np.max(labels) < num_classes, f"Labels must be < {num_classes}"
    
    # Create one-hot encoding
    n_samples = labels.shape[0]
    one_hot = np.zeros((n_samples, num_classes), dtype=np.float32)
    one_hot[np.arange(n_samples), labels] = 1
    
    return one_hot


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def calculate_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Calculate classification accuracy.
    
    Computes the proportion of correct predictions. Handles both one-hot
    encoded and integer label formats.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted class probabilities (n_samples, n_classes) or
        predicted class indices (n_samples,)
    labels : np.ndarray
        True labels in same format as predictions
        
    Returns
    -------
    accuracy : float
        Accuracy score in [0, 1] range
        
    Examples
    --------
    >>> # One-hot encoded
    >>> predictions = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    >>> labels = np.array([[1, 0], [0, 1], [1, 0]])
    >>> calculate_accuracy(predictions, labels)
    1.0
    
    >>> # Integer labels
    >>> predictions = np.array([0, 1, 0])
    >>> labels = np.array([0, 1, 1])
    >>> calculate_accuracy(predictions, labels)
    0.6666666666666666
    
    Notes
    -----
    For one-hot encoded inputs, the class with highest probability is selected.
    """
    # Convert one-hot to class indices if needed
    if predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=1)
    if labels.ndim == 2:
        labels = np.argmax(labels, axis=1)
    
    # Calculate accuracy
    correct = np.sum(predictions == labels)
    total = len(labels)
    
    return correct / total


def confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: Optional[int] = None
) -> np.ndarray:
    """
    Compute confusion matrix for classification results.
    
    Creates a matrix where entry (i, j) represents the number of samples
    with true label i that were predicted as class j.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted class probabilities (n_samples, n_classes) or
        predicted class indices (n_samples,)
    labels : np.ndarray
        True labels in same format as predictions
    num_classes : int, optional
        Number of classes. If None, inferred from data
        
    Returns
    -------
    cm : np.ndarray, shape (num_classes, num_classes)
        Confusion matrix
        
    Examples
    --------
    >>> predictions = np.array([0, 1, 2, 0, 1, 2])
    >>> labels = np.array([0, 1, 1, 0, 2, 2])
    >>> confusion_matrix(predictions, labels, num_classes=3)
    array([[2, 0, 0],
           [0, 1, 1],
           [0, 1, 1]])
    
    Notes
    -----
    Diagonal entries represent correct predictions.
    Off-diagonal entries represent misclassifications.
    Row sums equal the number of samples per true class.
    """
    # Convert one-hot to class indices if needed
    if predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=1)
    if labels.ndim == 2:
        labels = np.argmax(labels, axis=1)
    
    # Infer number of classes if not provided
    if num_classes is None:
        num_classes = max(int(np.max(predictions)), int(np.max(labels))) + 1
    
    # Initialize confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    # Populate confusion matrix
    for true_label, pred_label in zip(labels, predictions):
        cm[int(true_label), int(pred_label)] += 1
    
    return cm


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def shuffle_data(
    X: np.ndarray,
    y: np.ndarray,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffle features and labels in unison.
    
    Randomly permutes samples while maintaining correspondence between
    features and labels. Useful for breaking ordering biases in data.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, ...)
        Input features
    y : np.ndarray, shape (n_samples, ...)
        Target labels
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    X_shuffled, y_shuffled : tuple of np.ndarray
        Shuffled features and labels
        
    Examples
    --------
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = np.array([10, 20, 30, 40])
    >>> X_shuffled, y_shuffled = shuffle_data(X, y, random_seed=42)
    >>> # Order is randomized but X and y still correspond
    
    Notes
    -----
    Uses the same random permutation for both X and y to maintain pairing.
    """
    # Validate inputs
    assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
    
    # Set random seed if provided
    if random_seed is not None:
        set_random_seed(random_seed)
    
    # Generate random permutation
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    # Shuffle both arrays using same permutation
    return X[indices], y[indices]


def set_random_seed(seed: int) -> None:
    """
    Set random seed for NumPy random number generator.
    
    Ensures reproducibility of random operations across runs.
    Particularly important for research and debugging.
    
    Parameters
    ----------
    seed : int
        Random seed value
        
    Examples
    --------
    >>> set_random_seed(42)
    >>> np.random.randn(3)
    array([ 0.49671415, -0.1382643 ,  0.64768854])
    >>> set_random_seed(42)
    >>> np.random.randn(3)
    array([ 0.49671415, -0.1382643 ,  0.64768854])
    
    Notes
    -----
    Call this at the start of your script for reproducible results.
    Different seeds produce different random sequences.
    """
    np.random.seed(seed)


# =============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# =============================================================================

def get_mnist_class_names() -> List[str]:
    """
    Get class names for MNIST dataset.
    
    Returns
    -------
    class_names : list of str
        List of class names (digits 0-9)
        
    Examples
    --------
    >>> get_mnist_class_names()
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    """
    return [str(i) for i in range(10)]


def print_data_info(X: np.ndarray, y: np.ndarray, name: str = "Dataset") -> None:
    """
    Print informative summary of dataset.
    
    Displays shape, data type, and value ranges for features and labels.
    Helpful for quick data inspection.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Target labels
    name : str, default="Dataset"
        Name to display for this dataset
        
    Examples
    --------
    >>> X = np.random.randn(1000, 784)
    >>> y = np.random.randint(0, 10, size=(1000,))
    >>> print_data_info(X, y, name="Training Data")
    Training Data Information:
    X shape: (1000, 784), dtype: float64
    X range: [-3.45, 3.89]
    y shape: (1000,), dtype: int64
    y unique values: [0 1 2 3 4 5 6 7 8 9]
    """
    print(f"\n{name} Information:")
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"X range: [{np.min(X):.2f}, {np.max(X):.2f}]")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")
    
    # Handle one-hot encoded labels
    if y.ndim == 2:
        y_classes = np.argmax(y, axis=1)
        print(f"y unique classes: {np.unique(y_classes)}")
    else:
        print(f"y unique values: {np.unique(y)}")


if __name__ == "__main__":
    """
    Demo script showing usage of utility functions.
    """
    print("=" * 70)
    print("Neural Networks Tutorial - Utility Functions Demo")
    print("=" * 70)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    print("\n✓ Random seed set to 42 for reproducibility")
    
    # Load MNIST dataset
    print("\n📦 Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_mnist()
    print("✓ MNIST loaded successfully!")
    
    # Display data info
    print_data_info(X_train, y_train, "Training Data")
    print_data_info(X_test, y_test, "Test Data")
    
    # Create train/val split
    print("\n✂️  Splitting training data into train/val sets...")
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(
        X_train, y_train,
        train_ratio=0.8,
        val_ratio=0.2,
        shuffle=True
    )
    print_data_info(X_train, y_train, "Train Split")
    print_data_info(X_val, y_val, "Validation Split")
    
    # Create mini-batches
    print("\n📦 Creating mini-batches...")
    batches = create_batches(X_train, y_train, batch_size=128)
    print(f"✓ Created {len(batches)} batches of size 128")
    print(f"First batch shapes: X={batches[0][0].shape}, y={batches[0][1].shape}")
    
    # Calculate accuracy example
    print("\n📊 Testing metrics...")
    sample_preds = np.argmax(y_test[:100], axis=1)
    sample_labels = np.argmax(y_test[:100], axis=1)
    acc = calculate_accuracy(sample_preds, sample_labels)
    print(f"Sample accuracy (should be 1.0): {acc:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(sample_preds, sample_labels, num_classes=10)
    print(f"Confusion matrix shape: {cm.shape}")
    
    print("\n" + "=" * 70)
    print("✓ All utility functions tested successfully!")
    print("=" * 70)