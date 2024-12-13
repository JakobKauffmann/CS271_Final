# filename: feature_engineering.py
import numpy as np
from scipy.stats import entropy

def compute_stat_features(byte_seq):
    """
    Compute statistical features from a byte sequence.

    Args:
        byte_seq (np.ndarray): Array of byte values.

    Returns:
        list: List containing mean, variance, entropy, and histogram features.
    """
    mean_val = np.mean(byte_seq)
    var_val = np.var(byte_seq)
    hist, _ = np.histogram(byte_seq, bins=256, range=(0,255))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-9)  # Normalize histogram
    ent = entropy(hist + 1e-9)   # Compute entropy
    features = [mean_val, var_val, ent] + hist.tolist()
    return features

def compute_markov_features(byte_seq, bins=16):
    """
    Compute Markov transition features from a byte sequence.

    Args:
        byte_seq (np.ndarray): Array of byte values.
        bins (int): Number of bins to categorize byte values.

    Returns:
        np.ndarray: Flattened Markov transition matrix as feature vector.
    """
    # Downsample byte values into bins
    # Each byte is in [0,255], map it to [0, bins-1]
    bin_size = 256 // bins
    binned_seq = byte_seq // bin_size

    # Construct bins x bins transition matrix
    transitions = np.zeros((bins, bins), dtype=float)
    for i in range(len(binned_seq)-1):
        transitions[binned_seq[i], binned_seq[i+1]] += 1

    # Normalize so each row sums to 1 (stochastic)
    row_sums = transitions.sum(axis=1, keepdims=True) + 1e-9
    transitions /= row_sums

    # Flatten to a 1D feature vector
    return transitions.flatten()

def compute_sequential_features(byte_seq, window_size=4):
    """
    Compute lightweight sequential features from byte sequences.

    Args:
        byte_seq (np.ndarray): Byte sequence.
        window_size (int): Size of the rolling window for statistics.

    Returns:
        np.ndarray: Sequential features.
    """
    byte_seq = np.array(byte_seq, dtype=float)
    seq_len = len(byte_seq)

    # Ensure the rolling mean covers the entire sequence
    seq_mean = np.convolve(byte_seq, np.ones(window_size) / window_size, mode='same')

    # Compute rolling variance (squared difference from mean)
    seq_var = np.convolve((byte_seq - seq_mean) ** 2, np.ones(window_size) / window_size, mode='same')

    # Calculate min and max over non-overlapping windows
    seq_min = [np.min(byte_seq[i:i + window_size]) for i in range(0, seq_len, window_size)]
    seq_max = [np.max(byte_seq[i:i + window_size]) for i in range(0, seq_len, window_size)]

    # Extend min and max features to match the original sequence length
    seq_min_full = np.repeat(seq_min, window_size)[:seq_len]
    seq_max_full = np.repeat(seq_max, window_size)[:seq_len]

    # Concatenate all features
    return np.concatenate([seq_mean, seq_var, seq_min_full, seq_max_full], axis=0)
