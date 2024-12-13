import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from joblib import load, dump, delayed, Parallel
from tqdm import tqdm
from feature_engineering import compute_stat_features, compute_markov_features
from utils import get_data_samples_from_path
import gc


def compute_features_for_sample(byte_seq, bin_count=16):
    """
    Compute features for a single sample, combining statistical and Markov features.

    Args:
        byte_seq (np.ndarray): Byte sequence.
        bin_count (int): Number of bins for Markov features.

    Returns:
        np.ndarray: Combined feature vector.
    """
    stat_features = compute_stat_features(byte_seq)
    markov_features = compute_markov_features(byte_seq, bins=bin_count)
    return np.concatenate([stat_features, markov_features])


def main():
    # Configuration
    data_dir = "/Users/jake/Desktop/data"  # Adjust path as needed
    bin_count = 16
    batch_size = 1000
    output_dir = "models_gb_finetuned"
    os.makedirs(output_dir, exist_ok=True)

    # Load scaler and label encoder
    print("Loading supporting objects...")
    scaler = load("models_gb_optimized/scaler.joblib")
    le = load("models_gb_optimized/label_encoder.joblib")

    # Load data
    print("Loading data...")
    samples_dict = get_data_samples_from_path(data_dir)
    all_samples, all_labels = [], []
    for klass, samples in samples_dict.items():
        if klass == "unknown":
            continue  # Exclude 'unknown' (not a predicted class)
        for sample_obj in samples:
            all_samples.append(sample_obj.data)
            all_labels.append(klass)
    all_samples = np.array(all_samples)
    all_labels = np.array(all_labels)

    print(f"Number of samples: {len(all_samples)}")
    print(f"Shape of one sample: {all_samples[0].shape}")

    # Encode labels
    y = le.transform(all_labels)

    # Filter for classes A and B
    A_label = le.transform(['A'])[0]
    B_label = le.transform(['B'])[0]
    A_B_indices = np.where((y == A_label) | (y == B_label))[0]
    A_B_samples = all_samples[A_B_indices]
    A_B_labels = y[A_B_indices]

    # Remap labels for binary classification
    binary_label_map = {A_label: 0, B_label: 1}
    A_B_labels = np.array([binary_label_map[label] for label in A_B_labels])

    # Compute features for A and B
    print("\nProcessing A and B data...")
    A_B_features = []
    for i in range(0, len(A_B_samples), batch_size):
        batch_samples = A_B_samples[i:i + batch_size]
        features = Parallel(n_jobs=4, backend="loky")(
            delayed(compute_features_for_sample)(seq, bin_count) for seq in tqdm(batch_samples, desc=f"Batch {i // batch_size + 1}")
        )
        A_B_features.extend(features)
    A_B_features = np.array(A_B_features)

    # Scale features for A and B
    print("\nScaling A/B features...")
    ab_scaler = StandardScaler()
    A_B_features = ab_scaler.fit_transform(A_B_features)

    # Initialize a new Gradient Boosting model for binary classification
    print("\nInitializing a new Gradient Boosting model for binary classification...")
    ab_model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=50,
        validation_fraction=0.2,
        n_iter_no_change=5,
        random_state=42
    )

    # Train the model
    print("\nFine-tuning Gradient Boosting model on A and B...")
    ab_model.fit(A_B_features, A_B_labels)

    # Evaluate the fine-tuned model
    y_pred_ab = ab_model.predict(A_B_features)

    # Metrics for A and B
    acc = accuracy_score(A_B_labels, y_pred_ab)
    precision = precision_score(A_B_labels, y_pred_ab, average="binary", zero_division=0)
    recall = recall_score(A_B_labels, y_pred_ab, average="binary", zero_division=0)
    f1 = f1_score(A_B_labels, y_pred_ab, average="binary", zero_division=0)
    print(f"\nFine-tuned Model Metrics (A and B):")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(A_B_labels, y_pred_ab, target_names=['A', 'B']))

    # Save fine-tuned model and scaler
    print("\nSaving fine-tuned model and scaler...")
    dump(ab_model, os.path.join(output_dir, "fine_tuned_gb_model_ab.joblib"))
    dump(ab_scaler, os.path.join(output_dir, "scaler.joblib"))
    print("Fine-tuned model and scaler saved successfully.")

    # Garbage collection
    del A_B_features, A_B_labels, A_B_samples, ab_model, ab_scaler, scaler, le
    gc.collect()


if __name__ == "__main__":
    main()