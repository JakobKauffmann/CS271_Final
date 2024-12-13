import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from joblib import Parallel, delayed, dump
from tqdm import tqdm
import pandas as pd
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
    # Compute statistical and Markov transition features
    stat_features = compute_stat_features(byte_seq)
    markov_features = compute_markov_features(byte_seq, bins=bin_count)
    return np.concatenate([stat_features, markov_features])


def main():
    data_dir = "/Users/jake/Desktop/data"  # Adjust path as needed
    bin_count = 16
    batch_size = 1000
    output_dir = "models_gb_optimized"
    os.makedirs(output_dir, exist_ok=True)

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
    le = LabelEncoder()
    y = le.fit_transform(all_labels)
    print("Encoded classes:", le.classes_)

    # Split data into training and validation sets
    print("\nSplitting data into training (80%) and validation (20%) sets...")
    train_idx, val_idx = train_test_split(
        np.arange(len(all_samples)), test_size=0.2, stratify=y, random_state=42
    )
    X_train, y_train = [], y[train_idx]
    X_val, y_val = [], y[val_idx]

    # Process training data in batches
    print("Processing training data...")
    for i in range(0, len(train_idx), batch_size):
        batch_indices = train_idx[i:i + batch_size]
        batch_samples = all_samples[batch_indices]
        features = Parallel(n_jobs=4, backend="loky")(  # Reduced parallel jobs for efficiency
            delayed(compute_features_for_sample)(seq, bin_count) for seq in tqdm(batch_samples, desc=f"Training Batch {i // batch_size + 1}")
        )
        X_train.extend(features)
    X_train = np.array(X_train)

    # Process validation data in batches
    print("Processing validation data...")
    for i in range(0, len(val_idx), batch_size):
        batch_indices = val_idx[i:i + batch_size]
        batch_samples = all_samples[batch_indices]
        features = Parallel(n_jobs=4, backend="loky")(
            delayed(compute_features_for_sample)(seq, bin_count) for seq in tqdm(batch_samples, desc=f"Validation Batch {i // batch_size + 1}")
        )
        X_val.extend(features)
    X_val = np.array(X_val)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Train GB
    print("Training Gradient Boosting Model...")
    gb_model = HistGradientBoostingClassifier(
        learning_rate=0.2,
        #max_depth=10,
        max_iter=50,
        validation_fraction=0.2,
        n_iter_no_change=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = gb_model.predict(X_val)

    # Overall metrics
    acc = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    print(f"\nValidation Accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    # by class metrics
    by_class_metrics = []
    for class_idx, class_name in enumerate(le.classes_):
        class_precision = precision_score(y_val, y_pred, labels=[class_idx], average="micro", zero_division=0)
        class_recall = recall_score(y_val, y_pred, labels=[class_idx], average="micro", zero_division=0)
        class_f1 = f1_score(y_val, y_pred, labels=[class_idx], average="micro", zero_division=0)
        by_class_metrics.append({
            "class": class_name,
            "precision": class_precision,
            "recall": class_recall,
            "f1_score": class_f1,
        })

    # Save metrics to CSV for quick visualization in Tableau
    pd.DataFrame([{
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }]).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    pd.DataFrame(by_class_metrics).to_csv(os.path.join(output_dir, "by_class_metrics.csv"), index=False)
    print("\nMetrics saved to 'metrics.csv' and 'by_class_metrics.csv'.")

    # Save final model and scaler for classifying unknown
    print("\nSaving final model and scaler...")
    dump(gb_model, os.path.join(output_dir, "final_gb_model.joblib"))
    dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    dump(le, os.path.join(output_dir, "label_encoder.joblib"))
    print("Final model, scaler, and label encoder saved successfully.")

    # Garbage collection at the end for efficiency
    del X_train, X_val, scaler, gb_model, train_idx, val_idx, y_train, y_val
    gc.collect()


if __name__ == "__main__":
    main()