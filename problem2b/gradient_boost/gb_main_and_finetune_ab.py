import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from joblib import load
from tqdm import tqdm
from feature_engineering import compute_stat_features, compute_markov_features
from utils import get_data_samples_from_path
import gc
import pandas as pd


def compute_features_for_sample(byte_seq, bin_count=16):
    stat_features = compute_stat_features(byte_seq)
    markov_features = compute_markov_features(byte_seq, bins=bin_count)
    return np.concatenate([stat_features, markov_features])


def refine_predictions_with_ab_model(main_pred, byte_seq, ab_model, scaler, label_encoder):
    if main_pred in ['A', 'B']:
        ab_features = compute_features_for_sample(byte_seq)
        ab_features = scaler.transform(ab_features.reshape(1, -1))
        ab_pred = ab_model.predict(ab_features)
        ab_class = label_encoder.inverse_transform([ab_pred])[0]
        return ab_class
    return main_pred


def main():
    # Configuration
    data_dir = "/Users/jake/Desktop/data"  # Adjust path as needed
    bin_count = 16
    batch_size = 1000
    output_dir = "gb_ab_metrics"
    os.makedirs(output_dir, exist_ok=True)

    # Load models, scalers, and label encoder for base gb and ab binary gb
    print("Loading models and scalers...")
    gb_model = load("models_gb_optimized/final_gb_model.joblib")
    ab_model = load("models_gb_finetuned/fine_tuned_gb_model_ab.joblib")
    scaler = load("models_gb_optimized/scaler.joblib")
    ab_scaler = load("models_gb_finetuned/scaler.joblib")
    le = load("models_gb_optimized/label_encoder.joblib")

    # Load data using utils
    print("Loading data...")
    samples_dict = get_data_samples_from_path(data_dir)
    all_samples, all_labels = [], []
    for klass, samples in samples_dict.items():
        if klass == "unknown":
            continue
        for sample_obj in samples:
            all_samples.append(sample_obj.data)
            all_labels.append(klass)
    all_samples = np.array(all_samples)
    all_labels = np.array(all_labels)

    print(f"Number of samples: {len(all_samples)}")
    print(f"Shape of one sample: {all_samples[0].shape}")

    # Encode labels
    y = le.transform(all_labels)

    # Split data into training and validation sets
    print("\nSplitting data into training (80%) and validation (20%) sets...")
    train_idx, val_idx = train_test_split(
        np.arange(len(all_samples)), test_size=0.2, stratify=y, random_state=42
    )
    X_val, y_val = [], y[val_idx]
    val_samples = all_samples[val_idx]

    # Process validation data in batches
    print("Processing validation data...")
    for i in range(0, len(val_idx), batch_size):
        batch_indices = val_idx[i:i + batch_size]
        batch_samples = all_samples[batch_indices]
        features = [
            compute_features_for_sample(seq, bin_count)
            for seq in tqdm(batch_samples, desc=f"Validation Batch {i // batch_size + 1}")
        ]
        X_val.extend(features)
    X_val = np.array(X_val)

    # Scale validation features
    X_val = scaler.transform(X_val)

    # Predict on validation set with the main model
    print("\nPredicting with the main model...")
    y_pred = gb_model.predict(X_val)

    # Refine predictions for A and B using the fine-tuned model
    print("\nRefining predictions for A and B...")
    refined_predictions = [
        refine_predictions_with_ab_model(
            le.inverse_transform([main_pred])[0], byte_seq, ab_model, ab_scaler, le
        )
        for main_pred, byte_seq in zip(y_pred, val_samples)
    ]
    refined_predictions = le.transform(refined_predictions)

    # Evaluate the combined model
    acc = accuracy_score(y_val, refined_predictions)
    precision = precision_score(y_val, refined_predictions, average="weighted", zero_division=0)
    recall = recall_score(y_val, refined_predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_val, refined_predictions, average="weighted", zero_division=0)
    print(f"\nCombined Model Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, refined_predictions, target_names=le.classes_))

    # Save overall metrics to CSV
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "combined_metrics.csv"), index=False)

    # Calculate by clsss
    by_class_metrics = []
    for class_idx, class_name in enumerate(le.classes_):
        class_precision = precision_score(y_val, refined_predictions, labels=[class_idx], average="micro", zero_division=0)
        class_recall = recall_score(y_val, refined_predictions, labels=[class_idx], average="micro", zero_division=0)
        class_f1 = f1_score(y_val, refined_predictions, labels=[class_idx], average="micro", zero_division=0)
        by_class_metrics.append({
            "class": class_name,
            "precision": class_precision,
            "recall": class_recall,
            "f1_score": class_f1,
        })

    # Save  by lass metrics
    pd.DataFrame(by_class_metrics).to_csv(os.path.join(output_dir, "combined_by_class_metrics.csv"), index=False)

    print("\nMetrics saved in 'combined_metrics.csv' and 'combined_by_class_metrics.csv'.")

    # collect garbage
    del X_val, y_val, val_samples, gb_model, ab_model, scaler, ab_scaler, le
    gc.collect()


if __name__ == "__main__":
    main()