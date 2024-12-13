import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import load, Parallel, delayed
from tqdm import tqdm
from feature_engineering import compute_stat_features, compute_markov_features
from utils import get_data_samples_from_path
import pandas as pd
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


def refine_predictions_with_ab_model(main_pred, byte_seq, ab_model, scaler, label_encoder):
    """
    Refine predictions using the fine-tuned A/B model.

    Args:
        main_pred (int): Encoded prediction from the main model.
        byte_seq (np.ndarray): Original byte sequence for the sample.
        ab_model: Fine-tuned Gradient Boosting model for A and B.
        scaler: Scaler used during training of the A/B model.
        label_encoder: Label encoder for class labels.

    Returns:
        int: Refined prediction in encoded form.
    """
    if main_pred in label_encoder.transform(['A', 'B']):
        # Compute features for the binary A/B model
        byte_seq = np.array(byte_seq)  # Ensure byte_seq is a numpy array
        ab_features = compute_features_for_sample(byte_seq)
        ab_features = scaler.transform(ab_features.reshape(1, -1))
        ab_pred = ab_model.predict(ab_features)[0]  # Get the predicted class (0 or 1)
        refined_class = label_encoder.transform(['A', 'B'])[ab_pred]
        return refined_class
    return main_pred


def classify_unknown():
    # Configuration
    data_dir = "/Users/jake/Desktop/data"  # Adjust path as needed
    bin_count = 16
    batch_size = 500  # Adjust batch size based on available resources
    output_dir = "unknown_predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Load models, scalers, and label encoder
    print("Loading models and supporting objects...")
    gb_model = load("models_gb_optimized/final_gb_model.joblib")
    ab_model = load("models_gb_finetuned/fine_tuned_gb_model_ab.joblib")
    scaler = load("models_gb_optimized/scaler.joblib")
    ab_scaler = load("models_gb_finetuned/scaler.joblib")
    le = load("models_gb_optimized/label_encoder.joblib")

    # Load unknown samples
    print("Loading unknown samples...")
    samples_dict = get_data_samples_from_path(data_dir)
    unknown_samples = samples_dict.get("unknown", [])
    if not unknown_samples:
        print("No unknown samples found.")
        return
    all_unknown_samples = np.array([np.array(sample.data) for sample in unknown_samples])  # Convert to np.ndarray

    print(f"Number of unknown samples: {len(all_unknown_samples)}")

    # Process unknown samples in batches
    predictions = []
    for i in range(0, len(all_unknown_samples), batch_size):
        batch_samples = all_unknown_samples[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}...")

        # Compute features for the batch
        features = Parallel(n_jobs=4, backend="loky")(
            delayed(compute_features_for_sample)(seq, bin_count) for seq in tqdm(batch_samples, desc=f"Batch {i // batch_size + 1}")
        )
        features = np.array(features)
        features = scaler.transform(features)

        # Predict with the main model
        main_preds = gb_model.predict(features)

        # Refine predictions for A and B
        refined_preds = [
            refine_predictions_with_ab_model(
                main_pred, byte_seq, ab_model, ab_scaler, le
            )
            for main_pred, byte_seq in zip(main_preds, batch_samples)
        ]
        predictions.extend(refined_preds)

    # Decode predictions back to class labels
    decoded_predictions = le.inverse_transform(np.array(predictions))

    # Save predictions to a text file in the required format
    output_file = os.path.join(output_dir, "unknown_predictions.txt")
    with open(output_file, "w") as f:
        for idx, pred in enumerate(decoded_predictions):
            f.write(f"sample {idx}, {pred}\n")
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    classify_unknown()
