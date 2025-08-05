import argparse
import os
import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# It's better to import from the project structure if this script is run as a module
# But for a standalone script, we might need to adjust python path or do this:
try:
    from src.task import (
        Discriminator,
        TonIoTDataset,
        test,
        INPUT_DIM,
        MODEL_FEATURE_COLUMNS,
        LABEL_COLUMN
    )
except ImportError:
    print("Could not import from 'src'.")
    print("Please ensure the script is run from the root of the project (e.g., 'python -m visualization.evaluate_final_model')")
    print("Or that the project root is in the PYTHONPATH.")
    # As a fallback for simple execution, assume task.py is in a sibling directory
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from task import (
        Discriminator,
        TonIoTDataset,
        test,
        INPUT_DIM,
        MODEL_FEATURE_COLUMNS,
        LABEL_COLUMN
    )


def load_final_model_weights(model, filepath):
    """Loads weights from .npz file into a PyTorch model."""
    try:
        npz_file = np.load(filepath)
        # The number of parameter tensors for the Generator model.
        # This is brittle and depends on the model architecture defined in client_app.py and server_app.py
        # G has 3 linear layers (weight+bias) = 6 tensors.
        num_g_params = 6
        
        # Load all tensors
        all_tensors = [npz_file[key] for key in sorted(npz_file.keys())]
        
        if len(all_tensors) < num_g_params:
             raise ValueError(f"NPZ file contains {len(all_tensors)} arrays, but expected more for G and D.")

        # We only need the Discriminator weights, which come after the Generator weights
        d_weights_np = all_tensors[num_g_params:]

        params_dict = zip(model.state_dict().keys(), d_weights_np)
        state_dict = torch.utils.collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        print(f"Successfully loaded Discriminator weights from {filepath}")
    except Exception as e:
        print(f"Error loading model weights from {filepath}: {e}")
        raise

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix on Global Hold-out Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    output_path = os.path.join(output_dir, "final_model_confusion_matrix.png")
    plt.savefig(output_path)
    print(f"Saved confusion matrix plot to {output_path}")
    plt.close()

def main():
    """Main function to load model, data, and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a final trained GAN model on a hold-out test set.")
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to the instance-specific log directory (e.g., 'logs/20250608_1030/')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/final_evaluation",
        help="Directory to save the output plots (e.g., confusion matrix)."
    )
    args = parser.parse_args()

    # Construct file paths from the log directory
    model_weights_path = os.path.join(args.log_dir, 'final_model_parameters.npz')
    holdout_data_path = os.path.join(args.log_dir, 'holdout_test_set.csv')
    scaler_path = os.path.join(args.log_dir, 'min_max_scaler.pkl')

    print(f"--- Final Model Evaluation ---")
    print(f"Log Directory: {args.log_dir}")
    print(f"  > Model Weights: {model_weights_path}")
    print(f"  > Hold-out Data: {holdout_data_path}")
    print(f"  > Scaler: {scaler_path}")
    print(f"Output plots will be saved to: {args.output_dir}")

    # --- 1. Load the hold-out test data ---
    if not os.path.exists(holdout_data_path):
        print(f"Error: Hold-out data file not found at {holdout_data_path}")
        return
    
    df_holdout = pd.read_csv(holdout_data_path)
    print(f"Successfully loaded hold-out data with {len(df_holdout)} samples.")

    X_holdout_raw = df_holdout[MODEL_FEATURE_COLUMNS]
    y_holdout = df_holdout[LABEL_COLUMN]

    # --- 2. Load the scaler ---
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found at {scaler_path}")
        return

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Successfully loaded the data scaler.")

    X_holdout_scaled = scaler.transform(X_holdout_raw)

    # --- 3. Create DataLoader ---
    holdout_dataset = TonIoTDataset(X_holdout_scaled, y_holdout.values)
    holdout_loader = DataLoader(holdout_dataset, batch_size=256, shuffle=False)

    # --- 4. Load the final model ---
    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights file not found at {model_weights_path}")
        return

    print("Loading final model weights...")
    try:
        # Load the saved NumPy arrays
        loaded_ndarrays = np.load(model_weights_path, allow_pickle=True)
        # The arrays are stored with keys 'arr_0', 'arr_1', etc.
        parameters_list = [loaded_ndarrays[key] for key in sorted(loaded_ndarrays.keys())]

        # Initialize model architecture (make sure it matches what was trained)
        # Using hyperparameters from task.py for the trained model
        # NOTE: If these were changed in your training, you must update them here
        latent_dim = 100
        g_hidden1_size = 128
        g_hidden2_size = 32
        d_hidden1_size = 128
        d_hidden2_size = 32
        
        final_generator = Generator(INPUT_DIM, latent_dim, g_hidden1_size, g_hidden2_size)
        final_discriminator = Discriminator(INPUT_DIM, d_hidden1_size, d_hidden2_size)

        num_g_params = len(list(final_generator.state_dict().keys()))
        
        g_weights = parameters_list[:num_g_params]
        d_weights = parameters_list[num_g_params:]

        set_weights(final_generator, g_weights)
        set_weights(final_discriminator, d_weights)
        print("Successfully loaded final model weights into Generator and Discriminator.")

    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # --- 5. Run evaluation ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    # The 'test' function from task.py will be used.
    # We pass the final loaded discriminator (netD)
    _, _, final_metrics = test(netD=final_discriminator, testloader=holdout_loader, device=device)

    print("\n--- Evaluation Results ---")
    print(f"  F1 Score: {final_metrics.get('f1_score', 'N/A'):.4f}")
    print(f"  Accuracy: {final_metrics.get('accuracy', 'N/A'):.4f}")
    print(f"  Precision: {final_metrics.get('precision', 'N/A'):.4f}")
    print(f"  Recall: {final_metrics.get('recall', 'N/A'):.4f}")
    print(f"  Avg Anomaly Score: {final_metrics.get('avg_anomaly_score', 'N/A'):.4f}")
    print(f"  Threshold Used: {final_metrics.get('threshold', 'N/A'):.4f}")

    # --- 6. Generate and save visualizations ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Recreate predictions for confusion matrix
    anomaly_scores = final_metrics.get("anomaly_scores", [])
    true_labels = final_metrics.get("labels", [])
    threshold = final_metrics.get("threshold", 0.5)
    
    if anomaly_scores and true_labels:
        predictions = [1 if score > threshold else 0 for score in anomaly_scores]
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix on Hold-out Test Set')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"Saved confusion matrix plot to {cm_path}")
        plt.close()

        # Classification Report
        report = classification_report(true_labels, predictions, target_names=['Normal', 'Anomaly'])
        print("\n--- Classification Report ---")
        print(report)
        report_path = os.path.join(args.output_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Saved classification report to {report_path}")

if __name__ == "__main__":
    main() 