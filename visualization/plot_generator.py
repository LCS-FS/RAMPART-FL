import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import argparse
import os
import logging
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROFILE_COLORS = {
    "Low-End IoT": "blue",
    "Mid-Range Edge": "orange",
    "High-End Edge CPU": "green",
    "unknown_profile": "grey"
}

def get_profile_color(profile_name):
    return PROFILE_COLORS.get(profile_name, PROFILE_COLORS["unknown_profile"])

def plot_global_performance(df_global_eval, output_dir, timestamp_str):
    """Plots global F1, accuracy, precision, and recall over server rounds."""
    if df_global_eval.empty:
        logger.warning("Global evaluation data is empty. Skipping global performance plot.")
        return

    plt.figure(figsize=(12, 8))
    
    metrics_to_plot = {
        "gm_eval_f1": "Global F1-Score",
        "gm_eval_accuracy": "Global Accuracy",
        "gm_eval_precision": "Global Precision",
        "gm_eval_recall": "Global Recall"
    }
    
    for metric_col, label in metrics_to_plot.items():
        if metric_col in df_global_eval.columns and not df_global_eval[metric_col].dropna().empty:
            plt.plot(df_global_eval["server_round"], df_global_eval[metric_col].astype(float), marker='o', linestyle='-', label=label)
        else:
            logger.warning(f"Global metric {metric_col} not found or is all NaNs. Skipping its line in the plot.")

    plt.title("Global Model Performance Over Server Rounds")
    plt.xlabel("Server Round")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, f"global_model_performance_{timestamp_str}.png")
    plt.savefig(plot_filename)
    logger.info(f"Saved global model performance plot to {plot_filename}")
    plt.close()

def plot_client_f1_performance(df_client_eval, output_dir, timestamp_str):
    """Plots F1-score for each client over server rounds, color-coded by profile with hue variations."""
    if df_client_eval.empty:
        logger.warning("Client evaluation data is empty. Skipping client F1 performance plot.")
        return

    plt.figure(figsize=(15, 10))
    
    client_color_map = {}
    clients_grouped_by_profile = df_client_eval.groupby("c_eval_profile_name")

    HUE_SHIFT_TOTAL_RANGE = 0.18
    VALUE_SHIFT_TOTAL_RANGE = 0.5
    MIN_VALUE = 0.3
    MAX_VALUE = 1.0
    MIN_SATURATION = 0.35

    for profile_name_for_group, profile_df_for_group in clients_grouped_by_profile:
        base_color_name = PROFILE_COLORS.get(profile_name_for_group, PROFILE_COLORS["unknown_profile"])
        try:
            base_rgb = mcolors.to_rgb(base_color_name)
        except ValueError:
            logger.warning(f"Could not convert base_color_name '{base_color_name}' to RGB. Defaulting to grey for profile '{profile_name_for_group}'.")
            base_rgb = mcolors.to_rgb(PROFILE_COLORS["unknown_profile"])
            base_color_name = PROFILE_COLORS["unknown_profile"]

        base_hsv = mcolors.rgb_to_hsv(base_rgb)
        
        unique_clients_in_this_profile = profile_df_for_group["c_eval_partition_id"].unique()
        num_clients_in_this_profile = len(unique_clients_in_this_profile)

        for i, client_id_in_profile in enumerate(unique_clients_in_this_profile):
            if num_clients_in_this_profile == 1:
                client_color_map[client_id_in_profile] = base_color_name
            else:
                shift_factor = (i / (num_clients_in_this_profile - 1)) - 0.5 if num_clients_in_this_profile > 1 else 0
                
                h_new = base_hsv[0]
                s_new = base_hsv[1]
                v_new = base_hsv[2]

                if base_hsv[1] > 0.05:
                    hue_shift_amount = shift_factor * HUE_SHIFT_TOTAL_RANGE
                    h_new = (base_hsv[0] + hue_shift_amount) % 1.0
                
                value_shift_amount = shift_factor * VALUE_SHIFT_TOTAL_RANGE
                v_new = np.clip(base_hsv[2] + value_shift_amount, MIN_VALUE, MAX_VALUE)
                
                if base_hsv[1] <= 0.05:
                    s_new = np.clip(base_hsv[1], 0.0, 0.05)
                else:
                    s_new = np.clip(base_hsv[1], MIN_SATURATION, 1.0)
                
                varied_color_rgb = mcolors.hsv_to_rgb((h_new, s_new, v_new))
                client_color_map[client_id_in_profile] = varied_color_rgb

    for client_id, group_data in df_client_eval.groupby("c_eval_partition_id"):
        profile_name = group_data["c_eval_profile_name"].iloc[0]
        color = client_color_map.get(str(client_id), PROFILE_COLORS["unknown_profile"])
        
        group_data_sorted = group_data.sort_values("server_round")
        
        if "c_eval_f1" in group_data_sorted.columns and not group_data_sorted["c_eval_f1"].dropna().empty:
            plt.plot(group_data_sorted["server_round"], group_data_sorted["c_eval_f1"].astype(float), marker='o', linestyle='-', 
                    label=f"Client {client_id} ({profile_name})", color=color, alpha=0.7)
        else:
            logger.warning(f"Client {client_id} F1-score data (c_eval_f1) not found or is all NaNs. Skipping its line.")

    plt.title("Per-Client F1-Score Over Server Rounds (Color-coded by Device Profile)")
    plt.xlabel("Server Round")
    plt.ylabel("F1-Score")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) > 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
    else:
        plt.legend(fontsize='small')

    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plot_filename = os.path.join(output_dir, f"per_client_f1_performance_{timestamp_str}.png")
    plt.savefig(plot_filename)
    logger.info(f"Saved per-client F1 performance plot to {plot_filename}")
    plt.close()

def plot_global_gan_losses(df_global_fit, output_dir, timestamp_str):
    """Plots the aggregated Generator and Discriminator losses over server rounds."""
    if df_global_fit.empty:
        logger.warning("Global fit data is empty. Skipping global GAN loss plot.")
        return

    plt.figure(figsize=(12, 8))
    
    if 'gm_fit_g_loss' in df_global_fit.columns and not df_global_fit['gm_fit_g_loss'].dropna().empty:
        plt.plot(df_global_fit["server_round"], df_global_fit["gm_fit_g_loss"].astype(float), marker='^', linestyle='--', label="Aggregated Generator Loss", color='blue')
    else:
        logger.warning("Global metric gm_fit_g_loss not found or is all NaNs. Skipping its line.")

    if 'gm_fit_d_loss' in df_global_fit.columns and not df_global_fit['gm_fit_d_loss'].dropna().empty:
        plt.plot(df_global_fit["server_round"], df_global_fit["gm_fit_d_loss"].astype(float), marker='v', linestyle=':', label="Aggregated Discriminator Loss", color='red')
    else:
        logger.warning("Global metric gm_fit_d_loss not found or is all NaNs. Skipping its line.")

    plt.title("Aggregated GAN Losses Over Server Rounds")
    plt.xlabel("Server Round")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, f"global_gan_losses_{timestamp_str}.png")
    plt.savefig(plot_filename)
    logger.info(f"Saved global GAN losses plot to {plot_filename}")
    plt.close()

def plot_selection_dynamics(df_selection_info, output_dir, timestamp_str):
    """Plots softmax temperature and the standard deviation of selection probabilities."""
    if df_selection_info.empty:
        logger.warning("Selection info data is empty. Skipping selection dynamics plot.")
        return

    dynamics_data = df_selection_info.groupby("server_round").agg(
        prob_std=('s_client_selection_prob', 'std'),
        temperature=('s_softmax_temperature', 'first')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 8))

    color = 'tab:red'
    ax1.set_xlabel('Server Round')
    ax1.set_ylabel('Softmax Temperature', color=color)
    ax1.plot(dynamics_data['server_round'], dynamics_data['temperature'], color=color, marker='.', linestyle='-', label='Temperature')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='major', axis='y', linestyle='--', color=color, alpha=0.6)


    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Std Dev of Selection Probabilities', color=color)
    ax2.plot(dynamics_data['server_round'], dynamics_data['prob_std'], color=color, marker='.', linestyle='--', label='Prob. Std Dev')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('RL Selection Dynamics: Exploration vs. Exploitation')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plot_filename = os.path.join(output_dir, f"selection_dynamics_{timestamp_str}.png")
    plt.savefig(plot_filename)
    logger.info(f"Saved selection dynamics plot to {plot_filename}")
    plt.close()

def plot_client_fit_loss_distribution(df_client_fit, output_dir, timestamp_str, bin_size=10):
    """Plots the distribution of client-level G and D losses as boxplots, binned by server rounds."""
    if df_client_fit.empty:
        logger.warning("Client fit data is empty. Skipping client fit loss distribution plot.")
        return
        
    df_plot = df_client_fit.copy()
    
    for col in ['c_fit_g_loss', 'c_fit_d_loss']:
        if col in df_plot.columns:
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
        else:
            logger.warning(f"Column {col} not found in client fit data. Cannot generate loss distribution plot.")
            return
            
    df_plot.dropna(subset=['c_fit_g_loss', 'c_fit_d_loss'], inplace=True)
    
    if df_plot.empty:
        logger.warning("No valid client fit loss data to plot after cleaning NaNs.")
        return

    df_plot['round_bin'] = (df_plot['server_round'] // bin_size) * bin_size
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    sns.boxplot(ax=axes[0], x='round_bin', y='c_fit_g_loss', data=df_plot, whis=[5, 95], showfliers=False)
    axes[0].set_title("Distribution of Client-Level Generator (G) Loss Per Round (Binned)")
    axes[0].set_ylabel("G-Loss")
    axes[0].grid(True)
    
    sns.boxplot(ax=axes[1], x='round_bin', y='c_fit_d_loss', data=df_plot, whis=[5, 95], showfliers=False)
    axes[1].set_title("Distribution of Client-Level Discriminator (D) Loss Per Round (Binned)")
    axes[1].set_xlabel(f"Server Round (Binned by {bin_size})")
    axes[1].set_ylabel("D-Loss")
    axes[1].grid(True)
    
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"client_fit_loss_distribution_{timestamp_str}.png")
    plt.savefig(plot_filename)
    logger.info(f"Saved client fit loss distribution plot to {plot_filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate performance plots from RL training CSV data.")
    parser.add_argument("csv_filepath", type=str, help="Path to the rl_training_data.csv file.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save the generated plots (default: plots).")
    
    args = parser.parse_args()

    if not os.path.exists(args.csv_filepath):
        logger.error(f"Error: CSV file not found at {args.csv_filepath}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Reading data from: {args.csv_filepath}")
    logger.info(f"Plots will be saved to: {args.output_dir}")

    timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Using timestamp for plot filenames: {timestamp_now}")

    try:
        df = pd.read_csv(args.csv_filepath)
        logger.info(f"Successfully loaded CSV. Shape: {df.shape}")
        
        df["server_round"] = pd.to_numeric(df["server_round"], errors='coerce')
        df = df.dropna(subset=["server_round"])
        df["server_round"] = df["server_round"].astype(int)


        df_global_eval = df[df["event_type"] == "global_eval_metrics"].copy()
        for col in ["gm_eval_f1", "gm_eval_accuracy", "gm_eval_precision", "gm_eval_recall"]:
            if col in df_global_eval.columns:
                df_global_eval[col] = pd.to_numeric(df_global_eval[col], errors='coerce')
        df_global_eval.sort_values("server_round", inplace=True)
        
        df_global_fit = df[df["event_type"] == "global_fit_metrics"].copy()
        for col in ["gm_fit_g_loss", "gm_fit_d_loss"]:
            if col in df_global_fit.columns:
                df_global_fit[col] = pd.to_numeric(df_global_fit[col], errors='coerce')
        df_global_fit.sort_values("server_round", inplace=True)
        
        df_client_eval = df[df["event_type"] == "client_eval_metrics"].copy()
        for col in ["c_eval_f1", "c_eval_accuracy", "c_eval_precision", "c_eval_recall", "c_eval_num_samples", "c_eval_time_seconds", "c_eval_cores"]:
            if col in df_client_eval.columns:
                df_client_eval[col] = pd.to_numeric(df_client_eval[col], errors='coerce')
        if "c_eval_partition_id" in df_client_eval.columns:
            df_client_eval["c_eval_partition_id"] = df_client_eval["c_eval_partition_id"].astype(str)

        df_client_eval.sort_values(["c_eval_partition_id", "server_round"], inplace=True)

        df_client_fit = df[df["event_type"] == "client_fit_metrics"].copy()

        df_selection_info = df[df["event_type"] == "selection_info"].copy()
        for col in ["s_client_selection_prob", "s_softmax_temperature"]:
             if col in df_selection_info.columns:
                df_selection_info[col] = pd.to_numeric(df_selection_info[col], errors='coerce')
        df_selection_info.sort_values("server_round", inplace=True)

        plot_global_performance(df_global_eval, args.output_dir, timestamp_now)
        plot_client_f1_performance(df_client_eval, args.output_dir, timestamp_now)
        plot_global_gan_losses(df_global_fit, args.output_dir, timestamp_now)
        plot_selection_dynamics(df_selection_info, args.output_dir, timestamp_now)
        plot_client_fit_loss_distribution(df_client_fit, args.output_dir, timestamp_now)
        
        logger.info("Plot generation completed.")

    except FileNotFoundError:
        logger.error(f"Could not find the CSV file at {args.csv_filepath}")
    except pd.errors.EmptyDataError:
        logger.error(f"The CSV file at {args.csv_filepath} is empty.")
    except Exception as e:
        logger.error(f"An error occurred during plot generation: {e}", exc_info=True)

if __name__ == "__main__":
    main() 