import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import matplotlib.colors as mcolors
from datetime import datetime

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 12

PROFILE_COLORS = {
    "Low-End IoT": "blue",
    "Mid-Range Edge": "orange",
    "High-End Edge CPU": "green",
    "unknown_profile": "grey"
}

def plot_global_performance(df, output_dir):
    """Plots global evaluation metrics (F1, Accuracy, Precision, Recall) over server rounds."""
    df_global_eval = df[df['event_type'] == 'global_eval_metrics'].dropna(
        subset=['server_round', 'gm_eval_f1', 'gm_eval_accuracy', 'gm_eval_precision', 'gm_eval_recall']
    ).copy()
    
    if df_global_eval.empty:
        print("No 'global_eval_metrics' data found to plot global performance.")
        return

    df_global_eval['server_round'] = pd.to_numeric(df_global_eval['server_round'])
    
    plt.figure()
    plt.plot(df_global_eval['server_round'], df_global_eval['gm_eval_f1'], marker='o', linestyle='-', label='Global F1 Score')
    plt.plot(df_global_eval['server_round'], df_global_eval['gm_eval_accuracy'], marker='s', linestyle='--', label='Global Accuracy', color='#FF69B4')
    plt.plot(df_global_eval['server_round'], df_global_eval['gm_eval_precision'], marker='^', linestyle=':', label='Global Precision')
    plt.plot(df_global_eval['server_round'], df_global_eval['gm_eval_recall'], marker='x', linestyle='-.', label='Global Recall')
    
    plt.title('Global Model Performance Metrics Over Server Rounds', fontsize=32)
    plt.xlabel('Server Round', fontsize=24)
    plt.ylabel('Score', fontsize=24)
    plt.legend(fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "global_performance_metrics.png")
    plt.savefig(output_path)
    print(f"Saved global performance plot to {output_path}")
    plt.close()

def plot_rewards_and_components(df, output_dir):
    """Plots the total reward and its components over server rounds."""
    df_learning = df[df['event_type'] == 'learning_update'].copy()
    if df_learning.empty:
        print("No 'learning_update' data found to plot rewards.")
        return
        
    df_rewards = df_learning.groupby('server_round').first().reset_index()

    plt.figure()
    plt.plot(df_rewards['server_round'], df_rewards['l_reward_total_global_for_action'], marker='o', linestyle='-', label='Total Global Reward')
    plt.plot(df_rewards['server_round'], df_rewards['l_reward_performance_component'], marker='s', linestyle='--', label='Performance Component')
    plt.plot(df_rewards['server_round'], -df_rewards['l_reward_fairness_penalty_component'], marker='^', linestyle=':', label='Fairness Penalty (inverted)')
    plt.plot(df_rewards['server_round'], -df_rewards['l_reward_resource_cost_component'], marker='x', linestyle='-.', label='Resource Cost (inverted)')

    plt.title('RL Agent: Reward & Components Over Server Rounds')
    plt.xlabel('Server Round')
    plt.ylabel('Reward Value / Penalty Value')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "rl_rewards_and_components.png")
    plt.savefig(output_path)
    print(f"Saved rewards and components plot to {output_path}")
    plt.close()

def plot_q_value_evolution(df, output_dir, num_states_to_track=5):
    """Tracks and plots the Q-value evolution for a few selected states."""
    df_learning = df[df['event_type'] == 'learning_update'].copy()
    if df_learning.empty:
        print("No 'learning_update' data found to plot Q-value evolution.")
        return

    top_states = df_learning['l_state_at_selection'].value_counts().nlargest(num_states_to_track).index.tolist()
    
    if not top_states:
        print("Could not identify any states to track for Q-value evolution.")
        return

    print(f"Tracking Q-value evolution for top {len(top_states)} states: {top_states}")

    plt.figure()
    for state_str in top_states:
        df_state = df_learning[df_learning['l_state_at_selection'] == state_str]
        plt.plot(df_state['server_round'], df_state['l_q_updated_value_S'], marker='o', linestyle='-', label=f'State: {state_str}')

    plt.title(f'Q-Value Evolution for Top {len(top_states)} States')
    plt.xlabel('Server Round')
    plt.ylabel('Q-Value')
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "q_value_evolution.png")
    plt.savefig(output_path)
    print(f"Saved Q-value evolution plot to {output_path}")
    plt.close()


def plot_client_selection_distribution(df, output_dir):
    """Plots the distribution of selected clients based on their profile and core count."""
    df_client_eval = df[df['event_type'] == 'client_eval_metrics'].copy()
    df_selection = df[(df['event_type'] == 'selection_info') & (df['s_was_selected'] == 1)].copy()

    if df_client_eval.empty or df_selection.empty:
        print("Not enough 'client_eval_metrics' or 'selection_info' data to plot client selection distribution.")
        return
    
    available_info_cols = ['client_cid']
    if 'c_eval_profile_name' in df_client_eval.columns:
        available_info_cols.append('c_eval_profile_name')
    else:
        print("Warning: 'c_eval_profile_name' column not found in data. Skipping selection by profile plot.")

    if 'c_eval_cores' in df_client_eval.columns:
        available_info_cols.append('c_eval_cores')
    else:
        print("Warning: 'c_eval_cores' column not found in data. Skipping selection by cores plot.")

    client_info = df_client_eval.drop_duplicates(subset='client_cid', keep='last')[available_info_cols]
    
    df_selected_with_info = pd.merge(df_selection, client_info, on='client_cid')
    
    if df_selected_with_info.empty:
        print("Could not merge selection data with client info. No plots generated for selection distribution.")
        return

    if 'c_eval_profile_name' in df_selected_with_info.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(y='c_eval_profile_name', data=df_selected_with_info, order=df_selected_with_info['c_eval_profile_name'].value_counts().index)
        plt.title('Total Times Each Client Profile Was Selected')
        plt.xlabel('Number of Selections')
        plt.ylabel('Device Profile')
        plt.tight_layout()
        output_path_profile = os.path.join(output_dir, "client_selection_by_profile.png")
        plt.savefig(output_path_profile)
        print(f"Saved client selection by profile plot to {output_path_profile}")
        plt.close()

    if 'c_eval_cores' in df_selected_with_info.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(y='c_eval_cores', data=df_selected_with_info, order=df_selected_with_info['c_eval_cores'].value_counts().index, palette='viridis')
        plt.title('Total Times Clients Were Selected, by Core Count')
        plt.xlabel('Number of Selections')
        plt.ylabel('Number of CPU Cores')
        plt.tight_layout()
        output_path_cores = os.path.join(output_dir, "client_selection_by_cores.png")
        plt.savefig(output_path_cores)
        print(f"Saved client selection by cores plot to {output_path_cores}")
        plt.close()


def plot_softmax_temperature_decay(df, output_dir):
    """Plots the softmax temperature decay over server rounds."""
    df_selection = df[df['event_type'] == 'selection_info'].copy()
    if df_selection.empty:
        print("No 'selection_info' data found to plot softmax temperature decay.")
        return

    df_params = df_selection.groupby('server_round').agg(
        s_softmax_temperature=('s_softmax_temperature', 'first')
    ).reset_index()

    plt.figure()
    plt.plot(df_params['server_round'], df_params['s_softmax_temperature'], marker='.', linestyle='-', label='Softmax Temperature')
    
    plt.title('Softmax Temperature Decay Over Server Rounds')
    plt.xlabel('Server Round')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "softmax_temperature_decay.png")
    plt.savefig(output_path)
    print(f"Saved softmax temperature decay plot to {output_path}")
    plt.close()


def plot_clients_selected_per_round(df, output_dir):
    """Plots the number of clients selected per round."""
    df_selection = df[df['event_type'] == 'selection_info'].copy()
    if df_selection.empty:
        print("No 'selection_info' data found to plot clients selected per round.")
        return

    df_params = df_selection.groupby('server_round').agg(
        num_selected=('s_was_selected', 'sum')
    ).reset_index()

    plt.figure()
    plt.plot(df_params['server_round'], df_params['num_selected'], marker='o', linestyle='-', label='Clients Selected')
    
    plt.title('Number of Clients Selected per Round')
    plt.xlabel('Server Round')
    plt.ylabel('Number of Clients')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "clients_selected_per_round.png")
    plt.savefig(output_path)
    print(f"Saved clients selected per round plot to {output_path}")
    plt.close()

def plot_per_client_performance(df, output_dir):
    """Plots F1-score for each client over server rounds, color-coded by profile with hue variations."""
    df_client_eval = df[df['event_type'] == 'client_eval_metrics'].dropna(
        subset=['c_eval_f1', 'client_cid', 'c_eval_partition_id', 'c_eval_profile_name']
    ).copy()
    
    if df_client_eval.empty:
        print("Client evaluation data is empty. Skipping client F1 performance plot.")
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
            print(f"Warning: Could not convert base_color_name '{base_color_name}' to RGB. Defaulting to grey for profile '{profile_name_for_group}'.")
            base_rgb = mcolors.to_rgb(PROFILE_COLORS["unknown_profile"])
            base_color_name = PROFILE_COLORS["unknown_profile"]

        base_hsv = mcolors.rgb_to_hsv(base_rgb)
        
        profile_df_for_group['c_eval_partition_id'] = profile_df_for_group['c_eval_partition_id'].astype(str)
        unique_clients_in_this_profile = profile_df_for_group["c_eval_partition_id"].unique()
        num_clients_in_this_profile = len(unique_clients_in_this_profile)

        for i, client_id_in_profile in enumerate(unique_clients_in_this_profile):
            if num_clients_in_this_profile == 1:
                client_color_map[client_id_in_profile] = base_color_name
            else:
                shift_factor = (i / (num_clients_in_this_profile - 1)) - 0.5 if num_clients_in_this_profile > 1 else 0
                
                h_new, s_new, v_new = base_hsv[0], base_hsv[1], base_hsv[2]

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
    
    df_client_eval['c_eval_partition_id'] = df_client_eval['c_eval_partition_id'].astype(str)
    for client_id, group_data in df_client_eval.groupby("c_eval_partition_id"):
        profile_name = group_data["c_eval_profile_name"].iloc[0]
        color = client_color_map.get(client_id, PROFILE_COLORS["unknown_profile"])
        
        group_data_sorted = group_data.sort_values("server_round")
        
        if "c_eval_f1" in group_data_sorted.columns and not group_data_sorted["c_eval_f1"].dropna().empty:
            plt.plot(group_data_sorted["server_round"], group_data_sorted["c_eval_f1"].astype(float), marker='o', linestyle='-', 
                    label=f"Client {client_id} ({profile_name})", color=color, alpha=0.7)
        else:
            print(f"Warning: Client {client_id} F1-score data (c_eval_f1) not found or is all NaNs. Skipping its line.")

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

    plot_filename = os.path.join(output_dir, "per_client_f1_performance.png")
    plt.savefig(plot_filename)
    print(f"Saved per-client F1 performance plot to {plot_filename}")
    plt.close()

def plot_client_selection_heatmap(df, output_dir):
    """Plots a heatmap of which clients were selected in each round."""
    df_selection = df[(df['event_type'] == 'selection_info') & (df['s_was_selected'] == 1)].copy()
    if df_selection.empty:
        print("No 'selection_info' data found to plot selection heatmap.")
        return

    df_client_info = df[df['event_type'] == 'client_eval_metrics'].drop_duplicates(
        subset='client_cid', keep='last'
    )

    if not df_client_info.empty:
        cid_to_partition_map = df_client_info.set_index('client_cid')['c_eval_partition_id'].to_dict()
        cid_to_profile_map = df_client_info.set_index('client_cid')['c_eval_profile_name'].to_dict()
        df_selection['c_eval_partition_id'] = df_selection['client_cid'].map(cid_to_partition_map)
        df_selection['c_eval_profile_name'] = df_selection['client_cid'].map(cid_to_profile_map)

    def create_display_label(row):
        partition_id = row.get('c_eval_partition_id')
        profile = row.get('c_eval_profile_name')

        if pd.notna(partition_id) and pd.notna(profile):
            return f"Client {int(partition_id)} ({profile})"
        elif pd.notna(partition_id):
            return f"Client {int(partition_id)}"
        else:
            return str(row['client_cid'])

    df_selection['display_label'] = df_selection.apply(create_display_label, axis=1)

    selection_pivot = df_selection.pivot_table(index='display_label', columns='server_round', values='s_was_selected', fill_value=0)

    if selection_pivot.empty:
        print("Could not create pivot table for selection heatmap.")
        return

    num_clients = len(selection_pivot)
    fig_height = max(10, num_clients / 1.2)
    
    fig_width = fig_height * (16 / 9)
    plt.figure(figsize=(fig_width, fig_height))
    
    ax = sns.heatmap(selection_pivot, cmap="YlGnBu", cbar=False, xticklabels=10)
    plt.title('Client Selection Heatmap per Round', fontsize=26)
    plt.xlabel('Server Round', fontsize=22)
    plt.ylabel('Client (Number and Type)', fontsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=18)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "client_selection_heatmap.png")
    plt.savefig(output_path)
    print(f"Saved client selection heatmap to {output_path}")
    plt.close()

def plot_total_resource_usage(df, output_dir):
    """Plots the total CPU utilization of selected clients per round."""
    df_fit = df[df['event_type'] == 'client_fit_metrics'].dropna(subset=['c_fit_cpu_percent']).copy()
    if df_fit.empty:
        print("No 'client_fit_metrics' data found to plot CPU resource usage.")
        return
        
    df_resources = df_fit.groupby('server_round').agg(
        total_cpu_percent=('c_fit_cpu_percent', 'sum')
    ).reset_index()

    plt.figure(figsize=(14, 8))

    plt.plot(df_resources['server_round'], df_resources['total_cpu_percent'], color='tab:red', marker='x', linestyle='--', label='Total CPU (%)')

    plt.title('Total CPU Utilization of Selected Clients per Round', fontsize=32)
    plt.xlabel('Server Round', fontsize=24)
    plt.ylabel('Total CPU Usage (%)', fontsize=24)
    plt.ylim(0, 200)
    plt.legend(loc='upper left', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "total_resource_usage.png")
    plt.savefig(output_path)
    print(f"Saved total resource usage plot to {output_path}")
    plt.close()

def plot_f1_deltas(df, output_dir):
    """Plots a more readable, binned boxplot of the delta between global and local F1 scores."""
    df_global = df[df['event_type'] == 'global_eval_metrics'].dropna(subset=['gm_eval_f1'])[['server_round', 'gm_eval_f1']]
    df_local = df[df['event_type'] == 'client_fit_metrics'].dropna(subset=['c_fit_post_local_f1'])[['server_round', 'c_fit_post_local_f1']]
    
    if df_global.empty or df_local.empty:
        print("Not enough 'global_eval_metrics' or 'client_fit_metrics' data to plot F1 deltas.")
        return

    df_merged = pd.merge(df_local, df_global, on='server_round')
    if df_merged.empty:
        print("Could not merge local and global F1 data for delta plot.")
        return

    df_merged['f1_delta'] = df_merged['gm_eval_f1'] - df_merged['c_fit_post_local_f1']

    max_round = df_merged['server_round'].max()
    bin_size = max(10, int(np.ceil(max_round / 20.0))) 
    
    df_merged['round_bin'] = (df_merged['server_round'] // bin_size)
    
    df_merged = df_merged.sort_values('round_bin')
    
    unique_bins = df_merged['round_bin'].unique()
    bin_labels = [f"{int(b*bin_size)}-{int((b+1)*bin_size-1)}" for b in unique_bins]

    plt.figure(figsize=(20, 10))

    ax = sns.boxplot(x='round_bin', y='f1_delta', data=df_merged, palette="coolwarm", showfliers=False)

    median_trend = df_merged.groupby('round_bin')['f1_delta'].median()
    plt.plot(range(len(median_trend)), median_trend.values, 'o-', color='black', label='Median Delta Trend', linewidth=2, markersize=5, zorder=10)

    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.title('F1 Score Delta (Global vs. Local Models), Binned by Round')
    plt.xlabel('Server Round Bins')
    plt.ylabel('Delta (Global F1 - Local F1)')
    
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "f1_delta_boxplot.png")
    plt.savefig(output_path)
    print(f"Saved improved F1 delta boxplot to {output_path}")
    plt.close()

def plot_specialist_vs_generalist_performance(df, output_dir):
    """
    Plots the evolution of the average performance of specialized local models 
    vs. the generalized global model over binned server rounds.
    """
    df_local_fit = df[df['event_type'] == 'client_fit_metrics'].dropna(subset=['server_round', 'c_fit_post_local_f1'])
    df_global_eval = df[df['event_type'] == 'client_eval_metrics'].dropna(subset=['server_round', 'c_eval_f1'])

    if df_local_fit.empty or df_global_eval.empty:
        print("Not enough data to compare specialist vs. generalist performance.")
        return

    max_round = max(df_local_fit['server_round'].max(), df_global_eval['server_round'].max())
    bin_size = max(10, int(np.ceil(max_round / 20.0)))
    
    df_local_fit['round_bin'] = (df_local_fit['server_round'] // bin_size)
    df_global_eval['round_bin'] = (df_global_eval['server_round'] // bin_size)
    
    avg_specialist_f1_per_bin = df_local_fit.groupby('round_bin')['c_fit_post_local_f1'].mean()
    avg_generalist_f1_per_bin = df_global_eval.groupby('round_bin')['c_eval_f1'].mean()
    
    unique_bins = sorted(list(set(avg_specialist_f1_per_bin.index) | set(avg_generalist_f1_per_bin.index)))
    bin_labels = [f"{int(b*bin_size)}-{int((b+1)*bin_size-1)}" for b in unique_bins]

    plt.figure(figsize=(18, 9))
    
    plt.plot(avg_specialist_f1_per_bin.index, avg_specialist_f1_per_bin.values, marker='o', linestyle='-', label='Average Local "Specialist" F1')
    plt.plot(avg_generalist_f1_per_bin.index, avg_generalist_f1_per_bin.values, marker='s', linestyle='--', label='Average Global "Generalist" F1')

    plt.title('Performance Evolution: Specialist vs. Generalist Models')
    plt.xlabel('Server Round Bins')
    plt.ylabel('Average F1 Score')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.xticks(ticks=unique_bins, labels=bin_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "specialist_vs_generalist_f1_evolution.png")
    plt.savefig(output_path)
    print(f"Saved specialist vs. generalist performance evolution plot to {output_path}")
    plt.close()

def plot_global_gan_losses(df, output_dir):
    """Plots the aggregated Generator and Discriminator losses over server rounds."""
    df_global_fit = df[df['event_type'] == 'global_fit_metrics'].dropna(
        subset=['server_round', 'gm_fit_g_loss', 'gm_fit_d_loss']
    ).copy()
    
    if df_global_fit.empty:
        print("No 'global_fit_metrics' data found to plot global GAN losses.")
        return

    df_global_fit['server_round'] = pd.to_numeric(df_global_fit['server_round'])

    plt.figure()
    plt.plot(df_global_fit["server_round"], df_global_fit["gm_fit_g_loss"].astype(float), marker='^', linestyle='--', label="Aggregated Generator Loss", color='blue')
    plt.plot(df_global_fit["server_round"], df_global_fit["gm_fit_d_loss"].astype(float), marker='v', linestyle=':', label="Aggregated Discriminator Loss", color='red')

    plt.title("Aggregated GAN Losses Over Server Rounds")
    plt.xlabel("Server Round")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "global_gan_losses.png")
    plt.savefig(output_path)
    print(f"Saved global GAN losses plot to {output_path}")
    plt.close()

def plot_selection_dynamics(df, output_dir):
    """Plots softmax temperature and the standard deviation of selection probabilities."""
    df_selection_info = df[df['event_type'] == 'selection_info'].dropna(
        subset=['server_round', 's_client_selection_prob', 's_softmax_temperature']
    ).copy()
    
    if df_selection_info.empty:
        print("No 'selection_info' data found to plot selection dynamics.")
        return

    df_selection_info['server_round'] = pd.to_numeric(df_selection_info['server_round'])

    dynamics_data = df_selection_info.groupby("server_round").agg(
        prob_std=('s_client_selection_prob', 'std'),
        temperature=('s_softmax_temperature', 'first')
    ).reset_index()

    fig, ax1 = plt.subplots()

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

    output_path = os.path.join(output_dir, "selection_dynamics.png")
    plt.savefig(output_path)
    print(f"Saved selection dynamics plot to {output_path}")
    plt.close()

def plot_client_fit_loss_distribution(df, output_dir, bin_size=10):
    """Plots the distribution of client-level G and D losses on the same canvas, binned by server rounds."""
    df_client_fit = df[df['event_type'] == 'client_fit_metrics'].dropna(
        subset=['server_round', 'c_fit_g_loss', 'c_fit_d_loss']
    ).copy()

    if df_client_fit.empty:
        print("No 'client_fit_metrics' data found to plot client fit loss distribution.")
        return
        
    df_client_fit['server_round'] = pd.to_numeric(df_client_fit['server_round'])

    df_client_fit['round_bin'] = (df_client_fit['server_round'] // bin_size) * bin_size
    
    df_melted = pd.melt(df_client_fit, 
                        id_vars=['round_bin'], 
                        value_vars=['c_fit_g_loss', 'c_fit_d_loss'],
                        var_name='loss_type', 
                        value_name='loss_value')

    df_melted['loss_type'] = df_melted['loss_type'].map({
        'c_fit_g_loss': 'Generator Loss',
        'c_fit_d_loss': 'Discriminator Loss'
    })

    plt.figure(figsize=(18, 9))
    
    ax = sns.boxplot(x='round_bin', y='loss_value', hue='loss_type', data=df_melted, whis=[5, 95], showfliers=False)
    
    ax.set_title("Distribution of Client-Level Generator (G) and Discriminator (D) Losses (Binned)")
    ax.set_xlabel(f"Server Round (Binned by {bin_size})")
    ax.set_ylabel("Loss Value")
    ax.grid(True)
    
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Loss Type')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "client_fit_loss_distribution.png")
    plt.savefig(output_path)
    print(f"Saved client fit loss distribution plot to {output_path}")
    plt.close()

def main():
    """Main function to parse arguments and call plotting functions."""
    parser = argparse.ArgumentParser(description="Generate plots from RL training data CSV.")
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to the instance-specific log directory containing 'rl_training_data.csv'."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/rl_analysis",
        help="Base directory to save the generated plots. A timestamp will be appended."
    )
    args = parser.parse_args()

    csv_path = os.path.join(args.log_dir, 'rl_training_data.csv')

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = f"{args.output_dir}_{timestamp_str}"

    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Output directory: {final_output_dir}")

    if not os.path.exists(csv_path):
        print(f"Error: RL training data file not found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {csv_path} with {len(df)} rows.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    plot_global_performance(df, final_output_dir)
    plot_rewards_and_components(df, final_output_dir)
    plot_q_value_evolution(df, final_output_dir)
    plot_client_selection_distribution(df, final_output_dir)
    plot_softmax_temperature_decay(df, final_output_dir)
    plot_clients_selected_per_round(df, final_output_dir)
    plot_per_client_performance(df, final_output_dir)
    plot_client_selection_heatmap(df, final_output_dir)
    plot_total_resource_usage(df, final_output_dir)
    plot_f1_deltas(df, final_output_dir)
    plot_specialist_vs_generalist_performance(df, final_output_dir)
    plot_global_gan_losses(df, final_output_dir)
    plot_selection_dynamics(df, final_output_dir)
    plot_client_fit_loss_distribution(df, final_output_dir)


if __name__ == "__main__":
    main() 