from collections import OrderedDict
import warnings
import logging
import os
import pickle

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

task_logger = logging.getLogger("ClientAppLogger")


MODEL_FEATURE_COLUMNS = [
    'src_port', 'dst_port', 'duration', 'src_bytes', 'dst_bytes',
    'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
    'dns_qclass', 'dns_qtype', 'dns_rcode', 'http_trans_depth',
    'http_request_body_len', 'http_response_body_len', 'http_status_code'
]
LABEL_COLUMN = "label"
INPUT_DIM = len(MODEL_FEATURE_COLUMNS)

HOLDOUT_FRACTION = 0.15

DATA_PATH = "src/datasets/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv"
DATA_SAMPLE_FRACTION = 1
BATCH_SIZE = 32

NORMAL_LABELED_FRACTION = 0.20
FRACTION_OF_LABELED_POOL_FOR_TRAIN = 0.70
FRACTION_OF_LABELED_POOL_FOR_VAL = 0.15
FRACTION_OF_LABELED_POOL_FOR_TEST = 0.15

RANDOM_SEED = 42
MIN_TRAIN_SAMPLES_PER_PARTITION = 100
MIN_EVAL_SAMPLES_PER_PARTITION = 100
DIRICHLET_ALPHA = 0.3
THRESHOLD_PERCENTILE = 40


class Generator(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 20, hidden1_size: int = 32, hidden2_size: int = 24):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, input_dim)

    def forward(self, z):
        h1 = F.leaky_relu(self.fc1(z), 0.2)
        h2 = F.leaky_relu(self.fc2(h1), 0.2)
        output = torch.sigmoid(self.fc3(h2))
        return output

class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden1_size: int = 24, hidden2_size: int = 16):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 1)

    def forward(self, x):
        h1 = F.leaky_relu(self.fc1(x), 0.2)
        h2 = F.leaky_relu(self.fc2(h1), 0.2)
        output = torch.sigmoid(self.fc3(h2))
        return output


class TonIoTDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


scaler = MinMaxScaler()

def load_data(partition_id: int, num_partitions: int, instance_log_dir: str):
    global scaler

    task_logger.info(f"[LoadData {partition_id}] Starting load_data function.")
    try:
        columns_to_load = MODEL_FEATURE_COLUMNS + [LABEL_COLUMN, "type"]
        task_logger.info(f"[LoadData {partition_id}] Attempting to read CSV from {DATA_PATH} with columns: {columns_to_load}")
        df = pd.read_csv(DATA_PATH, low_memory=False, usecols=lambda c: c in columns_to_load)
        task_logger.info(f"[LoadData {partition_id}] CSV read successfully. DataFrame shape: {df.shape}")
    except FileNotFoundError:
        task_logger.error(f"Error: Dataset not found at {DATA_PATH}")
        task_logger.error("Please ensure the ToN-IoT dataset CSV is placed correctly.")
        return None, None, None

    if DATA_SAMPLE_FRACTION < 1.0:
        task_logger.info(f"Sampling {DATA_SAMPLE_FRACTION*100:.1f}% of the data.")
        df = df.sample(frac=DATA_SAMPLE_FRACTION, random_state=RANDOM_SEED)
        task_logger.info(f"[LoadData {partition_id}] Data sampled. New DataFrame shape: {df.shape}")

    all_columns_from_csv_to_use = MODEL_FEATURE_COLUMNS + [LABEL_COLUMN, "type"]
    task_logger.info(f"[LoadData {partition_id}] Starting '-' replacement and numeric conversion.")
    for col in all_columns_from_csv_to_use:
        if col in df.columns:
            df[col] = df[col].replace('-', 0)
        else:
            task_logger.warning(f"Column '{col}' not found in DataFrame during '-' replacement.")

    for col in MODEL_FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            task_logger.error(f"Feature column '{col}' not found in DataFrame for numeric conversion.")
    
    if LABEL_COLUMN in df.columns:
        df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors='coerce')
    else:
        task_logger.error(f"Label column '{LABEL_COLUMN}' not found in DataFrame.")
        return None, None, None 
    task_logger.info(f"[LoadData {partition_id}] Numeric conversion completed.")

    df[MODEL_FEATURE_COLUMNS] = df[MODEL_FEATURE_COLUMNS].fillna(0)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna(0)
    task_logger.info(f"[LoadData {partition_id}] NA values filled.")

    if HOLDOUT_FRACTION > 0:
        task_logger.info(f"Splitting off {HOLDOUT_FRACTION*100:.1f}% of data for global hold-out test set.")
        
        original_indices = df.index
        federated_indices, holdout_indices = train_test_split(
            original_indices,
            test_size=HOLDOUT_FRACTION,
            random_state=RANDOM_SEED,
            stratify=df[LABEL_COLUMN]
        )

        df_holdout = df.loc[holdout_indices]
        
        if partition_id == 0:
            holdout_path = os.path.join(instance_log_dir, 'holdout_test_set.csv')
            if not os.path.exists(holdout_path):
                df_holdout.to_csv(holdout_path, index=False)
                task_logger.info(f"Global hold-out test set with {len(df_holdout)} samples saved to {holdout_path}")
        
        df = df.loc[federated_indices].reset_index(drop=True)
        task_logger.info(f"Proceeding with federated dataset of size {len(df)}.")

    y = df[LABEL_COLUMN].astype(int)
    X = df[MODEL_FEATURE_COLUMNS].astype(float)
    task_logger.info(f"[LoadData {partition_id}] X and y created. X shape: {X.shape}, y shape: {y.shape}")

    if not hasattr(scaler, 'n_features_in_') or scaler.n_features_in_ != X.shape[1]:
        task_logger.info(f"Fitting MinMaxScaler on X with shape {X.shape}")
    scaler.fit(X)
    if partition_id == 0:
        scaler_path = os.path.join(instance_log_dir, 'min_max_scaler.pkl')
        if not os.path.exists(scaler_path):
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            task_logger.info(f"Scaler object saved by client 0 to {scaler_path}")
    task_logger.info(f"[LoadData {partition_id}] Scaler fitting process completed.")
    
    if X.shape[1] == scaler.n_features_in_:
        X_scaled = scaler.transform(X)
        task_logger.info(f"[LoadData {partition_id}] X_scaled successfully. Shape: {X_scaled.shape}")
    else:
        task_logger.error(f"Scaler was fit on {scaler.n_features_in_} features, but current X has {X.shape[1]} features. Aborting.")
        return None, None, None

    task_logger.info(f"[LoadData {partition_id}] Starting semi-supervised data splitting logic.")
    normal_indices_pos = np.where(y.values == 0)[0]
    initial_anomaly_indices_pos = np.where(y.values == 1)[0]

    task_logger.info(f"Initial number of anomaly samples (positions in sampled data): {len(initial_anomaly_indices_pos)}")
    task_logger.info(f"[LoadData {partition_id}] Initial normal indices count: {len(normal_indices_pos)}, anomaly indices count: {len(initial_anomaly_indices_pos)}")

    final_selected_anomaly_indices_pos = initial_anomaly_indices_pos
    task_logger.info(f"[LoadData {partition_id}] Starting anomaly reduction logic.")
    if len(initial_anomaly_indices_pos) > 0 and "type" in df.columns:
        df_anomalies_types = df.iloc[initial_anomaly_indices_pos]['type']
        
        unique_attack_types = df_anomalies_types.unique()
        task_logger.info(f"Found attack types in initial sampled anomalies: {unique_attack_types}")
        
        reduced_anomaly_positions_list = []
        for attack_type in unique_attack_types:
            relative_positions_of_type = np.where(df_anomalies_types.values == attack_type)[0]
            
            absolute_positions_of_type_in_sampled_df = initial_anomaly_indices_pos[relative_positions_of_type]
            
            num_to_select = max(1, int(len(absolute_positions_of_type_in_sampled_df) / 3))
            
            np.random.seed(RANDOM_SEED)
            selected_absolute_positions = np.random.choice(
                absolute_positions_of_type_in_sampled_df, 
                size=num_to_select, 
                replace=False
            )
            reduced_anomaly_positions_list.extend(selected_absolute_positions.tolist())
            task_logger.info(f"  Attack type '{attack_type}': Had {len(absolute_positions_of_type_in_sampled_df)} instances (in sampled), selected {len(selected_absolute_positions)} (target: {num_to_select}).")
        
        final_selected_anomaly_indices_pos = np.array(reduced_anomaly_positions_list)
        np.random.shuffle(final_selected_anomaly_indices_pos)
        task_logger.info(f"Total number of anomaly samples (positions) after reduction: {len(final_selected_anomaly_indices_pos)}")
        if len(final_selected_anomaly_indices_pos) > 0:
            final_anomaly_types_dist = df.iloc[final_selected_anomaly_indices_pos]['type'].value_counts()
            task_logger.info(f"""  Distribution of remaining attack types: 
{final_anomaly_types_dist}""")
    elif len(initial_anomaly_indices_pos) == 0:
        task_logger.info("No initial anomalies found in sampled data to reduce or use.")
        final_selected_anomaly_indices_pos = np.array([])
    else:
        task_logger.info(f"Proceeding with {len(final_selected_anomaly_indices_pos)} initial anomaly positions (no reduction performed as 'type' column missing or other).")
    task_logger.info(f"[LoadData {partition_id}] Anomaly reduction logic completed.")


    np.random.seed(RANDOM_SEED)
    np.random.shuffle(normal_indices_pos)
    if not (len(initial_anomaly_indices_pos) > 0 and "type" in df.columns and len(df_anomalies_types.unique()) > 0) :
        np.random.shuffle(final_selected_anomaly_indices_pos)
    task_logger.info(f"[LoadData {partition_id}] Normal and anomaly indices shuffled.")


    task_logger.info(f"[LoadData {partition_id}] Creating unlabeled training base.")
    n_normal = len(normal_indices_pos)
    n_normal_for_labeled_pool = int(n_normal * NORMAL_LABELED_FRACTION)
    n_normal_for_unlabeled_train = n_normal - n_normal_for_labeled_pool

    unlabeled_train_normal_indices = normal_indices_pos[n_normal_for_labeled_pool:]
    X_train_unlabeled_normal = X_scaled[unlabeled_train_normal_indices]
    y_train_unlabeled_normal = y.iloc[unlabeled_train_normal_indices].values

    task_logger.info(f"Unlabeled normal data for AE training base: {len(X_train_unlabeled_normal)} samples.")

    task_logger.info(f"[LoadData {partition_id}] Creating labeled pool.")
    labeled_pool_normal_indices = normal_indices_pos[:n_normal_for_labeled_pool]
    
    X_labeled_pool_list = []
    y_labeled_pool_list = []

    if len(labeled_pool_normal_indices) > 0:
        X_labeled_pool_list.append(X_scaled[labeled_pool_normal_indices])
        y_labeled_pool_list.append(y.iloc[labeled_pool_normal_indices].values)
    if len(final_selected_anomaly_indices_pos) > 0:
        X_labeled_pool_list.append(X_scaled[final_selected_anomaly_indices_pos])
        y_labeled_pool_list.append(y.iloc[final_selected_anomaly_indices_pos].values)

    if not X_labeled_pool_list:
        task_logger.warning("Labeled pool is empty. This might happen if there are no anomalies and NORMAL_LABELED_FRACTION is 0.")
        X_labeled_pool = np.array([])
        y_labeled_pool = np.array([])
    else:
        X_labeled_pool = np.concatenate(X_labeled_pool_list)
        y_labeled_pool = np.concatenate(y_labeled_pool_list)
        perm_labeled_pool = np.random.permutation(len(X_labeled_pool))
        X_labeled_pool = X_labeled_pool[perm_labeled_pool]
        y_labeled_pool = y_labeled_pool[perm_labeled_pool]

    task_logger.info(f"Labeled Pool (for train augmentation, val, test): {len(X_labeled_pool)} samples.")
    if len(X_labeled_pool) > 0:
        unique_lp_labels, counts_lp_labels = np.unique(y_labeled_pool, return_counts=True)
        task_logger.debug(f"  Labeled Pool Label Distribution: {dict(zip(unique_lp_labels, counts_lp_labels))}")
    task_logger.info(f"[LoadData {partition_id}] Labeled pool creation completed.")


    task_logger.info(f"[LoadData {partition_id}] Splitting labeled pool for train_aug, val, test.")
    X_labeled_train_aug, y_labeled_train_aug = np.array([]), np.array([])
    X_val, y_val = np.array([]), np.array([])
    X_test_final, y_test_final = np.array([]), np.array([])

    if len(X_labeled_pool) > 0:
        if FRACTION_OF_LABELED_POOL_FOR_TRAIN > 0 and FRACTION_OF_LABELED_POOL_FOR_TRAIN < 1.0:
            X_labeled_train_aug, X_temp_eval, y_labeled_train_aug, y_temp_eval = train_test_split(
                X_labeled_pool, y_labeled_pool,
                test_size=(1.0 - FRACTION_OF_LABELED_POOL_FOR_TRAIN),
                random_state=RANDOM_SEED,
                stratify=y_labeled_pool if len(np.unique(y_labeled_pool)) > 1 and np.min(np.bincount(y_labeled_pool)) > 1 else None
            )
        elif FRACTION_OF_LABELED_POOL_FOR_TRAIN == 1.0:
            X_labeled_train_aug, y_labeled_train_aug = X_labeled_pool, y_labeled_pool
            X_temp_eval, y_temp_eval = np.array([]), np.array([])
        else:
            X_temp_eval, y_temp_eval = X_labeled_pool, y_labeled_pool
        
        task_logger.info(f"Labeled data for AE training augmentation: {len(X_labeled_train_aug)} samples.")

        if len(X_temp_eval) > 0:
            remaining_pool_fraction_for_val = FRACTION_OF_LABELED_POOL_FOR_VAL
            remaining_pool_fraction_for_test = FRACTION_OF_LABELED_POOL_FOR_TEST
            sum_val_test_fractions_in_remaining_pool = remaining_pool_fraction_for_val + remaining_pool_fraction_for_test

            if sum_val_test_fractions_in_remaining_pool > 0:
                test_split_ratio_for_final_test = remaining_pool_fraction_for_test / sum_val_test_fractions_in_remaining_pool
                
                if test_split_ratio_for_final_test > 0 and test_split_ratio_for_final_test < 1.0:
                    X_val, X_test_final, y_val, y_test_final = train_test_split(
                                X_temp_eval, y_temp_eval,
                                test_size=test_split_ratio_for_final_test,
                                random_state=RANDOM_SEED,
                                stratify=y_temp_eval if len(np.unique(y_temp_eval)) > 1 and np.min(np.bincount(y_temp_eval)) > 1 else None
                            )
                elif test_split_ratio_for_final_test == 1.0:
                    X_test_final, y_test_final = X_temp_eval, y_temp_eval
                else:
                    X_val, y_val = X_temp_eval, y_temp_eval
            else:
                task_logger.warning("FRACTION_OF_LABELED_POOL_FOR_VAL and ..._TEST are both zero. No val/test data from labeled pool remainder.")


    task_logger.info(f"[LoadData {partition_id}] Combining data for final AE training set.")
    X_train_combined_list = []
    y_train_combined_list = []

    if len(X_train_unlabeled_normal) > 0:
        X_train_combined_list.append(X_train_unlabeled_normal)
        y_train_combined_list.append(y_train_unlabeled_normal)
    if len(X_labeled_train_aug) > 0:
        X_train_combined_list.append(X_labeled_train_aug)
        y_train_combined_list.append(y_labeled_train_aug)

    if not X_train_combined_list:
        task_logger.error("AE training data (X_train_combined) is empty. Check data splitting logic and input data.")
        X_train_combined = np.array([])
        y_train_combined = np.array([])
    else:
        X_train_combined = np.concatenate(X_train_combined_list)
        y_train_combined = np.concatenate(y_train_combined_list)
        perm_train_combined = np.random.permutation(len(X_train_combined))
        X_train_combined = X_train_combined[perm_train_combined]
        y_train_combined = y_train_combined[perm_train_combined]

    task_logger.info(f"[LoadData {partition_id}] Final AE training data combination completed.")


    task_logger.info(f"[LoadData {partition_id}] Logging final dataset sizes and distributions pre-partitioning.")
    n_train = len(X_train_combined)
    n_val = len(X_val)
    n_test_final = len(X_test_final)

    task_logger.info(f"Data Splitting Summary (Semi-Supervised AE):")
    task_logger.info(f"  AE Training (X_train_combined): {n_train} samples.")
    if n_train > 0:
        unique_train_labels, counts_train_labels = np.unique(y_train_combined, return_counts=True)
        task_logger.debug(f"    AE Training Label Distribution (for info): {dict(zip(unique_train_labels, counts_train_labels))}")
    
    task_logger.info(f"  Validation (X_val): {n_val} samples.")
    if n_val > 0:
        unique_val_labels, counts_val_labels = np.unique(y_val, return_counts=True)
        task_logger.debug(f"    Validation Label Distribution: {dict(zip(unique_val_labels, counts_val_labels))}")

    task_logger.info(f"  Test (X_test_final): {n_test_final} samples.")
    if n_test_final > 0:
        unique_test_labels, counts_test_labels = np.unique(y_test_final, return_counts=True)
        task_logger.debug(f"    Test Label Distribution: {dict(zip(unique_test_labels, counts_test_labels))}")

    task_logger.info(f"[Partitioning Debug {partition_id}] START of partitioning block.")
    task_logger.info(f"[LoadData {partition_id}] Starting Dirichlet-based partitioning with minimum sample guarantee.")

    def distribute_with_minimum(
        total_samples: int,
        num_partitions: int,
        alpha: float,
        min_per_partition: int,
        task_logger: logging.Logger,
        dataset_name: str
    ) -> np.ndarray:
        
        if total_samples < min_per_partition * num_partitions:
            task_logger.warning(
                f"[{dataset_name} Partitioning] Total samples ({total_samples}) is less than the required minimum "
                f"for all partitions ({min_per_partition * num_partitions}). "
                f"Distributing samples one by one. Some partitions may have fewer than the minimum."
            )
            counts = np.zeros(num_partitions, dtype=int)
            if total_samples > 0:
                for i in range(total_samples):
                    counts[i % num_partitions] += 1
            return counts

        base_counts = np.full(num_partitions, min_per_partition, dtype=int)
        
        remaining_samples = total_samples - (min_per_partition * num_partitions)
        
        if remaining_samples > 0:
            proportions = np.random.dirichlet(np.full(num_partitions, alpha))
            additional_counts = (proportions * remaining_samples).astype(int)
            
            remainder_after_proportions = remaining_samples - additional_counts.sum()
            if remainder_after_proportions > 0:
                for i in range(remainder_after_proportions):
                    additional_counts[np.argmin(additional_counts)] += 1
            
            final_counts = base_counts + additional_counts
        else:
            final_counts = base_counts
            
        if final_counts.sum() != total_samples:
            diff = total_samples - final_counts.sum()
            final_counts[0] += diff
            
        return final_counts


    np.random.seed(RANDOM_SEED)
    train_indices_full = np.arange(n_train)
    val_indices_full = np.arange(n_val) if n_val > 0 else np.array([])
    test_indices_full = np.arange(n_test_final) if n_test_final > 0 else np.array([])
    task_logger.info(f"[Partitioning Debug {partition_id}] Step 1: Full indices created.")
    task_logger.info(f"[LoadData {partition_id}] Partitioning: Full train/val/test indices created.")

    if n_train > 0: np.random.shuffle(train_indices_full)
    if n_val > 0: np.random.shuffle(val_indices_full)
    if n_test_final > 0: np.random.shuffle(test_indices_full)
    task_logger.info(f"[Partitioning Debug {partition_id}] Step 2: Full indices shuffled.")
    task_logger.info(f"[LoadData {partition_id}] Partitioning: Full indices shuffled.")

    train_counts = distribute_with_minimum(n_train, num_partitions, DIRICHLET_ALPHA, MIN_TRAIN_SAMPLES_PER_PARTITION, task_logger, "Train") if n_train > 0 else np.zeros(num_partitions, dtype=int)
    val_counts = distribute_with_minimum(n_val, num_partitions, DIRICHLET_ALPHA, MIN_EVAL_SAMPLES_PER_PARTITION, task_logger, "Val") if n_val > 0 else np.zeros(num_partitions, dtype=int)
    test_counts = distribute_with_minimum(n_test_final, num_partitions, DIRICHLET_ALPHA, MIN_EVAL_SAMPLES_PER_PARTITION, task_logger, "Test") if n_test_final > 0 else np.zeros(num_partitions, dtype=int)

    task_logger.info(f"[Partitioning Debug {partition_id}] Step 5d: Counts calculated. Final sums: Train={np.sum(train_counts) if n_train > 0 else 'N/A'}, Val={np.sum(val_counts) if n_val > 0 else 'N/A'}, Test={np.sum(test_counts) if n_test_final > 0 else 'N/A'}")
    task_logger.info(f"[LoadData {partition_id}] Partitioning: Remainder distribution completed.")
    task_logger.info(f"[Partitioning Debug {partition_id}] Step 5e: Final train_counts example: {train_counts[:5].tolist() if n_train > 0 and num_partitions > 0 else 'N/A'}")
    task_logger.info(f"[Partitioning Debug {partition_id}] Step 5f: Final val_counts example: {val_counts[:5].tolist() if n_val > 0 and num_partitions > 0 else 'N/A'}")
    task_logger.info(f"[Partitioning Debug {partition_id}] Step 5g: Final test_counts example: {test_counts[:5].tolist() if n_test_final > 0 and num_partitions > 0 else 'N/A'}")


    train_start = train_counts[:partition_id].sum()
    val_start = val_counts[:partition_id].sum()
    test_start = test_counts[:partition_id].sum()
    task_logger.info(f"[Partitioning Debug {partition_id}] Step 6: Start indices calculated. train_start={train_start}")
    task_logger.info(f"[LoadData {partition_id}] Partitioning: Start indices calculated. Train_start={train_start}, Val_start={val_start}, Test_start={test_start}")

    train_end = train_start + train_counts[partition_id]
    val_end = val_start + val_counts[partition_id]
    test_end = test_start + test_counts[partition_id]
    task_logger.info(f"[Partitioning Debug {partition_id}] Step 7: End indices calculated. train_end={train_end}")
    task_logger.info(f"[LoadData {partition_id}] Partitioning: End indices calculated. Train_end={train_end}, Val_end={val_end}, Test_end={test_end}")

    partition_train_indices = train_indices_full[train_start:train_end] if n_train > 0 else np.array([])
    partition_val_indices = val_indices_full[val_start:val_end] if n_val > 0 else np.array([])
    partition_test_indices = test_indices_full[test_start:test_end] if n_test_final > 0 else np.array([])
    task_logger.info(f"[Partitioning Debug {partition_id}] Step 8: Partition-specific indices extracted. Num train indices: {len(partition_train_indices)}")
    task_logger.info(f"[LoadData {partition_id}] Partitioning: Specific indices extracted. Train idx: {len(partition_train_indices)}, Val idx: {len(partition_val_indices)}, Test idx: {len(partition_test_indices)}")

    X_train_part_full = X_train_combined[partition_train_indices] if n_train > 0 else np.array([])
    y_train_part_full = y_train_combined[partition_train_indices] if n_train > 0 else np.array([])
    task_logger.info(f"[Partitioning Debug {partition_id}] Step 9a: Train data sliced for partition. X_train_part_full shape: {X_train_part_full.shape}")
    task_logger.info(f"[LoadData {partition_id}] Partitioning: Train data sliced. Shape: {X_train_part_full.shape}")

    X_val_part = X_val[partition_val_indices] if n_val > 0 else np.array([])
    y_val_part = y_val[partition_val_indices] if n_val > 0 else np.array([])
    task_logger.info(f"[Partitioning Debug {partition_id}] Step 9b: Validation data sliced for partition. X_val_part shape: {X_val_part.shape}")
    task_logger.info(f"[LoadData {partition_id}] Partitioning: Validation data sliced. Shape: {X_val_part.shape}")

    X_test_part = X_test_final[partition_test_indices] if n_test_final > 0 else np.array([])
    y_test_part = y_test_final[partition_test_indices] if n_test_final > 0 else np.array([])
    task_logger.info(f"[Partitioning Debug {partition_id}] Step 9c: Test data sliced for partition. X_test_part shape: {X_test_part.shape}")
    task_logger.info(f"[LoadData {partition_id}] Partitioning: Test data sliced. Shape: {X_test_part.shape}")

    task_logger.info(f"[Partitioning Debug {partition_id}] Step 10: Creating TonIoTDataset for train_dataset_full_potential.")
    task_logger.info(f"[LoadData {partition_id}] Creating TonIoTDataset objects.")
    train_dataset_full_potential = TonIoTDataset(X_train_part_full, y_train_part_full)
    task_logger.info(f"[Partitioning Debug {partition_id}] Step 11: Creating TonIoTDataset for val_dataset_static.")
    val_dataset_static = TonIoTDataset(X_val_part, y_val_part)
    task_logger.info(f"[Partitioning Debug {partition_id}] Step 12: Creating TonIoTDataset for test_dataset_static.")
    test_dataset_static = TonIoTDataset(X_test_part, y_test_part)
    task_logger.info(f"[Partitioning Debug {partition_id}] Step 13: All TonIoTDataset objects created.")
    task_logger.info(f"[LoadData {partition_id}] TonIoTDataset objects created.")
    
    task_logger.info(f"Partition {partition_id} (Static Datasets): Train Potential {len(train_dataset_full_potential) if train_dataset_full_potential else 0}, Val {len(val_dataset_static) if val_dataset_static else 0}, Test {len(test_dataset_static) if test_dataset_static else 0}")
    task_logger.info(f"[Partitioning Debug {partition_id}] END of partitioning block, returning datasets.")
    task_logger.info(f"[LoadData {partition_id}] load_data function completed.")
    return train_dataset_full_potential, val_dataset_static, test_dataset_static


def train(netG, netD, trainloader, epochs, device, learning_rate_g: float = 0.0002, learning_rate_d: float = 0.0002, latent_dim: int = 20, evaluate_on_train_data=True):
    netG.to(device)
    netD.to(device)
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate_g, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate_d, betas=(0.5, 0.999))

    netG.train()
    netD.train()

    total_avg_epoch_g_loss = 0.0
    total_avg_epoch_d_loss = 0.0

    for epoch in range(epochs):
        epoch_g_loss_sum = 0.0
        epoch_d_loss_sum = 0.0
        num_batches_processed = 0
        num_samples_processed_in_epoch = 0


        for real_features, true_labels in trainloader:
            real_features = real_features.to(device)
            true_labels = true_labels.to(device)
            batch_size_current = real_features.size(0)

            d_real_targets = torch.full((batch_size_current, 1), 0.9, device=device, dtype=torch.float32)
            anomaly_mask = (true_labels == 1).unsqueeze(1).float()
            
            effective_d_real_targets = (1.0 - anomaly_mask) * 0.9 + anomaly_mask * 0.1

            d_fake_targets = torch.full((batch_size_current, 1), 0.1, device=device, dtype=torch.float32)

            optimizerD.zero_grad()
            
            outputs_real = netD(real_features)
            d_loss_real_data = criterion(outputs_real, effective_d_real_targets)
            
            noise = torch.randn(batch_size_current, latent_dim, device=device)
            fake_features_from_g = netG(noise)
            outputs_fake_from_g = netD(fake_features_from_g.detach()) 
            d_loss_fake_data = criterion(outputs_fake_from_g, d_fake_targets)
            
            d_loss = d_loss_real_data + d_loss_fake_data
            d_loss.backward()
            optimizerD.step()

            optimizerG.zero_grad()
            outputs_fake_for_g = netD(fake_features_from_g)
            g_loss = criterion(outputs_fake_for_g, d_real_targets)
            
            g_loss.backward()

            epoch_d_loss_sum += d_loss.item() * batch_size_current
            epoch_g_loss_sum += g_loss.item() * batch_size_current
            num_samples_processed_in_epoch += batch_size_current
            num_batches_processed +=1


        if num_samples_processed_in_epoch > 0:
            avg_epoch_d_loss = epoch_d_loss_sum / num_samples_processed_in_epoch
            avg_epoch_g_loss = epoch_g_loss_sum / num_samples_processed_in_epoch
            task_logger.debug(f"Epoch [{epoch+1}/{epochs}], Avg D Loss: {avg_epoch_d_loss:.4f}, Avg G Loss: {avg_epoch_g_loss:.4f}")
            total_avg_epoch_d_loss += avg_epoch_d_loss
            total_avg_epoch_g_loss += avg_epoch_g_loss
        else:
            task_logger.debug(f"Epoch [{epoch+1}/{epochs}], No batches processed.")

    final_avg_g_loss = total_avg_epoch_g_loss / epochs if epochs > 0 else 0.0
    final_avg_d_loss = total_avg_epoch_d_loss / epochs if epochs > 0 else 0.0
    task_logger.info(f"Training finished. Avg G Loss over {epochs} epochs: {final_avg_g_loss:.4f}, Avg D Loss: {final_avg_d_loss:.4f}")

    return final_avg_g_loss, final_avg_d_loss


def test(netD, testloader, device, threshold_percentile=THRESHOLD_PERCENTILE):
    if testloader is None or not hasattr(testloader, 'dataset') or len(testloader.dataset) == 0:
        task_logger.warning("[Task] Testloader is None or empty. Skipping evaluation.")
        return 0.0, 0, {
            "avg_anomaly_score": 0.0, 
            "accuracy": 0.0, 
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "threshold": 0.0,
            "anomaly_scores": [], 
            "labels": [],
            "evaluate_metric_error": "No data in testloader"
        }

    netD.to(device)
    
    all_sample_anomaly_scores_list = []
    all_sample_labels_list = []
    num_samples = 0

    netD.eval()
    with torch.no_grad():
        for features, labels in testloader:
            if torch.all(labels == -1): 
                task_logger.debug("[Task-Debug] Skipping batch in test() due to all labels being -1.")
                continue 

            features = features.to(device)
            
            discriminator_outputs = netD(features)
            sample_anomaly_scores_tensor = 1.0 - discriminator_outputs.squeeze(dim=-1)
            
            all_sample_anomaly_scores_list.extend(sample_anomaly_scores_tensor.cpu().tolist())
            all_sample_labels_list.extend(labels.cpu().tolist())
            num_samples += features.size(0)

    avg_anomaly_score_val = np.mean(all_sample_anomaly_scores_list) if num_samples > 0 and all_sample_anomaly_scores_list else 0.0
    
    current_accuracy = 0.0
    current_f1_score = 0.0
    current_precision = 0.0
    current_recall = 0.0
    chosen_threshold = 0.0
    error_message = None

    valid_indices = [i for i, label in enumerate(all_sample_labels_list) if label != -1]
    if not valid_indices and len(all_sample_labels_list) > 0 :
        task_logger.info("[Task-Debug] No valid labels (0 or 1) found in testloader for metric calculation.")
        error_message = "No valid labels (0 or 1) for metric calculation"
    elif not all_sample_anomaly_scores_list:
        task_logger.warning("[Task-Debug] Empty anomaly scores list before metric calculation.")
        error_message = "Empty anomaly_scores list"

    if error_message is None and num_samples > 0 and valid_indices:
        filtered_anomaly_scores = np.array(all_sample_anomaly_scores_list)[valid_indices]
        filtered_labels = np.array(all_sample_labels_list)[valid_indices]

        if len(filtered_anomaly_scores) > 0 and len(filtered_labels) > 0 :
            try:
                anomaly_score_array = filtered_anomaly_scores
                labels_array = filtered_labels
                
                task_logger.debug(f"[Task-Debug] Test Evaluation - Batch Info (using {len(labels_array)} valid samples):")
                unique_labels, counts_labels = np.unique(labels_array, return_counts=True)
                task_logger.debug(f"  Unique true labels in batch: {dict(zip(unique_labels, counts_labels))}")
                task_logger.debug(f"  Anomaly Scores: Min={np.min(anomaly_score_array):.4f}, Max={np.max(anomaly_score_array):.4f}, Median={np.median(anomaly_score_array):.4f}, Avg={np.mean(anomaly_score_array):.4f}")

                if len(unique_labels) > 1 :
                    chosen_threshold = float(np.percentile(anomaly_score_array, threshold_percentile)) 
                    task_logger.debug(f"  Threshold strategy: {threshold_percentile}th percentile of all anomaly scores ({len(anomaly_score_array)} samples).")
                else:
                    chosen_threshold = float(np.median(anomaly_score_array))
                    task_logger.debug(f"  Threshold strategy: Fallback - Median of all available anomaly scores ({len(anomaly_score_array)} samples).")
                
                task_logger.debug(f"  Chosen Threshold for anomaly score: {chosen_threshold:.4f}")

                predictions = [1 if score > chosen_threshold else 0 for score in anomaly_score_array]
                predictions_array = np.array(predictions)
                unique_preds, counts_preds = np.unique(predictions_array, return_counts=True)
                task_logger.debug(f"  Unique predictions in batch: {dict(zip(unique_preds, counts_preds))}")
                
                current_accuracy = float(accuracy_score(labels_array, predictions_array))
                current_f1_score = float(f1_score(labels_array, predictions_array, zero_division=0))
                current_precision = float(precision_score(labels_array, predictions_array, zero_division=0))
                current_recall = float(recall_score(labels_array, predictions_array, zero_division=0))
                
                task_logger.info(f"[Task] Test Evaluation: Samples (valid for metrics)={len(labels_array)}, Avg Anomaly Score (all samples)={avg_anomaly_score_val:.4f}, Threshold={chosen_threshold:.4f}, Accuracy={current_accuracy:.4f}, F1={current_f1_score:.4f}, Precision={current_precision:.4f}, Recall={current_recall:.4f}")

            except Exception as e:
                task_logger.error(f"[Task] Error calculating metrics in test: {e}", exc_info=True)
                error_message = f"Error in metric calculation: {str(e)}"
        else:
            task_logger.warning("[Task-Debug] Filtered anomaly scores or labels list is empty after validation checks. Skipping metric calculation.")
            if not error_message: error_message = "Filtered anomaly_scores/labels empty post-validation"
            
    metrics_to_return = {
        "avg_anomaly_score": float(avg_anomaly_score_val),
        "accuracy": current_accuracy,
        "f1_score": current_f1_score,
        "precision": current_precision,
        "recall": current_recall,
        "threshold": chosen_threshold, 
        "anomaly_scores": all_sample_anomaly_scores_list,
        "labels": all_sample_labels_list 
    }
    if error_message:
        metrics_to_return["evaluate_metric_error"] = error_message
    
    return avg_anomaly_score_val, num_samples, metrics_to_return


def get_weights(net):
    task_logger.debug('get weights')
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
