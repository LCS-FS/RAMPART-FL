"""quickstart-docker: Flower client app for ToN-IoT anomaly detection."""

import torch
import time 
import psutil 
import os 
import logging
from datetime import datetime
import numpy as np

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.task import (
    Generator,
    Discriminator,
    get_weights,
    load_data,
    set_weights,
    test,
    train,
    BATCH_SIZE,
    INPUT_DIM,
)

# Module-level logger instance (can be used by other parts of the module like FlowerClient)
module_client_logger = logging.getLogger("ClientAppLogger")

ANOMALY_FRACTION_FOR_TRAINING = 0.3

class FlowerClient(NumPyClient):
    def __init__(self, netG, netD, full_train_potential_dataset, valloader_static, local_epochs, total_server_rounds, batch_size, allocated_cores, client_id_for_log, learning_rate_g, learning_rate_d, latent_dim, profile_name):
        self.netG = netG
        self.netD = netD
        self.full_train_potential_dataset = full_train_potential_dataset
        self.valloader = valloader_static
        self.local_epochs = local_epochs
        self.total_server_rounds = total_server_rounds
        self.batch_size = batch_size
        self.allocated_cores = int(os.getenv('ALLOCATED_CORES', 1))
        self.client_id_for_log = client_id_for_log
        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d
        self.latent_dim = latent_dim
        self.profile_name = profile_name

        # Store the number of parameter tensors for the Generator model
        # This is used to correctly split the flat list of parameters received by set_parameters
        self.num_g_params = len(list(self.netG.state_dict().keys()))
        module_client_logger.info(f"[Client] Generator model has {self.num_g_params} parameter tensors.")

        # --- Force CPU Usage ---
        self.device = torch.device("cpu")
        # torch.set_num_threads() is now called in client_fn, before net instantiation
        module_client_logger.info(f"[Client] Initialized. Device: {self.device}. Operating with {self.allocated_cores} allocated core(s) for PyTorch tasks. Current PyTorch intra-op threads: {torch.get_num_threads()}.")
        # --- End CPU Usage ---
        self.netG.to(self.device)
        self.netD.to(self.device)

    def get_parameters(self, config):
        """Return model parameters as a flat list of NumPy ndarrays (G_params + D_params)."""
        module_client_logger.info("[Client] get_parameters called")
        g_weights = get_weights(self.netG)
        d_weights = get_weights(self.netD)
        # Concatenate the lists of ndarrays
        return g_weights + d_weights

    def set_parameters(self, parameters):
        """Update model parameters from a flat list of NumPy ndarrays."""
        module_client_logger.info(f"[Client] set_parameters called with {len(parameters)} total parameter tensors.")
        if not parameters:
            module_client_logger.error("[Client] Received empty parameters list in set_parameters. Skipping update.")
            return

        if self.num_g_params <= 0:
            module_client_logger.error("[Client] Number of Generator parameters (self.num_g_params) is not positive. Cannot split parameters. Model weights not updated.")
            return
            
        if len(parameters) < self.num_g_params:
            module_client_logger.error(f"[Client] Received {len(parameters)} tensors, but expected at least {self.num_g_params} for the Generator. Model weights not updated.")
            return

        g_weights = parameters[:self.num_g_params]
        d_weights = parameters[self.num_g_params:]

        module_client_logger.info(f"[Client] Splitting parameters: {len(g_weights)} for G, {len(d_weights)} for D.")

        if not d_weights: # Check if d_weights is empty after slicing
             module_client_logger.warning(f"[Client] Discriminator received no weights after splitting (num_g_params={self.num_g_params}, total_received={len(parameters)}). This might be an issue if D is expected to have weights.")


        set_weights(self.netG, g_weights)
        
        # Only try to set D weights if d_weights is not empty AND D actually has parameters
        if d_weights or len(list(self.netD.state_dict().keys())) > 0:
            if len(d_weights) == len(list(self.netD.state_dict().keys())):
                 set_weights(self.netD, d_weights)
            else:
                module_client_logger.error(f"[Client] Mismatch for Discriminator weights. Expected {len(list(self.netD.state_dict().keys()))}, got {len(d_weights)}. D weights not updated.")
        elif not d_weights and len(list(self.netD.state_dict().keys())) > 0 :
             module_client_logger.error(f"[Client] Discriminator expects {len(list(self.netD.state_dict().keys()))} weights, but received none after split. D weights not updated.")
        # If d_weights is empty and D has no parameters, it's fine.

    def fit(self, parameters, config):
        """Train the model on accumulating data, measure resources, and return results."""
        current_server_round = config.get("server_round", 1)
        module_client_logger.info(f"[Client] Fit started for round {current_server_round}/{self.total_server_rounds}.") 
        
        fit_start_time = time.time()
        process = psutil.Process(os.getpid())
        process.cpu_percent(interval=0.1) 
        ram_mb_before_fit = process.memory_info().rss / (1024 * 1024)

        self.set_parameters(parameters)
        
        combined_metrics = {
            "cpu_cores_available": self.allocated_cores,
            "client_cid_for_attributes": str(self.client_id_for_log),
            "server_round": current_server_round,
            "allocated_cores_for_pytorch": self.allocated_cores,
            "profile_name": self.profile_name
        }
        
        current_trainloader_for_round = None
        num_examples_for_fit = 0

        if self.full_train_potential_dataset and len(self.full_train_potential_dataset) > 0:
            n_total_potential_train_samples = len(self.full_train_potential_dataset)
            effective_total_rounds = self.total_server_rounds if self.total_server_rounds > 0 else 1
            round_to_use_for_slice = min(current_server_round, effective_total_rounds)
            samples_per_pseudo_round = n_total_potential_train_samples / effective_total_rounds
            num_samples_to_include_cumulative = int(round_to_use_for_slice * samples_per_pseudo_round)
            num_samples_to_include_cumulative = max(0, min(num_samples_to_include_cumulative, n_total_potential_train_samples))

            if num_samples_to_include_cumulative > 0:
                all_features_in_slice = self.full_train_potential_dataset.features[:num_samples_to_include_cumulative]
                all_labels_in_slice = self.full_train_potential_dataset.labels[:num_samples_to_include_cumulative]
                
                normal_indices = (all_labels_in_slice == 0).nonzero(as_tuple=True)[0]
                
                if len(normal_indices) > 0:
                    normal_features_for_gan = all_features_in_slice[normal_indices]
                    normal_labels_for_gan = torch.zeros(len(normal_features_for_gan), dtype=torch.long) 
                    
                    anomalous_indices = (all_labels_in_slice == 1).nonzero(as_tuple=True)[0]
                    num_anomalies_to_add = int(len(normal_features_for_gan) * ANOMALY_FRACTION_FOR_TRAINING)
                    
                    final_features_for_gan_list = [normal_features_for_gan]
                    final_labels_for_gan_list = [normal_labels_for_gan]
                    
                    num_anomalies_actually_added = 0
                    if len(anomalous_indices) > 0 and num_anomalies_to_add > 0:
                        num_anomalies_to_select = min(len(anomalous_indices), num_anomalies_to_add)
                        
                        anomalous_indices_np = anomalous_indices.cpu().numpy()
                        selected_anomaly_indices_relative_to_anomalous_subset = np.random.choice(
                            len(anomalous_indices_np), 
                            size=num_anomalies_to_select, 
                            replace=False
                        )
                        actual_selected_anomaly_indices = anomalous_indices[selected_anomaly_indices_relative_to_anomalous_subset]

                        selected_anomalous_features = all_features_in_slice[actual_selected_anomaly_indices]
                        selected_anomalous_labels = torch.ones(len(selected_anomalous_features), dtype=torch.long) 
                        
                        final_features_for_gan_list.append(selected_anomalous_features)
                        final_labels_for_gan_list.append(selected_anomalous_labels)
                        num_anomalies_actually_added = len(selected_anomalous_features)
                        module_client_logger.info(f"[Client] Semi-supervised: Added {num_anomalies_actually_added} anomalous samples to GAN training data (target: {num_anomalies_to_add}).")
                    else:
                        module_client_logger.info(f"[Client] Semi-supervised: No anomalies to add (available: {len(anomalous_indices)}, target: {num_anomalies_to_add}). GAN training with normal data only.")

                    final_features_for_gan = torch.cat(final_features_for_gan_list, dim=0)
                    final_labels_for_gan = torch.cat(final_labels_for_gan_list, dim=0)
                    
                    perm = torch.randperm(len(final_features_for_gan))
                    shuffled_features = final_features_for_gan[perm]
                    shuffled_labels = final_labels_for_gan[perm]

                    gan_train_dataset = TensorDataset(shuffled_features, shuffled_labels)
                    
                    dataloader_num_workers = 0 
                    current_trainloader_for_round = DataLoader(gan_train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=dataloader_num_workers)
                    num_examples_for_fit = len(gan_train_dataset)
                    module_client_logger.info(f"[Client] Round {current_server_round}: Training GAN with {num_examples_for_fit} samples ({len(normal_features_for_gan)} normal, {num_anomalies_actually_added} anomalous).")
                else:
                    module_client_logger.info(f"[Client] Round {current_server_round}: No NORMAL samples found in the {num_samples_to_include_cumulative} cumulative samples. Skipping GAN training logic.")
                    num_examples_for_fit = 0
            else:
                module_client_logger.info(f"[Client] Round {current_server_round}: No training samples to include ({num_samples_to_include_cumulative} calculated). Skipping GAN training logic.")
        else:
            module_client_logger.warning(f"[Client] Warning: No potential training data available. Skipping fit.")

        if current_trainloader_for_round and num_examples_for_fit > 0 :
            try:
                module_client_logger.info("[Client] Starting GAN training...")
                avg_g_loss, avg_d_loss = train(
                    self.netG,
                    self.netD,
                    current_trainloader_for_round, 
                    epochs=self.local_epochs,
                    device=self.device,
                    learning_rate_g=self.learning_rate_g,
                    learning_rate_d=self.learning_rate_d,
                    latent_dim=self.latent_dim 
                )
                combined_metrics["g_loss"] = float(avg_g_loss)
                combined_metrics["d_loss"] = float(avg_d_loss)
                module_client_logger.info(f"GAN Training completed. Avg G Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")

                # --- NEW: Evaluate local model on local validation set post-fit ---
                if self.valloader and len(self.valloader.dataset) > 0:
                    module_client_logger.info("[Client] Evaluating updated local model on local validation set...")
                    _, _, post_fit_metrics = test(self.netD, self.valloader, self.device)
                    local_f1 = post_fit_metrics.get("f1_score", 0.0)
                    combined_metrics["post_fit_local_f1"] = float(local_f1)
                    module_client_logger.info(f"[Client] Post-fit local evaluation F1: {local_f1:.4f}")
                else:
                    combined_metrics["post_fit_local_f1"] = 0.0 # No valloader to test against
                # --- END NEW ---

            except Exception as e:
                module_client_logger.error(f"[Client] ERROR during GAN training: {e}", exc_info=True)
                combined_metrics["fit_error_training"] = str(e)
        elif num_examples_for_fit == 0:
            if self.full_train_potential_dataset and len(self.full_train_potential_dataset) > 0:
                combined_metrics["fit_status"] = "skipped_no_normal_samples_for_round"
        else:
            combined_metrics["fit_status"] = "skipped_no_initial_train_data"

        cpu_usage_fit = process.cpu_percent() / psutil.cpu_count() if psutil.cpu_count() else process.cpu_percent()
        ram_mb_after_fit = process.memory_info().rss / (1024 * 1024)
        fit_duration = time.time() - fit_start_time
        combined_metrics["fit_time_seconds"] = fit_duration
        combined_metrics["cpu_percent_fit"] = cpu_usage_fit
        combined_metrics["ram_mb_fit"] = ram_mb_after_fit - ram_mb_before_fit
        combined_metrics["accumulated_train_samples_this_round"] = num_examples_for_fit

        module_client_logger.info(f"Fit completed for round {current_server_round}. Duration: {fit_duration:.2f}s. Metrics: {combined_metrics}")
        
        updated_weights = self.get_parameters(config={})
        return updated_weights, num_examples_for_fit, combined_metrics

    def evaluate(self, parameters, config):
        """Evaluate the network using the static validation set with the Discriminator."""
        current_server_round = config.get("server_round", -1)
        module_client_logger.info(f"[Client] Evaluate method started for round {current_server_round} using Discriminator.") 
        eval_start_time = time.time()

        client_metrics_for_aggregation = {
            "client_cid_for_attributes": str(self.client_id_for_log),
            "allocated_cores_for_pytorch": self.allocated_cores,
            "server_round": current_server_round,
            "profile_name": self.profile_name
        }

        if self.valloader is None:
            module_client_logger.warning("[Client] ValLoader is None, skipping evaluation")
            error_metrics = {"avg_anomaly_score": 0.0, "accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0, "eval_status": "skipped_no_data", "eval_time_seconds": time.time() - eval_start_time}
            client_metrics_for_aggregation.update(error_metrics)
            return 0.0, 0, client_metrics_for_aggregation

        try:
            self.set_parameters(parameters)
            module_client_logger.info("[Client] Starting evaluation with Discriminator...")
            
            avg_anomaly_score, num_samples, metrics_from_task = test(self.netD, self.valloader, self.device) 
            
            eval_duration = time.time() - eval_start_time 

            current_accuracy = metrics_from_task.get("accuracy", 0.0)
            current_f1_score = metrics_from_task.get("f1_score", 0.0)
            current_precision = metrics_from_task.get("precision", 0.0)
            current_recall = metrics_from_task.get("recall", 0.0)

            module_client_logger.info(f"[Client] Evaluation finished. Samples: {num_samples}, Acc: {current_accuracy:.4f}, F1: {current_f1_score:.4f}, Prec: {current_precision:.4f}, Rec: {current_recall:.4f}, Avg Anomaly Score: {avg_anomaly_score:.4f}")

            performance_metrics = {
                "avg_anomaly_score": float(avg_anomaly_score),
                "accuracy": float(current_accuracy),
                "f1_score": float(current_f1_score),
                "precision": float(current_precision),
                "recall": float(current_recall),
                "eval_time_seconds": float(eval_duration)
            }
            client_metrics_for_aggregation.update(performance_metrics)
            
            if "evaluate_metric_error" in metrics_from_task: 
                client_metrics_for_aggregation["evaluate_metric_error"] = metrics_from_task["evaluate_metric_error"]
            if "threshold" in metrics_from_task: 
                client_metrics_for_aggregation["threshold"] = metrics_from_task.get("threshold", 0.0)

            return avg_anomaly_score, num_samples, client_metrics_for_aggregation
        except Exception as e:
            module_client_logger.error(f"[Client] ERROR during evaluation: {e}", exc_info=True)
            eval_duration = time.time() - eval_start_time 
            error_metrics = {"avg_anomaly_score": 0.0, "accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0, "eval_time_seconds": eval_duration, "evaluate_error": str(e)}
            client_metrics_for_aggregation.update(error_metrics)
            return 0.0, 0, client_metrics_for_aggregation


def client_fn(context: Context):
    """Create and return an instance of FlowerClient."""
    
    client_logger_fn_scope = logging.getLogger("ClientAppLogger")

    partition_id = context.node_config.get("partition-id", os.environ.get("PARTITION_ID", "unknown_cid"))
    profile_name_from_env = os.environ.get("PROFILE_NAME", "unknown_profile")
    base_log_dir = os.environ.get("LOG_DIR", "/app/logs") 
    
    now = datetime.now()
    rounded_minutes = (now.minute // 30) * 30
    timestamp = now.replace(minute=rounded_minutes, second=0, microsecond=0).strftime("%Y%m%d_%H%M")
    instance_log_dir = os.path.join(base_log_dir, timestamp)
    os.makedirs(instance_log_dir, exist_ok=True)
    
    client_log_file = os.path.join(instance_log_dir, f"client_{partition_id}.log")

    if client_logger_fn_scope.hasHandlers():
        client_logger_fn_scope.handlers.clear()

    client_logger_fn_scope.setLevel(logging.INFO)
    fh = logging.FileHandler(client_log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(f'%(asctime)s - %(name)s [Client {partition_id}] - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    client_logger_fn_scope.addHandler(fh)
    
    client_logger_fn_scope.info(f"ClientApp starting for partition_id: {partition_id}. Instance log dir: {instance_log_dir}")

    num_partitions = context.node_config.get("num-partitions", int(os.environ.get("NUM_CLIENTS", 1)))
    total_server_rounds = context.run_config.get("num-server-rounds", 3)

    lr_g = 5e-05
    lr_d = 0.0001
    latent_dim = 100
    g_hidden1_size = 128
    g_hidden2_size = 32
    d_hidden1_size = 128
    d_hidden2_size = 32

    local_epochs = 10
    current_batch_size = 32

    DEFAULT_PYTORCH_THREADS = 1
    allocated_cores_for_pytorch = DEFAULT_PYTORCH_THREADS
    
    env_cores_str = os.environ.get('ALLOCATED_CORES')
    if env_cores_str:
        try:
            parsed_env_cores = int(env_cores_str)
            if parsed_env_cores > 0:
                allocated_cores_for_pytorch = parsed_env_cores
                client_logger_fn_scope.info(f"ALLOCATED_CORES='{env_cores_str}' found. Setting PyTorch threads to {allocated_cores_for_pytorch}.")
            else:
                client_logger_fn_scope.warning(f"ALLOCATED_CORES='{env_cores_str}' is invalid (<=0). Defaulting PyTorch threads to {DEFAULT_PYTORCH_THREADS}.")
        except ValueError:
            client_logger_fn_scope.warning(f"ALLOCATED_CORES='{env_cores_str}' is not a valid integer. Defaulting PyTorch threads to {DEFAULT_PYTORCH_THREADS}.")

    torch.set_num_threads(allocated_cores_for_pytorch)
    client_logger_fn_scope.info(f"Called torch.set_num_threads({allocated_cores_for_pytorch}). Current PyTorch intra-op threads: {torch.get_num_threads()}. Client will operate with {allocated_cores_for_pytorch} core(s).")

    client_logger_fn_scope.info(f"Loading data for partition {partition_id}/{num_partitions}...")
    data_load_result = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        instance_log_dir=instance_log_dir
    )

    if data_load_result is None or not any(data_load_result):
        client_logger_fn_scope.error(f"Failed to load data for partition {partition_id}. Exiting client_fn.")
        raise ValueError(f"Data loading failed for client {partition_id}")

    full_train_potential_dataset, val_dataset_static, _ = data_load_result
    
    client_logger_fn_scope.info(f"Data loaded for partition {partition_id}: Train samples: {len(full_train_potential_dataset) if full_train_potential_dataset else 0}, Val samples: {len(val_dataset_static) if val_dataset_static else 0}")

    client_logger_fn_scope.info(f"Instantiating GAN models: G(latent_dim={latent_dim}, h1={g_hidden1_size}, h2={g_hidden2_size}), D(h1={d_hidden1_size}, h2={d_hidden2_size})")
    netG = Generator(
        input_dim=INPUT_DIM,
        latent_dim=latent_dim,
        hidden1_size=g_hidden1_size,
        hidden2_size=g_hidden2_size
    )
    netD = Discriminator(
        input_dim=INPUT_DIM,
        hidden1_size=d_hidden1_size,
        hidden2_size=d_hidden2_size
    )

    valloader_static = None
    if val_dataset_static and len(val_dataset_static) > 0:
        val_dataloader_num_workers = 0
        valloader_static = DataLoader(val_dataset_static, batch_size=current_batch_size, shuffle=False, num_workers=val_dataloader_num_workers)
    else:
        client_logger_fn_scope.info(f"No validation data assigned or dataset empty for partition {partition_id}.")


    return FlowerClient(
        netG=netG,
        netD=netD,
        full_train_potential_dataset=full_train_potential_dataset,
        valloader_static=valloader_static,
        local_epochs=local_epochs,
        total_server_rounds=total_server_rounds,
        batch_size=current_batch_size,
        allocated_cores=allocated_cores_for_pytorch,
        client_id_for_log=str(partition_id),
        learning_rate_g=lr_g,
        learning_rate_d=lr_d,
        latent_dim=latent_dim,
        profile_name=profile_name_from_env
    ).to_client()


app = ClientApp(
    client_fn=client_fn,
)
