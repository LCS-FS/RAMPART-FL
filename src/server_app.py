from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import os
import logging
from datetime import datetime

try:
    from flwr.common.record import ConfigRecord
except ImportError:
    ConfigRecord = type(None)

from flwr.common import Context, Metrics, ndarrays_to_parameters, Parameters, FitIns, EvaluateIns, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from src.task import (
    Generator,
    Discriminator,
    get_weights, 
    INPUT_DIM
)
from src.participant_selection import RLClientSelectionStrategy
from src import rl_data_logger

logger = logging.getLogger("FedAnomalyServer")

CLIENT_ATTRIBUTES: Dict[str, Dict[str, Union[int, float, str]]] = {}

def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    try:
        logger.info("="*50)
        logger.info("EVALUATE METRICS AGGREGATION FUNCTION CALLED")
        logger.info(f"Received metrics from {len(metrics)} clients for evaluation.")
        logger.info("="*50)

        if not metrics: 
            logger.warning("No metrics received for evaluation aggregation, returning empty dict.")
            return {}

        total_weighted_anomaly_score = 0.0
        total_weighted_accuracy = 0.0
        total_weighted_f1 = 0.0
        total_weighted_precision = 0.0
        total_weighted_recall = 0.0
        total_eval_time = 0.0
        
        total_examples = 0 
        valid_clients = 0 
        evaluate_errors = 0 

        for num_ex, client_metrics_obj in metrics:
            client_metrics_dict = None 
            error_reported_by_client = False

            if hasattr(client_metrics_obj, 'get'): 
                if client_metrics_obj.get("evaluate_error") is not None:
                    logger.warning(f"Client (num_ex={num_ex}) reported evaluate error: {client_metrics_obj['evaluate_error']}")
                    evaluate_errors += 1
                    error_reported_by_client = True
                else:
                    client_metrics_dict = client_metrics_obj 
            else: 
                logger.warning(f"Received unexpected metrics object type {type(client_metrics_obj)} from client (num_ex={num_ex}).")
                evaluate_errors += 1 
                error_reported_by_client = True
            
            if error_reported_by_client:
                continue 

            if client_metrics_dict and num_ex > 0:
                anomaly_score = client_metrics_dict.get("avg_anomaly_score")
                accuracy = client_metrics_dict.get("accuracy")
                f1_score_val = client_metrics_dict.get("f1_score")
                precision_val = client_metrics_dict.get("precision")
                recall_val = client_metrics_dict.get("recall")
                eval_time = client_metrics_dict.get("eval_time_seconds")

                if anomaly_score is not None and isinstance(anomaly_score, (float, int)) and \
                    accuracy is not None and isinstance(accuracy, (float, int)) and \
                    f1_score_val is not None and isinstance(f1_score_val, (float, int)) and \
                    precision_val is not None and isinstance(precision_val, (float, int)) and \
                    recall_val is not None and isinstance(recall_val, (float, int)):
                    
                    total_weighted_anomaly_score += num_ex * anomaly_score
                    total_weighted_accuracy += num_ex * accuracy
                    total_weighted_f1 += num_ex * f1_score_val
                    total_weighted_precision += num_ex * precision_val
                    total_weighted_recall += num_ex * recall_val
                    
                    if eval_time is not None and isinstance(eval_time, (float, int)):
                        total_eval_time += eval_time 

                    total_examples += num_ex
                    valid_clients += 1
                else:
                    logger.warning(f"Client (num_ex={num_ex}) missing/invalid performance metrics. Metrics: {client_metrics_dict}")
                    continue 
            elif num_ex == 0: 
                logger.info(f"Client (num_ex={num_ex}) reported 0 examples during evaluation. Metrics: {client_metrics_dict}")

        metrics_aggregated = {}
        metrics_aggregated["evaluate_clients_attempted_processing"] = len(metrics) 
        metrics_aggregated["evaluate_clients_reported_valid_perf_metrics"] = valid_clients
        metrics_aggregated["evaluate_clients_reported_errors"] = evaluate_errors
        
        if total_examples == 0 or valid_clients == 0: 
            logger.warning("Evaluate phase: No clients reported valid performance metrics and examples for aggregation.")
            return metrics_aggregated

        aggregated_anomaly_score = total_weighted_anomaly_score / total_examples
        aggregated_accuracy = total_weighted_accuracy / total_examples
        aggregated_f1 = total_weighted_f1 / total_examples
        aggregated_precision = total_weighted_precision / total_examples
        aggregated_recall = total_weighted_recall / total_examples
        avg_eval_time = total_eval_time / valid_clients 
        
        metrics_aggregated["global_avg_anomaly_score"] = aggregated_anomaly_score
        metrics_aggregated["global_accuracy"] = aggregated_accuracy
        metrics_aggregated["global_f1_score"] = aggregated_f1
        metrics_aggregated["global_precision"] = aggregated_precision
        metrics_aggregated["global_recall"] = aggregated_recall
        metrics_aggregated["eval_time_avg_seconds_per_client"] = avg_eval_time

        logger.info(f"Evaluate phase: Aggregated Results ({total_examples} examples from {valid_clients} clients):")
        logger.info(f"  Global Avg Anomaly Score (Wgtd Avg): {aggregated_anomaly_score:.4f}")
        logger.info(f"  Global Accuracy (Wgtd Avg): {aggregated_accuracy:.4f}")
        logger.info(f"  Global F1 Score (Wgtd Avg): {aggregated_f1:.4f}")
        logger.info(f"  Global Precision (Wgtd Avg): {aggregated_precision:.4f}")
        logger.info(f"  Global Recall (Wgtd Avg): {aggregated_recall:.4f}")
        logger.info(f"  Avg Eval Time per Client: {avg_eval_time:.2f}s")
        
        return metrics_aggregated

    except Exception as e:
        logger.critical(f"CRITICAL ERROR in evaluate_metrics_aggregation_fn: {str(e)}", exc_info=True)
        return {"evaluate_aggregation_critical_error": str(e)}


def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        logger.warning("Fit phase aggregation: Received no metrics.")
        return {}

    total_train_examples = 0
    valid_clients_count_for_loss = 0 
    weighted_g_loss_sum = 0.0
    weighted_d_loss_sum = 0.0
    total_fit_time_sum = 0.0
    
    total_allocated_cores_sum = 0.0
    clients_reporting_allocated_cores = 0

    fit_errors_count = 0
    fit_skipped_count = 0

    logger.info(f"Fit phase aggregation: Received metrics from {len(metrics)} clients.")

    for num_train_ex, client_metrics_obj in metrics:
        client_metrics = None
        error_reported_in_fit_logic = False 
        status_skipped = False

        if hasattr(client_metrics_obj, 'get'):
            fit_error_keys = ["fit_error_training", "fit_error_final_return", "fit_error_initial_weight_get", "fit_error_training_and_weight_get"]
            if any(client_metrics_obj.get(key) for key in fit_error_keys):
                error_key = next((k for k in fit_error_keys if client_metrics_obj.get(k)), "fit_error_unknown")
                logger.warning(f"Client (train_ex={num_train_ex}) reported fit error: {client_metrics_obj.get(error_key)}")
                error_reported_in_fit_logic = True
                fit_errors_count +=1
            elif client_metrics_obj.get("fit_status") and "skipped" in client_metrics_obj.get("fit_status"):
                logger.info(f"Client (train_ex={num_train_ex}) reported status: {client_metrics_obj.get('fit_status')}")
                status_skipped = True
                fit_skipped_count +=1
            client_metrics = client_metrics_obj
        else:
            logger.warning(f"Received unexpected metrics object type {type(client_metrics_obj)} in fit phase.")
            fit_errors_count +=1 
            continue 

        fit_time = client_metrics.get("fit_time_seconds")
        if fit_time is not None and isinstance(fit_time, (float, int)):
            total_fit_time_sum += fit_time
        
        allocated_cores = client_metrics.get("allocated_cores_for_pytorch")
        if allocated_cores is not None and isinstance(allocated_cores, (int, float)):
            total_allocated_cores_sum += allocated_cores
            clients_reporting_allocated_cores += 1
        
        if not error_reported_in_fit_logic and not status_skipped:
            g_loss = client_metrics.get("g_loss")
            d_loss = client_metrics.get("d_loss")

            if g_loss is not None and isinstance(g_loss, (float, int)) and \
               d_loss is not None and isinstance(d_loss, (float, int)) and \
               num_train_ex > 0:
                weighted_g_loss_sum += num_train_ex * g_loss
                weighted_d_loss_sum += num_train_ex * d_loss
                total_train_examples += num_train_ex 
                valid_clients_count_for_loss += 1
            else:
                if not (g_loss is not None and isinstance(g_loss, (float, int))):
                    logger.warning(f"Client (train_ex={num_train_ex}) provided invalid/missing 'g_loss'. Metrics: {client_metrics}")
                if not (d_loss is not None and isinstance(d_loss, (float, int))):
                    logger.warning(f"Client (train_ex={num_train_ex}) provided invalid/missing 'd_loss'. Metrics: {client_metrics}")
                elif num_train_ex == 0:
                    logger.info(f"Client (train_ex={num_train_ex}) reported 0 examples for loss calculation after passing error/skip checks. Metrics: {client_metrics}")


    metrics_aggregated = {}
    metrics_aggregated["fit_clients_total_received"] = len(metrics)
    metrics_aggregated["fit_clients_reported_errors"] = fit_errors_count
    metrics_aggregated["fit_clients_skipped_training"] = fit_skipped_count
    metrics_aggregated["fit_clients_valid_for_loss_aggregation"] = valid_clients_count_for_loss

    if clients_reporting_allocated_cores > 0:
        avg_allocated_cores = total_allocated_cores_sum / clients_reporting_allocated_cores
        metrics_aggregated["allocated_cores_avg_per_client"] = avg_allocated_cores
        logger.info(f"Fit phase aggregation: Avg Allocated Cores per Client = {avg_allocated_cores:.2f}")
    else:
        metrics_aggregated["allocated_cores_avg_per_client"] = None


    if total_train_examples > 0 and valid_clients_count_for_loss > 0:
        aggregated_g_loss = weighted_g_loss_sum / total_train_examples
        aggregated_d_loss = weighted_d_loss_sum / total_train_examples
        metrics_aggregated["aggregated_g_loss"] = aggregated_g_loss
        metrics_aggregated["aggregated_d_loss"] = aggregated_d_loss
        logger.info(f"Fit phase aggregation: Aggregated G_Loss={aggregated_g_loss:.4f}, Aggregated D_Loss={aggregated_d_loss:.4f} from {valid_clients_count_for_loss} clients ({total_train_examples} examples)")
    else:
        logger.info("Fit phase aggregation: No valid training losses received from clients for aggregation.")
        metrics_aggregated["aggregated_g_loss"] = None
        metrics_aggregated["aggregated_d_loss"] = None

    num_reporting_clients = len(metrics) 
    if num_reporting_clients > 0:
        avg_fit_time = total_fit_time_sum / num_reporting_clients
        metrics_aggregated["fit_time_avg_seconds_all_reporting"] = avg_fit_time
        logger.info(f"Fit phase aggregation: Average Fit Time (all reporting clients)={avg_fit_time:.2f}s across {num_reporting_clients} clients")
    else:
        metrics_aggregated["fit_time_avg_seconds_all_reporting"] = 0.0
        
    logger.info("Fit phase aggregation: Summary")
    for key, value in metrics_aggregated.items():
        if isinstance(value, float): logger.info(f"  {key}: {value:.4f}")
        else: logger.info(f"  {key}: {value}")
    return metrics_aggregated


def fit_config(server_round: int) -> Dict:
    config = {
        "server_round": server_round,
    }
    return config


def server_fn(context: Context):
    base_log_dir = os.environ.get("LOG_DIR", "/app/logs")
    
    now = datetime.now()
    rounded_minutes = (now.minute // 30) * 30
    timestamp = now.replace(minute=rounded_minutes, second=0, microsecond=0).strftime("%Y%m%d_%H%M")
    instance_log_dir = os.path.join(base_log_dir, timestamp)
    os.makedirs(instance_log_dir, exist_ok=True)
    
    server_log_file = os.path.join(instance_log_dir, "server.log")

    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    
    file_handler = logging.FileHandler(server_log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') 
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    flwr_core_logger = logging.getLogger("flwr")
    flwr_core_logger.setLevel(logging.INFO)
    
    is_our_handler_on_flwr = any(h is file_handler for h in flwr_core_logger.handlers)
    if not is_our_handler_on_flwr:
        flwr_core_logger.addHandler(file_handler)

    logger.info(f"Instance log directory: {instance_log_dir}")
    logger.info(f"NOTE: 'flwr' core logger level temporarily set to INFO for detailed framework messages during debugging.")
    
    rl_data_csv_path = rl_data_logger.setup_rl_data_csv(instance_log_dir)
    if rl_data_csv_path:
        logger.info(f"RL data will be logged to: {rl_data_csv_path}")
    else:
        logger.error("Failed to initialize RL data CSV logger. RL data will not be saved to CSV.")

    num_rounds = context.run_config.get("num-server-rounds", 3)
    rl_learning_rate = context.run_config.get("rl-learning-rate", 0.1)
    rl_discount_factor = context.run_config.get("rl-discount-factor", 0.9)
    rl_temperature_initial = context.run_config.get("rl-temperature-initial", 1.0)
    rl_temperature_decay = context.run_config.get("rl-temperature-decay", 0.995)
    rl_temperature_min = context.run_config.get("rl-temperature-min", 0.01)
    rl_history_length = context.run_config.get("rl-history-length", 3)
    rl_recency_window = context.run_config.get("rl-recency-window", 5)
    rl_fairness_penalty_factor = context.run_config.get("rl-fairness-penalty-factor", 0.1)
    rl_fairness_participation_threshold_factor = context.run_config.get("rl-fairness-participation-threshold-factor", 0.5)
    rl_w_f1 = context.run_config.get("rl-w-f1", 0.5)
    rl_w_precision = context.run_config.get("rl-w-precision", 0.25)
    rl_w_recall = context.run_config.get("rl-w-recall", 0.25)
    rl_num_clients_to_select_fallback = context.run_config.get("rl-num-clients-to-select-fallback", 5)
    rl_prob_threshold_factor = context.run_config.get("rl-prob-threshold-factor", 1.2)
    rl_resource_cost_factor = context.run_config.get("rl-resource-cost-factor", 0.005)
    rl_ablation_mode = context.run_config.get("rl-ablation-mode", "multi-criteria")

    logger.info(f"RL Agent parameters loaded: lr={rl_learning_rate}, gamma={rl_discount_factor}, temp_init={rl_temperature_initial}, hist_len={rl_history_length}")
    logger.info(f"RL Ablation Mode set to: '{rl_ablation_mode}'")
    
    
    
    min_available_clients_from_config = context.run_config.get("min-available-clients", 1)
    min_available_clients_for_server_start = min_available_clients_from_config
    
    effective_min_fit_clients = 1
    effective_min_evaluate_clients = min_available_clients_from_config


    logger.info(f"Server Config: num_rounds={num_rounds}")
    logger.info(f"EVALUATION Selection: Targeting ALL available clients each round for EVALUATION.")
    logger.info(f"Strategy settings: min_fit_clients={effective_min_fit_clients}, min_evaluate_clients={effective_min_evaluate_clients} (fraction_evaluate=1.0 implied for 'all')")
    logger.info(f"Server start requires min_available_clients={min_available_clients_for_server_start}")

    initial_latent_dim = 100
    initial_g_hidden1_size = 128
    initial_g_hidden2_size = 32
    initial_d_hidden1_size = 128 
    initial_d_hidden2_size = 32 

    logger.info(f"Initializing global GAN models (Generator, Discriminator) with: input_dim={INPUT_DIM}, latent_dim={initial_latent_dim}, g_h1={initial_g_hidden1_size}, g_h2={initial_g_hidden2_size}, d_h1={initial_d_hidden1_size}, d_h2={initial_d_hidden2_size}")
    initial_generator = Generator(
        input_dim=INPUT_DIM,
        latent_dim=initial_latent_dim,
        hidden1_size=initial_g_hidden1_size,
        hidden2_size=initial_g_hidden2_size
    )
    initial_discriminator = Discriminator(
        input_dim=INPUT_DIM,
        hidden1_size=initial_d_hidden1_size,
        hidden2_size=initial_d_hidden2_size
    )
    
    g_ndarrays = get_weights(initial_generator)
    d_ndarrays = get_weights(initial_discriminator)
    
    flat_initial_ndarrays = g_ndarrays + d_ndarrays
    
    initial_parameters_obj = ndarrays_to_parameters(flat_initial_ndarrays)

    strategy = RLClientSelectionStrategy(
        client_attributes_store=CLIENT_ATTRIBUTES,
        strategy_logger=logger,
        num_rounds=num_rounds,
        instance_log_dir=instance_log_dir,
        rl_learning_rate=rl_learning_rate,
        rl_discount_factor=rl_discount_factor,
        rl_temperature_initial=rl_temperature_initial,
        rl_temperature_decay=rl_temperature_decay,
        rl_temperature_min=rl_temperature_min,
        rl_history_length=rl_history_length,
        rl_recency_window=rl_recency_window,
        rl_fairness_penalty_factor=rl_fairness_penalty_factor,
        rl_fairness_participation_threshold_factor=rl_fairness_participation_threshold_factor,
        rl_w_f1=rl_w_f1,
        rl_w_precision=rl_w_precision,
        rl_w_recall=rl_w_recall,
        rl_num_clients_to_select_fallback=rl_num_clients_to_select_fallback,
        rl_prob_threshold_factor=rl_prob_threshold_factor,
        rl_resource_cost_factor=rl_resource_cost_factor,
        rl_ablation_mode=rl_ablation_mode,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=effective_min_fit_clients,
        min_evaluate_clients=effective_min_evaluate_clients,
        min_available_clients=min_available_clients_for_server_start,
        initial_parameters=initial_parameters_obj,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_fit_config_fn=fit_config,
    )

    config = ServerConfig(num_rounds=num_rounds)
    logger.info("RLClientSelectionStrategy and ServerConfig initialized.")
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
