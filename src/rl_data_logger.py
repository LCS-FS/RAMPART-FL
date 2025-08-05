import csv
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_simulation_start_time = None

CSV_HEADERS = [
    "elapsed_seconds_since_start",
    "server_round",
    "event_type",
    "client_cid",
    "s_available_cids_count",
    "s_client_state_tuple",
    "s_client_q_value",
    "s_client_selection_prob",
    "s_was_selected",
    "s_softmax_temperature",
    "s_selection_context_global_f1_prev_round",
    "s_selection_context_max_global_f1_achieved",
    "l_state_at_selection",
    "l_reward_performance_component",
    "l_reward_fairness_penalty_component",
    "l_reward_total_global_for_action",
    "l_q_current_value_S",
    "l_q_next_max_value_S_prime",
    "l_q_updated_value_S",
    "l_rl_learning_rate",
    "l_rl_discount_factor",
    "l_learn_context_current_global_f1",
    "l_learn_context_last_global_f1",
    "l_learn_context_last_global_precision",
    "l_learn_context_last_global_recall",
    "l_learn_context_max_global_f1",
    "l_reward_resource_cost_component",
    "gm_eval_avg_anomaly_score",
    "gm_eval_accuracy",
    "gm_eval_f1",
    "gm_eval_precision",
    "gm_eval_recall",
    "gm_eval_avg_client_time_seconds",
    "gm_eval_clients_attempted",
    "gm_eval_clients_valid_metrics",
    "gm_fit_g_loss",
    "gm_fit_d_loss",
    "gm_fit_avg_client_time_seconds",
    "gm_fit_clients_total_received",
    "gm_fit_clients_valid_for_loss",
    "c_eval_partition_id",
    "c_eval_profile_name",
    "c_eval_cores",
    "c_eval_f1",
    "c_eval_accuracy",
    "c_eval_precision",
    "c_eval_recall",
    "c_eval_num_samples",
    "c_eval_time_seconds",
    "c_fit_partition_id",
    "c_fit_cpu_percent",
    "c_fit_ram_mb",
    "c_fit_time_seconds",
    "c_fit_post_local_f1",
    "c_fit_g_loss",
    "c_fit_d_loss"
]

_rl_csv_filepath = None

def setup_rl_data_csv(instance_log_dir: str, filename="rl_training_data.csv") -> str:
    """
    Creates the RL data CSV file in the specified directory, writes the header,
    and records the simulation start time.
    Returns the full path to the CSV file.
    """
    global _rl_csv_filepath, _simulation_start_time
    _simulation_start_time = datetime.now()
    
    if not os.path.exists(instance_log_dir):
        os.makedirs(instance_log_dir, exist_ok=True)
    
    _rl_csv_filepath = os.path.join(instance_log_dir, filename)
    
    try:
        with open(_rl_csv_filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
            writer.writeheader()
        logger.info(f"RL data CSV initialized at: {_rl_csv_filepath}")
    except IOError as e:
        logger.error(f"Failed to initialize RL data CSV at {_rl_csv_filepath}: {e}")
        _rl_csv_filepath = None
        
    return _rl_csv_filepath

def log_rl_data_row(data_dict: dict):
    """
    Appends a new row to the RL data CSV file.
    The data_dict should contain keys corresponding to CSV_HEADERS.
    Missing keys will result in empty cells for those columns.
    """
    if _rl_csv_filepath is None:
        return

    elapsed_seconds = ""
    if _simulation_start_time:
        elapsed_duration = datetime.now() - _simulation_start_time
        elapsed_seconds = elapsed_duration.total_seconds()
    else:
        logger.warning("Simulation start time not recorded; elapsed time will be empty or incorrect.")

    data_dict["elapsed_seconds_since_start"] = elapsed_seconds

    try:
        with open(_rl_csv_filepath, 'a', newline='') as csvfile:
            row_to_write = {header: data_dict.get(header, "") for header in CSV_HEADERS}
            writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
            writer.writerow(row_to_write)
    except IOError as e:
        logger.error(f"Failed to write RL data row to {_rl_csv_filepath}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error writing RL data row: {e}. Data was: {data_dict}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_log_dir = "temp_logs/rl_data_test"
    
    csv_path = setup_rl_data_csv(test_log_dir, "test_rl_data.csv")
    
    if csv_path:
        selection_event_1 = {
            "server_round": 3,
            "event_type": "selection_info",
            "client_cid": "client_A",
            "s_available_cids_count": 10,
            "s_client_state_tuple": "(1,0,2,1)",
            "s_client_q_value": 0.5,
            "s_client_selection_prob": 0.15,
            "s_was_selected": 1,
            "s_softmax_temperature": 0.8,
            "s_selection_context_global_f1_prev_round": 0.65
        }
        log_rl_data_row(selection_event_1)

        learning_event_1 = {
            "server_round": 3,
            "event_type": "learning_update",
            "client_cid": "client_A",
            "l_state_at_selection": "(1,0,2,1)",
            "l_reward_performance_component": 0.05,
            "l_reward_fairness_penalty_component": -0.01,
            "l_reward_total_global_for_action": 0.04,
            "l_q_current_value_S": 0.5,
            "l_q_next_max_value_S_prime": 0.6,
            "l_q_updated_value_S": 0.52,
            "l_rl_learning_rate": 0.1,
            "l_rl_discount_factor": 0.9,
            "l_learn_context_current_global_f1": 0.70,
            "l_learn_context_last_global_f1": 0.65,
            "l_learn_context_last_global_precision": 0.60,
            "l_learn_context_last_global_recall": 0.70
        }
        log_rl_data_row(learning_event_1)
        
        logger.info(f"Test RL data written to {csv_path}") 