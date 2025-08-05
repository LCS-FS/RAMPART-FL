import logging
from typing import Dict, List, Tuple, Union, Optional 
import math
import numpy as np
from flwr.common import parameters_to_ndarrays
import os

from flwr.common import FitIns, EvaluateIns, Parameters, FitRes, EvaluateRes, Scalar 
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.rl_agent import RLAgent
from src.rl_data_logger import log_rl_data_row


class RLClientSelectionStrategy(FedAvg):
    def __init__(
        self,
        client_attributes_store: Dict[str, Dict[str, Union[int, float, str]]],
        strategy_logger: logging.Logger,
        num_rounds: int,
        instance_log_dir: str,
        rl_learning_rate=0.1,
        rl_discount_factor=0.9,
        rl_temperature_initial=1.0,
        rl_temperature_decay=0.995,
        rl_temperature_min=0.01,
        rl_history_length=3,
        rl_recency_window=5,
        rl_fairness_penalty_factor=0.1,
        rl_fairness_participation_threshold_factor=0.5,
        rl_w_f1=0.5,
        rl_w_precision=0.25,
        rl_w_recall=0.25,
        rl_num_clients_to_select_fallback=5,
        rl_prob_threshold_factor=1.2,
        rl_resource_cost_factor=0.005,
        rl_ablation_mode="multi-criteria",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.client_attributes_store = client_attributes_store
        self.strategy_logger = strategy_logger
        self.num_rounds = num_rounds
        self.instance_log_dir = instance_log_dir
        self.rl_resource_cost_factor = rl_resource_cost_factor
        self.rl_ablation_mode = rl_ablation_mode

        self.rl_agent = RLAgent(
            client_attributes_store=self.client_attributes_store,
            learning_rate=rl_learning_rate,
            discount_factor=rl_discount_factor,
            temperature_initial=rl_temperature_initial,
            temperature_decay=rl_temperature_decay,
            temperature_min=rl_temperature_min,
            history_length=rl_history_length,
            recency_window=rl_recency_window,
            fairness_penalty_factor=rl_fairness_penalty_factor,
            fairness_participation_threshold_factor=rl_fairness_participation_threshold_factor,
            w_f1=rl_w_f1,
            w_precision=rl_w_precision,
            w_recall=rl_w_recall,
            passed_logger=self.strategy_logger,
            rl_ablation_mode=self.rl_ablation_mode,
        )
        self.rl_num_clients_to_select_fallback = rl_num_clients_to_select_fallback
        self.rl_prob_threshold_factor = rl_prob_threshold_factor

        self.last_global_f1: float = 0.0
        self.last_global_precision: float = 0.0
        self.last_global_recall: float = 0.0
        self.max_global_f1_achieved_so_far: float = 0.0

        self.current_round_rl_selected_client_states: Dict[
            str, Optional[tuple]
        ] = {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if results:
            first_client_proxy, first_eval_res = results[0]
            self.strategy_logger.info(f"  Sample of first result (CID {first_client_proxy.cid}): Metrics: {first_eval_res.metrics}")
        if failures:
            first_failure = failures[0]
            if isinstance(first_failure, BaseException):
                self.strategy_logger.info(f"  Sample of first failure (Exception): {str(first_failure)}")
            else:
                failed_client_proxy, failed_eval_res = first_failure
                self.strategy_logger.info(f"  Sample of first failure (Client {failed_client_proxy.cid}): Status: {failed_eval_res.status}, Metrics: {failed_eval_res.metrics}")

        if not results:
            self.strategy_logger.info(f"[Round {server_round}, AggrEval] No results to aggregate for CLIENT_ATTRIBUTES update.")
        else:
            self.strategy_logger.info(f"[Round {server_round}, AggrEval] Received {len(results)} evaluation results. Updating CLIENT_ATTRIBUTES with historical data.")
            for client_proxy, eval_res in results:
                proxy_cid = client_proxy.cid
                metrics = eval_res.metrics
                num_eval_examples = eval_res.num_examples

                if proxy_cid not in self.client_attributes_store:
                    self.client_attributes_store[proxy_cid] = {
                        'partition_id': metrics.get("client_cid_for_attributes", "unknown_partition"),
                        'cores': int(metrics.get("allocated_cores_for_pytorch", 0)),
                        'fit_rounds_participated': [],
                        'fit_train_loss_history': [],
                        'fit_g_loss_history': [],
                        'fit_d_loss_history': [],
                        'fit_time_seconds_history': [],
                        'fit_cpu_percent_history': [],
                        'fit_ram_mb_history': [],
                        'fit_accumulated_train_samples_history': [],
                        'eval_rounds_participated': [],
                        'eval_avg_anomaly_score_history': [],
                        'eval_accuracy_history': [],
                        'eval_f1_score_history': [],
                        'eval_precision_history': [],
                        'eval_recall_history': [],
                        'eval_time_seconds_history': [],
                        'eval_num_samples_history': []
                    }
                    self.strategy_logger.info(f"  Initialized entry for new proxy_cid {proxy_cid} with partition_id: {self.client_attributes_store[proxy_cid]['partition_id']} and cores: {self.client_attributes_store[proxy_cid]['cores']}.")
                
                client_entry = self.client_attributes_store[proxy_cid]

                client_entry['partition_id'] = metrics.get("client_cid_for_attributes", client_entry.get('partition_id', "unknown_partition"))
                client_entry['cores'] = int(metrics.get("allocated_cores_for_pytorch", client_entry.get('cores', 0)))
                client_entry['last_eval_round'] = server_round

                client_entry['eval_rounds_participated'].append(server_round)
                client_entry['eval_avg_anomaly_score_history'].append(metrics.get("avg_anomaly_score"))
                client_entry['eval_accuracy_history'].append(metrics.get("accuracy"))
                client_entry['eval_f1_score_history'].append(metrics.get("f1_score"))
                client_entry['eval_precision_history'].append(metrics.get("precision"))
                client_entry['eval_recall_history'].append(metrics.get("recall"))
                client_entry['eval_time_seconds_history'].append(metrics.get("eval_time_seconds"))
                client_entry['eval_num_samples_history'].append(num_eval_examples)

                self.strategy_logger.info(f"  Appended eval metrics for proxy_cid {proxy_cid} (partition_id: {client_entry['partition_id']}) for round {server_round}.")
            
            self.strategy_logger.debug(f"[Round {server_round}, AggrEval] CLIENT_ATTRIBUTES after update: {self.client_attributes_store}")
        
        if results:
            self.strategy_logger.info(f"[Round {server_round}, AggrEval] Logging detailed evaluation metrics for {len(results)} clients to CSV.")
            for client_proxy_log, eval_res_log in results:
                metrics_log = eval_res_log.metrics
                if metrics_log:
                    client_eval_log_entry = {
                        "server_round": server_round,
                        "event_type": "client_eval_metrics",
                        "client_cid": client_proxy_log.cid,
                        "c_eval_partition_id": metrics_log.get("client_cid_for_attributes"),
                        "c_eval_profile_name": metrics_log.get("profile_name"),
                        "c_eval_cores": metrics_log.get("allocated_cores_for_pytorch"),
                        "c_eval_f1": metrics_log.get("f1_score"),
                        "c_eval_accuracy": metrics_log.get("accuracy"),
                        "c_eval_precision": metrics_log.get("precision"),
                        "c_eval_recall": metrics_log.get("recall"),
                        "c_eval_num_samples": eval_res_log.num_examples,
                        "c_eval_time_seconds": metrics_log.get("eval_time_seconds")
                    }
                    log_rl_data_row(client_eval_log_entry)
                else:
                    self.strategy_logger.warning(f"  Skipping CSV log for client {client_proxy_log.cid}, no metrics in EvaluateRes.")

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if server_round > 2 and self.current_round_rl_selected_client_states:
            current_global_f1 = aggregated_metrics.get("global_f1_score", 0.0)
            current_global_precision = aggregated_metrics.get("global_precision", 0.0)
            current_global_recall = aggregated_metrics.get("global_recall", 0.0)
            prev_max_global_f1_for_reward = self.max_global_f1_achieved_so_far
            self.max_global_f1_achieved_so_far = max(
                self.max_global_f1_achieved_so_far, current_global_f1
            )
            f1_reward_component = (
                (current_global_f1 - prev_max_global_f1_for_reward) * 2.0 + 0.1
                if current_global_f1 > prev_max_global_f1_for_reward
                else (current_global_f1 - self.last_global_f1)
                * (0.5 if current_global_f1 >= self.last_global_f1 else 1.0)
            )

            performance_reward = (
                self.rl_agent.w_f1 * f1_reward_component
                + self.rl_agent.w_precision * (current_global_precision - self.last_global_precision)
                + self.rl_agent.w_recall * (current_global_recall - self.last_global_recall)
            )

            total_fairness_penalty_score = sum(
                max(0, sum(1 for r_fit in self.client_attributes_store[cid].get("fit_rounds_participated", []) if (server_round - 1) - r_fit < self.rl_agent.recency_window and r_fit < server_round) - self.rl_agent.fairness_participation_threshold)
                for cid in self.current_round_rl_selected_client_states.keys()
                if cid in self.client_attributes_store
            )
            actual_fairness_penalty = total_fairness_penalty_score * self.rl_agent.fairness_penalty_factor

            total_cores_selected_by_rl = sum(
                int(self.client_attributes_store.get(cid, {}).get("cores", 0))
                for cid in self.current_round_rl_selected_client_states.keys()
            )
            resource_cost_component = total_cores_selected_by_rl * self.rl_resource_cost_factor

            if self.rl_ablation_mode == "performance-only":
                final_global_reward = performance_reward
                self.strategy_logger.info(f"  [Ablation Mode: performance-only] Final reward set to performance component: {final_global_reward:.4f}")
            elif self.rl_ablation_mode == "resource-only":
                final_global_reward = -resource_cost_component
                self.strategy_logger.info(f"  [Ablation Mode: resource-only] Final reward set to the negative resource cost: {final_global_reward:.4f}")
            elif self.rl_ablation_mode == "fairness-only":
                final_global_reward = -actual_fairness_penalty
                self.strategy_logger.info(f"  [Ablation Mode: fairness-only] Final reward set to the negative fairness penalty: {final_global_reward:.4f}")
            else:
                final_global_reward = performance_reward - actual_fairness_penalty - resource_cost_component
                self.strategy_logger.info(f"  [Mode: multi-criteria] Final reward is Perf({performance_reward:.4f}) - FairPenalty({actual_fairness_penalty:.4f}) - ResourceCost({resource_cost_component:.4f}) = {final_global_reward:.4f}")

            self.rl_agent.learn_from_round_outcome(
                client_states_at_selection=self.current_round_rl_selected_client_states,
                reward_performance_component=performance_reward,
                reward_fairness_penalty_component=actual_fairness_penalty,
                reward_resource_cost_component=resource_cost_component,
                final_global_reward_for_action=final_global_reward,
                next_round_global_f1=current_global_f1,
                current_round_for_next_state_calc=server_round,
                learn_context_last_global_f1=self.last_global_f1,
                learn_context_last_global_precision=self.last_global_precision,
                learn_context_last_global_recall=self.last_global_recall,
                learn_context_max_global_f1=self.max_global_f1_achieved_so_far,
            )

        self.last_global_f1 = aggregated_metrics.get(
            "global_f1_score", self.last_global_f1
        )
        self.last_global_precision = aggregated_metrics.get(
            "global_precision", self.last_global_precision
        )
        self.last_global_recall = aggregated_metrics.get(
            "global_recall", self.last_global_recall
        )
        self.current_round_rl_selected_client_states = {}

        if aggregated_metrics:
            log_rl_data_row({
                "server_round": server_round,
                "event_type": "global_eval_metrics",
                "gm_eval_avg_anomaly_score": aggregated_metrics.get("global_avg_anomaly_score"),
                "gm_eval_accuracy": aggregated_metrics.get("global_accuracy"),
                "gm_eval_f1": aggregated_metrics.get("global_f1_score"),
                "gm_eval_precision": aggregated_metrics.get("global_precision"),
                "gm_eval_recall": aggregated_metrics.get("global_recall"),
                "gm_eval_avg_client_time_seconds": aggregated_metrics.get("eval_time_avg_seconds_per_client"),
                "gm_eval_clients_attempted": aggregated_metrics.get("evaluate_clients_attempted_processing"),
                "gm_eval_clients_valid_metrics": aggregated_metrics.get("evaluate_clients_reported_valid_perf_metrics"),
            })

        return aggregated_loss, aggregated_metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.strategy_logger.info(
            f"[Round {server_round}, AggrFit START] Called. "
            f"Received {len(results)} results, {len(failures)} failures."
        )
        if results:
            for client_proxy, fit_res in results:
                metrics_log = fit_res.metrics
                if metrics_log:
                    log_rl_data_row({
                        "server_round": server_round,
                        "event_type": "client_fit_metrics",
                        "client_cid": client_proxy.cid,
                        "c_fit_partition_id": metrics_log.get("client_cid_for_attributes"),
                        "c_fit_cpu_percent": metrics_log.get("cpu_percent_fit"),
                        "c_fit_ram_mb": metrics_log.get("ram_mb_fit"),
                        "c_fit_time_seconds": metrics_log.get("fit_time_seconds"),
                        "c_fit_post_local_f1": metrics_log.get("post_fit_local_f1"),
                        "c_fit_g_loss": metrics_log.get("g_loss"),
                        "c_fit_d_loss": metrics_log.get("d_loss"),
                    })

            if all(fit_res.num_examples == 0 for _, fit_res in results):
                for client_proxy, fit_res in results:
                    proxy_cid = client_proxy.cid
                    metrics = fit_res.metrics
                    if proxy_cid not in self.client_attributes_store:
                        self.client_attributes_store[proxy_cid] = {
                            "partition_id": metrics.get("client_cid_for_attributes", "unknown_partition"),
                            "cores": int(metrics.get("allocated_cores_for_pytorch", 0)),
                            "fit_rounds_participated": [], "fit_g_loss_history": [], "fit_d_loss_history": [], "fit_time_seconds_history": [],
                            "fit_cpu_percent_history": [], "fit_ram_mb_history": [], "fit_accumulated_train_samples_history": [],
                            "eval_rounds_participated": [], "eval_avg_anomaly_score_history": [], "eval_accuracy_history": [],
                            "eval_f1_score_history": [], "eval_precision_history": [], "eval_recall_history": [],
                            "eval_time_seconds_history": [], "eval_num_samples_history": [],
                        }
                    
                    client_entry = self.client_attributes_store[proxy_cid]
                    client_entry["last_fit_round"] = server_round
                    client_entry["fit_rounds_participated"].append(server_round)
                    client_entry["fit_g_loss_history"].append(metrics.get("g_loss"))
                    client_entry["fit_d_loss_history"].append(metrics.get("d_loss"))
                    client_entry["fit_time_seconds_history"].append(metrics.get("fit_time_seconds"))
                    client_entry["fit_cpu_percent_history"].append(metrics.get("cpu_percent_fit"))
                    client_entry["fit_ram_mb_history"].append(metrics.get("ram_mb_fit"))
                    client_entry["fit_accumulated_train_samples_history"].append(fit_res.num_examples)
                
                aggregated_fit_metrics = {}
                if self.fit_metrics_aggregation_fn:
                    fit_res_metrics = [(res.num_examples, res.metrics) for _, res in results]
                    aggregated_fit_metrics = self.fit_metrics_aggregation_fn(fit_res_metrics)
                return None, aggregated_fit_metrics

            for client_proxy, fit_res in results:
                proxy_cid, metrics, num_fit_examples = client_proxy.cid, fit_res.metrics, fit_res.num_examples
                if proxy_cid not in self.client_attributes_store:
                    self.client_attributes_store[proxy_cid] = {
                        'partition_id': metrics.get("client_cid_for_attributes", "unknown_partition"),
                        'cores': int(metrics.get("allocated_cores_for_pytorch", 0)),
                        'fit_rounds_participated': [],
                        'fit_train_loss_history': [],
                        'fit_g_loss_history': [],
                        'fit_d_loss_history': [],
                        'fit_time_seconds_history': [],
                        'fit_cpu_percent_history': [],
                        'fit_ram_mb_history': [],
                        'fit_accumulated_train_samples_history': [],
                        'eval_rounds_participated': [],
                        'eval_avg_anomaly_score_history': [],
                        'eval_accuracy_history': [],
                        'eval_f1_score_history': [],
                        'eval_precision_history': [],
                        'eval_recall_history': [],
                        'eval_time_seconds_history': [],
                        'eval_num_samples_history': []
                    }
                    self.strategy_logger.info(f"  Initialized entry for new proxy_cid {proxy_cid} during AggrFit with partition_id: {self.client_attributes_store[proxy_cid]['partition_id']} and cores: {self.client_attributes_store[proxy_cid]['cores']}.")
                
                client_entry = self.client_attributes_store[proxy_cid]
                client_entry.update({
                    "partition_id": metrics.get("client_cid_for_attributes", client_entry.get("partition_id", "unknown_partition")),
                    "cores": int(metrics.get("allocated_cores_for_pytorch", client_entry.get("cores", 0))),
                    "last_fit_round": server_round,
                })
                for key, value in {
                    "fit_rounds_participated": server_round,
                    "fit_g_loss_history": metrics.get("g_loss"),
                    "fit_d_loss_history": metrics.get("d_loss"),
                    "fit_time_seconds_history": metrics.get("fit_time_seconds"),
                    "fit_cpu_percent_history": metrics.get("cpu_percent_fit"),
                    "fit_ram_mb_history": metrics.get("ram_mb_fit"),
                    "fit_accumulated_train_samples_history": num_fit_examples,
                }.items():
                    client_entry[key].append(value)

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        if server_round == self.num_rounds and parameters_aggregated:
            ndarrays = parameters_to_ndarrays(parameters_aggregated)
            np.savez(f"{self.instance_log_dir}/final_model_parameters.npz", *ndarrays)

        if metrics_aggregated:
            log_rl_data_row({
                "server_round": server_round,
                "event_type": "global_fit_metrics",
                "gm_fit_g_loss": metrics_aggregated.get("aggregated_g_loss"),
                "gm_fit_d_loss": metrics_aggregated.get("aggregated_d_loss"),
                "gm_fit_avg_client_time_seconds": metrics_aggregated.get("fit_time_avg_seconds_all_reporting"),
                "gm_fit_clients_total_received": metrics_aggregated.get("fit_clients_total_received"),
                "gm_fit_clients_valid_for_loss": metrics_aggregated.get("fit_clients_valid_for_loss_aggregation"),
            })

        return parameters_aggregated, metrics_aggregated

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        fit_ins = FitIns(parameters, self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {})
        available_clients_map = client_manager.all()
        if not available_clients_map:
            return []

        selected_clients_for_fit: List[ClientProxy] = []

        if server_round == 1:
            self.strategy_logger.info(f"--- [Round {server_round}, Fit] GHOST ROUND for attribute collection --- ")
            self.strategy_logger.info(f"  No clients will be selected for FIT in this round. Evaluation at the end of this round will populate attributes.")
            self.strategy_logger.info(f"--- [Round {server_round}, Fit] GHOST ROUND COMPLETE. Expected 0 clients for FIT. --- ")
            return [] 

        elif server_round == 2:
            if self.client_attributes_store:
                max_found_cores = max(
                    (int(attrs.get("cores", 0)) for attrs in self.client_attributes_store.values()),
                    default=0,
                )
                if max_found_cores > 0:
                    selection_threshold = math.floor(max_found_cores * 0.75)
                    selected_clients_for_fit = [
                        available_clients_map[cid]
                        for cid, attrs in self.client_attributes_store.items()
                        if cid in available_clients_map
                        and int(attrs.get("cores", 0)) >= selection_threshold
                    ]
        else:
            selected_proxy_cids, selected_client_states = self.rl_agent.choose_actions_softmax(
                available_proxy_cids=list(available_clients_map.keys()),
                current_round=server_round,
                global_f1_prev_round=self.last_global_f1,
                max_global_f1_achieved=self.max_global_f1_achieved_so_far,
                num_clients_to_select=self.rl_num_clients_to_select_fallback,
                probability_threshold_factor=self.rl_prob_threshold_factor,
            )
            self.current_round_rl_selected_client_states = selected_client_states
            selected_clients_for_fit = [
                available_clients_map[cid] for cid in selected_proxy_cids if cid in available_clients_map
            ]

        if len(selected_clients_for_fit) < self.min_fit_clients:
            self.strategy_logger.warning(
                f"Selected clients ({len(selected_clients_for_fit)}) is less than min_fit_clients ({self.min_fit_clients})."
            )

        return [(client, fit_ins) for client in selected_clients_for_fit]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        eval_ins = EvaluateIns(parameters, self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {})
        all_available_clients_map = client_manager.all()
        if not all_available_clients_map:
            return []
        
        clients_to_evaluate = list(all_available_clients_map.values())
        if server_round > 0 and len(clients_to_evaluate) < self.min_evaluate_clients:
            return []

        if len(clients_to_evaluate) < self.min_evaluate_clients:
            self.strategy_logger.warning(
                f"Selected clients for evaluation ({len(clients_to_evaluate)}) is less than min_evaluate_clients ({self.min_evaluate_clients})."
            )
        return [(client, eval_ins) for client in clients_to_evaluate] 