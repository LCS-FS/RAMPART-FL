import random
import math
import logging
from collections import defaultdict
import numpy as np # Ensure numpy is imported
from src.rl_data_logger import log_rl_data_row # Import for CSV logging

class RLAgent:
    def __init__(self, client_attributes_store, 
                learning_rate=0.1, discount_factor=0.9,
                temperature_initial=1.0, temperature_decay=0.995, temperature_min=0.01, # For softmax
                history_length=3, 
                recency_window=5,
                fairness_penalty_factor=0.1, # Penalty factor for over-selection
                fairness_participation_threshold_factor=0.5, # Factor of recency_window
                w_f1=0.5, w_precision=0.25, w_recall=0.25, # Reward weights
                passed_logger=None, # Optional logger passed from strategy
                optimistic_q_init_value=0.05, # Value for optimistic initialization for high-core clients
                rl_ablation_mode="multi-criteria" # To be passed from strategy
                ):
        self.client_attributes_store = client_attributes_store  # Reference
        self.lr = learning_rate
        self.gamma = discount_factor
        
        self.temperature = temperature_initial
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min
        
        self.history_length = history_length
        self.recency_window = recency_window
        self.fairness_penalty_factor = fairness_penalty_factor
        self.fairness_participation_threshold = math.floor(recency_window * fairness_participation_threshold_factor)

        self.w_f1 = w_f1
        self.w_precision = w_precision
        self.w_recall = w_recall
        self.rl_ablation_mode = rl_ablation_mode

        # Q-table: dict where keys are state tuples (derived from client features), 
        # and values are the Q-values (expected utility of selecting that client given its state).
        self.q_table = defaultdict(float)
        
        if passed_logger:
            self.agent_logger = passed_logger
        else:
            self.agent_logger = logging.getLogger(f"RLAgent:{id(self)}") # Unique logger per agent instance
            # Configure logger if it doesn't have handlers (e.g. if not configured by strategy)
            if not self.agent_logger.hasHandlers():
                handler = logging.StreamHandler() # Default to console output
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.agent_logger.addHandler(handler)
                self.agent_logger.setLevel(logging.INFO) # Default level, can be overridden
        self.agent_logger.info(f"RLAgent initialized with: lr={learning_rate}, gamma={discount_factor}, temp_init={temperature_initial}, hist_len={history_length}, fair_thresh={self.fairness_participation_threshold}")
        self.optimistic_q_init_value = optimistic_q_init_value # Value for optimistic initialization for high-core clients

    def _get_client_features(self, proxy_cid, current_round, global_f1_prev_round, max_global_f1_achieved):
        """
        Extracts and discretizes features for a single client to form a state tuple.
        Args:
            proxy_cid: The client's proxy CID.
            current_round: The current server round.
            global_f1_prev_round: Global F1 score from the previous round.
            max_global_f1_achieved: Maximum global F1 score achieved so far in the simulation.
        Returns:
            A tuple representing the discretized state for the client, or None if client data not found.
        """
        # self.agent_logger.debug(f"[RLAgent _get_client_features] For CID {proxy_cid}, Round {current_round}")

        if proxy_cid not in self.client_attributes_store:
            self.agent_logger.debug(f"Client {proxy_cid} not in attributes store for feature extraction.")
            return None # Or a default "unknown" state tuple

        client_data = self.client_attributes_store[proxy_cid]
        
        # 1. Average F1 score from last N eval participations
        eval_f1_history = client_data.get('eval_f1_score_history', [])
        # Filter out None values that might have been appended if a metric was missing
        recent_f1_scores = [f for f in eval_f1_history if f is not None][-self.history_length:]
        avg_f1 = sum(recent_f1_scores) / len(recent_f1_scores) if recent_f1_scores else 0.0

        # 2. Number of times selected for fit in the last M rounds
        fit_rounds_participated = client_data.get('fit_rounds_participated', [])
        # Calculate participation for rounds strictly *before* the current round being decided
        recent_fit_participations = sum(1 for r_fit in fit_rounds_participated if (current_round - 1) - r_fit < self.recency_window and r_fit < current_round)
        
        # 3. Core count
        cores = client_data.get('cores', 0)

        # Discretize features
        f1_bin = math.floor(avg_f1 * 10) # Bin into 10 segments (0-9)
        participation_bin = min(recent_fit_participations, self.recency_window) # Max value is recency_window
        # Core binning: Bin 0: <=2 cores, Bin 1: 3-4 cores, Bin 2: >=5 cores
        if isinstance(cores, (int, float)):
            if cores <= 2:
                cores_bin = 0
            elif cores <= 4:
                cores_bin = 1
            else:
                cores_bin = 2 # High core bin
        else:
            cores_bin = 0
        global_f1_bin = math.floor(global_f1_prev_round * 10) # Bin into 10 segments
        max_f1_bin = math.floor(max_global_f1_achieved * 10) # Bin max_f1 into 10 segments

        state_tuple = (f1_bin, participation_bin, cores_bin, global_f1_bin, max_f1_bin)
        
        # Optimistic Q-value initialization for new, high-core states
        if state_tuple not in self.q_table and cores_bin == 2: # Check if state is new and high-core
            self.q_table[state_tuple] = self.optimistic_q_init_value
            self.agent_logger.info(f"  Optimistic init for new high-core state {state_tuple} (CID {proxy_cid}): Q set to {self.optimistic_q_init_value}")
        # For other new states, defaultdict(float) will initialize to 0.0 automatically on first access in get_q_value

        # self.agent_logger.debug(f"Client {proxy_cid} (part_id: {client_data.get('partition_id', 'N/A')}): Raw features for state calc (round {current_round}): avg_f1={avg_f1:.2f} (hist: {recent_f1_scores}), recent_fits_before_this_round={recent_fit_participations} (hist: {fit_rounds_participated}), cores={cores}. Prev Global F1: {global_f1_prev_round:.2f}. Discretized state: {state_tuple}")
        return state_tuple

    def get_q_value(self, client_state_tuple):
        """Gets the Q-value for a given client state tuple."""
        q_val = self.q_table[client_state_tuple]
        # self.agent_logger.debug(f"[RLAgent get_q_value] State: {client_state_tuple}, Q-value: {q_val:.4f}")
        return q_val

    def choose_actions_softmax(self, available_proxy_cids, current_round, global_f1_prev_round, 
                            max_global_f1_achieved, # Added parameter
                            num_clients_to_select=None, probability_threshold_factor=1.2):
        """
        Chooses a dynamic subset of clients using softmax selection over Q-values.
        Clients are selected if their probability > (1/num_available) * probability_threshold_factor.
        If num_clients_to_select is specified, it will take top-k if threshold yields too many/few.
        (Currently, dynamic selection by threshold is primary, num_clients_to_select is a fallback/cap)
        """
        self.agent_logger.info(f"[RLAgent choose_actions_softmax] Round {current_round}. Temperature: {self.temperature:.4f}")
        self.agent_logger.info(f"  Available CIDs: {available_proxy_cids}")
        if not available_proxy_cids:
            self.agent_logger.info("  No CIDs available, returning empty selection.")
            return [], {}

        client_q_values = {}
        client_states = {}
        for cid in available_proxy_cids:
            state = self._get_client_features(cid, current_round, global_f1_prev_round, max_global_f1_achieved)
            client_states[cid] = state
            client_q_values[cid] = self.get_q_value(state)
        
        self.agent_logger.info(f"  Calculated Q-values for available clients: {client_q_values}")

        # --- Tie-breaking for ablation modes ---
        q_values_array = np.array(list(client_q_values.values()))
        # Check for tie condition: more than one client, and all Q-values are very close
        if len(q_values_array) > 1 and np.allclose(q_values_array, q_values_array[0]):
            self.agent_logger.info(f"  All Q-values are nearly identical. Applying tie-breaking for mode: '{self.rl_ablation_mode}'.")
            
            TIE_BREAKING_FACTOR = 1e-5 # A small constant to nudge probabilities
            bonuses = {}

            if self.rl_ablation_mode == "fairness-only":
                self.agent_logger.info("    Tie-breaking with recency bonus (less recent is better).")
                for cid in client_q_values.keys():
                    if cid in self.client_attributes_store:
                        client_data = self.client_attributes_store[cid]
                        fit_rounds = client_data.get('fit_rounds_participated', [])
                        last_fit_round = max(fit_rounds) if fit_rounds else 0
                        # Higher bonus for clients that haven't participated for longer
                        bonus = (current_round - last_fit_round) * TIE_BREAKING_FACTOR
                        client_q_values[cid] += bonus
                        bonuses[cid] = bonus

            elif self.rl_ablation_mode == "resource-only":
                self.agent_logger.info("    Tie-breaking with resource bonus (fewer cores is better).")
                for cid in client_q_values.keys():
                     if cid in self.client_attributes_store:
                        client_data = self.client_attributes_store[cid]
                        cores = client_data.get('cores', 1) # Default to 1 to avoid division by zero
                        # Higher bonus for clients with fewer cores
                        bonus = (1 / (cores + 1)) * TIE_BREAKING_FACTOR 
                        client_q_values[cid] += bonus
                        bonuses[cid] = bonus
            
            else:
                 self.agent_logger.info(f"    No specific tie-breaker for current mode ('{self.rl_ablation_mode}'). Skipping bonus application.")

            if bonuses:
                self.agent_logger.info(f"    Applied tie-breaking bonuses: {bonuses}")
                self.agent_logger.info(f"  Q-values after tie-breaking: {client_q_values}")


        # Softmax calculation
        if not client_q_values:
            self.agent_logger.info("  No Q-values calculated (e.g. no clients), returning empty selection.")
            return [], {}

        q_values_list = np.array(list(client_q_values.values())) # Use np.array
        # Adding a small epsilon for numerical stability if temperature is very low or all q_values are same
        # Corrected softmax implementation using numpy for vectorized operations
        temp_adjusted_q_values = (q_values_list - np.max(q_values_list)) / (self.temperature + 1e-6)
        exp_q_values = np.exp(temp_adjusted_q_values)
        probabilities = exp_q_values / np.sum(exp_q_values)
        
        client_probs = {cid: prob for cid, prob in zip(client_q_values.keys(), probabilities)}
        self.agent_logger.info(f"  Selection probabilities: {client_probs}")

        selected_cids_for_current_factor = []
        num_available = len(available_proxy_cids)
        current_probability_threshold_factor = probability_threshold_factor # Start with the configured factor
        min_probability_threshold_factor = 0.1 # Minimum sensible factor
        factor_reduction_step = 0.05
        selection_threshold_value = 0 # For logging
        iteration_count = 0
        # Max iterations: how many steps to get from initial factor to min_factor
        max_iterations = int((current_probability_threshold_factor - min_probability_threshold_factor) / factor_reduction_step) + 2 # +2 for initial try and final min_factor try

        if num_available > 0:
            while iteration_count < max_iterations:
                iteration_count += 1
                selection_threshold_value = (1.0 / num_available) * current_probability_threshold_factor
                self.agent_logger.info(f"  Attempt {iteration_count} with factor {current_probability_threshold_factor:.2f}. Threshold value: {selection_threshold_value:.4f}")
                
                current_selection_attempt = []
                for cid, prob in client_probs.items():
                    if prob >= selection_threshold_value:
                        current_selection_attempt.append(cid)
                self.agent_logger.info(f"    Selected {len(current_selection_attempt)} CIDs with this factor: {current_selection_attempt}")

                if current_selection_attempt: # If we found at least one client with this factor
                    selected_cids_for_current_factor = current_selection_attempt
                    break # Found a working factor, proceed with this selection
                
                # If no clients selected and factor can be reduced further
                if current_probability_threshold_factor <= min_probability_threshold_factor:
                    self.agent_logger.info(f"    Factor at minimum {min_probability_threshold_factor:.2f}, and still no clients selected. Stopping factor reduction.")
                    break # Stop if we are at or below the minimum factor
                
                current_probability_threshold_factor = max(min_probability_threshold_factor, current_probability_threshold_factor - factor_reduction_step)
                self.agent_logger.info(f"    No clients selected, reducing factor for next attempt to {current_probability_threshold_factor:.2f}.")
            
            if not selected_cids_for_current_factor:
                self.agent_logger.warning(f"  Adaptive threshold factor reduction completed. No clients selected by probability threshold (final factor tried: {current_probability_threshold_factor:.2f}).")

        # Capping Logic / Fallback to top N if thresholding yielded nothing AND num_clients_to_select is specified
        final_selected_cids = []
        if selected_cids_for_current_factor: # If adaptive thresholding found clients
            if num_clients_to_select is not None and len(selected_cids_for_current_factor) > num_clients_to_select:
                self.agent_logger.info(f"  Selection by factor {current_probability_threshold_factor:.2f} yielded {len(selected_cids_for_current_factor)} clients, which is > cap ({num_clients_to_select}). Capping to top-{num_clients_to_select} by probability.")
                # Sort the clients selected by the chosen factor, then cap
                probs_of_selected_by_factor = {cid: client_probs[cid] for cid in selected_cids_for_current_factor}
                sorted_selected_clients = sorted(probs_of_selected_by_factor.items(), key=lambda item: item[1], reverse=True)
                final_selected_cids = [cid for cid, prob in sorted_selected_clients[:num_clients_to_select]]
            else:
                final_selected_cids = selected_cids_for_current_factor # Use all clients found by the chosen factor
        elif num_clients_to_select is not None: # If adaptive thresholding found NO clients, AND a fallback number is specified
             self.agent_logger.info(f"  Adaptive thresholding yielded 0 clients. Using fallback to select top-{num_clients_to_select} by probability if available.")
             sorted_clients_by_prob = sorted(client_probs.items(), key=lambda item: item[1], reverse=True)
             final_selected_cids = [cid for cid, prob in sorted_clients_by_prob[:num_clients_to_select]]
        
        self.agent_logger.info(f"  Final selected CIDs after adaptive thresholding & capping: {final_selected_cids}")

        # --- Log selection info to CSV for all available clients ---
        for cid_log in available_proxy_cids:
            log_entry = {
                "server_round": current_round,
                "event_type": "selection_info",
                "client_cid": cid_log,
                "s_available_cids_count": num_available,
                "s_client_state_tuple": str(client_states.get(cid_log)),
                "s_client_q_value": client_q_values.get(cid_log),
                "s_client_selection_prob": client_probs.get(cid_log),
                "s_was_selected": 1 if cid_log in final_selected_cids else 0, # Use final_selected_cids
                "s_softmax_temperature": self.temperature, # Log temperature at time of selection
                "s_selection_context_global_f1_prev_round": global_f1_prev_round,
                "s_selection_context_max_global_f1_achieved": max_global_f1_achieved # Log max F1
            }
            log_rl_data_row(log_entry)
        # --- End CSV logging for selection ---

        # Decay temperature
        self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
        self.agent_logger.info(f"  New temperature for next round: {self.temperature:.4f}")
        
        final_selected_client_states = {cid: client_states[cid] for cid in final_selected_cids if cid in client_states} # Use final_selected_cids
        self.agent_logger.info(f"  Final selected CIDs to return: {final_selected_cids}. Corresponding states: {final_selected_client_states}")
        return final_selected_cids, final_selected_client_states

    def learn_from_round_outcome(self, client_states_at_selection, 
                                # Reward components
                                reward_performance_component, 
                                reward_fairness_penalty_component, 
                                reward_resource_cost_component, # New parameter
                                final_global_reward_for_action,
                                # Context for next state Q value
                                next_round_global_f1, 
                                current_round_for_next_state_calc, 
                                # Additional context for logging
                                learn_context_last_global_f1,
                                learn_context_last_global_precision,
                                learn_context_last_global_recall,
                                learn_context_max_global_f1 # Already passed from strategy for logging
                                ):
        """Update Q-values for selected clients based on the global reward and next state."""
        self.agent_logger.info(f"[RLAgent learn_from_round_outcome] Learning from outcome of round (completed) {current_round_for_next_state_calc}.")
        self.agent_logger.info(f"  Final Global Reward for action: {final_global_reward_for_action:.4f}. Client states at selection: {client_states_at_selection}")
        self.agent_logger.info(f"  Performance Reward: {reward_performance_component:.4f}, Fairness Penalty: {reward_fairness_penalty_component:.4f}, Resource Cost: {reward_resource_cost_component:.4f}")
        self.agent_logger.info(f"  Next round's global F1 (for next state Q-val): {next_round_global_f1:.4f}")

        if not client_states_at_selection:
            self.agent_logger.info("  No client states provided, skipping learning.")
            return

        for proxy_cid, state_at_selection in client_states_at_selection.items():
            if state_at_selection is None: # Should not happen if logic is correct
                self.agent_logger.warning(f"  Skipping client {proxy_cid} as its state_at_selection is None.")
                continue

            # Determine the client's state in the *next* round (s')
            # The next state is based on metrics *after* the current_round_for_next_state_calc has completed.
            # The global_f1_prev_round for this next_state calculation is the global_f1 achieved in current_round_for_next_state_calc.
            next_state_for_client = self._get_client_features(
                proxy_cid=proxy_cid, 
                current_round=current_round_for_next_state_calc + 1, # Simulating features for the start of the next logical round
                global_f1_prev_round=next_round_global_f1, # This is crucial: global F1 from the round that just completed
                max_global_f1_achieved=learn_context_max_global_f1 # Use the max F1 known at the end of the learning round for next state context
            )
            # self.agent_logger.debug(f"    Client {proxy_cid}: State at selection (s): {state_at_selection}")
            # self.agent_logger.debug(f"    Client {proxy_cid}: Calculated next state (s'): {next_state_for_client}")

            # Q-learning update rule
            current_q = self.get_q_value(state_at_selection)
            next_max_q_for_client = self.get_q_value(next_state_for_client) # Q(s', a') -> for this client, it's just Q(s') as action is implicit
            
            # Standard Q-learning: Q(s) = Q(s) + lr * (reward + gamma * max_a' Q(s',a') - Q(s))
            # Here, since we evaluate the state of a client if selected, max_a' Q(s',a') becomes Q(s') for that client.
            new_q = current_q + self.lr * (final_global_reward_for_action + self.gamma * next_max_q_for_client - current_q)
            self.q_table[state_at_selection] = new_q
            self.agent_logger.info(f"    Client {proxy_cid} (Partition ID: {self.client_attributes_store.get(proxy_cid, {}).get('partition_id', 'N/A')}): ")
            self.agent_logger.info(f"      State(s): {state_at_selection} -> NextState(s'): {next_state_for_client}")
            self.agent_logger.info(f"      Q(s) updated from {current_q:.4f} to {new_q:.4f} (TotalReward: {final_global_reward_for_action:.4f}, NextMaxQ: {next_max_q_for_client:.4f})")

            # --- Log learning update to CSV ---
            log_entry = {
                "server_round": current_round_for_next_state_calc, # The round that just completed
                "event_type": "learning_update",
                "client_cid": proxy_cid,
                "l_state_at_selection": str(state_at_selection),
                "l_reward_performance_component": reward_performance_component,
                "l_reward_fairness_penalty_component": reward_fairness_penalty_component,
                "l_reward_resource_cost_component": reward_resource_cost_component, # Log new component
                "l_reward_total_global_for_action": final_global_reward_for_action,
                "l_q_current_value_S": current_q,
                "l_q_next_max_value_S_prime": next_max_q_for_client,
                "l_q_updated_value_S": new_q,
                "l_rl_learning_rate": self.lr,
                "l_rl_discount_factor": self.gamma,
                "l_learn_context_current_global_f1": next_round_global_f1, # F1 of round that just completed
                "l_learn_context_last_global_f1": learn_context_last_global_f1,
                "l_learn_context_last_global_precision": learn_context_last_global_precision,
                "l_learn_context_last_global_recall": learn_context_last_global_recall,
                "l_learn_context_max_global_f1": learn_context_max_global_f1 # Log max F1
            }
            log_rl_data_row(log_entry)
            # --- End CSV logging for learning update ---

        self.agent_logger.info(f"[RLAgent learn_from_round_outcome] Q-table after updates (showing first 5 items for brevity if large): {dict(list(self.q_table.items())[:5])}")

    # TODO: Methods for saving/loading Q-table
    # def save_q_table(self, filepath="q_table.pkl"):
    #     import pickle
    #     with open(filepath, 'wb') as f:
    #         pickle.dump(dict(self.q_table), f) # Store as dict for broader compatibility
    #     self.logger.info(f"Q-table saved to {filepath}")

    # def load_q_table(self, filepath="q_table.pkl"):
    #     import pickle
    #     import os
    #     if os.path.exists(filepath):
    #         with open(filepath, 'rb') as f:
    #             self.q_table = defaultdict(float, pickle.load(f))
    #         self.logger.info(f"Q-table loaded from {filepath}")
    #     else:
    #         self.logger.info(f"No Q-table found at {filepath}, starting with an empty one.") 