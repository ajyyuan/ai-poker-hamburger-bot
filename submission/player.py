import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from treys import Evaluator
from gym_env import PokerEnv  # int_to_card is defined within PokerEnv
from agents.agent import Agent

# Access the enum and card conversion from PokerEnv
action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

# Constants
TOTAL_MATCH_HANDS = 1000
AVERAGE_FORCED_LOSS = 3  # Average stack difference lost per hand when auto-folding

# Define a simple feedforward network as our policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class PlayerAgent(Agent):
    def __init__(self, stream: bool = False):
        super().__init__(stream)
        self.evaluator = Evaluator()
        # Track opponent's actions
        self.opp_action_counts = {"raise": 0, "call": 0, "check": 0, "discard": 0, "fold": 0}
        self.last_opp_bet = 0
        self.has_discarded = False

        # Net advantage: (our chips - opponent's chips)
        self.total_reward = 0

        # Hand counter and fingerprinting to detect new hands
        self.hand_count = 0
        self.last_hand_fingerprint = None

        # --- RL Module: Contextual Bandit via Neural Network ---
        # Candidate multipliers (e.g., 0.8 => threshold = 0.65*0.8 = 0.52)
        self.strategy_candidates = [0.8, 0.85, 0.9, 0.95, 1.0]
        self.num_candidates = len(self.strategy_candidates)
        # Input features: [equity, pot_odds, opp_aggr, normalized_hand_count, normalized_total_reward]
        self.input_dim = 5  
        self.policy_net = PolicyNetwork(self.input_dim, self.num_candidates)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.epsilon = 0.2  # Exploration rate
        # Store the last features and chosen action index for training
        self.last_features = None
        self.last_action_index = None

    def update_hand_counter(self, observation):
        """
        Update the hand counter using a simple fingerprint (your hole cards).
        Called when street == 0.
        """
        if observation["street"] == 0:
            current_fingerprint = tuple(observation["my_cards"])
            if current_fingerprint != self.last_hand_fingerprint:
                self.last_hand_fingerprint = current_fingerprint
                self.hand_count += 1
                self.logger.debug(f"New hand detected. Hand count: {self.hand_count}")

    def reset_hand(self):
        """Reset per-hand flags."""
        self.last_opp_bet = 0
        self.has_discarded = False

    def compute_equity(self, observation, num_simulations=200):
        """
        Monte Carlo simulation to estimate the probability (0..1) that our final hand beats the opponent's.
        """
        my_cards = observation["my_cards"]
        community_cards = [c for c in observation["community_cards"] if c != -1]
        opp_known = []
        if observation["opp_discarded_card"] != -1:
            opp_known.append(observation["opp_discarded_card"])
        if observation["opp_drawn_card"] != -1:
            opp_known.append(observation["opp_drawn_card"])
        shown_cards = set(my_cards + community_cards + opp_known)
        deck = list(range(27))  # 27-card deck (no clubs, no face cards)
        non_shown_cards = [card for card in deck if card not in shown_cards]

        wins = 0
        for _ in range(num_simulations):
            opp_needed = 2 - len(opp_known)
            board_needed = 5 - len(community_cards)
            sample_size = opp_needed + board_needed
            if sample_size > len(non_shown_cards):
                continue
            sample = random.sample(non_shown_cards, sample_size)
            opp_cards = opp_known + sample[:opp_needed]
            board_sample = community_cards + sample[opp_needed:]
            my_hand = [int_to_card(card) for card in my_cards]
            opp_hand = [int_to_card(card) for card in opp_cards]
            board = [int_to_card(card) for card in board_sample]
            my_rank = self.evaluator.evaluate(my_hand, board)
            opp_rank = self.evaluator.evaluate(opp_hand, board)
            if my_rank < opp_rank:  # lower rank is better
                wins += 1
        return wins / num_simulations if num_simulations > 0 else 0.0

    def update_opponent_model(self, observation):
        """Track opponent actions."""
        current_opp_bet = observation["opp_bet"]
        diff = current_opp_bet - self.last_opp_bet
        if diff > 0:
            if diff >= observation["min_raise"]:
                self.opp_action_counts["raise"] += 1
            else:
                self.opp_action_counts["call"] += 1
        else:
            self.opp_action_counts["check"] += 1
        self.last_opp_bet = current_opp_bet
        if observation["opp_discarded_card"] != -1:
            self.opp_action_counts["discard"] += 1

    def get_opponent_aggressiveness(self):
        """Compute opponent aggression ratio."""
        raises = self.opp_action_counts.get("raise", 0)
        calls = self.opp_action_counts.get("call", 0)
        checks = self.opp_action_counts.get("check", 0)
        total = raises + calls + checks + 1e-6
        return raises / total

    def get_features(self, observation, equity, pot_odds, opp_aggr):
        """
        Construct a feature vector:
          - equity: estimated win probability
          - pot_odds: ratio of call cost to total pot
          - opp_aggr: opponent aggression ratio
          - normalized hand count (hand_count / TOTAL_MATCH_HANDS)
          - normalized total reward (total_reward / 100)
        """
        norm_hand = self.hand_count / TOTAL_MATCH_HANDS
        norm_reward = self.total_reward / 100.0
        features = np.array([equity, pot_odds, opp_aggr, norm_hand, norm_reward], dtype=np.float32)
        return features

    def select_strategy_nn(self, features):
        """
        Use the policy network with ε-greedy selection to choose a raise threshold multiplier.
        """
        state = torch.tensor(features).unsqueeze(0)  # shape: [1, input_dim]
        q_values = self.policy_net(state)  # shape: [1, num_candidates]
        q_values_np = q_values.detach().cpu().numpy().flatten()
        if random.random() < self.epsilon:
            action_index = random.randrange(self.num_candidates)
            self.logger.debug("Exploring strategy (NN)")
        else:
            action_index = int(np.argmax(q_values_np))
            self.logger.debug("Exploiting strategy (NN)")
        self.last_features = state  # Store state for update
        self.last_action_index = action_index
        chosen_multiplier = self.strategy_candidates[action_index]
        return chosen_multiplier

    def update_strategy_nn(self, reward):
        """
        After a hand, update the network by treating the observed reward as the target
        for the Q-value of the chosen action.
        """
        if self.last_features is None or self.last_action_index is None:
            return
        target = torch.tensor([reward], dtype=torch.float32)
        self.optimizer.zero_grad()
        q_values = self.policy_net(self.last_features)
        q_value = q_values[0, self.last_action_index]
        loss = self.loss_fn(q_value, target)
        loss.backward()
        self.optimizer.step()
        # Decay exploration slightly.
        self.epsilon = max(0.05, self.epsilon * 0.99)
        self.logger.debug(f"Updated NN strategy: loss={loss.item():.4f}, new epsilon={self.epsilon:.4f}")
        self.last_features = None
        self.last_action_index = None

    def act(self, observation, reward, terminated, truncated, info):
        # Update hand counter
        self.update_hand_counter(observation)
        if observation["street"] == 0:
            self.reset_hand()
            remaining_hands = TOTAL_MATCH_HANDS - self.hand_count
            needed_for_safe_fold = remaining_hands * AVERAGE_FORCED_LOSS
            self.logger.debug(
                f"Hand {self.hand_count}, net_advantage={self.total_reward:.2f}, needed={needed_for_safe_fold:.2f}"
            )
            if self.total_reward > 0 and self.total_reward >= needed_for_safe_fold:
                self.logger.info(
                    f"Folding rest of match. net_advantage={self.total_reward:.2f}, needed={needed_for_safe_fold:.2f}"
                )
                return (action_types.FOLD.value, 0, -1)
        # Compute contextual features
        equity = self.compute_equity(observation)
        continue_cost = observation["opp_bet"] - observation["my_bet"]
        pot_size = observation["opp_bet"] + observation["my_bet"]
        pot_odds = continue_cost / (pot_size + continue_cost) if continue_cost > 0 else 0.0
        opp_aggr = self.get_opponent_aggressiveness()
        features = self.get_features(observation, equity, pot_odds, opp_aggr)
        # Select raise threshold multiplier using the NN-based strategy
        chosen_multiplier = self.select_strategy_nn(features)
        normal_threshold = 0.65
        raise_threshold = normal_threshold * chosen_multiplier
        self.logger.debug(f"Using NN strategy multiplier: {chosen_multiplier} (raise_threshold={raise_threshold:.2f})")
        # Update opponent model and log state
        self.update_opponent_model(observation)
        self.logger.debug(
            f"Hand {self.hand_count}, Equity={equity:.2f}, PotOdds={pot_odds:.2f}, OppAgg={opp_aggr:.2f}"
        )
        valid = observation["valid_actions"]
        action_type = None
        raise_amount = 0
        card_to_discard = -1
        # Decision tree for action selection:
        if valid[action_types.RAISE.value] and equity > raise_threshold:
            action_type = action_types.RAISE.value
            factor = 0.75 if opp_aggr < 0.5 else 0.5
            base_raise = int(pot_size * factor)
            random_adjustment = random.randint(-2, 2)
            calc_raise = base_raise + random_adjustment
            raise_amount = max(observation["min_raise"], min(calc_raise, observation["max_raise"]))
            self.logger.info(
                f"Raising {raise_amount} with equity={equity:.2f} threshold={raise_threshold:.2f}"
            )
        elif valid[action_types.CALL.value] and equity >= pot_odds:
            action_type = action_types.CALL.value
            self.logger.debug(f"Calling with equity={equity:.2f}, pot_odds={pot_odds:.2f}")
        elif valid[action_types.CHECK.value]:
            action_type = action_types.CHECK.value
            self.logger.debug("Checking")
        elif valid[action_types.DISCARD.value] and not self.has_discarded:
            action_type = action_types.DISCARD.value
            card_to_discard = 0 if observation["my_cards"][0] < observation["my_cards"][1] else 1
            self.has_discarded = True
            self.logger.debug(f"Discarding card index={card_to_discard}")
        else:
            action_type = action_types.FOLD.value
            self.logger.info(f"Folding with equity={equity:.2f}, pot_odds={pot_odds:.2f}")
        return (action_type, raise_amount, card_to_discard)

    def observe(self, observation, reward, terminated, truncated, info):
        self.update_hand_counter(observation)
        self.total_reward += reward
        if terminated:
            # Update the NN-based strategy using the hand reward.
            self.update_strategy_nn(reward)
            if abs(reward) > 20:
                self.logger.info(f"Hand completed with reward={reward}")
