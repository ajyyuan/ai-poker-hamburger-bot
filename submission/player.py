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
AVERAGE_FORCED_LOSS = 1.5  # Forced loss per hand (in chips)

# Define a simple feedforward network as our policy network.
# This network will output logits over candidate multipliers.
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
        return x  # logits

# Define a value network for the critic.
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)  # Single value estimate
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        value = self.fc2(x)
        return value

class PlayerAgent(Agent):
    def __init__(self, stream: bool = False):
        super().__init__(stream)
        self.evaluator = Evaluator()
        # Track opponent's actions.
        self.opp_action_counts = {"raise": 0, "call": 0, "check": 0, "discard": 0, "fold": 0}
        self.last_opp_bet = 0
        self.has_discarded = False

        # Net advantage: (our chips - opponent's chips)
        self.total_reward = 0

        # Hand counting: incremented on hand termination.
        self.hand_count = 0
        self.hand_counted = False  # flag to ensure one count per terminated hand

        # --- RL Module: Actor-Critic for Multiplier Selection ---
        # Candidate multipliers for RL mode (for actor-critic adaptation).
        self.strategy_candidates = [0.8, 0.85, 0.9, 0.95, 1.0]
        self.num_candidates = len(self.strategy_candidates)
        # We'll use an expanded feature vector now (10 elements).
        # Features: [equity, pot_odds, opp_aggr, normalized_hand_count, normalized_total_reward,
        #            normalized street, bet diff ratio, normalized pot size, normalized board texture, board connectivity]
        self.input_dim = 10  
        self.policy_net = PolicyNetwork(self.input_dim, self.num_candidates)
        self.value_net = ValueNetwork(self.input_dim)
        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.critic_optimizer = optim.Adam(self.value_net.parameters(), lr=0.01)
        self.last_log_prob = None
        self.last_state = None

        # --- Mode Switching for Surprise Play ---
        # Modes: "normal", "aggressive", "conservative"
        self.mode = "normal"
        self.mode_duration = 0  # number of hands the override lasts
        # Override multipliers: aggressive forces 0.8 (lowest) and conservative forces 1.1.
        self.aggressive_multiplier = 0.8
        self.conservative_multiplier = 1.1

        # --- Learning Start Threshold ---
        # Until this hand count is reached, play near GTO (fixed multiplier = 1.0).
        self.learning_start_hand = random.randint(100, 200)
        self.logger.info(f"RL adaptation will start at hand {self.learning_start_hand}")

    def reset_hand(self):
        """Reset per-hand flags; called at the start of each hand."""
        self.last_opp_bet = 0
        self.has_discarded = False
        self.hand_counted = False

    def compute_equity(self, observation, num_simulations=200):
        """Monte Carlo simulation to estimate win probability."""
        my_cards = observation["my_cards"]
        community_cards = [c for c in observation["community_cards"] if c != -1]
        opp_known = []
        if observation["opp_discarded_card"] != -1:
            opp_known.append(observation["opp_discarded_card"])
        if observation["opp_drawn_card"] != -1:
            opp_known.append(observation["opp_drawn_card"])
        shown_cards = set(my_cards + community_cards + opp_known)
        deck = list(range(27))
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
            if my_rank < opp_rank:
                wins += 1
        return wins / num_simulations if num_simulations > 0 else 0.0

    def update_opponent_model(self, observation):
        """Update opponent action counts."""
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
        """Return opponent aggression ratio."""
        raises = self.opp_action_counts.get("raise", 0)
        calls = self.opp_action_counts.get("call", 0)
        checks = self.opp_action_counts.get("check", 0)
        total = raises + calls + checks + 1e-6
        return raises / total

    def get_features(self, observation, equity, pot_odds, opp_aggr):
        """
        Construct a 10-element feature vector:
          1. equity (0-1)
          2. pot_odds (ratio)
          3. opp_aggr (ratio)
          4. normalized hand count: hand_count/TOTAL_MATCH_HANDS
          5. normalized total reward: total_reward/100
          6. normalized street: observation["street"]/3
          7. bet difference ratio: (opp_bet - my_bet)/pot_size (or 0 if pot_size==0)
          8. normalized pot size: pot_size/200
          9. board texture: unique suits among visible community cards normalized by 3
          10. board connectivity: (max(rank)-min(rank)) among visible community cards normalized by 8
        """
        norm_hand = self.hand_count / TOTAL_MATCH_HANDS
        norm_reward = self.total_reward / 100.0
        street = observation["street"] / 3.0

        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        pot_size = my_bet + opp_bet
        bet_diff = (opp_bet - my_bet) / pot_size if pot_size > 0 else 0.0
        norm_pot = pot_size / 200.0

        community_cards = [card for card in observation["community_cards"] if card != -1]
        if community_cards:
            unique_suits = len(set(card // 9 for card in community_cards))
        else:
            unique_suits = 0
        norm_texture = unique_suits / 3.0

        if community_cards:
            ranks = [card % 9 for card in community_cards]
            connectivity = (max(ranks) - min(ranks)) / 8.0
        else:
            connectivity = 0.0

        features = np.array([
            equity, pot_odds, opp_aggr, norm_hand, norm_reward,
            street, bet_diff, norm_pot, norm_texture, connectivity
        ], dtype=np.float32)
        return features

    def select_action_actor(self, features):
        """
        Use the policy network (actor) with softmax to select a raise threshold multiplier.
        Stores the log probability and state for later update.
        """
        state = torch.tensor(features).unsqueeze(0)
        logits = self.policy_net(state)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.last_log_prob = dist.log_prob(action)
        self.last_state = state
        chosen_multiplier = self.strategy_candidates[action.item()]
        self.logger.debug(f"Actor selected multiplier: {chosen_multiplier} (log_prob={self.last_log_prob.item():.4f})")
        return chosen_multiplier

    def update_actor_critic(self, reward):
        """
        Update both the actor and critic using an actor-critic update.
        Advantage = reward - value_estimate.
        """
        if self.last_state is None or self.last_log_prob is None:
            return
        value_estimate = self.value_net(self.last_state)
        advantage = reward - value_estimate.item()
        actor_loss = -self.last_log_prob * advantage
        critic_loss = nn.MSELoss()(value_estimate, torch.tensor([[reward]], dtype=torch.float32))
        total_loss = actor_loss + critic_loss
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.logger.debug(f"Actor-Critic update: actor_loss={actor_loss.item():.4f}, critic_loss={critic_loss.item():.4f}")
        self.last_log_prob = None
        self.last_state = None

    def maybe_switch_mode(self):
        """
        With a small probability, switch mode for a spurts of hands.
        If mode_duration is active, decrement it.
        If not, with probability p, randomly switch to 'aggressive' or 'conservative' for a random duration.
        """
        if self.mode_duration > 0:
            self.mode_duration -= 1
        else:
            # With a small probability, switch mode.
            if random.random() < 0.1:  # 10% chance per hand to switch mode
                self.mode = random.choice(["aggressive", "conservative"])
                self.mode_duration = random.randint(3, 10)  # mode lasts between 3 and 10 hands
                self.logger.info(f"Mode switch: entering {self.mode} mode for {self.mode_duration} hands")
            else:
                self.mode = "normal"

    def act(self, observation, reward, terminated, truncated, info):
        # Possibly switch mode at the beginning of each hand.
        if observation["street"] == 0:
            self.reset_hand()
            self.maybe_switch_mode()
            remaining_rounds = TOTAL_MATCH_HANDS - self.hand_count
            needed_for_safe_fold = remaining_rounds * AVERAGE_FORCED_LOSS
            self.logger.debug(f"Pre-hand: hand_count={self.hand_count}, total_reward={self.total_reward:.2f}, threshold={needed_for_safe_fold:.2f}")
            if self.total_reward > 0 and self.total_reward >= needed_for_safe_fold:
                self.logger.info(f"Auto-fold activated: total_reward={self.total_reward:.2f} >= threshold={needed_for_safe_fold:.2f}")
                return (action_types.FOLD.value, 0, -1)

        # Check auto-flop mode.
        rounds_left = TOTAL_MATCH_HANDS - self.hand_count
        if self.total_reward > AVERAGE_FORCED_LOSS * rounds_left:
            self.logger.info(f"Auto-flop mode triggered. hand_count={self.hand_count}, total_reward={self.total_reward}, rounds_left={rounds_left}")
            valid = observation["valid_actions"]
            if observation["street"] == 0:
                if valid[action_types.CALL.value]:
                    self.logger.info("Auto-flop (pre-flop): Calling.")
                    return (action_types.CALL.value, 0, -1)
                elif valid[action_types.CHECK.value]:
                    self.logger.info("Auto-flop (pre-flop): Checking.")
                    return (action_types.CHECK.value, 0, -1)
                else:
                    self.logger.info("Auto-flop (pre-flop): Folding.")
                    return (action_types.FOLD.value, 0, -1)
            else:
                if valid[action_types.CHECK.value]:
                    self.logger.info("Auto-flop (post-flop): Checking.")
                    return (action_types.CHECK.value, 0, -1)
                elif valid[action_types.CALL.value]:
                    self.logger.info("Auto-flop (post-flop): Calling.")
                    return (action_types.CALL.value, 0, -1)
                else:
                    self.logger.info("Auto-flop (post-flop): Folding.")
                    return (action_types.FOLD.value, 0, -1)

        # Normal behavior.
        equity = self.compute_equity(observation)
        continue_cost = observation["opp_bet"] - observation["my_bet"]
        pot_size = observation["opp_bet"] + observation["my_bet"]
        pot_odds = continue_cost / (pot_size + continue_cost) if continue_cost > 0 else 0.0
        opp_aggr = self.get_opponent_aggressiveness()
        features = self.get_features(observation, equity, pot_odds, opp_aggr)

        # Determine which mode to use.
        # Before learning_start_hand, use fixed GTO (multiplier=1.0).
        # After that, normally use actor-critic adaptation.
        # However, if mode override is active, use that.
        if self.hand_count < self.learning_start_hand:
            chosen_multiplier = 1.0
            self.logger.debug("Using default GTO strategy (multiplier=1.0)")
        else:
            # Check mode override:
            if self.mode == "aggressive":
                chosen_multiplier = self.aggressive_multiplier
                self.logger.info("Overriding mode: aggressive")
            elif self.mode == "conservative":
                chosen_multiplier = self.conservative_multiplier
                self.logger.info("Overriding mode: conservative")
            else:
                chosen_multiplier = self.select_action_actor(features)
                self.logger.debug(f"Using actor-critic strategy multiplier: {chosen_multiplier}")

        normal_threshold = 0.65
        raise_threshold = normal_threshold * chosen_multiplier
        self.logger.debug(f"Computed raise threshold: {raise_threshold:.2f}")
        self.update_opponent_model(observation)
        self.logger.debug(f"hand_count={self.hand_count}, Equity={equity:.2f}, PotOdds={pot_odds:.2f}, OppAgg={opp_aggr:.2f}")
        valid = observation["valid_actions"]
        action_type = None
        raise_amount = 0
        card_to_discard = -1
        if valid[action_types.RAISE.value] and equity > raise_threshold:
            action_type = action_types.RAISE.value
            factor = 0.75 if opp_aggr < 0.5 else 0.5
            base_raise = int(pot_size * factor)
            random_adjustment = random.randint(-2, 2)
            calc_raise = base_raise + random_adjustment
            raise_amount = max(observation["min_raise"], min(calc_raise, observation["max_raise"]))
            self.logger.info(f"Raising {raise_amount} with equity={equity:.2f} and threshold={raise_threshold:.2f}")
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
        # Update total reward.
        self.total_reward += reward
        if terminated and not self.hand_counted:
            self.hand_count += 1
            self.hand_counted = True
            self.logger.debug(f"Hand terminated. Updated hand count: {self.hand_count}")
            self.update_actor_critic(reward)
            self.logger.info(f"Hand completed with reward={reward}")
