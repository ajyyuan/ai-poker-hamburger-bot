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

##############################################
# Recurrent Networks for Actor and Critic (Expert)
##############################################

class RecurrentPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=16):
        super(RecurrentPolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]  # use last output
        logits = self.fc(out)
        return logits, hidden

class RecurrentValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size=16):
        super(RecurrentValueNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]
        value = self.fc(out)
        return value, hidden

##############################################
# Meta-Controller Network
##############################################

class MetaController(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_size=16):
        super(MetaController, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_experts)
        
    def forward(self, x):
        # x shape: [batch, input_dim]
        out = self.fc1(x)
        out = self.relu(out)
        logits = self.fc2(out)
        return logits  # logits over experts

##############################################
# Updated Agent with Ensemble of Experts and Meta-Controller
##############################################

class PlayerAgent(Agent):
    def __init__(self, stream: bool = False):
        super().__init__(stream)
        self.evaluator = Evaluator()
        # Opponent modeling
        self.opp_action_counts = {"raise": 0, "call": 0, "check": 0, "discard": 0, "fold": 0}
        self.last_opp_bet = 0
        self.has_discarded = False
        self.total_reward = 0

        # Hand counting: update on hand termination.
        self.hand_count = 0
        self.hand_counted = False

        # --- Ensemble of Experts ---
        self.num_experts = 3
        # Candidate multipliers remain as before.
        self.strategy_candidates = [0.8, 0.85, 0.9, 0.95, 1.0]
        self.num_candidates = len(self.strategy_candidates)
        # Expanded feature vector: 10 elements.
        self.input_dim = 10  
        # Create an ensemble of recurrent actor networks and critics.
        self.expert_actors = nn.ModuleList([
            RecurrentPolicyNetwork(self.input_dim, self.num_candidates, hidden_size=16)
            for _ in range(self.num_experts)
        ])
        self.expert_critics = nn.ModuleList([
            RecurrentValueNetwork(self.input_dim, hidden_size=16)
            for _ in range(self.num_experts)
        ])
        # We'll maintain separate hidden states for each expert.
        self.expert_hidden = [None] * self.num_experts
        self.critic_hidden = [None] * self.num_experts

        # Meta-Controller: to select which expert to use.
        self.meta_controller = MetaController(self.input_dim, self.num_experts, hidden_size=16)
        self.meta_optimizer = optim.Adam(self.meta_controller.parameters(), lr=0.01)

        # Actor and critic optimizers for experts.
        self.actor_optimizers = [optim.Adam(expert.parameters(), lr=0.01) for expert in self.expert_actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=0.01) for critic in self.expert_critics]

        # For storing last meta log probability and chosen expert index.
        self.last_meta_log_prob = None
        self.last_expert_index = None
        self.last_expert_log_prob = None
        self.last_state = None

        # --- Mode Switching for Surprise Play ---
        self.mode = "normal"
        self.mode_duration = 0
        self.aggressive_multiplier = 0.8
        self.conservative_multiplier = 1.1

        # --- Learning Start Threshold ---
        self.learning_start_hand = random.randint(175, 225)
        self.logger.info(f"RL adaptation will start at hand {self.learning_start_hand}")

    def reset_hand(self):
        """Reset per-hand flags and hidden states at the start of each hand."""
        self.last_opp_bet = 0
        self.has_discarded = False
        self.hand_counted = False
        # Reset hidden states for all experts.
        self.expert_hidden = [None] * self.num_experts
        self.critic_hidden = [None] * self.num_experts

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
          2. pot_odds
          3. opp_aggr
          4. normalized hand count: hand_count/TOTAL_MATCH_HANDS
          5. normalized total reward: total_reward/100
          6. normalized street: observation["street"]/3
          7. bet difference ratio: (opp_bet - my_bet)/pot_size (or 0)
          8. normalized pot size: pot_size/200
          9. board texture: unique suits among visible community cards normalized by 3
          10. board connectivity: (max(rank)-min(rank)) normalized by 8
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

    def dynamic_gto_multiplier(self, equity, pot_odds, opp_aggr, street):
        """
        Compute a dynamic multiplier for the GTO phase based on the current state.
        - If equity is significantly above pot odds, be more aggressive (lower multiplier).
        - If equity is below pot odds, be more conservative (higher multiplier).
        - Adjust based on opponent aggressiveness and street.
        """
        multiplier = 1.0
        # Adjust for equity vs. pot odds.
        if equity > pot_odds + 0.1:
            multiplier -= 0.1
        elif equity < pot_odds - 0.1:
            multiplier += 0.1

        # Adjust for opponent aggression.
        if opp_aggr > 0.6:
            multiplier += 0.1
        elif opp_aggr < 0.4:
            multiplier -= 0.1

        # Adjust for betting street (pre-flop more aggressive, river more conservative).
        if street == 0:  # pre-flop
            multiplier *= 0.95
        elif street == 3:  # river
            multiplier *= 1.05

        # Bound multiplier within a reasonable range.
        multiplier = max(0.8, min(multiplier, 1.2))
        return multiplier

    def select_action_actor(self, features):
        """
        Use the meta-controller to select an expert, then use that expert's recurrent actor network
        to choose a raise threshold multiplier.
        """
        # First, get meta-controller probabilities.
        state_meta = torch.tensor(features).unsqueeze(0)  # shape: [1, input_dim]
        meta_logits = self.meta_controller(state_meta)
        meta_probs = torch.softmax(meta_logits, dim=1)
        meta_dist = torch.distributions.Categorical(meta_probs)
        expert_index = meta_dist.sample().item()
        self.last_meta_log_prob = meta_dist.log_prob(torch.tensor(expert_index))
        self.last_expert_index = expert_index

        # Now, use the chosen expert's actor.
        # Prepare state for LSTM: shape [1, 1, input_dim]
        state = torch.tensor(features).unsqueeze(0).unsqueeze(0)
        # If this expert's hidden state is not initialized, do so.
        if self.expert_hidden[expert_index] is None:
            self.expert_hidden[expert_index] = (torch.zeros(1, 1, 16), torch.zeros(1, 1, 16))
        logits, new_hidden = self.expert_actors[expert_index](state, self.expert_hidden[expert_index])
        self.expert_hidden[expert_index] = new_hidden
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.last_expert_log_prob = dist.log_prob(action)
        self.last_state = state  # store state for actor-critic update
        chosen_multiplier = self.strategy_candidates[action.item()]
        self.logger.debug(f"Expert {expert_index} selected multiplier: {chosen_multiplier} (log_prob={self.last_expert_log_prob.item():.4f})")
        return chosen_multiplier

    def update_actor_critic(self, reward):
        """
        Update the chosen expert's actor and critic networks and the meta-controller using an actor-critic update.
        Advantage = reward - value_estimate.
        """
        if self.last_state is None or self.last_expert_log_prob is None or self.last_meta_log_prob is None:
            return
        expert_index = self.last_expert_index
        # Update critic for the chosen expert.
        if self.critic_hidden[expert_index] is None:
            self.critic_hidden[expert_index] = (torch.zeros(1, 1, 16), torch.zeros(1, 1, 16))
        value_estimate, new_hidden = self.expert_critics[expert_index](self.last_state, self.critic_hidden[expert_index])
        self.critic_hidden[expert_index] = new_hidden
        advantage = reward - value_estimate.item()
        # Actor loss for the chosen expert.
        actor_loss = -self.last_expert_log_prob * advantage
        critic_loss = nn.MSELoss()(value_estimate, torch.tensor([[reward]], dtype=torch.float32))
        total_loss = actor_loss + critic_loss
        self.actor_optimizers[expert_index].zero_grad()
        self.critic_optimizers[expert_index].zero_grad()
        total_loss.backward()
        self.actor_optimizers[expert_index].step()
        self.critic_optimizers[expert_index].step()
        # Also update meta-controller using the same advantage.
        meta_loss = -self.last_meta_log_prob * advantage
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        self.logger.debug(f"Actor-Critic update for expert {expert_index}: actor_loss={actor_loss.item():.4f}, critic_loss={critic_loss.item():.4f}, meta_loss={meta_loss.item():.4f}")
        self.last_expert_log_prob = None
        self.last_meta_log_prob = None
        self.last_state = None

    def maybe_switch_mode(self):
        """
        Occasionally switch mode (aggressive or conservative) for a brief period.
        Aggressive forces multiplier=0.8; conservative forces multiplier=1.1.
        """
        if self.mode_duration > 0:
            self.mode_duration -= 1
        else:
            if random.random() < 0.1:  # 10% chance to switch mode
                self.mode = random.choice(["aggressive", "conservative"])
                self.mode_duration = random.randint(3, 5)  # override lasts 3-5 hands
                self.logger.info(f"Mode switch: entering {self.mode} mode for {self.mode_duration} hands")
            else:
                self.mode = "normal"

    def act(self, observation, reward, terminated, truncated, info):
        # At street 0, reset hand and possibly switch mode.
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
        
        # Schedule-based strategy adjustment:
        if self.hand_count < 50:
            chosen_multiplier = self.aggressive_multiplier
            self.logger.info("Early phase: Using aggressive multiplier (first 50 hands)")
        elif self.hand_count < 100:
            chosen_multiplier = self.conservative_multiplier
            self.logger.info("Early phase: Using conservative multiplier (hands 50-100)")
        elif self.hand_count < self.learning_start_hand:
            # Use dynamic, state-dependent GTO strategy.
            chosen_multiplier = self.dynamic_gto_multiplier(equity, pot_odds, opp_aggr, observation["street"])
            self.logger.info(f"GTO phase: Using dynamic GTO multiplier: {chosen_multiplier:.2f}")
        else:
            chosen_multiplier = self.select_action_actor(features)
            self.logger.debug(f"Using actor-critic ensemble; chosen multiplier: {chosen_multiplier}")
        
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
