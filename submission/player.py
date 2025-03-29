import random
import numpy as np
from treys import Evaluator
from gym_env import PokerEnv  # int_to_card is defined within PokerEnv
from agents.agent import Agent

# Access the enum and card conversion from PokerEnv
action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

# Constants
TOTAL_MATCH_HANDS = 1000
AVERAGE_FORCED_LOSS = 3  # Avg stack diff. lost per hand when auto-fold (1 or 2 BB)

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

        # Count of hands played and fingerprint of last hand's hole cards
        self.hand_count = 0
        self.last_hand_fingerprint = None

    def update_hand_counter(self, observation):
        """
        Update hand counter if a new hand has started.
        Uses the fact that at the start of a hand (street 0), the hole cards should be new.
        """
        if observation["street"] == 0:
            # Use the tuple of hole cards as a simple fingerprint
            current_fingerprint = tuple(observation["my_cards"])
            if current_fingerprint != self.last_hand_fingerprint:
                self.last_hand_fingerprint = current_fingerprint
                self.hand_count += 1
                self.logger.debug(f"New hand detected. Updated hand count: {self.hand_count}")

    def reset_hand(self):
        """Called at the start of each new hand (street 0) if acting."""
        self.last_opp_bet = 0
        self.has_discarded = False
        # Note: hand_count update is now handled by update_hand_counter

    def compute_equity(self, observation, num_simulations=200):
        """
        Monte Carlo simulation to estimate the probability (0..1) that our final hand beats the opponent's.
        """
        my_cards = observation["my_cards"]
        community_cards = [c for c in observation["community_cards"] if c != -1]

        # Include opponent's known cards (if any)
        opp_known = []
        if observation["opp_discarded_card"] != -1:
            opp_known.append(observation["opp_discarded_card"])
        if observation["opp_drawn_card"] != -1:
            opp_known.append(observation["opp_drawn_card"])

        # Build deck excluding known cards
        shown_cards = set(my_cards + community_cards + opp_known)
        deck = list(range(27))  # 27-card deck (no clubs, no face cards)
        non_shown_cards = [card for card in deck if card not in shown_cards]

        wins = 0
        for _ in range(num_simulations):
            opp_needed = 2 - len(opp_known)
            board_needed = 5 - len(community_cards)
            sample_size = opp_needed + board_needed

            if sample_size > len(non_shown_cards):
                continue  # Skip simulation if not enough cards remain

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
        """Track opponent actions: raises, calls, checks, etc."""
        current_opp_bet = observation["opp_bet"]
        diff = current_opp_bet - self.last_opp_bet

        if diff > 0:
            # Consider it a raise if diff >= min_raise; otherwise, a call.
            if diff >= observation["min_raise"]:
                self.opp_action_counts["raise"] += 1
            else:
                self.opp_action_counts["call"] += 1
        else:
            self.opp_action_counts["check"] += 1

        self.last_opp_bet = current_opp_bet

        # Record discard action if opponent discarded a card.
        if observation["opp_discarded_card"] != -1:
            self.opp_action_counts["discard"] += 1

    def get_opponent_aggressiveness(self):
        """Return opponent aggression as: raises / (raises + calls + checks)."""
        raises = self.opp_action_counts.get("raise", 0)
        calls = self.opp_action_counts.get("call", 0)
        checks = self.opp_action_counts.get("check", 0)
        total = raises + calls + checks + 1e-6  # prevent division by zero
        return raises / total

    def act(self, observation, reward, terminated, truncated, info):
        # Update hand counter based on observation (in case this hand hasn't been counted yet)
        self.update_hand_counter(observation)

        # If acting on a new hand, also reset hand-specific flags.
        if observation["street"] == 0:
            self.reset_hand()

            remaining_hands = TOTAL_MATCH_HANDS - self.hand_count
            needed_for_safe_fold = remaining_hands * AVERAGE_FORCED_LOSS

            self.logger.debug(
                f"Hand {self.hand_count}, net_advantage={self.total_reward:.2f}, needed={needed_for_safe_fold:.2f}"
            )

            # If net advantage is sufficient, fold all remaining hands.
            if self.total_reward > 0 and self.total_reward >= needed_for_safe_fold:
                self.logger.info(
                    f"Folding rest of match. net_advantage={self.total_reward:.2f}, needed={needed_for_safe_fold:.2f}"
                )
                return (action_types.FOLD.value, 0, -1)
            # End safe-fold check

        # Normal decision-making:
        self.update_opponent_model(observation)
        equity = self.compute_equity(observation)
        continue_cost = observation["opp_bet"] - observation["my_bet"]
        pot_size = observation["opp_bet"] + observation["my_bet"]
        pot_odds = continue_cost / (pot_size + continue_cost) if continue_cost > 0 else 0.0
        opp_aggr = self.get_opponent_aggressiveness()

        self.logger.debug(
            f"Hand {self.hand_count}, Equity={equity:.2f}, PotOdds={pot_odds:.2f}, OppAgg={opp_aggr:.2f}"
        )

        # Early aggression logic:
        # To be aggressive early (i.e., require lower equity to raise), we set the multiplier low at first,
        # then gradually raise it over the first 500 hands.
        early_mult = min(1.0, 0.8 + 0.2 * (self.hand_count / (TOTAL_MATCH_HANDS * 0.5)))
        normal_threshold = 0.65
        raise_threshold = normal_threshold * early_mult
        # For example: at hand 0, raise_threshold = 0.65 * 0.8 = 0.52; at hand 500, raise_threshold = 0.65.

        valid = observation["valid_actions"]
        action_type = None
        raise_amount = 0
        card_to_discard = -1

        # RAISE if equity exceeds the threshold.
        if valid[action_types.RAISE.value] and equity > raise_threshold:
            action_type = action_types.RAISE.value
            # Base raise scales with pot size and opponent aggressiveness.
            factor = 0.75 if opp_aggr < 0.5 else 0.5
            base_raise = int(pot_size * factor)
            # Add a small random offset for variability.
            random_adjustment = random.randint(-2, 2)
            calc_raise = base_raise + random_adjustment
            raise_amount = max(observation["min_raise"], min(calc_raise, observation["max_raise"]))
            self.logger.info(
                f"Raising {raise_amount} with equity={equity:.2f} threshold={raise_threshold:.2f}"
            )
        # CALL if equity is at least as good as pot odds.
        elif valid[action_types.CALL.value] and equity >= pot_odds:
            action_type = action_types.CALL.value
            self.logger.debug(f"Calling with equity={equity:.2f}, pot_odds={pot_odds:.2f}")
        # CHECK if available.
        elif valid[action_types.CHECK.value]:
            action_type = action_types.CHECK.value
            self.logger.debug("Checking")
        # DISCARD if available and not already done.
        elif valid[action_types.DISCARD.value] and not self.has_discarded:
            action_type = action_types.DISCARD.value
            # Discard the lower card.
            card_to_discard = 0 if observation["my_cards"][0] < observation["my_cards"][1] else 1
            self.has_discarded = True
            self.logger.debug(f"Discarding card index={card_to_discard}")
        else:
            # Otherwise, fold.
            action_type = action_types.FOLD.value
            self.logger.info(f"Folding with equity={equity:.2f}, pot_odds={pot_odds:.2f}")

        return (action_type, raise_amount, card_to_discard)

    def observe(self, observation, reward, terminated, truncated, info):
        # Update hand counter in observe() as well to ensure every new hand is captured,
        # even if act() wasn't called on a street==0 observation.
        self.update_hand_counter(observation)
        # Update the net reward difference after each hand.
        self.total_reward += reward
        if terminated and abs(reward) > 20:
            self.logger.info(f"Hand completed with reward={reward}")
