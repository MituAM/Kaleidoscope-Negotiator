"""
Kaleidoscope Negotiator — ANL 2026 Strategic Agent
Collaborative Artificial Intelligence (CAI)

Design objective:  maximise  Score = Advantage + Concealing

  Advantage  = ufun(agreement) − reserved_value
  Concealing = 1 − normalise(opponent's modelling accuracy of us)

Three interacting strategies
─────────────────────────────
1. Kaleidoscope Bidding  [→ Concealing]
   Draw exclusively from the top-25 % of rational outcomes (maintaining
   high utility), but choose *which* outcome to bid by maximising a
   combined score of normalised utility and pattern-diversity from recent
   bids.  Diverse bids rotate issue-value combinations so the opponent
   cannot reliably learn our issue weights, keeping their model inaccurate.

   score(o) = α · u_norm(o) + (1−α) · diversity(o)
   α shifts from 0.4 (diversity-first, early) to 0.9 (utility-first, late).

2. Frequency Opponent Model  [→ better decisions + ANL scoring]
   After every observed offer we count how often each issue value has
   appeared.  Frequently offered values are inferred to be preferred.
   The estimate is stored in private_info["opponent_ufun"] for evaluation.

   û_opp(outcome) = mean_i [ freq_i(outcome[i]) / total_i ]

3. Time-Adaptive Boulware Acceptance  [→ Advantage]
   threshold(t) = rv + (max_u − rv) · max(0.1, 0.9 − 0.8·t)
   Starts demanding (~90 % of utility range) and concedes smoothly to
   just above the reserved value as the deadline approaches.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from negmas import ResponseType, SAOState
from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun
from negmas.preferences.preferences import PreferencesChangeType
from negmas.sao import SAOCallNegotiator, SAOResponse


class MyNegotiator(SAOCallNegotiator):
    """
    Kaleidoscope Negotiator: deceptive bidding + frequency opponent model
    + Boulware acceptance.  Targets ANL 2026 Score = Advantage + Concealing.
    """

    rational_outcomes: tuple[Outcome, ...] = tuple()

    # ------------------------------------------------------------------ init --

    def on_preferences_changed(self, changes: list[Any]) -> None:
        """
        Initialise agent state when the utility function is set.

        Called twice by the framework: once at negotiation start (Initialization)
        and once at negotiation end to clear the preference owner (Dissociated).
        We skip the Dissociated call so the trained opponent model stays readable
        for post-negotiation scoring.
        """
        if any(c.type == PreferencesChangeType.Dissociated for c in changes):
            return  # cleanup call — preserve trained model for scoring
        if self.ufun is None:
            return

        rv = float(self.ufun.reserved_value or 0.0)

        # Build sorted list of rational outcomes (utility > reserved value)
        scored = [
            (float(self.ufun(o) or 0.0), o)
            for o in self.nmi.outcome_space.enumerate_or_sample()
            if (self.ufun(o) or 0.0) > rv
        ]
        self.rational_outcomes = tuple(o for _, o in sorted(scored, reverse=True))

        # Bid pool: top 25 % — high utility with enough variety to rotate
        n_pool = max(1, len(self.rational_outcomes) // 4)
        self._bid_pool: list[Outcome] = list(self.rational_outcomes[:n_pool])

        # Sliding window of our recent bids (max 8) for diversity calculation
        self._my_bids: list[Outcome] = []

        # Issue list kept stable for indexing into outcomes
        self._issues = list(self.nmi.outcome_space.issues)

        # Per-issue value frequency tables for opponent modelling
        self._opp_freq: list[dict[Any, int]] = [defaultdict(int) for _ in self._issues]
        self._n_opp_offers: int = 0

        # Uniform uninformed opponent model — updated as offers arrive
        self.private_info["opponent_ufun"] = LambdaMultiFun(f=lambda x: 0.5)

    # --------------------------------------------------------- main callback --

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        offer = state.current_offer

        # Opening move — nothing to evaluate yet
        if offer is None:
            bid = self.concealing_bidding_strategy(state)
            return SAOResponse(ResponseType.REJECT_OFFER, bid)

        self.update_opponent_model(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        bid = self.concealing_bidding_strategy(state)
        return SAOResponse(ResponseType.REJECT_OFFER, bid)

    # ------------------------------------------------------- acceptance ------

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        Boulware-shaped acceptance: demand high utility early, concede late.
        Accept anything rational in the final 10 % of negotiation time.
        """
        assert self.ufun

        offer = state.current_offer
        if offer is None:
            return False

        u = float(self.ufun(offer) or 0.0)
        rv = float(self.ufun.reserved_value or 0.0)

        if u <= rv:
            return False
        if not self.rational_outcomes:
            return True

        max_u = float(self.ufun(self.rational_outcomes[0]) or 0.0)
        t = state.relative_time

        # Smooth concession curve: fraction of utility range above rv
        # t=0 → 0.9 × range;  t=1 → 0.1 × range
        fraction = max(0.1, 0.9 - 0.8 * t)
        threshold = rv + (max_u - rv) * fraction

        return u >= threshold

    # ------------------------------------------------------- bidding ---------

    def concealing_bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        Kaleidoscope bidding: pick a high-utility outcome that looks *different*
        from our recent bids, preventing the opponent from learning our issue weights.

        Deception mechanism
        -------------------
        If we always offer the same top outcome the opponent quickly infers our
        preferences.  By rotating which issue values appear in our bids — while
        staying within the top-25 % utility tier — we create a confusing pattern
        that keeps opponent modelling accuracy low (→ high Concealing score).

        Near the deadline we stop rotating and go for our best outcome, trading
        deception for the highest possible Advantage if a deal is made.
        """
        if not self.rational_outcomes:
            return None

        t = state.relative_time

        # Final phase: maximise utility, no more deception games
        if t > 0.88:
            return self.rational_outcomes[0]

        if not self._bid_pool:
            return self.rational_outcomes[0]

        # Sample candidates to avoid an O(n) scan every round
        n_candidates = min(len(self._bid_pool), 15)
        candidates = random.sample(self._bid_pool, n_candidates)

        rv = float(self.ufun.reserved_value or 0.0)
        max_u = float(self.ufun(self.rational_outcomes[0]) or 0.0)
        u_range = (max_u - rv) or 1.0

        # α: weight on utility vs. diversity — shifts as time passes
        alpha = 0.4 + 0.5 * t  # 0.4 early (diversity-first) → 0.9 late (utility-first)

        best, best_score = None, -1.0
        for o in candidates:
            u_norm = (float(self.ufun(o) or 0.0) - rv) / u_range
            div = self._diversity_score(o)
            score = alpha * u_norm + (1.0 - alpha) * div
            if score > best_score:
                best_score = score
                best = o

        if best is not None:
            self._my_bids.append(best)
            if len(self._my_bids) > 8:
                self._my_bids.pop(0)

        return best

    def _diversity_score(self, outcome: Outcome) -> float:
        """
        How surprising is this outcome relative to our 5 most recent bids?
        Returns 1.0 (fully novel) → 0.0 (identical to recent pattern).
        """
        if not self._my_bids:
            return 1.0
        recent = self._my_bids[-5:]
        total = sum(
            1.0 - sum(1 for b in recent if b[i] == outcome[i]) / len(recent)
            for i in range(len(self._issues))
        )
        return total / max(1, len(self._issues))

    # ------------------------------------------------------- opponent model --

    def update_opponent_model(self, state: SAOState) -> None:
        """
        Frequency-based opponent modelling.

        For each issue we accumulate how often the opponent has offered each
        possible value.  The normalised frequencies become a proxy utility:
        values offered more frequently are assumed to be more preferred.

        û_opp(outcome) = mean_i [ freq_i(outcome[i]) / total_i ]

        The model is stored in private_info["opponent_ufun"] for ANL scoring.
        """
        assert self.ufun and self.opponent_ufun

        offer = state.current_offer
        if offer is None:
            return

        self._n_opp_offers += 1

        for i in range(len(self._issues)):
            self._opp_freq[i][offer[i]] += 1

        # Snapshot current frequencies so the closure is stable
        freq_snap = [dict(f) for f in self._opp_freq]
        n_issues = len(self._issues)

        def estimate(x: Outcome) -> float:
            """Average normalised frequency across all issues."""
            if x is None:
                return 0.0
            score = 0.0
            for i, counts in enumerate(freq_snap):
                total = sum(counts.values()) or 1
                score += counts.get(x[i], 0) / total
            return score / max(1, n_issues)

        self.private_info["opponent_ufun"] = LambdaMultiFun(f=estimate)
