# Kaleidoscope Negotiator — ANL 2026

[![CI](https://github.com/MituAM/Kaleidoscope-Negotiator/actions/workflows/ci.yml/badge.svg)](https://github.com/MituAM/Kaleidoscope-Negotiator/actions/workflows/ci.yml)

CAI course portfolio — a bilateral negotiation agent that maximises **Score = Advantage + Concealing** in the [ANL 2026](https://anac.cs.brown.edu/anl) competition framework.

---

## Agent design

ANL 2026 rewards two things simultaneously:

| Component | Formula | Meaning |
|---|---|---|
| **Advantage** | `ufun(agreement) - reserved_value` | How much better than our walkaway value? |
| **Concealing** | `1 - normalise(opponent model accuracy)` | How well did we hide our preferences? |

Three interlocking strategies address both:

### 1 — Kaleidoscope Bidding *(targets Concealing)*

Offers are drawn exclusively from the top-25 % of rational outcomes (high utility), but the specific outcome chosen each round maximises a combined diversity-and-utility score:

```
score(o) = alpha * u_norm(o) + (1 - alpha) * diversity(o)
alpha: 0.4 early (diversity-first) -> 0.9 near deadline (utility-first)
```

Rotating issue-value combinations across bids prevents the opponent from reliably learning our issue weights, keeping their model inaccurate.

### 2 — Frequency Opponent Model *(targets Concealing + Advantage)*

After each observed offer, we count how often each issue value has appeared. Frequently offered values are inferred to be preferred:

```
u_opp(outcome) = mean_i [ freq_i(outcome[i]) / total_i ]
```

The model is stored in `private_info["opponent_ufun"]` for ANL evaluation.

### 3 — Boulware Acceptance *(targets Advantage)*

A time-adaptive threshold smoothly concedes toward the deadline:

```
threshold(t) = rv + (max_u - rv) * max(0.1, 0.9 - 0.8*t)
t=0 -> demands 90% of utility range above rv
t=1 -> accepts anything > 10% of utility range above rv
```

---

## Quick start

```bash
# Install dependencies
uv sync

# Interactive educational showcase (recommended first run)
python showcase.py

# Faster demo — skip tournament
python showcase.py --quick --scenario Laptop

# Run a single negotiation via CLI
uv run anl2026 run --no-plot

# Run a full local tournament
uv run anl2026 tournament
```

## Run tests

```bash
uv run pytest tests/test_mynegotiator.py tests/test_examples.py -v
```

---

## Project layout

```
mynegotiator.py          <- Kaleidoscope Negotiator (our agent)
showcase.py              <- Interactive educational showcase
main.py                  <- ANL 2026 CLI  (anl2026 run / tournament)
examples/
  boa.py                 <- BOA reference agent
  map.py                 <- MAP reference agent
  simple.py              <- Minimal reference agent
scenarios/               <- Negotiation domains (Camera, Car, Laptop, ...)
tests/
  test_mynegotiator.py   <- Agent unit tests
  test_examples.py       <- Reference agent tests
  test_cli.py            <- CLI integration tests
pyproject.toml           <- Project config & dependencies
```

---

## Course learning objectives (CAI)

| # | Objective | How this project addresses it |
|---|---|---|
| 1 | **Centralised vs Collaborative AI** | No coordinator assigns outcomes. Each agent independently maximises utility through bilateral offers — a pure collaborative paradigm. |
| 2 | **Co-active design** | Three clearly separated, human-interpretable modules (bidding, opponent modelling, acceptance), each tunable independently. |
| 3 | **Automated negotiation principles** | Full SAO protocol with utility functions, reserved values, Boulware concession, and the ANL 2026 Advantage + Concealing scoring formula. |
| 4 | **Agent coordination mechanisms** | Deadline pressure drives implicit convergence; deceptive Kaleidoscope bidding blocks the opponent from exploiting knowledge of our preferences. |
| 5 | **Agent interaction protocols** | Alternating offers, Accept/Reject/End responses, round-limit enforcement, and Kendall-tau opponent-model evaluation (ANL 2026 standard). |

---

## Results (Camera scenario, 8 matchups)

| Opponent | Avg Score | Rating |
|---|---|---|
| Boulware | ~1.57 | Strong |
| BOA | ~1.61 | Strong |
| Simple | ~1.58 | Strong |
| MAP | ~1.55 | Strong |

MyNegotiator consistently achieves **Concealing ≈ 1.0** — the opponent's model accuracy share collapses to near zero while ours dominates.
