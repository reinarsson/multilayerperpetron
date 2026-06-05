# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run tests**
```bash
pytest tests/
```

**Run a single test**
```bash
pytest -k "test_xor_learns"
```

**Run the digit recognition example**
```bash
python recognition_cases/digital.py
```

**Dependencies**
```bash
pip install numpy pytest
```

## Architecture

```
src/
  activations.py             # sigmoid + ActivationFunction type alias
  perceptron.py              # Single neuron, injectable activation
  multilayer_perceptron.py   # Fully connected MLP with backpropagation
tests/
  test_perceptron.py
  test_multilayer_perceptron.py
recognition_cases/
  digital.py       # 7-segment digit training demo
```

**`Perceptron`** — holds a weight vector (length = inputs + 1 for bias). Accepts any `ActivationFunction` at construction. `run(x)` returns the activated weighted sum.

**`MultiLayerPerceptron`** — stores neurons in `network[layer][neuron]`, activations in `values[layer][neuron]`, and backprop error terms in `d[layer][neuron]`. Layer 0 is the input layer and has no neuron objects. `bp(x, y)` runs one forward pass + weight update and returns MSE.

**`activations.py`** — defines `ActivationFunction = Callable[[float], float]` and `sigmoid`. Add new activation functions here and pass them to `Perceptron` or `MultiLayerPerceptron`.

## Code Standards

- No magic numbers — define named constants
- `from __future__ import annotations` in every file
- Type hints on all signatures; Google-style docstrings on all public methods
- No I/O (print statements) inside model classes — that is the caller's responsibility

## Git workflow — mandatory for all agents

1. **Always start from the latest `main`:**
   ```bash
   git checkout main && git pull origin main
   ```
2. **Never commit directly to `main`.** Always create a feature branch from it:
   ```bash
   git checkout -b feat/your-feature-name
   ```
3. Do all work on that branch and commit changes there.
4. Before opening a PR, run `pytest tests/`. Do not open a PR if tests fail.
5. Open a PR to merge into `main` using `gh pr create`.
6. After opening the PR, monitor its status with `gh pr checks`. If any checks fail or review comments are posted, fix the issues on the same branch, commit, and push. Repeat until all checks pass and comments are resolved.
