# Contrastive RL for Massive Action Spaces

Research platform for studying how contrastive embeddings (SimCSE, CLIP) enable efficient exploration-exploitation in large action spaces through optimal geometric properties (alignment + uniformity).

## Project Overview

This project implements experiments for a research paper on scaling reinforcement learning to massive action spaces using contrastive embeddings. The key insight is that contrastive uniformity reduces the coverage metric ρ(k,K) by ensuring that a small random sample k covers the full action space K efficiently.

## Project Structure

```
contrastive-rl/
├── app.py                    # Main Streamlit dashboard
├── src/
│   ├── embeddings/          # Embedding models (SimCSE, CLIP, BERT)
│   │   ├── cache.py         # Embedding caching utilities
│   │   ├── simcse.py        # SimCSE and BERT encoders
│   │   └── clip_encoder.py  # CLIP encoder for multimodal
│   ├── agents/              # RL agents
│   │   ├── bandit.py        # Thompson Sampling, UCB bandits
│   │   └── a2c.py           # A2C with Transformer encoder
│   ├── datasets/            # Dataset loaders
│   │   ├── amazon.py        # Amazon Electronics (synthetic)
│   │   ├── toolbench.py     # ToolBench API dataset
│   │   └── math_data.py     # GSM8K and MATH-500
│   └── utils/               # Utility functions
│       ├── sampling.py      # Uniform sampling, coverage metrics
│       └── metrics.py       # Experiment metrics
├── experiments/             # Experiment runners
│   ├── recsys_experiments.py
│   ├── tool_experiments.py
│   └── math_experiments.py
├── config/                  # Configuration files
│   ├── recsys_config.yaml
│   ├── tool_config.yaml
│   └── math_config.yaml
├── cache/                   # Cached embeddings
└── results/                 # Experiment results
```

## Three Use Cases

### 1. Personalized Recommendation (RecSys)
- **Dataset**: Amazon Electronics (synthetic, extensible)
- **Action Space**: ~10K-500K items
- **Formulations**: Contextual Bandit (single-step), A2C (10-step sessions)
- **Embeddings**: SimCSE (text), CLIP (multimodal)

### 2. LLM Tool Selection
- **Dataset**: ToolBench-inspired (16K APIs)
- **Tasks**: I1 (single-tool), I2 (same-category), I3 (cross-category)
- **Formulations**: Contextual Bandit, A2C for tool chaining
- **Embeddings**: SimCSE on tool descriptions

### 3. Math Reasoning
- **Dataset**: GSM8K (train) → MATH-500 (test)
- **Action Space**: 50-200 reasoning step candidates per state
- **Verification**: SymPy for arithmetic (FREE!)
- **Positioning**: vs rStar-Math (ICML 2025 Oral)
- **Key Metric**: Rollouts needed to reach solve rate threshold

## Key Theoretical Concepts

### Coverage Metric ρ(k,K)
Expected distance from any action to its nearest sample in a k-sized random subset. Lower is better.

### Uniformity Advantage
SimCSE embeddings are uniformly distributed on the hypersphere, leading to better coverage than anisotropic BERT embeddings.

### Eluder Dimension
Contrastive uniformity reduces effective Eluder dimension, improving regret bounds in bandit/RL settings.

## Running the Application

The Streamlit dashboard runs on port 5000:
```bash
streamlit run app.py --server.port 5000
```

## Key Dependencies
- torch, transformers, sentence-transformers
- sympy (math verification)
- streamlit, plotly (visualization)
- scipy, scikit-learn (metrics)

## Configuration

Each use case has its own config file in `config/`:
- `recsys_config.yaml`: RecSys experiment parameters
- `tool_config.yaml`: Tool selection parameters
- `math_config.yaml`: Math reasoning parameters

## Usage Notes

1. **Demo Mode**: The app includes a demo mode with synthetic data for quick testing
2. **Full Mode**: Load actual embeddings from pretrained models for real experiments
3. **Caching**: Embeddings are cached to disk to avoid recomputation

## Recent Changes

- Initial implementation of all three use cases
- Streamlit dashboard with tabs for each experiment type
- Visualization: coverage plots, regret curves, strategy distribution
- Synthetic datasets for rapid prototyping
