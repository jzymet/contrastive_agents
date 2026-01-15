# Contrastive RL for Massive Action Spaces

Research platform for ICML 2026 paper studying how contrastive embeddings (SimCSE, Jina, LLM2Vec) enable efficient exploration-exploitation in large action spaces through optimal geometric properties (alignment + uniformity).

## Core Thesis

Reconstruction-based embeddings (BERT, RoBERTa, LLaMA) produce anisotropic representations with effective dimension ~50, causing linear regret in RL. Contrastive embeddings produce uniform representations with effective dimension ~200, enabling sublinear regret and efficient exploration.

## Project Structure

```
contrastive-rl/
├── app.py                    # Main Streamlit dashboard
├── src/
│   ├── embeddings/           # Embedding extractors
│   │   ├── base.py           # BaseEmbeddingExtractor, EmbeddingCache
│   │   ├── extractors.py     # BERT, RoBERTa, SimCSE, Jina extractors
│   │   ├── simcse.py         # Legacy SimCSE encoder
│   │   └── clip_encoder.py   # CLIP encoder for multimodal
│   ├── models/               # RL agents
│   │   ├── neural_ts.py      # NeuralTSBandit, SimpleNeuralBandit
│   │   └── reward_transformer.py  # RewardTransformer, A2CTransformerAgent
│   ├── analysis/             # Theory validation
│   │   ├── eigenvalues.py    # Eigenvalue spectra, d_eff computation
│   │   ├── rkhs.py           # RKHS norm analysis
│   │   └── coverage.py       # Coverage metric ρ(k,K)
│   ├── datasets/             # Dataset loaders
│   │   ├── amazon.py         # Amazon Electronics
│   │   ├── toolbench.py      # ToolBench API dataset
│   │   └── math_data.py      # GSM8K and MATH-500
│   └── utils/                # Utility functions
│       ├── sampling.py       # Uniform sampling, coverage metrics
│       └── metrics.py        # Experiment metrics
├── experiments/              # Experiment runners
│   ├── 01_embedding_analysis.py   # Priority 0: Theory validation
│   ├── 02_recsys_bandit.py        # Priority 1: Neural TS bandit
│   ├── 03_tools_a2c.py            # Priority 2: Tool selection A2C
│   └── 04_math_a2c.py             # Priority 3: Math reasoning (optional)
├── config/                   # Configuration files
├── data/                     # Data and embeddings cache
│   └── embeddings/           # Cached embeddings
└── results/                  # Experiment results
    ├── plots/                # Generated figures
    └── metrics/              # JSON results
```

## Embedding Models (6 total)

### Anisotropic (Reconstruction-based - predicted to fail):
1. **BERT-base-uncased** - Expected d_eff ≈ 40
2. **RoBERTa-base** - Expected d_eff ≈ 50
3. **LLaMA-3-8B-base** - Expected d_eff ≈ 60

### Contrastive (Uniform - predicted to succeed):
4. **SimCSE-base** - Expected d_eff ≈ 200
5. **Jina-embeddings-v3** - Expected d_eff ≈ 220
6. **LLM2Vec-LLaMA-3** - Expected d_eff ≈ 210

## Three Use Cases

### 1. Personalized Recommendation (RecSys) - Priority 1
- **Dataset**: Amazon Electronics (10K items)
- **Agent**: Neural Thompson Sampling
- **Metric**: Cumulative regret over 10,000 rounds

### 2. LLM Tool Selection - Priority 2
- **Dataset**: ToolBench (16K APIs)
- **Tasks**: I3 cross-category tool chaining
- **Agent**: A2C with RewardTransformer (36M params)
- **Metric**: Task success rate

### 3. Math Reasoning - Priority 3 (Optional)
- **Dataset**: GSM8K → MATH-500
- **Agent**: A2C with reasoning step selection
- **Metric**: Solve rate, rollouts to threshold

## Key Theoretical Concepts

### Effective Dimension (d_eff)
Participation ratio: d_eff = (Σλ_i)² / (Σλ_i²)
Higher d_eff = more uniformly distributed = better coverage

### Coverage Metric ρ(k,K)
Expected distance from any action to its nearest sample in a k-sized random subset.
Lower ρ = better coverage = more efficient exploration.

### RKHS Norm
||R||_RKHS = sqrt(Σ w_i² / λ_i)
High RKHS norm = reward function poorly represented = linear regret.

## Running the Application

```bash
streamlit run app.py --server.port 5000
```

## Key Dependencies
- torch, transformers, sentence-transformers
- scipy, scikit-learn (metrics and analysis)
- streamlit, plotly (visualization)
- sympy (math verification)

## Recent Changes (January 15, 2026)

- Major revamp based on new ICML 2026 prompt
- Expanded to 6 embedding models (3 anisotropic + 3 contrastive)
- Added unified EmbeddingExtractor with PCA dimension standardization
- Implemented NeuralTSBandit for RecSys experiments
- Added RewardTransformer (36M param critic) for A2C
- Created comprehensive embedding analysis module (eigenvalues, d_eff, RKHS, coverage)
- Updated Streamlit dashboard with theory validation visualizations
- Restructured project with src/models, src/analysis directories

## Timeline (13 days to deadline)
- Days 1-2: Embedding Analysis (eigenvalue spectra, d_eff, coverage)
- Days 3-6: RecSys Neural Bandit experiments
- Days 7-12: Tools A2C experiments
- Days 13-14: Paper writing
