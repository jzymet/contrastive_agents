"""
Priority 0: Embedding Analysis
Validates theoretical claims from Sections 3-4 of the paper.

Computes:
1. Eigenvalue spectra
2. Effective dimension (d_eff)
3. Coverage metric ρ(k,K)

Expected results:
- Anisotropic (BERT, RoBERTa, LLaMA): d_eff ≈ 40-60
- Contrastive (SimCSE, Jina, LLM2Vec): d_eff ≈ 200-220
"""

import sys
import os
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import get_extractor, EmbeddingCache, ALL_MODELS
from src.analysis import (
    analyze_all_embeddings,
    plot_eigenvalue_spectra,
    analyze_coverage_all_models,
    plot_coverage_curves,
    generate_demo_eigenvalue_data,
    generate_demo_coverage_data
)


def load_or_generate_embeddings(
    dataset_name: str = 'amazon_10k',
    n_samples: int = 1000,
    use_demo: bool = True
) -> dict:
    """Load cached embeddings or generate demo data."""
    
    if use_demo:
        print("Using demo embeddings...")
        embeddings = {}
        for model_name in ALL_MODELS:
            is_uniform = model_name in ['simcse', 'jina', 'llm2vec']
            extractor = get_extractor(model_name, use_dummy=True)
            
            dummy_texts = [f"Sample text {i}" for i in range(n_samples)]
            embeddings[model_name] = extractor.encode(dummy_texts)
        
        return embeddings
    
    cache = EmbeddingCache()
    embeddings = {}
    
    for model_name in ALL_MODELS:
        cached = cache.load(model_name, dataset_name)
        if cached is not None:
            emb_matrix = np.stack(list(cached.values()))
            embeddings[model_name] = emb_matrix[:n_samples]
        else:
            print(f"No cache for {model_name}, using demo data")
            extractor = get_extractor(model_name, use_dummy=True)
            dummy_texts = [f"Sample text {i}" for i in range(n_samples)]
            embeddings[model_name] = extractor.encode(dummy_texts)
    
    return embeddings


def run_embedding_analysis(
    use_demo: bool = True,
    n_samples: int = 1000,
    save_results: bool = True
) -> dict:
    """Run full embedding analysis."""
    
    print("\n" + "="*60)
    print("EMBEDDING ANALYSIS - Theory Validation")
    print("="*60)
    
    if use_demo:
        print("\n[Using demo data for visualization]")
        eigen_results = generate_demo_eigenvalue_data()
        coverage_results = generate_demo_coverage_data()
    else:
        embeddings = load_or_generate_embeddings(
            n_samples=n_samples,
            use_demo=False
        )
        
        print("\n1. Computing Eigenvalue Spectra...")
        eigen_results = analyze_all_embeddings(embeddings)
        
        print("\n2. Computing Coverage Metrics...")
        coverage_results = analyze_coverage_all_models(embeddings)
    
    print("\n" + "-"*60)
    print("EFFECTIVE DIMENSION RESULTS")
    print("-"*60)
    print(f"{'Model':<12} {'Type':<12} {'d_eff':>8}")
    print("-"*40)
    
    for model in ['bert', 'roberta', 'llama3', 'simcse', 'jina', 'llm2vec']:
        if model in eigen_results:
            d_eff = eigen_results[model]['d_eff']
            model_type = 'Anisotropic' if model in ['bert', 'roberta', 'llama3'] else 'Contrastive'
            print(f"{model:<12} {model_type:<12} {d_eff:>8.1f}")
    
    print("\n" + "-"*60)
    print("COVERAGE METRIC ρ(100, K)")
    print("-"*60)
    
    for model in coverage_results:
        rho_100 = coverage_results[model].get(100, 'N/A')
        if isinstance(rho_100, float):
            print(f"{model:<12}: ρ(100) = {rho_100:.4f}")
    
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    
    print("\n3. Generating plots...")
    fig_eigen = plot_eigenvalue_spectra(eigen_results, 'results/plots/eigenvalue_spectra.png')
    fig_coverage = plot_coverage_curves(coverage_results, 'results/plots/coverage_metric.png')
    
    if save_results:
        results = {
            'eigenvalues': {
                model: {
                    'd_eff': data['d_eff'],
                    'd_99': data.get('d_99', 0),
                    'top_10': data.get('top_10_eigenvalues', [])
                }
                for model, data in eigen_results.items()
            },
            'coverage': {
                model: {str(k): v for k, v in rho.items()}
                for model, rho in coverage_results.items()
            }
        }
        
        with open('results/metrics/embedding_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to results/metrics/embedding_analysis.json")
    
    return {
        'eigenvalues': eigen_results,
        'coverage': coverage_results
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embedding Analysis')
    parser.add_argument('--demo', action='store_true', default=True,
                       help='Use demo data (default: True)')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples for analysis')
    args = parser.parse_args()
    
    results = run_embedding_analysis(
        use_demo=args.demo,
        n_samples=args.n_samples
    )
