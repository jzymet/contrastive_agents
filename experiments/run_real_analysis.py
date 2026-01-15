"""
Run real embedding analysis with BERT and SimCSE.
Validates theoretical claims with actual embeddings.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_sample_texts(n_samples: int = 500) -> list:
    """Generate diverse sample texts for embedding analysis."""
    categories = [
        "Electronics", "Clothing", "Home & Kitchen", "Books", "Sports",
        "Toys", "Beauty", "Health", "Automotive", "Garden"
    ]
    
    adjectives = [
        "premium", "affordable", "high-quality", "durable", "lightweight",
        "compact", "professional", "versatile", "ergonomic", "innovative"
    ]
    
    products = [
        "headphones", "speaker", "cable", "adapter", "charger",
        "case", "stand", "holder", "organizer", "accessory",
        "shirt", "jacket", "pants", "shoes", "bag",
        "lamp", "chair", "table", "shelf", "container"
    ]
    
    texts = []
    for i in range(n_samples):
        cat = categories[i % len(categories)]
        adj = adjectives[i % len(adjectives)]
        prod = products[i % len(products)]
        
        text = f"{adj.capitalize()} {prod} for {cat.lower()}. Great product with excellent reviews and fast shipping. Perfect for everyday use."
        texts.append(text)
    
    return texts


def run_real_embedding_analysis():
    """Run real embedding extraction and analysis."""
    
    print("\n" + "="*60)
    print("REAL EMBEDDING ANALYSIS")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    n_samples = 500
    texts = generate_sample_texts(n_samples)
    print(f"\nGenerated {len(texts)} sample texts")
    
    from src.analysis.eigenvalues import compute_eigenvalue_spectrum, compute_effective_dimension
    from src.analysis.coverage import compute_coverage_metric
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': n_samples,
        'models': {}
    }
    
    models_to_test = ['bert', 'simcse']
    
    for model_name in models_to_test:
        print(f"\n--- {model_name.upper()} ---")
        
        try:
            if model_name == 'bert':
                from transformers import AutoModel, AutoTokenizer
                import torch
                
                print("Loading BERT model...")
                model = AutoModel.from_pretrained('bert-base-uncased')
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                model.eval()
                
                print(f"Encoding {len(texts)} texts...")
                embeddings = []
                batch_size = 32
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    inputs = tokenizer(
                        batch,
                        return_tensors='pt',
                        truncation=True,
                        max_length=128,
                        padding=True
                    )
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        batch_embs = outputs.last_hidden_state.mean(dim=1).numpy()
                        embeddings.append(batch_embs)
                    
                    if (i // batch_size) % 5 == 0:
                        print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)}")
                
                embeddings = np.vstack(embeddings)
                
                del model, tokenizer
                import gc
                gc.collect()
                
            elif model_name == 'simcse':
                from sentence_transformers import SentenceTransformer
                
                print("Loading SimCSE model...")
                model = SentenceTransformer('princeton-nlp/sup-simcse-bert-base-uncased')
                
                print(f"Encoding {len(texts)} texts...")
                embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
                
                del model
                import gc
                gc.collect()
            
            print(f"Embeddings shape: {embeddings.shape}")
            
            print("Computing eigenvalue spectrum...")
            eigenvalues, _ = compute_eigenvalue_spectrum(embeddings)
            d_eff = compute_effective_dimension(eigenvalues)
            
            print("Computing coverage metrics...")
            k_values = [10, 50, 100, 200]
            coverage = compute_coverage_metric(embeddings, k_values=k_values, n_trials=30)
            
            results['models'][model_name] = {
                'embedding_dim': int(embeddings.shape[1]),
                'd_eff': float(d_eff),
                'eigenvalues_top20': eigenvalues[:20].tolist(),
                'eigenvalues_sum': float(eigenvalues.sum()),
                'coverage': {str(k): float(v) for k, v in coverage.items()}
            }
            
            print(f"  d_eff = {d_eff:.1f}")
            print(f"  ρ(100) = {coverage.get(100, 'N/A'):.4f}")
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results['models'][model_name] = {'error': str(e)}
    
    print("\n" + "-"*60)
    print("SUMMARY")
    print("-"*60)
    
    for model_name, data in results['models'].items():
        if 'error' not in data:
            print(f"{model_name:12s}: d_eff = {data['d_eff']:.1f}, ρ(100) = {data['coverage'].get('100', 'N/A'):.4f}")
    
    if len([m for m in results['models'].values() if 'error' not in m]) >= 2:
        bert_data = results['models'].get('bert', {})
        simcse_data = results['models'].get('simcse', {})
        
        if 'error' not in bert_data and 'error' not in simcse_data:
            d_eff_ratio = simcse_data['d_eff'] / bert_data['d_eff']
            coverage_improvement = (bert_data['coverage']['100'] - simcse_data['coverage']['100']) / bert_data['coverage']['100'] * 100
            
            print(f"\nTheory Validation:")
            print(f"  SimCSE d_eff / BERT d_eff = {d_eff_ratio:.2f}x")
            print(f"  Coverage improvement = {coverage_improvement:.1f}%")
            
            if d_eff_ratio > 1.5:
                print("  ✓ Confirmed: Contrastive embeddings have higher effective dimension")
            if coverage_improvement > 10:
                print("  ✓ Confirmed: Contrastive embeddings have better coverage")
    
    os.makedirs('results/metrics', exist_ok=True)
    output_path = 'results/metrics/real_embedding_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == '__main__':
    results = run_real_embedding_analysis()
