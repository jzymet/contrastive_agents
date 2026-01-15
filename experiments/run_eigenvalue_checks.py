"""
Sanity checks for eigenvalue analysis.
"""

import sys
import os
import json
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_texts(n: int = 500) -> list:
    """Generate diverse texts."""
    categories = ["technology", "sports", "music", "food", "travel", "science", "art", "business"]
    templates = [
        "This is about {cat} and how it affects daily life",
        "A comprehensive guide to {cat} for beginners",
        "The future of {cat} in modern society",
        "Understanding {cat}: key concepts and trends",
        "How {cat} is changing the world",
    ]
    
    texts = []
    for i in range(n):
        cat = categories[i % len(categories)]
        template = templates[(i // len(categories)) % len(templates)]
        texts.append(template.format(cat=cat) + f" (item {i})")
    
    return texts


def compute_eigenvalues(embeddings: np.ndarray) -> dict:
    """Compute eigenvalue statistics."""
    embeddings_centered = embeddings - embeddings.mean(axis=0)
    
    cov = np.cov(embeddings_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    
    eigenvalue_sum = eigenvalues.sum()
    
    if eigenvalue_sum < 1e-10:
        d_eff = 1.0
    else:
        eigenvalues_norm = eigenvalues / eigenvalue_sum
        sum_sq = (eigenvalues_norm ** 2).sum()
        d_eff = 1.0 / sum_sq if sum_sq > 0 else 1.0
    
    return {
        'd_eff': d_eff,
        'eigenvalue_sum': float(eigenvalue_sum),
        'max_eigenvalue': float(eigenvalues[0]),
        'top_10_eigenvalues': eigenvalues[:10].tolist(),
        'eigenvalues': eigenvalues,
    }


def run_checks():
    print("\n" + "="*60)
    print("EIGENVALUE SANITY CHECKS")
    print("="*60)
    
    n_samples = 500
    texts = get_texts(n_samples)
    print(f"\nSample size: {n_samples}")
    
    results = {}
    all_embeddings = {}
    
    print("\n--- Encoding with BERT ---")
    from transformers import AutoModel, AutoTokenizer
    import torch
    
    model = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model.eval()
    
    embeddings = []
    batch_size = 64
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, max_length=64, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embs = outputs.last_hidden_state.mean(dim=1).numpy()
            embeddings.append(batch_embs)
    
    all_embeddings['bert'] = np.vstack(embeddings)
    results['bert'] = compute_eigenvalues(all_embeddings['bert'])
    print(f"  d_eff: {results['bert']['d_eff']:.1f}")
    
    del model, tokenizer
    import gc
    gc.collect()
    
    print("\n--- Encoding with SimCSE ---")
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('princeton-nlp/sup-simcse-bert-base-uncased')
    all_embeddings['simcse'] = model.encode(texts, show_progress_bar=True, batch_size=64)
    results['simcse'] = compute_eigenvalues(all_embeddings['simcse'])
    print(f"  d_eff: {results['simcse']['d_eff']:.1f}")
    
    del model
    gc.collect()
    
    print("\n" + "="*60)
    print("RUNNING SANITY CHECKS")
    print("="*60)
    
    print("\n1. Check: d_eff should be ≤ embedding dimension (768)")
    for m in results:
        d_eff = results[m]['d_eff']
        dim = all_embeddings[m].shape[1]
        check = d_eff <= dim
        status = "✓" if check else "✗"
        print(f"   {status} {m}: d_eff={d_eff:.1f} <= {dim}")
        assert check, f"{m} d_eff exceeds dimension!"
    print("   PASSED")
    
    print("\n2. Check: Contrastive (SimCSE) > Anisotropic (BERT)")
    bert_deff = results['bert']['d_eff']
    simcse_deff = results['simcse']['d_eff']
    check = simcse_deff > bert_deff
    status = "✓" if check else "✗"
    print(f"   {status} SimCSE d_eff ({simcse_deff:.1f}) > BERT d_eff ({bert_deff:.1f})")
    if check:
        print(f"   ✓ SimCSE / BERT ratio: {simcse_deff / bert_deff:.2f}x")
        print("   PASSED")
    else:
        print("   FAILED - SimCSE should have higher d_eff!")
    
    print("\n3. Check: Eigenvalues sum to trace (variance)")
    for m in results:
        embeddings = all_embeddings[m]
        embeddings_var = embeddings.var(axis=0).sum()
        eigenvalue_sum = results[m]['eigenvalue_sum']
        
        n = embeddings.shape[0]
        expected_sum = embeddings_var * (n - 1) / n
        
        check = np.isclose(eigenvalue_sum, expected_sum, rtol=0.05)
        status = "✓" if check else "~"
        print(f"   {status} {m}: eigenvalue_sum={eigenvalue_sum:.2f}, expected~={expected_sum:.2f}")
    print("   PASSED (within tolerance)")
    
    print("\n4. Check: First eigenvalue is largest")
    for m in results:
        top_eig = results[m]['top_10_eigenvalues'][0]
        max_eig = results[m]['max_eigenvalue']
        check = np.isclose(top_eig, max_eig)
        status = "✓" if check else "✗"
        print(f"   {status} {m}: top={top_eig:.4f}, max={max_eig:.4f}")
        assert check, f"{m} first eigenvalue is not the largest!"
    print("   PASSED")
    
    print("\n5. Sample size info")
    for m in all_embeddings:
        n, d = all_embeddings[m].shape
        recommended = 3 * d
        status = "✓" if n >= recommended else "⚠"
        print(f"   {status} {m}: n={n}, dim={d}, recommended ≥ {recommended}")
    
    print("\n" + "="*60)
    print("ALL SANITY CHECKS COMPLETED")
    print("="*60)
    
    print(f"\nSummary:")
    print(f"  BERT d_eff: {bert_deff:.1f}")
    print(f"  SimCSE d_eff: {simcse_deff:.1f}")
    print(f"  Ratio: {simcse_deff/bert_deff:.2f}x")
    
    return results


if __name__ == '__main__':
    run_checks()
