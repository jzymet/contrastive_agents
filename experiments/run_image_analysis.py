"""
Run image embedding analysis to compare anisotropic (ResNet) vs contrastive (CLIP).
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_synthetic_images(n: int = 500) -> list:
    """Generate synthetic image data (colored squares) for testing."""
    from PIL import Image
    
    images = []
    np.random.seed(42)
    
    for i in range(n):
        r = int(50 + 150 * (i % 10) / 10)
        g = int(50 + 150 * ((i // 10) % 10) / 10)
        b = int(50 + 150 * ((i // 100) % 10) / 10)
        
        img = Image.new('RGB', (224, 224), (r, g, b))
        
        for _ in range(3):
            x1 = np.random.randint(0, 200)
            y1 = np.random.randint(0, 200)
            x2 = x1 + np.random.randint(10, 50)
            y2 = y1 + np.random.randint(10, 50)
            
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            draw.rectangle([x1, y1, x2, y2], fill=color)
        
        images.append(img)
    
    return images


def compute_d_eff(embeddings: np.ndarray) -> tuple:
    """Compute effective dimension from eigenvalue spectrum."""
    from sklearn.preprocessing import StandardScaler
    
    embeddings_centered = embeddings - embeddings.mean(axis=0)
    
    cov = np.cov(embeddings_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    
    eigenvalues = eigenvalues[:min(100, len(eigenvalues))]
    
    total = eigenvalues.sum()
    if total < 1e-10:
        return 1.0, eigenvalues
    
    eigenvalues_norm = eigenvalues / total
    sum_sq = (eigenvalues_norm ** 2).sum()
    d_eff = 1.0 / sum_sq if sum_sq > 0 else 1.0
    
    return d_eff, eigenvalues


def run_image_analysis():
    """Run image embedding analysis on CLIP."""
    
    print("\n" + "="*60)
    print("IMAGE EMBEDDING ANALYSIS")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    n_samples = 300
    print(f"\nGenerating {n_samples} synthetic images...")
    images = generate_synthetic_images(n_samples)
    
    from src.embeddings.image_extractors import CLIPImageExtractor
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': n_samples,
        'models': {}
    }
    
    print("\n--- CLIP (Contrastive - Multimodal) ---")
    try:
        clip = CLIPImageExtractor(target_dim=768)
        clip_embs = clip.encode(images, batch_size=32)
        d_eff_clip, eigs_clip = compute_d_eff(clip_embs)
        
        results['models']['clip'] = {
            'type': 'contrastive',
            'd_eff': float(d_eff_clip),
            'top_eigenvalue_pct': float(eigs_clip[0] / eigs_clip.sum() * 100) if eigs_clip.sum() > 0 else 0,
            'eigenvalues': eigs_clip[:20].tolist(),
        }
        print(f"  d_eff: {d_eff_clip:.1f}")
        print(f"  Top eigenvalue: {eigs_clip[0]/eigs_clip.sum()*100:.1f}%")
    except Exception as e:
        print(f"  Error: {e}")
        results['models']['clip'] = {'error': str(e)}
    
    print("\n" + "-"*60)
    print("IMAGE EMBEDDING RESULTS")
    print("-"*60)
    
    if 'clip' in results['models'] and 'd_eff' in results['models']['clip']:
        clip_d = results['models']['clip']['d_eff']
        print(f"\n{'Model':<15} {'Type':<15} {'d_eff':>10}")
        print("-"*45)
        print(f"{'CLIP':<15} {'Contrastive':<15} {clip_d:>10.1f}")
        
        if clip_d > 15:
            print("\nCLIP shows reasonable effective dimension for contrastive image embeddings")
    
    os.makedirs('results/metrics', exist_ok=True)
    output_path = 'results/metrics/image_embedding_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == '__main__':
    results = run_image_analysis()
