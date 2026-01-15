"""
Run real embedding analysis with BERT and SimCSE using diverse texts.
Validates theoretical claims with actual embeddings.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_diverse_texts(n_samples: int = 1000) -> list:
    """Generate truly diverse sample texts from multiple domains."""
    
    texts = []
    
    electronics = [
        "Wireless Bluetooth headphones with active noise cancellation and 30-hour battery life",
        "4K Ultra HD Smart TV with HDR support and built-in streaming apps",
        "Mechanical gaming keyboard with RGB lighting and Cherry MX switches",
        "Portable power bank with 20000mAh capacity and fast charging support",
        "USB-C hub with HDMI output, SD card reader, and multiple USB ports",
        "Wireless gaming mouse with adjustable DPI and ergonomic design",
        "Smart home speaker with voice assistant and multi-room audio support",
        "Laptop cooling pad with dual fans and adjustable height settings",
        "Wireless earbuds with touch controls and water resistance rating",
        "External SSD drive with 1TB storage and USB 3.2 connectivity",
    ]
    
    clothing = [
        "Cotton blend t-shirt with crew neck and relaxed fit design",
        "Waterproof hiking boots with ankle support and grip sole",
        "Wool blend sweater with cable knit pattern and ribbed cuffs",
        "Athletic running shorts with moisture wicking fabric",
        "Leather wallet with RFID blocking and multiple card slots",
        "Winter parka with down insulation and fur-lined hood",
        "Formal dress shirt with wrinkle-free fabric and slim fit",
        "Canvas sneakers with vulcanized rubber sole and lace-up closure",
        "Silk scarf with floral print and hand-rolled edges",
        "Denim jeans with stretch fabric and classic five-pocket styling",
    ]
    
    books = [
        "A gripping thriller novel about an FBI agent hunting a serial killer",
        "Self-help guide for building better habits and achieving personal goals",
        "Historical fiction set during World War II in occupied France",
        "Cookbook featuring authentic Italian recipes from regional traditions",
        "Science fiction epic exploring humanity's first contact with aliens",
        "Biography of a legendary entrepreneur who changed the tech industry",
        "Children's picture book teaching kindness and friendship values",
        "Mystery novel featuring an amateur detective solving local crimes",
        "Philosophy book examining the meaning of life and happiness",
        "Travel guide with insider tips for exploring Japan on a budget",
    ]
    
    home = [
        "Stainless steel cookware set with non-stick coating and glass lids",
        "Memory foam mattress topper for pressure relief and comfort",
        "Robot vacuum cleaner with smart mapping and app control",
        "Ceramic plant pot with drainage hole and saucer included",
        "Blackout curtains with thermal insulation and noise reduction",
        "Cast iron skillet pre-seasoned and ready for stovetop or oven",
        "LED desk lamp with adjustable brightness and color temperature",
        "Bamboo cutting board with juice groove and carrying handles",
        "Microfiber bed sheet set with deep pockets and soft finish",
        "Air purifier with HEPA filter and quiet operation mode",
    ]
    
    sports = [
        "Yoga mat with non-slip surface and alignment markers",
        "Adjustable dumbbell set with quick-change weight plates",
        "Running shoes with responsive cushioning and breathable mesh",
        "Fitness tracker with heart rate monitor and sleep tracking",
        "Resistance bands set with varying tension levels for workouts",
        "Tennis racket with carbon fiber frame and spin-friendly strings",
        "Camping tent for four people with waterproof rainfly",
        "Mountain bike with front suspension and disc brakes",
        "Golf club driver with adjustable loft and large sweet spot",
        "Swimming goggles with anti-fog coating and UV protection",
    ]
    
    news_snippets = [
        "The central bank announced interest rate changes affecting mortgage markets",
        "Scientists discovered a new species of deep-sea fish near underwater volcanoes",
        "The championship game ended in overtime with a dramatic final shot",
        "Tech company unveiled next-generation smartphone with revolutionary features",
        "International summit addressed climate change mitigation strategies",
        "Local restaurant won prestigious culinary award for innovative cuisine",
        "Space agency successfully launched satellite for weather monitoring",
        "Film festival announced lineup including documentaries and foreign films",
        "Medical researchers published breakthrough findings on disease treatment",
        "Urban development project will transform downtown with mixed-use buildings",
    ]
    
    random_sentences = [
        "The quick brown fox jumps over the lazy dog near the riverbank",
        "Machine learning algorithms continue to improve natural language understanding",
        "Classical music concerts attract audiences seeking cultural experiences",
        "Environmental sustainability requires collective action from communities",
        "Digital transformation is reshaping how businesses operate globally",
        "Healthy eating habits contribute to overall wellness and longevity",
        "Creative writing workshops help aspiring authors develop their craft",
        "Renewable energy sources are becoming more cost-effective each year",
        "Ancient civilizations left behind remarkable architectural achievements",
        "Modern art exhibitions challenge traditional aesthetic conventions",
    ]
    
    all_templates = [electronics, clothing, books, home, sports, news_snippets, random_sentences]
    
    while len(texts) < n_samples:
        for template_list in all_templates:
            for text in template_list:
                if len(texts) < n_samples:
                    texts.append(text)
                    
                    variation = text + f" Item #{len(texts)} in our collection."
                    if len(texts) < n_samples:
                        texts.append(variation)
    
    np.random.shuffle(texts)
    return texts[:n_samples]


def run_real_embedding_analysis():
    """Run real embedding extraction and analysis."""
    
    print("\n" + "="*60)
    print("REAL EMBEDDING ANALYSIS (v2 - Diverse Texts)")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    n_samples = 800
    texts = get_diverse_texts(n_samples)
    print(f"\nGenerated {len(texts)} diverse sample texts")
    print(f"Sample: '{texts[0][:60]}...'")
    
    from src.analysis.eigenvalues import compute_eigenvalue_spectrum, compute_effective_dimension
    from src.analysis.coverage import compute_coverage_metric
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': n_samples,
        'models': {}
    }
    
    models_to_test = ['bert', 'simcse']
    all_embeddings = {}
    
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
            
            all_embeddings[model_name] = embeddings
            print(f"Embeddings shape: {embeddings.shape}")
            
            print("Computing eigenvalue spectrum...")
            eigenvalues, _ = compute_eigenvalue_spectrum(embeddings)
            d_eff = compute_effective_dimension(eigenvalues)
            
            variance_90 = 0
            cumsum = np.cumsum(eigenvalues) / eigenvalues.sum()
            for i, cs in enumerate(cumsum):
                if cs >= 0.9:
                    variance_90 = i + 1
                    break
            
            variance_99 = 0
            for i, cs in enumerate(cumsum):
                if cs >= 0.99:
                    variance_99 = i + 1
                    break
            
            print("Computing coverage metrics...")
            k_values = [10, 50, 100, 200]
            coverage = compute_coverage_metric(embeddings, k_values=k_values, n_trials=30)
            
            results['models'][model_name] = {
                'embedding_dim': int(embeddings.shape[1]),
                'd_eff': float(d_eff),
                'd_90': int(variance_90),
                'd_99': int(variance_99),
                'eigenvalues_top50': eigenvalues[:50].tolist(),
                'eigenvalues_sum': float(eigenvalues.sum()),
                'coverage': {str(k): float(v) for k, v in coverage.items()}
            }
            
            print(f"  d_eff = {d_eff:.1f}")
            print(f"  dims for 90% var = {variance_90}")
            print(f"  dims for 99% var = {variance_99}")
            print(f"  ρ(100) = {coverage.get(100, 'N/A'):.4f}")
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results['models'][model_name] = {'error': str(e)}
    
    print("\n" + "-"*60)
    print("DETAILED COMPARISON")
    print("-"*60)
    
    bert_data = results['models'].get('bert', {})
    simcse_data = results['models'].get('simcse', {})
    
    if 'error' not in bert_data and 'error' not in simcse_data:
        print(f"\n{'Metric':<25} {'BERT':>12} {'SimCSE':>12} {'Ratio':>10}")
        print("-"*60)
        
        print(f"{'Effective Dim (d_eff)':<25} {bert_data['d_eff']:>12.1f} {simcse_data['d_eff']:>12.1f} {simcse_data['d_eff']/bert_data['d_eff']:>10.2f}x")
        print(f"{'Dims for 90% variance':<25} {bert_data['d_90']:>12} {simcse_data['d_90']:>12}")
        print(f"{'Dims for 99% variance':<25} {bert_data['d_99']:>12} {simcse_data['d_99']:>12}")
        print(f"{'Coverage ρ(100)':<25} {bert_data['coverage']['100']:>12.4f} {simcse_data['coverage']['100']:>12.4f}")
        
        print("\nTop 10 eigenvalues comparison:")
        print(f"{'Index':<8} {'BERT':>15} {'SimCSE':>15}")
        for i in range(10):
            print(f"{i:<8} {bert_data['eigenvalues_top50'][i]:>15.4f} {simcse_data['eigenvalues_top50'][i]:>15.4f}")
        
        bert_eigs = np.array(bert_data['eigenvalues_top50'])
        simcse_eigs = np.array(simcse_data['eigenvalues_top50'])
        
        bert_concentration = bert_eigs[0] / bert_eigs.sum()
        simcse_concentration = simcse_eigs[0] / simcse_eigs.sum()
        
        print(f"\nTop eigenvalue concentration:")
        print(f"  BERT: {bert_concentration*100:.1f}% of variance in 1st component")
        print(f"  SimCSE: {simcse_concentration*100:.1f}% of variance in 1st component")
        
        if simcse_concentration < bert_concentration:
            print("  ✓ SimCSE has more uniform eigenvalue distribution (less concentrated)")
    
    os.makedirs('results/metrics', exist_ok=True)
    output_path = 'results/metrics/real_embedding_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == '__main__':
    results = run_real_embedding_analysis()
