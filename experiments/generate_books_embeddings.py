"""
Generate BERT and SimCSE embeddings for Books dataset.

This script:
1. Loads processed Books data
2. Extracts item texts (title + description)
3. Generates BERT and SimCSE embeddings
4. Computes user embeddings (average of history)
5. Saves all embeddings to disk
"""

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import get_extractor


def generate_item_embeddings(
    data_file='../data/amazon_books/Books_processed.pkl',
    output_dir='../data/embeddings'
):
    """Generate item embeddings for Books dataset."""
    
    print("="*60)
    print("STEP 1: Loading Processed Data")
    print("="*60)
    
    # Load processed data
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {data_path}\n"
            f"Run: python experiments/process_books_data.py first!"
        )
    
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    items_dict = dataset['items_dict']
    item_ids = list(items_dict.keys())
    
    print(f"✓ Loaded {len(item_ids)} items")
    
    print("\n" + "="*60)
    print("STEP 2: Extracting Item Texts")
    print("="*60)
    
    # Extract texts from metadata
    texts = []
    valid_item_ids = []
    
    for item_id in tqdm(item_ids, desc="Processing items"):
        item = items_dict[item_id]
        
        # Extract title
        title = item.get('title', '')
        
        # Extract description (can be list or string)
        desc_raw = item.get('description', '')
        if isinstance(desc_raw, list):
            desc = ' '.join(desc_raw[:3])  # First 3 sentences
        else:
            desc = str(desc_raw)
        
        # Extract categories
        categories = item.get('categories', [])
        if categories:
            # Flatten nested categories
            if isinstance(categories, list) and len(categories) > 0:
                if isinstance(categories[0], list):
                    flat_cats = [cat for sublist in categories for cat in sublist]
                else:
                    flat_cats = categories
                cat_str = ' '.join(flat_cats[:3])
            else:
                cat_str = ''
        else:
            cat_str = ''
        
        # Combine: title is most important
        text = title
        if desc:
            text += f". {desc}"
        if cat_str:
            text += f". Categories: {cat_str}"
        
        # Skip if no text
        if not text.strip():
            continue
        
        texts.append(text)
        valid_item_ids.append(item_id)
    
    print(f"✓ Extracted {len(texts)} valid item texts")
    print(f"  Sample: {texts[0][:200]}...")
    
    print("\n" + "="*60)
    print("STEP 3: Generating BERT Embeddings")
    print("="*60)
    
    # Generate BERT embeddings
    bert_encoder = get_extractor('bert')
    bert_embs_array = bert_encoder.encode(
        texts
    )
    
    print(f"✓ BERT embeddings shape: {bert_embs_array.shape}")
    
    # Create dict mapping
    bert_item_embs = {
        valid_item_ids[i]: bert_embs_array[i] 
        for i in range(len(valid_item_ids))
    }
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'bert_embeddings_books.pkl', 'wb') as f:
        pickle.dump(bert_item_embs, f)
    
    print(f"✓ Saved BERT item embeddings to: {output_path / 'bert_embeddings_books.pkl'}")
    
    print("\n" + "="*60)
    print("STEP 4: Generating SimCSE Embeddings")
    print("="*60)
    
    # Generate SimCSE embeddings
    simcse_encoder = get_extractor('simcse')
    simcse_embs_array = simcse_encoder.encode(
        texts
    )
    
    print(f"✓ SimCSE embeddings shape: {simcse_embs_array.shape}")
    
    # Create dict mapping
    simcse_item_embs = {
        valid_item_ids[i]: simcse_embs_array[i]
        for i in range(len(valid_item_ids))
    }
    
    # Save
    with open(output_path / 'simcse_embeddings_books.pkl', 'wb') as f:
        pickle.dump(simcse_item_embs, f)
    
    print(f"✓ Saved SimCSE item embeddings to: {output_path / 'simcse_embeddings_books.pkl'}")
    
    return bert_item_embs, simcse_item_embs


def main():
    """Generate all embeddings for Books dataset."""
    
    print("\n" + "="*60)
    print("GENERATING BOOKS ITEM EMBEDDINGS")
    print("="*60)
    print()
    
    # Generate item embeddings only
    bert_item_embs, simcse_item_embs = generate_item_embeddings()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Items with embeddings: {len(bert_item_embs)}")
    print(f"BERT embedding dim: {bert_item_embs[list(bert_item_embs.keys())[0]].shape[0]}")
    print(f"SimCSE embedding dim: {simcse_item_embs[list(simcse_item_embs.keys())[0]].shape[0]}")
    print("\n✓ Books item embeddings generated successfully!")
    print("\nUser embeddings will be computed on-the-fly from recent K items in history")
    print("\nNext step: Run experiments with:")
    print("  python experiments/03_ranking_recsys.py")


if __name__ == "__main__":
    main()