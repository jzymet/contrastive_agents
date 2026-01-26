"""
Process Amazon Books data with robust JSON error handling.

Fixes:
1. Handles malformed JSON lines gracefully (skip them)
2. Uses proper field names for Books dataset
3. Filters minimum interactions correctly
4. Caches processed data
"""

import numpy as np
import pandas as pd
import gzip
import json
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
from pathlib import Path
import pickle


def load_reviews_robust(file_path, max_reviews=None):
    """
    Load reviews with robust error handling for malformed JSON.
    
    Args:
        file_path: Path to reviews file (.jsonl.gz or .json.gz)
        max_reviews: Maximum number of reviews to load (None = all)
    
    Returns:
        list of review dicts
    """
    print(f"Loading reviews from: {file_path}")
    
    reviews = []
    errors = 0
    
    # Handle both .jsonl.gz and .json.gz formats
    open_func = gzip.open if str(file_path).endswith('.gz') else open
    
    try:
        with open_func(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading reviews")):
                if max_reviews and len(reviews) >= max_reviews:
                    break
                
                try:
                    # Try to parse JSON
                    review = json.loads(line)
                    reviews.append(review)
                    
                except json.JSONDecodeError as e:
                    errors += 1
                    if errors <= 10:  # Print first 10 errors
                        print(f"  [WARNING] Line {i+1}: JSON error - {str(e)[:100]}")
                    continue
                
                except Exception as e:
                    errors += 1
                    if errors <= 10:
                        print(f"  [WARNING] Line {i+1}: Unexpected error - {str(e)[:100]}")
                    continue
    
    except Exception as e:
        # Handle gzip corruption - use what we loaded so far
        print(f"\n[ERROR] File corruption detected: {str(e)[:200]}")
        print(f"Using {len(reviews)} reviews loaded before corruption")
    
    print(f"✓ Loaded {len(reviews)} reviews")
    if errors > 0:
        print(f"  Skipped {errors} malformed lines ({errors/(len(reviews)+errors)*100:.1f}%)")
    
    return reviews


def load_metadata_robust(file_path):
    """
    Load item metadata with robust error handling.
    
    Args:
        file_path: Path to metadata file
    
    Returns:
        dict mapping item_id -> metadata
    """
    print(f"Loading metadata from: {file_path}")
    
    items = {}
    errors = 0
    
    open_func = gzip.open if str(file_path).endswith('.gz') else open
    
    try:
        with open_func(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading metadata")):
                try:
                    item = json.loads(line)
                    
                    # Use correct field name for Books
                    item_id = item.get('parent_asin') or item.get('asin')
                    if item_id:
                        items[item_id] = item
                        
                except json.JSONDecodeError:
                    errors += 1
                    continue
                except Exception:
                    errors += 1
                    continue
    
    except Exception as e:
        # Handle gzip corruption
        print(f"\n[ERROR] File corruption detected: {str(e)[:200]}")
        print(f"Using {len(items)} items loaded before corruption")
    
    print(f"✓ Loaded {len(items)} items")
    if errors > 0:
        print(f"  Skipped {errors} malformed lines")
    
    return items


def process_books_dataset(
    reviews_file='../data/amazon_books/Books.jsonl.gz',
    meta_file='../data/amazon_books/meta_Books.jsonl.gz',
    min_user_interactions=10,
    min_item_interactions=10,
    output_file='../data/amazon_books/Books_processed.pkl'
):
    """
    Process Books dataset with robust error handling.
    
    Args:
        reviews_file: Path to reviews file
        meta_file: Path to metadata file
        min_user_interactions: Minimum reviews per user
        min_item_interactions: Minimum reviews per item
        output_file: Where to save processed data
    
    Returns:
        dict with processed data
    """
    
    # ===== STEP 1: Load reviews =====
    reviews = load_reviews_robust(reviews_file)
    
    if len(reviews) == 0:
        raise ValueError("No reviews loaded! Check file path and format.")
    
    print("\n" + "="*60)
    print("STEP 1: Raw Reviews Statistics")
    print("="*60)
    print(f"Total reviews: {len(reviews)}")
    
    # Check field names
    sample = reviews[0]
    print(f"\nSample review fields: {list(sample.keys())}")
    
    # ===== STEP 2: Extract core fields =====
    print("\n" + "="*60)
    print("STEP 2: Extracting Fields")
    print("="*60)
    
    processed_reviews = []
    for r in tqdm(reviews, desc="Processing reviews"):
        try:
            # Use correct field names for Books dataset
            user_id = r.get('reviewerID') or r.get('user_id')
            item_id = r.get('asin') or r.get('parent_asin')
            rating = r.get('overall') or r.get('rating')
            timestamp = r.get('unixReviewTime') or r.get('timestamp')
            
            if user_id and item_id and rating is not None:
                processed_reviews.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': float(rating),
                    'timestamp': int(timestamp) if timestamp else 0,
                })
        except Exception:
            continue
    
    print(f"✓ Processed {len(processed_reviews)} reviews with valid fields")
    
    # ===== STEP 3: Filter by minimum interactions =====
    print("\n" + "="*60)
    print("STEP 3: Filtering")
    print("="*60)
    
    df = pd.DataFrame(processed_reviews)
    
    print(f"Before filtering:")
    print(f"  Users: {df['user_id'].nunique()}")
    print(f"  Items: {df['item_id'].nunique()}")
    print(f"  Reviews: {len(df)}")
    
    # Iterative filtering
    for iteration in range(5):  # Max 5 iterations
        # Filter users
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Filter items
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df['item_id'].isin(valid_items)]
        
        print(f"  Iteration {iteration+1}: {len(df)} reviews, "
              f"{df['user_id'].nunique()} users, "
              f"{df['item_id'].nunique()} items")
    
    # ===== STEP 4: Build user histories =====
    print("\n" + "="*60)
    print("STEP 4: Building User Histories")
    print("="*60)
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    user_histories = defaultdict(list)
    for _, row in df.iterrows():
        user_histories[row['user_id']].append({
            'item_id': row['item_id'],
            'rating': row['rating'],
            'timestamp': row['timestamp']
        })
    
    # Convert to regular dict
    user_histories = dict(user_histories)
    
    # ===== STEP 5: Train/test split =====
    print("\n" + "="*60)
    print("STEP 5: Train/Test Split")
    print("="*60)
    
    train_histories = {}
    test_interactions = []
    
    for user_id, history in user_histories.items():
        if len(history) < 2:
            continue
        
        # Use all but last as train
        train_histories[user_id] = history[:-1]
        
        # Last item as test
        last_item = history[-1]
        test_interactions.append((
            user_id,
            last_item['item_id'],
            last_item['rating'] / 5.0  # Normalize to [0, 1]
        ))
    
    print(f"✓ Train users: {len(train_histories)}")
    print(f"✓ Test interactions: {len(test_interactions)}")
    
    # ===== STEP 6: Load metadata =====
    print("\n" + "="*60)
    print("STEP 6: Loading Item Metadata")
    print("="*60)
    
    # Get list of items we actually need
    valid_item_ids = set(df['item_id'].unique())
    print(f"Need metadata for {len(valid_item_ids)} items")
    
    # Load only needed items
    items_dict = {}
    errors = 0
    
    open_func = gzip.open if str(meta_file).endswith('.gz') else open
    
    try:
        with open_func(meta_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading metadata")):
                try:
                    item = json.loads(line)
                    item_id = item.get('parent_asin') or item.get('asin')
                    
                    # Only keep if we need it
                    if item_id and item_id in valid_item_ids:
                        items_dict[item_id] = item
                        
                        # Early exit if we got everything
                        if len(items_dict) >= len(valid_item_ids):
                            print(f"\n✓ Found all {len(items_dict)} needed items, stopping early")
                            break
                            
                except:
                    errors += 1
                    continue
                    
    except Exception as e:
        print(f"\n[WARNING] Metadata loading interrupted: {str(e)[:200]}")
    
    print(f"✓ Loaded {len(items_dict)} items with metadata (needed {len(valid_item_ids)})")
    
    # ===== STEP 7: Save processed data =====
    print("\n" + "="*60)
    print("STEP 7: Saving")
    print("="*60)
    
    dataset = {
        'reviews_df': df,
        'items_dict': items_dict,
        'user_histories': user_histories,
        'train_histories': train_histories,
        'test_interactions': test_interactions
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"✓ Saved to: {output_path}")
    
    # ===== SUMMARY =====
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Users: {len(train_histories)}")
    print(f"Items: {len(items_dict)}")
    print(f"Reviews: {len(df)}")
    print(f"Test interactions: {len(test_interactions)}")
    print(f"Avg reviews/user: {len(df) / len(train_histories):.1f}")
    print(f"Avg rating: {df['rating'].mean():.2f}")
    
    return dataset


if __name__ == "__main__":
    # Process Books dataset
    dataset = process_books_dataset(
        reviews_file='../data/amazon_books/Books.jsonl.gz',
        meta_file='../data/amazon_books/meta_Books.jsonl.gz',
        min_user_interactions=10,
        min_item_interactions=10,
        output_file='../data/amazon_books/Books_processed.pkl'
    )
    
    print("\n✓ Books dataset processed successfully!")