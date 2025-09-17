#!/usr/bin/env python3
"""
Test script to validate hybrid recommendation improvements.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from src.utils import load_books_data, load_ratings_data
from src.content_based import create_content_similarity_matrix
from src.collaborative import create_user_item_matrix, create_item_item_similarity_matrix
from src.hybrid import hybrid_recommend, find_optimal_hybrid_config, optimize_alpha

def test_hybrid_improvements():
    """Test the improved hybrid recommendation system."""
    print("Testing Hybrid Recommendation Improvements")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    books_df = load_books_data()
    ratings_df = load_ratings_data()
    
    if books_df.empty or ratings_df.empty:
        print("Error: No data available for testing")
        return
    
    # Create similarity matrices
    print("Creating similarity matrices...")
    content_sim_matrix, _ = create_content_similarity_matrix(books_df)
    user_item_matrix = create_user_item_matrix(ratings_df, books_df)
    collab_sim_matrix = create_item_item_similarity_matrix(user_item_matrix)
    
    # Test with a sample book
    test_book = books_df.iloc[0]['title']
    print(f"Testing with book: {test_book}")
    
    # Test different configurations
    configs = [
        {'name': 'Original (min-max)', 'use_candidate_union': False, 'use_rank_fusion': False, 'alpha': 0.5},
        {'name': 'Improved (rank-based)', 'use_candidate_union': True, 'use_rank_fusion': False, 'alpha': 0.5},
        {'name': 'Rank Fusion', 'use_candidate_union': True, 'use_rank_fusion': True, 'alpha': 0.5},
        {'name': 'Content-heavy', 'use_candidate_union': True, 'use_rank_fusion': False, 'alpha': 0.8},
        {'name': 'Collaborative-heavy', 'use_candidate_union': True, 'use_rank_fusion': False, 'alpha': 0.2},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        try:
            recommendations = hybrid_recommend(
                test_book, books_df, content_sim_matrix, collab_sim_matrix,
                top_n=10, **{k: v for k, v in config.items() if k != 'name'}
            )
            
            if recommendations:
                # Calculate metrics
                hybrid_scores = [rec['hybrid_score'] for rec in recommendations]
                content_scores = [rec['content_score'] for rec in recommendations]
                collab_scores = [rec['collab_score'] for rec in recommendations]
                
                # Publisher diversity
                publishers = [rec['genre'] for rec in recommendations]
                unique_publishers = len(set(publishers))
                diversity = unique_publishers / len(publishers) if publishers else 0
                
                # Score statistics
                avg_hybrid = np.mean(hybrid_scores)
                std_hybrid = np.std(hybrid_scores)
                avg_content = np.mean(content_scores)
                avg_collab = np.mean(collab_scores)
                
                results[config['name']] = {
                    'num_recs': len(recommendations),
                    'avg_hybrid_score': avg_hybrid,
                    'std_hybrid_score': std_hybrid,
                    'avg_content_score': avg_content,
                    'avg_collab_score': avg_collab,
                    'diversity': diversity,
                    'unique_publishers': unique_publishers
                }
                
                print(f"  Recommendations: {len(recommendations)}")
                print(f"  Avg Hybrid Score: {avg_hybrid:.4f} ± {std_hybrid:.4f}")
                print(f"  Avg Content Score: {avg_content:.4f}")
                print(f"  Avg Collab Score: {avg_collab:.4f}")
                print(f"  Diversity: {diversity:.4f} ({unique_publishers} unique publishers)")
                
            else:
                print("  No recommendations generated")
                results[config['name']] = None
                
        except Exception as e:
            print(f"  Error: {e}")
            results[config['name']] = None
    
    # Find optimal configuration
    print(f"\nFinding optimal configuration for {test_book}...")
    try:
        optimal_config = find_optimal_hybrid_config(test_book, books_df, content_sim_matrix, collab_sim_matrix)
        if optimal_config:
            print(f"Optimal config: {optimal_config}")
        else:
            print("Could not find optimal configuration")
    except Exception as e:
        print(f"Error finding optimal config: {e}")
    
    # Test alpha optimization
    print(f"\nTesting alpha optimization...")
    try:
        optimal_alpha, scores = optimize_alpha(test_book, books_df, content_sim_matrix, collab_sim_matrix, metric='f1')
        print(f"Optimal alpha for F1: {optimal_alpha}")
        print(f"Alpha scores: {[f'{s:.4f}' for s in scores]}")
    except Exception as e:
        print(f"Error optimizing alpha: {e}")
    
    # Summary
    print(f"\nSummary of Results:")
    print("-" * 30)
    for name, result in results.items():
        if result:
            print(f"{name:20} | Score: {result['avg_hybrid_score']:.4f} | Diversity: {result['diversity']:.4f}")
        else:
            print(f"{name:20} | No results")

if __name__ == "__main__":
    test_hybrid_improvements()
