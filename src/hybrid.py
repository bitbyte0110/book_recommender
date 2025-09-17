import pandas as pd
import numpy as np
from .content_based import get_content_based_recommendations
from .collaborative import get_collaborative_recommendations

def get_method_label(alpha, collab_data_available, matrices_match):
    """
    Determine the correct method label based on alpha value and data availability.
    
    Args:
        alpha: Weight for content-based filtering (0 = pure collaborative, 1 = pure content)
        collab_data_available: Whether collaborative filtering data is available
        matrices_match: Whether content and collaborative matrices have matching shapes
    
    Returns:
        method_label: String indicating the recommendation method used
    """
    if not collab_data_available or not matrices_match:
        return 'content_based'
    elif alpha == 1.0:
        return 'content_based'
    elif alpha == 0.0:
        return 'collaborative'
    else:
        return 'hybrid'

def get_actual_method(alpha, is_hybrid, collab_data_available):
    """
    Determine the actual method being used based on alpha value and data availability.
    
    Args:
        alpha: Weight for content-based filtering (0 = pure collaborative, 1 = pure content)
        is_hybrid: Whether hybrid scoring is being used
        collab_data_available: Whether collaborative filtering data is available
    
    Returns:
        method_label: String indicating the actual recommendation method used
    """
    if not collab_data_available:
        return 'content_based'
    elif alpha == 0.0:
        return 'collaborative'
    elif alpha == 1.0:
        return 'content_based'
    else:
        return 'hybrid'

def calculate_similarity_penalty(idx1, idx2, books_df, content_sim_matrix):
    """
    Calculate similarity penalty between two books for diversity filtering.
    
    Args:
        idx1: Index of first book
        idx2: Index of second book
        books_df: DataFrame with book information
        content_sim_matrix: Content-based similarity matrix
    
    Returns:
        penalty: Similarity penalty value (0.0-1.0)
    """
    # Use content similarity as the base penalty
    content_sim = content_sim_matrix[idx1, idx2]
    
    # Additional penalty based on same author
    author1 = str(books_df.iloc[idx1].get('authors', '')).lower()
    author2 = str(books_df.iloc[idx2].get('authors', '')).lower()
    author_penalty = 0.3 if author1 == author2 and author1 != '' else 0.0
    
    # Additional penalty based on same publisher
    publisher1 = str(books_df.iloc[idx1].get('publisher', '')).lower()
    publisher2 = str(books_df.iloc[idx2].get('publisher', '')).lower()
    publisher_penalty = 0.2 if publisher1 == publisher2 and publisher1 != '' else 0.0
    
    # Additional penalty based on same genre (if available)
    genre1 = str(books_df.iloc[idx1].get('genre', '')).lower()
    genre2 = str(books_df.iloc[idx2].get('genre', '')).lower()
    genre_penalty = 0.1 if genre1 == genre2 and genre1 != '' else 0.0
    
    # Combine penalties
    total_penalty = content_sim + author_penalty + publisher_penalty + genre_penalty
    return min(total_penalty, 1.0)  # Cap at 1.0

def hybrid_recommend(book_title, books_df, content_sim_matrix, collab_sim_matrix, 
                    alpha=0.6, top_n=10, min_similarity=0.1, diversity_weight=0.3, 
                    fallback_to_content=True):
    """
    Generate hybrid recommendations combining content-based and collaborative filtering.
    
    Args:
        book_title: Title of the book to find recommendations for
        books_df: DataFrame with book information
        content_sim_matrix: Content-based similarity matrix
        collab_sim_matrix: Collaborative filtering similarity matrix
        alpha: Weight for content-based filtering (0 = pure collaborative, 1 = pure content)
        top_n: Number of recommendations to return
        min_similarity: Minimum similarity score threshold (0.0-1.0)
        diversity_weight: Weight for diversity penalty (0.0-1.0)
        fallback_to_content: Whether to fall back to content-based if collaborative fails
    
    Returns:
        hybrid_recommendations: List of recommended books with hybrid scores
    """
    try:
        # Find the book index (handle partial matches and duplicates)
        import re
        escaped_title = re.escape(book_title.lower())
        matches = books_df[books_df['title'].str.lower().str.contains(escaped_title, regex=True, na=False)]
        
        if len(matches) == 0:
            raise IndexError("Book not found")
        
        # If multiple matches, prefer the one with the lowest book_id (usually the first one)
        if len(matches) > 1:
            # Sort by book_id and take the first one
            matches = matches.sort_values('book_id')
        
        book_idx = matches.index[0]
        
        # Get content-based scores
        content_scores = content_sim_matrix[book_idx]
        
        # Initialize variables
        collab_data_available = False
        collab_scores = np.zeros_like(content_scores)
        is_hybrid = False
        
        # Check if collaborative filtering matrix has the same shape
        if collab_sim_matrix is not None and collab_sim_matrix.shape[0] == content_sim_matrix.shape[0]:
            # Get collaborative scores
            collab_scores = collab_sim_matrix[book_idx]
            
            # Check if collaborative filtering has enough data
            collab_data_available = np.sum(collab_scores > 0) > 1
            
            if collab_data_available:
                # Enhanced normalization with better F1-score optimization
                content_scores_norm = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)
                collab_scores_norm = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min() + 1e-8)
                
                # Advanced ensemble with boosting for better F1-score
                # Apply exponential boosting to high-confidence predictions
                content_boosted = np.power(content_scores_norm, 0.8)  # Slight boost for content
                collab_boosted = np.power(collab_scores_norm, 0.9)   # Stronger boost for collaborative
                
                # Weighted combination with dynamic alpha adjustment
                dynamic_alpha = alpha * (1 + 0.1 * np.mean(collab_scores_norm))  # Boost alpha if collaborative is strong
                hybrid_scores = dynamic_alpha * content_boosted + (1 - dynamic_alpha) * collab_boosted
                
                # Determine the actual method based on alpha value
                if alpha == 0.0:
                    is_hybrid = False  # Pure collaborative
                elif alpha == 1.0:
                    is_hybrid = False  # Pure content-based
                else:
                    is_hybrid = True   # True hybrid
            else:
                # Fall back to content-based only
                if fallback_to_content:
                    hybrid_scores = content_scores
                else:
                    return []
        else:
            # Collaborative matrix has different shape or doesn't exist, use content-based only
            if fallback_to_content:
                hybrid_scores = content_scores
            else:
                return []
        
        # Apply minimum similarity filter
        valid_indices = np.where(hybrid_scores >= min_similarity)[0]
        # Exclude the book itself
        valid_indices = valid_indices[valid_indices != book_idx]
        
        if len(valid_indices) == 0:
            return []  # No recommendations meet the threshold
        
        # Apply diversity penalty if enabled
        if diversity_weight > 0:
            # Greedy selection with diversity penalty
            selected_indices = []
            remaining_indices = list(valid_indices)
            
            for _ in range(min(top_n, len(remaining_indices))):
                if not remaining_indices:
                    break
                
                # Calculate diversity-penalized scores for remaining books
                penalized_scores = hybrid_scores[remaining_indices].copy()
                
                for i, idx in enumerate(remaining_indices):
                    penalty = 0
                    for selected_idx in selected_indices:
                        similarity_penalty = calculate_similarity_penalty(
                            idx, selected_idx, books_df, content_sim_matrix
                        )
                        penalty += similarity_penalty * diversity_weight
                    
                    penalized_scores[i] = hybrid_scores[idx] - penalty
                
                # Select the book with the highest penalized score
                best_idx_pos = np.argmax(penalized_scores)
                best_idx = remaining_indices[best_idx_pos]
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            similar_indices = selected_indices
        else:
            # No diversity penalty, just sort by score
            sorted_indices = np.argsort(hybrid_scores[valid_indices])[::-1]
            similar_indices = valid_indices[sorted_indices][:top_n]
        
        # Create recommendations list
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'book_id': books_df.iloc[idx]['book_id'],
                'title': books_df.iloc[idx]['title'],
                'authors': books_df.iloc[idx]['authors'],
                'genre': books_df.iloc[idx].get('publisher', 'Unknown'),
                'hybrid_score': hybrid_scores[idx],
                'content_score': content_scores[idx],
                'collab_score': collab_scores[idx] if collab_data_available else 0,
                'rating': books_df.iloc[idx].get('average_rating', 0),
                'method': get_actual_method(alpha, is_hybrid, collab_data_available)
            })
        
        return recommendations
    
    except (IndexError, KeyError):
        return []

def get_separate_recommendations(book_title, books_df, content_sim_matrix, collab_sim_matrix, top_n=10):
    """
    Get separate content-based and collaborative recommendations for comparison.
    
    Args:
        book_title: Title of the book to find recommendations for
        books_df: DataFrame with book information
        content_sim_matrix: Content-based similarity matrix
        collab_sim_matrix: Collaborative filtering similarity matrix
        top_n: Number of recommendations to return
    
    Returns:
        dict: Dictionary containing separate recommendations
    """
    content_recs = get_content_based_recommendations(book_title, books_df, content_sim_matrix, top_n)
    collab_recs = get_collaborative_recommendations(book_title, books_df, collab_sim_matrix, top_n)
    
    return {
        'content_based': content_recs,
        'collaborative': collab_recs
    }

def analyze_recommendation_overlap(content_recs, collab_recs, hybrid_recs):
    """
    Analyze overlap between different recommendation methods.
    
    Args:
        content_recs: Content-based recommendations
        collab_recs: Collaborative filtering recommendations
        hybrid_recs: Hybrid recommendations
    
    Returns:
        overlap_analysis: Dictionary with overlap statistics
    """
    content_titles = set([rec['title'] for rec in content_recs])
    collab_titles = set([rec['title'] for rec in collab_recs])
    hybrid_titles = set([rec['title'] for rec in hybrid_recs])
    
    # Calculate overlaps
    content_collab_overlap = len(content_titles.intersection(collab_titles))
    content_hybrid_overlap = len(content_titles.intersection(hybrid_titles))
    collab_hybrid_overlap = len(collab_titles.intersection(hybrid_titles))
    
    # Calculate Jaccard similarity
    jaccard_content_collab = content_collab_overlap / len(content_titles.union(collab_titles)) if len(content_titles.union(collab_titles)) > 0 else 0
    jaccard_content_hybrid = content_hybrid_overlap / len(content_titles.union(hybrid_titles)) if len(content_titles.union(hybrid_titles)) > 0 else 0
    jaccard_collab_hybrid = collab_hybrid_overlap / len(collab_titles.union(hybrid_titles)) if len(collab_titles.union(hybrid_titles)) > 0 else 0
    
    return {
        'content_collab_overlap': content_collab_overlap,
        'content_hybrid_overlap': content_hybrid_overlap,
        'collab_hybrid_overlap': collab_hybrid_overlap,
        'jaccard_content_collab': jaccard_content_collab,
        'jaccard_content_hybrid': jaccard_content_hybrid,
        'jaccard_collab_hybrid': jaccard_collab_hybrid,
        'total_unique_recommendations': len(content_titles.union(collab_titles).union(hybrid_titles))
    }

def optimize_alpha(book_title, books_df, content_sim_matrix, collab_sim_matrix, 
                  test_alphas=np.arange(0, 1.1, 0.1), top_n=10):
    """
    Find optimal alpha value for hybrid recommendations.
    
    Args:
        book_title: Title of the book to test
        books_df: DataFrame with book information
        content_sim_matrix: Content-based similarity matrix
        collab_sim_matrix: Collaborative filtering similarity matrix
        test_alphas: Array of alpha values to test
        top_n: Number of recommendations to return
    
    Returns:
        optimal_alpha: Alpha value that maximizes diversity
    """
    diversity_scores = []
    
    for alpha in test_alphas:
        recommendations = hybrid_recommend(book_title, books_df, content_sim_matrix, 
                                         collab_sim_matrix, alpha, top_n)
        
        if recommendations:
            # Calculate diversity based on publisher variety
            publishers = [rec['genre'] for rec in recommendations]  # genre field contains publisher
            unique_publishers = len(set(publishers))
            diversity_score = unique_publishers / len(publishers) if len(publishers) > 0 else 0
            diversity_scores.append(diversity_score)
        else:
            diversity_scores.append(0)
    
    # Find alpha with maximum diversity
    optimal_alpha_idx = np.argmax(diversity_scores)
    optimal_alpha = test_alphas[optimal_alpha_idx]
    
    return optimal_alpha, diversity_scores

def get_personalized_recommendations(user_preferences, books_df, content_sim_matrix, 
                                   collab_sim_matrix, alpha=0.6, top_n=10):
    """
    Get personalized recommendations based on user preferences.
    
    Args:
        user_preferences: List of book titles the user likes
        books_df: DataFrame with book information
        content_sim_matrix: Content-based similarity matrix
        collab_sim_matrix: Collaborative filtering similarity matrix
        alpha: Weight for content-based filtering
        top_n: Number of recommendations to return
    
    Returns:
        personalized_recs: List of personalized recommendations
    """
    if not user_preferences:
        return []
    
    # Get recommendations for each preferred book
    all_recommendations = []
    for book_title in user_preferences:
        recs = hybrid_recommend(book_title, books_df, content_sim_matrix, 
                              collab_sim_matrix, alpha, top_n)
        all_recommendations.extend(recs)
    
    if not all_recommendations:
        return []
    
    # Aggregate scores by book
    book_scores = {}
    for rec in all_recommendations:
        book_id = rec['book_id']
        if book_id not in book_scores:
            book_scores[book_id] = {
                'title': rec['title'],
                'authors': rec['authors'],
                'genre': rec['genre'],
                'total_score': 0,
                'count': 0,
                'rating': rec['rating']
            }
        
        book_scores[book_id]['total_score'] += rec['hybrid_score']
        book_scores[book_id]['count'] += 1
    
    # Calculate average scores and sort
    personalized_recs = []
    for book_id, scores in book_scores.items():
        avg_score = scores['total_score'] / scores['count']
        personalized_recs.append({
            'book_id': book_id,
            'title': scores['title'],
            'authors': scores['authors'],
            'genre': scores['genre'],
            'personalized_score': avg_score,
            'recommendation_count': scores['count'],
            'rating': scores['rating']
        })
    
    # Sort by personalized score and return top recommendations
    personalized_recs.sort(key=lambda x: x['personalized_score'], reverse=True)
    return personalized_recs[:top_n]
