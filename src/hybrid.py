import pandas as pd
import numpy as np
from .content_based import get_content_based_recommendations
from .collaborative import get_collaborative_recommendations

def hybrid_recommend(book_title, books_df, content_sim_matrix, collab_sim_matrix, 
                    alpha=0.6, top_n=10, fallback_to_content=True):
    """
    Generate hybrid recommendations combining content-based and collaborative filtering.
    
    Args:
        book_title: Title of the book to find recommendations for
        books_df: DataFrame with book information
        content_sim_matrix: Content-based similarity matrix
        collab_sim_matrix: Collaborative filtering similarity matrix
        alpha: Weight for content-based filtering (0 = pure collaborative, 1 = pure content)
        top_n: Number of recommendations to return
        fallback_to_content: Whether to fall back to content-based if collaborative fails
    
    Returns:
        hybrid_recommendations: List of recommended books with hybrid scores
    """
    try:
        # Find the book index
        book_idx = books_df[books_df['title'].str.lower() == book_title.lower()].index[0]
        
        # Get content-based scores
        content_scores = content_sim_matrix[book_idx]
        
        # Get collaborative scores
        collab_scores = collab_sim_matrix[book_idx]
        
        # Check if collaborative filtering has enough data
        collab_data_available = np.sum(collab_scores > 0) > 1
        
        if collab_data_available:
            # Normalize scores to 0-1 range
            content_scores_norm = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min())
            collab_scores_norm = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min())
            
            # Calculate hybrid scores
            hybrid_scores = alpha * content_scores_norm + (1 - alpha) * collab_scores_norm
        else:
            # Fall back to content-based only
            if fallback_to_content:
                hybrid_scores = content_scores
            else:
                return []
        
        # Get top recommendations (excluding the book itself)
        similar_indices = np.argsort(hybrid_scores)[::-1][1:top_n+1]
        
        # Create recommendations list
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'book_id': books_df.iloc[idx]['book_id'],
                'title': books_df.iloc[idx]['title'],
                'author': books_df.iloc[idx]['author'],
                'genre': books_df.iloc[idx]['genre'],
                'hybrid_score': hybrid_scores[idx],
                'content_score': content_scores[idx],
                'collab_score': collab_scores[idx] if collab_data_available else 0,
                'rating': books_df.iloc[idx].get('rating', 0),
                'method': 'hybrid' if collab_data_available else 'content_only'
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
            # Calculate diversity based on genre variety
            genres = [rec['genre'] for rec in recommendations]
            unique_genres = len(set(genres))
            diversity_score = unique_genres / len(genres) if len(genres) > 0 else 0
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
                'author': rec['author'],
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
            'author': scores['author'],
            'genre': scores['genre'],
            'personalized_score': avg_score,
            'recommendation_count': scores['count'],
            'rating': scores['rating']
        })
    
    # Sort by personalized score and return top recommendations
    personalized_recs.sort(key=lambda x: x['personalized_score'], reverse=True)
    return personalized_recs[:top_n]
