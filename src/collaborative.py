import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

def create_user_item_matrix(ratings_df, books_df):
    """
    Create user-item rating matrix for collaborative filtering.
    
    Args:
        ratings_df: DataFrame with user ratings (user_id, book_id, rating)
        books_df: DataFrame with book information
    
    Returns:
        user_item_matrix: Pivot table with users as rows and books as columns
    """
    # Create pivot table
    user_item_matrix = ratings_df.pivot_table(
        index='user_id',
        columns='book_id',
        values='rating',
        fill_value=0
    )
    
    return user_item_matrix

def create_item_item_similarity_matrix(user_item_matrix):
    """
    Create item-item similarity matrix using cosine similarity.
    
    Args:
        user_item_matrix: User-item rating matrix
    
    Returns:
        item_similarity_matrix: Item-item similarity matrix
    """
    # Calculate item-item similarity using cosine similarity
    item_similarity_matrix = cosine_similarity(user_item_matrix.T)
    
    return item_similarity_matrix

def get_collaborative_recommendations(book_title, books_df, item_similarity_matrix, top_n=10):
    """
    Get collaborative filtering recommendations for a given book.
    
    Args:
        book_title: Title of the book to find recommendations for
        books_df: DataFrame with book information
        item_similarity_matrix: Pre-computed item-item similarity matrix
        top_n: Number of recommendations to return
    
    Returns:
        recommendations: List of recommended book indices and scores
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
        
        # Get similarity scores for this book
        book_similarities = item_similarity_matrix[book_idx]
        
        # Get indices of most similar books (excluding the book itself)
        similar_indices = np.argsort(book_similarities)[::-1][1:top_n+1]
        
        # Create recommendations list
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'book_id': books_df.iloc[idx]['book_id'],
                'title': books_df.iloc[idx]['title'],
                'authors': books_df.iloc[idx]['authors'],
                'genre': books_df.iloc[idx].get('publisher', 'Unknown'),
                'similarity_score': book_similarities[idx],
                'rating': books_df.iloc[idx].get('average_rating', 0)
            })
        
        return recommendations
    
    except (IndexError, KeyError):
        # If book not found, return empty list
        return []

def get_user_based_recommendations(user_id, user_item_matrix, books_df, top_n=10):
    """
    Get user-based collaborative filtering recommendations.
    
    Args:
        user_id: ID of the user
        user_item_matrix: User-item rating matrix
        books_df: DataFrame with book information
        top_n: Number of recommendations to return
    
    Returns:
        recommendations: List of recommended books for the user
    """
    try:
        # Get user's ratings
        user_ratings = user_item_matrix.loc[user_id]
        
        # Find books the user hasn't rated (rating = 0)
        unrated_books = user_ratings[user_ratings == 0].index
        
        if len(unrated_books) == 0:
            return []
        
        # Calculate predicted ratings for unrated books
        predicted_ratings = {}
        
        for book_id in unrated_books:
            # Find similar users who rated this book
            similar_users = user_item_matrix[user_item_matrix[book_id] > 0]
            
            if len(similar_users) > 0:
                # Calculate average rating for this book
                avg_rating = similar_users[book_id].mean()
                predicted_ratings[book_id] = avg_rating
        
        # Sort by predicted rating and get top recommendations
        sorted_books = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for book_id, predicted_rating in sorted_books[:top_n]:
            book_info = books_df[books_df['book_id'] == book_id].iloc[0]
            recommendations.append({
                'book_id': book_id,
                'title': book_info['title'],
                'authors': book_info['authors'],
                'genre': book_info.get('publisher', 'Unknown'),
                'predicted_rating': predicted_rating,
                'rating': book_info.get('average_rating', 0)
            })
        
        return recommendations
    
    except (KeyError, IndexError):
        return []

def save_collaborative_similarity_matrix(similarity_matrix, filepath='data/processed/collab_sim_matrix.npy'):
    """
    Save the collaborative similarity matrix to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, similarity_matrix)

def load_collaborative_similarity_matrix(filepath='data/processed/collab_sim_matrix.npy'):
    """
    Load the collaborative similarity matrix from disk.
    """
    try:
        return np.load(filepath)
    except FileNotFoundError:
        return None

def calculate_rating_statistics(ratings_df):
    """
    Calculate statistics about the ratings data.
    
    Args:
        ratings_df: DataFrame with user ratings
    
    Returns:
        stats: Dictionary with rating statistics
    """
    stats = {
        'total_ratings': len(ratings_df),
        'unique_users': ratings_df['user_id'].nunique(),
        'unique_books': ratings_df['book_id'].nunique(),
        'avg_rating': ratings_df['rating'].mean(),
        'rating_distribution': ratings_df['rating'].value_counts().sort_index().to_dict(),
        'sparsity': 1 - (len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['book_id'].nunique()))
    }
    
    return stats

def get_popular_books(ratings_df, books_df, top_n=10):
    """
    Get most popular books based on number of ratings.
    
    Args:
        ratings_df: DataFrame with user ratings
        books_df: DataFrame with book information
        top_n: Number of popular books to return
    
    Returns:
        popular_books: List of popular books with rating statistics
    """
    # Count ratings per book
    book_ratings = ratings_df.groupby('book_id').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    
    book_ratings.columns = ['book_id', 'rating_count', 'avg_rating']
    
    # Sort by rating count and get top books
    popular_books = book_ratings.sort_values('rating_count', ascending=False).head(top_n)
    
    # Merge with book information
    popular_books = popular_books.merge(books_df[['book_id', 'title', 'authors']], on='book_id')
    
    return popular_books.to_dict('records')
