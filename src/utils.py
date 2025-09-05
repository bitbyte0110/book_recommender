import pandas as pd
import numpy as np
import os
import pickle
from typing import Tuple, Optional

def load_sim_matrix(matrix_type: str) -> Optional[np.ndarray]:
    """
    Load similarity matrix from disk.
    
    Args:
        matrix_type: Type of matrix ('content' or 'collab')
    
    Returns:
        similarity_matrix: Loaded similarity matrix or None if not found
    """
    if matrix_type == 'content':
        filepath = 'data/processed/content_sim_matrix.npy'
    elif matrix_type == 'collab':
        filepath = 'data/processed/collab_sim_matrix.npy'
    else:
        raise ValueError("matrix_type must be 'content' or 'collab'")
    
    try:
        return np.load(filepath)
    except FileNotFoundError:
        return None

def save_sim_matrix(similarity_matrix: np.ndarray, matrix_type: str) -> None:
    """
    Save similarity matrix to disk.
    
    Args:
        similarity_matrix: Matrix to save
        matrix_type: Type of matrix ('content' or 'collab')
    """
    if matrix_type == 'content':
        filepath = 'data/processed/content_sim_matrix.npy'
    elif matrix_type == 'collab':
        filepath = 'data/processed/collab_sim_matrix.npy'
    else:
        raise ValueError("matrix_type must be 'content' or 'collab'")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, similarity_matrix)

def load_books_data() -> pd.DataFrame:
    """
    Load cleaned books data.
    
    Returns:
        books_df: DataFrame with book information
    """
    try:
        filepath = 'data/processed/books_clean.csv'
        if os.path.exists(filepath):
            books_df = pd.read_csv(filepath)
            return books_df
        else:
            # If no processed data exists, load and clean raw data
            from .data_processing import load_and_clean_data
            books_df = load_and_clean_data()
            return books_df
    except Exception as e:
        print(f"Error loading books data: {e}")
        return pd.DataFrame()

def load_ratings_data() -> pd.DataFrame:
    """
    Load ratings data.
    
    Returns:
        ratings_df: DataFrame with user ratings
    """
    try:
        filepath = 'data/processed/ratings.csv'
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            raise FileNotFoundError("Ratings data not found")
    except Exception as e:
        print(f"Error loading ratings data: {e}")
        return pd.DataFrame()

def get_book_titles(books_df: pd.DataFrame) -> list:
    """
    Get list of book titles for dropdown selection.
    
    Args:
        books_df: DataFrame with book information
    
    Returns:
        titles: List of book titles
    """
    return books_df['title'].tolist()

def get_genres(books_df: pd.DataFrame) -> list:
    """
    Get list of unique publishers (since genre column doesn't exist).
    
    Args:
        books_df: DataFrame with book information
    
    Returns:
        publishers: List of unique publishers
    """
    return sorted(books_df['publisher'].unique().tolist())

def filter_books_by_genre(books_df: pd.DataFrame, selected_genres: list) -> pd.DataFrame:
    """
    Filter books by selected publishers (since genre column doesn't exist).
    
    Args:
        books_df: DataFrame with book information
        selected_genres: List of publishers to filter by
    
    Returns:
        filtered_df: Filtered DataFrame
    """
    if not selected_genres:
        return books_df
    return books_df[books_df['publisher'].isin(selected_genres)]

def search_books(books_df: pd.DataFrame, search_term: str) -> pd.DataFrame:
    """
    Search books by title or author.
    
    Args:
        books_df: DataFrame with book information
        search_term: Search term
    
    Returns:
        filtered_df: Filtered DataFrame
    """
    if not search_term:
        return books_df
    
    search_term = search_term.lower()
    mask = (
        books_df['title'].str.lower().str.contains(search_term, na=False) |
        books_df['authors'].str.lower().str.contains(search_term, na=False)
    )
    return books_df[mask]

def format_rating(rating: float) -> str:
    """
    Format rating for display.
    
    Args:
        rating: Rating value
    
    Returns:
        formatted_rating: Formatted rating string
    """
    if pd.isna(rating) or rating == 0:
        return "No rating"
    return f"{rating:.1f} ⭐"

def format_similarity_score(score: float) -> str:
    """
    Format similarity score for display.
    
    Args:
        score: Similarity score
    
    Returns:
        formatted_score: Formatted score string
    """
    return f"{score:.3f}"

def get_book_cover_url(book_title: str, author: str) -> str:
    """
    Generate a placeholder book cover URL (in a real app, this would use a book cover API).
    
    Args:
        book_title: Book title
        author: Book author
    
    Returns:
        cover_url: Placeholder cover URL
    """
    # This is a placeholder - in a real app, you'd use Google Books API or similar
    return f"https://via.placeholder.com/150x200/4A90E2/FFFFFF?text={book_title[:20]}"

def calculate_recommendation_metrics(recommendations: list) -> dict:
    """
    Calculate metrics for recommendation quality.
    
    Args:
        recommendations: List of recommendations
    
    Returns:
        metrics: Dictionary with recommendation metrics
    """
    if not recommendations:
        return {}
    
    publishers = [rec.get('genre', 'Unknown') for rec in recommendations]  # Using genre field which contains publisher
    ratings = [rec.get('rating', 0) for rec in recommendations]
    scores = [rec.get('hybrid_score', 0) for rec in recommendations]
    
    metrics = {
        'total_recommendations': len(recommendations),
        'unique_genres': len(set(publishers)),
        'genre_diversity': len(set(publishers)) / len(publishers) if publishers else 0,
        'avg_rating': np.mean([r for r in ratings if r > 0]) if any(r > 0 for r in ratings) else 0,
        'avg_similarity_score': np.mean(scores) if scores else 0,
        'genre_distribution': pd.Series(publishers).value_counts().to_dict()
    }
    
    return metrics

def save_model(model, filepath: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Model to save
        filepath: Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath: str):
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the model file
    
    Returns:
        model: Loaded model
    """
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def validate_data_integrity(books_df: pd.DataFrame, ratings_df: pd.DataFrame) -> dict:
    """
    Validate the integrity of the data.
    
    Args:
        books_df: DataFrame with book information
        ratings_df: DataFrame with user ratings
    
    Returns:
        validation_results: Dictionary with validation results
    """
    results = {
        'books_total': len(books_df),
        'books_with_missing_titles': books_df['title'].isna().sum(),
        'books_with_missing_authors': books_df['authors'].isna().sum(),
        'books_with_missing_publishers': books_df['publisher'].isna().sum(),
        'unique_publishers': books_df['publisher'].nunique(),
        'ratings_total': len(ratings_df) if not ratings_df.empty else 0,
        'unique_users': ratings_df['user_id'].nunique() if not ratings_df.empty else 0,
        'unique_rated_books': ratings_df['book_id'].nunique() if not ratings_df.empty else 0,
        'avg_rating': ratings_df['rating'].mean() if not ratings_df.empty else 0
    }
    
    # Check for orphaned ratings (books that don't exist in books_df)
    if not ratings_df.empty:
        orphaned_ratings = ratings_df[~ratings_df['book_id'].isin(books_df['book_id'])]
        results['orphaned_ratings'] = len(orphaned_ratings)
    
    return results

def get_system_info() -> dict:
    """
    Get system information for debugging.
    
    Returns:
        info: Dictionary with system information
    """
    import sys
    import platform
    
    return {
        'python_version': sys.version,
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'working_directory': os.getcwd(),
        'data_directory_exists': os.path.exists('data'),
        'processed_directory_exists': os.path.exists('data/processed'),
        'raw_directory_exists': os.path.exists('data/raw')
    }
