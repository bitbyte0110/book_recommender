import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def create_content_similarity_matrix(books_df):
    """
    Create content-based similarity matrix using TF-IDF and cosine similarity.
    
    Args:
        books_df: DataFrame with book information including 'combined_features'
    
    Returns:
        similarity_matrix: Cosine similarity matrix
        tfidf_matrix: TF-IDF matrix for later use
    """
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9
    )
    
    # Fit and transform the combined features
    tfidf_matrix = tfidf.fit_transform(books_df['combined_features'])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return similarity_matrix, tfidf_matrix

def get_content_based_recommendations(book_title, books_df, similarity_matrix, top_n=10):
    """
    Get content-based recommendations for a given book.
    
    Args:
        book_title: Title of the book to find recommendations for
        books_df: DataFrame with book information
        similarity_matrix: Pre-computed similarity matrix
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
        book_similarities = similarity_matrix[book_idx]
        
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

def save_content_similarity_matrix(similarity_matrix, filepath='data/processed/content_sim_matrix.npy'):
    """
    Save the content similarity matrix to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, similarity_matrix)

def load_content_similarity_matrix(filepath='data/processed/content_sim_matrix.npy'):
    """
    Load the content similarity matrix from disk.
    """
    try:
        return np.load(filepath)
    except FileNotFoundError:
        return None

def get_book_features(books_df, book_title):
    """
    Extract features for a specific book.
    
    Args:
        books_df: DataFrame with book information
        book_title: Title of the book
    
    Returns:
        features: Dictionary with book features
    """
    try:
        import re
        escaped_title = re.escape(book_title.lower())
        matches = books_df[books_df['title'].str.lower().str.contains(escaped_title, regex=True, na=False)]
        
        if len(matches) == 0:
            raise IndexError("Book not found")
        
        # If multiple matches, prefer the one with the lowest book_id (usually the first one)
        if len(matches) > 1:
            # Sort by book_id and take the first one
            matches = matches.sort_values('book_id')
        
        book = matches.iloc[0]
        return {
            'title': book['title'],
            'authors': book['authors'],
            'genre': book.get('publisher', 'Unknown'),
            'description': book.get('description', ''),
            'rating': book.get('average_rating', 0),
            'pages': book.get('num_pages', 0)
        }
    except (IndexError, KeyError):
        return None

def analyze_genre_similarity(books_df, similarity_matrix):
    """
    Analyze similarity patterns within and across publishers.
    
    Args:
        books_df: DataFrame with book information
        similarity_matrix: Pre-computed similarity matrix
    
    Returns:
        genre_analysis: Dictionary with publisher similarity statistics
    """
    publishers = books_df['publisher'].unique()
    genre_analysis = {}
    
    for publisher in publishers:
        publisher_books = books_df[books_df['publisher'] == publisher]
        publisher_indices = publisher_books.index
        
        if len(publisher_indices) > 1:
            # Calculate average similarity within publisher
            within_similarities = []
            for i in publisher_indices:
                for j in publisher_indices:
                    if i != j:
                        within_similarities.append(similarity_matrix[i, j])
            
            genre_analysis[publisher] = {
                'count': len(publisher_indices),
                'avg_within_similarity': np.mean(within_similarities),
                'std_within_similarity': np.std(within_similarities)
            }
    
    return genre_analysis
