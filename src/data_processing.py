import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os

def load_and_clean_data():
    """
    Load and clean book data from the raw directory.
    Returns cleaned book data with merged features.
    """
    try:
        # Try to load from raw data directory
        books_path = 'data/raw/books.csv'
        if os.path.exists(books_path):
            try:
                books = pd.read_csv(books_path, on_bad_lines='skip')
            except Exception as e:
                print(f"Error loading books.csv: {e}")
                print("Creating sample books data instead")
                return create_sample_books()
        else:
            # Create sample data if no raw data exists
            books = create_sample_data()
        
        # Clean the data
        books_clean = clean_book_data(books)
        
        # Save cleaned data
        os.makedirs('data/processed', exist_ok=True)
        books_clean.to_csv('data/processed/books_clean.csv', index=False)
        
        return books_clean
    
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return sample data as fallback
        return create_sample_data()

def clean_book_data(books):
    """
    Clean and preprocess book data.
    """
    # Make a copy to avoid modifying original
    books_clean = books.copy()
    
    # Handle missing values
    books_clean['title'] = books_clean['title'].fillna('Unknown Title')
    books_clean['authors'] = books_clean['authors'].fillna('Unknown Author')
    books_clean['publisher'] = books_clean['publisher'].fillna('Unknown Publisher')
    books_clean['average_rating'] = books_clean['average_rating'].fillna(0.0)
    
    # Clean text fields
    books_clean['title'] = books_clean['title'].astype(str).str.strip()
    books_clean['authors'] = books_clean['authors'].astype(str).str.strip()
    books_clean['publisher'] = books_clean['publisher'].astype(str).str.strip()
    
    # Create combined features for content-based filtering
    books_clean['combined_features'] = (
        books_clean['title'] + ' ' + 
        books_clean['authors'] + ' ' + 
        books_clean['publisher']
    )
    
    # Remove special characters and normalize
    books_clean['combined_features'] = books_clean['combined_features'].apply(
        lambda x: re.sub(r'[^\w\s]', ' ', str(x)).lower()
    )
    
    # Ensure we have an index column
    if 'book_id' not in books_clean.columns:
        books_clean['book_id'] = range(len(books_clean))
    
    return books_clean

def create_sample_data():
    """
    Create sample book data for demonstration purposes.
    """
    sample_books = [
        {
            'book_id': 1,
            'title': 'The Great Gatsby',
            'authors': 'F. Scott Fitzgerald',
            'publisher': 'Scribner',
            'average_rating': 4.2,
            'num_pages': 180
        },
        {
            'book_id': 2,
            'title': 'To Kill a Mockingbird',
            'authors': 'Harper Lee',
            'publisher': 'Grand Central',
            'average_rating': 4.3,
            'num_pages': 281
        },
        {
            'book_id': 3,
            'title': '1984',
            'authors': 'George Orwell',
            'publisher': 'Signet',
            'description': 'A dystopian novel about totalitarianism and surveillance society.',
            'average_rating': 4.1,
            'num_pages': 328
        },
        {
            'book_id': 4,
            'title': 'Pride and Prejudice',
            'authors': 'Jane Austen',
            'publisher': 'Penguin',
            'description': 'The story of Elizabeth Bennet and Mr. Darcy in Georgian-era England.',
            'average_rating': 4.4,
            'num_pages': 432
        },
        {
            'book_id': 5,
            'title': 'The Hobbit',
            'authors': 'J.R.R. Tolkien',
            'publisher': 'Houghton Mifflin',
            'description': 'Bilbo Baggins embarks on an adventure with thirteen dwarves to reclaim their homeland.',
            'average_rating': 4.5,
            'num_pages': 310
        },
        {
            'book_id': 6,
            'title': 'The Catcher in the Rye',
            'authors': 'J.D. Salinger',
            'publisher': 'Little, Brown',
            'description': 'Holden Caulfield recounts his experiences in New York City after being expelled from prep school.',
            'average_rating': 4.0,
            'num_pages': 277
        },
        {
            'book_id': 7,
            'title': 'Lord of the Flies',
            'authors': 'William Golding',
            'publisher': 'Penguin',
            'description': 'A group of British boys stranded on an uninhabited island try to govern themselves.',
            'average_rating': 3.8,
            'num_pages': 224
        },
        {
            'book_id': 8,
            'title': 'Animal Farm',
            'authors': 'George Orwell',
            'publisher': 'Signet',
            'description': 'A farm is taken over by its overworked, mistreated animals with a dream of equality.',
            'average_rating': 4.0,
            'num_pages': 112
        },
        {
            'book_id': 9,
            'title': 'The Alchemist',
            'authors': 'Paulo Coelho',
            'publisher': 'HarperOne',
            'description': 'A shepherd boy named Santiago travels from his homeland in Spain to the Egyptian desert.',
            'average_rating': 4.2,
            'num_pages': 208
        },
        {
            'book_id': 10,
            'title': 'Brave New World',
            'authors': 'Aldous Huxley',
            'publisher': 'Signet',
            'description': 'A futuristic society where people are genetically bred and pharmaceutically anesthetized.',
            'average_rating': 4.1,
            'num_pages': 311
        }
    ]
    
    return pd.DataFrame(sample_books)

def create_sample_ratings(books_df):
    """
    Create sample user ratings for collaborative filtering.
    """
    np.random.seed(42)
    n_users = 100
    n_books = len(books_df)
    
    # Create random ratings (1-5 scale)
    ratings_data = []
    for user_id in range(1, n_users + 1):
        # Each user rates 5-15 random books (or all books if less than 5)
        max_ratings = min(n_books, np.random.randint(5, 16))
        n_ratings = min(max_ratings, n_books)  # Ensure we don't exceed available books
        
        if n_ratings > 0:
            rated_books = np.random.choice(books_df['book_id'], n_ratings, replace=False)
            
            for book_id in rated_books:
                rating = np.random.randint(1, 6)  # 1-5 scale
                ratings_data.append({
                    'user_id': user_id,
                    'book_id': book_id,
                    'rating': rating
                })
    
    ratings_df = pd.DataFrame(ratings_data)
    
    # Save ratings
    os.makedirs('data/processed', exist_ok=True)
    ratings_df.to_csv('data/processed/ratings.csv', index=False)
    
    return ratings_df
