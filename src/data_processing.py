import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os

def get_genre_keywords(publisher):
    """
    Add genre-like keywords based on publisher to improve content similarity.
    """
    publisher_lower = str(publisher).lower()
    
    # Map publishers to genre keywords
    genre_mapping = {
        'signet': 'classic literature dystopian fiction political satire',
        'penguin': 'classic literature british literature literary fiction',
        'harperone': 'spiritual self-help inspirational philosophy',
        'little brown': 'contemporary fiction coming of age literary fiction',
        'houghton mifflin': 'fantasy adventure children literature',
        'scribner': 'american literature classic fiction jazz age',
        'grand central': 'american literature social justice classic fiction'
    }
    
    for pub_key, keywords in genre_mapping.items():
        if pub_key in publisher_lower:
            return keywords
    
    return 'general literature fiction'

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
                return create_sample_data()
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
    books_clean['description'] = books_clean['description'].fillna('No description available')
    
    # Clean text fields
    books_clean['title'] = books_clean['title'].astype(str).str.strip()
    books_clean['authors'] = books_clean['authors'].astype(str).str.strip()
    books_clean['publisher'] = books_clean['publisher'].astype(str).str.strip()
    
    # Create combined features for content-based filtering
    # Include more descriptive content for better similarity calculation
    books_clean['combined_features'] = (
        books_clean['title'] + ' ' + 
        books_clean['authors'] + ' ' + 
        books_clean['publisher'] + ' ' +
        books_clean['description'].fillna('') + ' ' +
        # Add genre-like information based on publisher patterns
        books_clean['publisher'].apply(lambda x: get_genre_keywords(x))
    )
    
    # Remove special characters and normalize
    books_clean['combined_features'] = books_clean['combined_features'].apply(
        lambda x: re.sub(r'[^\w\s]', ' ', str(x)).lower()
    )
    
    # Clean up multiple spaces
    books_clean['combined_features'] = books_clean['combined_features'].apply(
        lambda x: re.sub(r'\s+', ' ', str(x)).strip()
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
        },
        {
            'book_id': 11,
            'title': 'The Lord of the Rings',
            'authors': 'J.R.R. Tolkien',
            'publisher': 'Houghton Mifflin',
            'description': 'The epic fantasy tale of Frodo Baggins and his quest to destroy the One Ring.',
            'average_rating': 4.6,
            'num_pages': 1216
        },
        {
            'book_id': 12,
            'title': 'Harry Potter and the Sorcerer\'s Stone',
            'authors': 'J.K. Rowling',
            'publisher': 'Scholastic',
            'description': 'A young wizard discovers his magical heritage and attends Hogwarts School of Witchcraft and Wizardry.',
            'average_rating': 4.5,
            'num_pages': 309
        },
        {
            'book_id': 13,
            'title': 'The Chronicles of Narnia',
            'authors': 'C.S. Lewis',
            'publisher': 'HarperCollins',
            'description': 'Four children discover a magical world through an old wardrobe.',
            'average_rating': 4.3,
            'num_pages': 767
        },
        {
            'book_id': 14,
            'title': 'The Handmaid\'s Tale',
            'authors': 'Margaret Atwood',
            'publisher': 'Anchor',
            'description': 'A dystopian novel set in the Republic of Gilead, a totalitarian state.',
            'average_rating': 4.2,
            'num_pages': 311
        },
        {
            'book_id': 15,
            'title': 'The Kite Runner',
            'authors': 'Khaled Hosseini',
            'publisher': 'Riverhead Books',
            'description': 'A story of friendship, betrayal, and redemption set in Afghanistan.',
            'average_rating': 4.4,
            'num_pages': 371
        },
        {
            'book_id': 16,
            'title': 'The Book Thief',
            'authors': 'Markus Zusak',
            'publisher': 'Knopf',
            'description': 'A young girl in Nazi Germany steals books and shares them with others.',
            'average_rating': 4.3,
            'num_pages': 552
        },
        {
            'book_id': 17,
            'title': 'The Giver',
            'authors': 'Lois Lowry',
            'publisher': 'Houghton Mifflin',
            'description': 'A young boy discovers the dark secrets of his seemingly perfect community.',
            'average_rating': 4.1,
            'num_pages': 208
        },
        {
            'book_id': 18,
            'title': 'The Fault in Our Stars',
            'authors': 'John Green',
            'publisher': 'Dutton Books',
            'description': 'A teenage girl with cancer falls in love with a boy she meets at a support group.',
            'average_rating': 4.2,
            'num_pages': 313
        },
        {
            'book_id': 19,
            'title': 'The Hunger Games',
            'authors': 'Suzanne Collins',
            'publisher': 'Scholastic',
            'description': 'In a dystopian future, teenagers fight to the death in televised games.',
            'average_rating': 4.3,
            'num_pages': 374
        },
        {
            'book_id': 20,
            'title': 'The Da Vinci Code',
            'authors': 'Dan Brown',
            'publisher': 'Doubleday',
            'description': 'A symbologist and cryptologist solve a murder mystery involving secret societies.',
            'average_rating': 3.9,
            'num_pages': 689
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
    
    # Ensure every book has at least one rating
    all_book_ids = set(books_df['book_id'])
    rated_book_ids = set([r['book_id'] for r in ratings_data])
    unrated_books = all_book_ids - rated_book_ids
    
    # Add ratings for unrated books
    for book_id in unrated_books:
        user_id = np.random.randint(1, n_users + 1)
        rating = np.random.randint(1, 6)
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
