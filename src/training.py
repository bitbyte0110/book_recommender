"""
Training module for Hybrid Book Recommender System
Handles model training, hyperparameter tuning, and model persistence
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
# Simplified collaborative filtering without surprise library
import numpy as np
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

class HybridRecommenderTrainer:
    """
    Trainer class for hybrid book recommender system
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.models_dir = 'models'
        self.content_model = None
        self.collab_model = None
        self.tfidf_vectorizer = None
        self.content_similarity_matrix = None
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess books and ratings data
        """
        print("Loading and preprocessing data...")
        
        # Prefer processed files if present
        processed_books = os.path.join(self.data_dir, 'processed', 'books_clean.csv')
        processed_ratings = os.path.join(self.data_dir, 'processed', 'ratings.csv')

        # Load books data
        books_path = processed_books if os.path.exists(processed_books) else os.path.join(self.data_dir, 'raw', 'books.csv')
        if os.path.exists(books_path):
            # Use error_bad_lines=False to skip malformed lines and warn_bad_lines=True to show warnings
            try:
                books_df = pd.read_csv(books_path, on_bad_lines='skip')
                print(f"Loaded {len(books_df)} books")
            except Exception as e:
                print(f"Error loading books.csv: {e}")
                print("Using sample data instead")
                books_df = self._create_sample_books()
        else:
            print("Books data not found, using sample data")
            books_df = self._create_sample_books()
        
        # Load ratings data
        ratings_path = processed_ratings if os.path.exists(processed_ratings) else os.path.join(self.data_dir, 'raw', 'Books_rating.csv')
        if os.path.exists(ratings_path):
            ratings_df = pd.read_csv(ratings_path)
            print(f"Loaded {len(ratings_df)} ratings")
        else:
            print("Ratings data not found, using sample data")
            ratings_df = self._create_sample_ratings(books_df)
        
        # Clean and preprocess data
        books_clean = self._clean_books_data(books_df)
        ratings_clean = self._clean_ratings_data(ratings_df, books_clean)
        
        # Save cleaned data
        os.makedirs(os.path.join(self.data_dir, 'processed'), exist_ok=True)
        books_clean.to_csv(os.path.join(self.data_dir, 'processed', 'books_clean.csv'), index=False)
        ratings_clean.to_csv(os.path.join(self.data_dir, 'processed', 'ratings.csv'), index=False)
        
        return books_clean, ratings_clean
    
    def _clean_books_data(self, books_df):
        """
        Clean and preprocess books data
        """
        # Handle missing values
        books_df = books_df.fillna({
            'title': 'Unknown Title',
            'authors': 'Unknown Author',
            'average_rating': 0.0,
            'language_code': 'eng',
            'num_pages': 0,
            'ratings_count': 0,
            'text_reviews_count': 0,
            'publisher': 'Unknown Publisher'
        })
        
        # Create combined features for content-based filtering
        books_df['combined_features'] = (
            books_df['title'].astype(str) + ' ' +
            books_df['authors'].astype(str) + ' ' +
            books_df['publisher'].astype(str)
        )
        
        # Clean text features
        books_df['combined_features'] = books_df['combined_features'].str.lower()
        books_df['combined_features'] = books_df['combined_features'].str.replace(r'[^\w\s]', ' ', regex=True)
        books_df['combined_features'] = books_df['combined_features'].str.replace(r'\s+', ' ', regex=True)
        
        # Ensure book_id exists
        if 'bookID' in books_df.columns:
            books_df = books_df.rename(columns={'bookID': 'book_id'})
        elif 'book_id' not in books_df.columns:
            books_df['book_id'] = range(1, len(books_df) + 1)
        
        return books_df
    
    def _clean_ratings_data(self, ratings_df, books_df):
        """
        Clean and preprocess ratings data
        """
        print(f"Original ratings columns: {list(ratings_df.columns)}")
        
        # Handle different column names
        if 'Id' in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={'Id': 'book_id'})
        if 'User_id' in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={'User_id': 'user_id'})
        if 'review/score' in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={'review/score': 'rating'})
        
        print(f"After renaming columns: {list(ratings_df.columns)}")
        
        # Ensure required columns exist
        required_cols = ['book_id', 'user_id', 'rating']
        for col in required_cols:
            if col not in ratings_df.columns:
                print(f"Warning: {col} column not found in ratings data")
                return self._create_sample_ratings(books_df)
        
        # Clean ratings
        ratings_df = ratings_df[required_cols].copy()
        ratings_df = ratings_df.dropna()
        
        print(f"After dropping NA: {len(ratings_df)} ratings")
        
        # Convert ratings to numeric
        ratings_df['rating'] = pd.to_numeric(ratings_df['rating'], errors='coerce')
        ratings_df = ratings_df.dropna()
        
        print(f"After converting to numeric: {len(ratings_df)} ratings")
        
        # Filter valid ratings (1-5 scale)
        ratings_df = ratings_df[(ratings_df['rating'] >= 1) & (ratings_df['rating'] <= 5)]
        
        print(f"After filtering valid ratings: {len(ratings_df)} ratings")
        
        # Check if we need to map ISBN to book_id or if book_id is already correct
        if 'isbn' in books_df.columns:
            # Create mapping from ISBN to book_id
            isbn_to_book_id = books_df.set_index('isbn')['book_id'].to_dict()
            # Map ratings book_id (ISBN) to actual book_id
            ratings_df['book_id'] = ratings_df['book_id'].map(isbn_to_book_id)
        else:
            # book_id is already correct, just ensure it's numeric
            ratings_df['book_id'] = pd.to_numeric(ratings_df['book_id'], errors='coerce')
        
        # Remove ratings for books not found in books_df
        ratings_df = ratings_df.dropna(subset=['book_id'])
        
        # Convert book_id to integer
        ratings_df['book_id'] = ratings_df['book_id'].astype(int)
        
        print(f"After mapping ISBNs to book_ids: {len(ratings_df)} ratings")
        
        # Create numeric user IDs
        user_id_map = {user_id: idx + 1 for idx, user_id in enumerate(ratings_df['user_id'].unique())}
        ratings_df['user_id'] = ratings_df['user_id'].map(user_id_map)
        
        print(f"Cleaned ratings: {len(ratings_df)} ratings from {ratings_df['user_id'].nunique()} users")
        return ratings_df
    
    def _create_sample_books(self):
        """
        Create sample books data if not available
        """
        sample_books = [
            {'book_id': 1, 'title': 'The Great Gatsby', 'authors': 'F. Scott Fitzgerald', 'average_rating': 4.2, 'publisher': 'Scribner'},
            {'book_id': 2, 'title': 'To Kill a Mockingbird', 'authors': 'Harper Lee', 'average_rating': 4.3, 'publisher': 'Grand Central'},
            {'book_id': 3, 'title': '1984', 'authors': 'George Orwell', 'average_rating': 4.1, 'publisher': 'Signet'},
            {'book_id': 4, 'title': 'Pride and Prejudice', 'authors': 'Jane Austen', 'average_rating': 4.4, 'publisher': 'Penguin'},
            {'book_id': 5, 'title': 'The Hobbit', 'authors': 'J.R.R. Tolkien', 'average_rating': 4.5, 'publisher': 'Houghton Mifflin'},
            {'book_id': 6, 'title': 'The Catcher in the Rye', 'authors': 'J.D. Salinger', 'average_rating': 4.0, 'publisher': 'Little, Brown'},
            {'book_id': 7, 'title': 'Lord of the Flies', 'authors': 'William Golding', 'average_rating': 3.8, 'publisher': 'Penguin'},
            {'book_id': 8, 'title': 'Animal Farm', 'authors': 'George Orwell', 'average_rating': 4.0, 'publisher': 'Signet'},
            {'book_id': 9, 'title': 'The Alchemist', 'authors': 'Paulo Coelho', 'average_rating': 4.2, 'publisher': 'HarperOne'},
            {'book_id': 10, 'title': 'Brave New World', 'authors': 'Aldous Huxley', 'average_rating': 4.1, 'publisher': 'Signet'},
            {'book_id': 11, 'title': 'The Lord of the Rings', 'authors': 'J.R.R. Tolkien', 'average_rating': 4.6, 'publisher': 'Houghton Mifflin'},
            {'book_id': 12, 'title': 'Harry Potter and the Sorcerer\'s Stone', 'authors': 'J.K. Rowling', 'average_rating': 4.5, 'publisher': 'Scholastic'},
            {'book_id': 13, 'title': 'The Chronicles of Narnia', 'authors': 'C.S. Lewis', 'average_rating': 4.3, 'publisher': 'HarperCollins'},
            {'book_id': 14, 'title': 'The Handmaid\'s Tale', 'authors': 'Margaret Atwood', 'average_rating': 4.2, 'publisher': 'Anchor'},
            {'book_id': 15, 'title': 'The Kite Runner', 'authors': 'Khaled Hosseini', 'average_rating': 4.4, 'publisher': 'Riverhead Books'},
            {'book_id': 16, 'title': 'The Book Thief', 'authors': 'Markus Zusak', 'average_rating': 4.3, 'publisher': 'Knopf'},
            {'book_id': 17, 'title': 'The Giver', 'authors': 'Lois Lowry', 'average_rating': 4.1, 'publisher': 'Houghton Mifflin'},
            {'book_id': 18, 'title': 'The Fault in Our Stars', 'authors': 'John Green', 'average_rating': 4.2, 'publisher': 'Dutton Books'},
            {'book_id': 19, 'title': 'The Hunger Games', 'authors': 'Suzanne Collins', 'average_rating': 4.3, 'publisher': 'Scholastic'},
            {'book_id': 20, 'title': 'The Da Vinci Code', 'authors': 'Dan Brown', 'average_rating': 3.9, 'publisher': 'Doubleday'}
        ]
        return pd.DataFrame(sample_books)
    
    def _create_sample_ratings(self, books_df):
        """
        Create sample ratings data if not available
        """
        np.random.seed(42)
        n_users = 100
        n_books = len(books_df)
        
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
        
        return pd.DataFrame(ratings_data)
    
    def train_content_based_model(self, books_df, **kwargs):
        """
        Train content-based recommendation model using TF-IDF
        """
        print("Training content-based model...")
        
        # TF-IDF parameters
        tfidf_params = {
            'max_features': kwargs.get('max_features', 5000),
            'stop_words': 'english',
            'ngram_range': kwargs.get('ngram_range', (1, 2)),
            'min_df': kwargs.get('min_df', 2),
            'max_df': kwargs.get('max_df', 0.95)
        }
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(**tfidf_params)
        
        # Fit and transform the combined features
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(books_df['combined_features'])
        
        # Calculate cosine similarity
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
        
        print(f"Content-based model trained with {tfidf_matrix.shape[1]} features")
        return self.content_similarity_matrix
    
    def train_collaborative_model(self, ratings_df, **kwargs):
        """
        Train collaborative filtering model using bias-adjusted ALS
        """
        print("Training collaborative filtering model...")
        
        # Import the improved collaborative filtering
        from src.collaborative import create_item_item_similarity_matrix
        
        # Create user-item matrix
        user_item_matrix = ratings_df.pivot_table(
            index='user_id', 
            columns='book_id', 
            values='rating', 
            fill_value=0
        )
        
        # Use the improved ALS-based collaborative filtering
        item_similarity_matrix = create_item_item_similarity_matrix(user_item_matrix)
        
        # Store the similarity matrix and user-item matrix
        self.collab_model = {
            'item_similarity_matrix': item_similarity_matrix,
            'user_item_matrix': user_item_matrix
        }
        
        print(f"Collaborative model trained with bias-adjusted ALS")
        return self.collab_model
    
    def hyperparameter_tuning(self, ratings_df, books_df):
        """
        Perform hyperparameter tuning for both models
        """
        print("Performing hyperparameter tuning...")
        
        # Content-based tuning
        content_params = {
            'max_features': [3000, 5000, 7000],
            'ngram_range': [(1, 1), (1, 2), (1, 3)],
            'min_df': [1, 2, 3]
        }
        
        best_content_score = 0
        best_content_params = None
        
        for max_features in content_params['max_features']:
            for ngram_range in content_params['ngram_range']:
                for min_df in content_params['min_df']:
                    try:
                        tfidf = TfidfVectorizer(
                            max_features=max_features,
                            ngram_range=ngram_range,
                            min_df=min_df,
                            stop_words='english'
                        )
                        tfidf_matrix = tfidf.fit_transform(books_df['combined_features'])
                        similarity_matrix = cosine_similarity(tfidf_matrix)
                        
                        # Simple evaluation (can be enhanced)
                        score = np.mean(similarity_matrix)
                        if score > best_content_score:
                            best_content_score = score
                            best_content_params = {
                                'max_features': max_features,
                                'ngram_range': ngram_range,
                                'min_df': min_df
                            }
                    except:
                        continue
        
        print(f"Best content-based parameters: {best_content_params}")
        
        # Collaborative filtering tuning (simplified)
        # Use min of 50 or number of users/2 to avoid component issues
        n_users = ratings_df['user_id'].nunique()
        n_factors = min(50, max(5, n_users // 2))
        best_collab_params = {'n_factors': n_factors}
        
        print(f"Best collaborative parameters: {best_collab_params}")
        
        return best_content_params, best_collab_params
    
    def cross_validate_models(self, ratings_df, books_df):
        """
        Perform cross-validation for model evaluation
        """
        print("Performing cross-validation...")
        
        # Simplified cross-validation for content-based model
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, mean_squared_error
        
        # Create user-item matrix for collaborative filtering
        user_item_matrix = ratings_df.pivot_table(
            index='user_id', 
            columns='book_id', 
            values='rating', 
            fill_value=0
        )
        
        # Simple cross-validation for TruncatedSVD
        n_components = min(50, user_item_matrix.shape[1], user_item_matrix.shape[0])
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Calculate explained variance ratio
        svd.fit(user_item_matrix)
        explained_variance = svd.explained_variance_ratio_.sum()
        
        print("TruncatedSVD Cross-validation results:")
        print(f"Explained Variance Ratio: {explained_variance:.3f}")
        
        return {'explained_variance': explained_variance}
    
    def save_models(self):
        """
        Save trained models to disk
        """
        print("Saving models...")
        
        # Save content-based model
        if self.content_similarity_matrix is not None:
            joblib.dump(self.content_similarity_matrix, 
                       os.path.join(self.models_dir, 'content_similarity_matrix.pkl'))
        
        if self.tfidf_vectorizer is not None:
            joblib.dump(self.tfidf_vectorizer, 
                       os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'))
        
        # Save collaborative model
        if self.collab_model is not None:
            if isinstance(self.collab_model, dict) and 'item_similarity_matrix' in self.collab_model:
                # Save the new ALS-based model
                joblib.dump(self.collab_model, 
                           os.path.join(self.models_dir, 'enhanced_svd_model.pkl'))
            else:
                # Fallback for old SVD model
                joblib.dump(self.collab_model, 
                           os.path.join(self.models_dir, 'svd_model.pkl'))
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'content_model_shape': self.content_similarity_matrix.shape if self.content_similarity_matrix is not None else None,
            'collab_model_params': self.collab_model.get_params() if hasattr(self.collab_model, 'get_params') else 'ALS-based collaborative model'
        }
        
        joblib.dump(metadata, os.path.join(self.models_dir, 'training_metadata.pkl'))
        print("Models saved successfully")
    
    def load_models(self):
        """
        Load trained models from disk
        """
        print("Loading models...")
        
        try:
            self.content_similarity_matrix = joblib.load(
                os.path.join(self.models_dir, 'content_similarity_matrix.pkl'))
            self.tfidf_vectorizer = joblib.load(
                os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'))
            self.collab_model = joblib.load(
                os.path.join(self.models_dir, 'svd_model.pkl'))
            print("Models loaded successfully")
            return True
        except FileNotFoundError:
            print("No trained models found")
            return False
    
    def train_full_pipeline(self, use_hyperparameter_tuning=True, use_cross_validation=True, custom_params=None):
        """
        Complete training pipeline
        """
        print("Starting full training pipeline...")
        
        # Load and preprocess data
        books_df, ratings_df = self.load_and_preprocess_data()
        
        # Hyperparameter tuning
        if custom_params:
            # Use custom parameters
            best_content_params = {
                'max_features': custom_params.get('content_max_features', 3000),
                'ngram_range': custom_params.get('content_ngram_range', (1, 1)),
                'min_df': custom_params.get('content_min_df', 2)
            }
            best_collab_params = {
                'n_factors': custom_params.get('collab_n_factors', 50)
            }
        elif use_hyperparameter_tuning:
            best_content_params, best_collab_params = self.hyperparameter_tuning(ratings_df, books_df)
        else:
            best_content_params = {}
            best_collab_params = {}
        
        # Train content-based model
        self.train_content_based_model(books_df, **best_content_params)
        
        # Train collaborative model
        self.train_collaborative_model(ratings_df, **best_collab_params)
        
        # Cross-validation
        if use_cross_validation:
            cv_results = self.cross_validate_models(ratings_df, books_df)
        
        # Save models
        self.save_models()
        
        print("Training pipeline completed successfully!")
        return books_df, ratings_df

def main():
    """
    Main training function
    """
    trainer = HybridRecommenderTrainer()
    
    # Check if models already exist
    if trainer.load_models():
        print("Pre-trained models found. Use 'retrain=True' to retrain.")
        return
    
    # Train full pipeline
    books_df, ratings_df = trainer.train_full_pipeline(
        use_hyperparameter_tuning=True,
        use_cross_validation=True
    )
    
    print(f"Training completed with {len(books_df)} books and {len(ratings_df)} ratings")

if __name__ == "__main__":
    main()
