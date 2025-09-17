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
    Create item-item similarity matrix using advanced ALS with regularization.
    
    Args:
        user_item_matrix: User-item rating matrix
    
    Returns:
        item_similarity_matrix: Item-item similarity matrix
    """
    print("Creating item-item similarity matrix using advanced ALS...")
    
    # Implement ALS (Alternating Least Squares) with regularization
    class ALSRecommender:
        def __init__(self, n_factors=20, regularization=0.1, iterations=50, random_state=42):
            self.n_factors = n_factors
            self.regularization = regularization
            self.iterations = iterations
            self.random_state = random_state
            self.user_factors = None
            self.item_factors = None
            # Bias terms
            self.global_bias = 0.0
            self.user_bias = None
            self.item_bias = None
            
        def fit(self, user_item_matrix):
            """Fit ALS model to user-item matrix"""
            np.random.seed(self.random_state)
            
            n_users, n_items = user_item_matrix.shape
            
            # Initialize factors randomly
            self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
            self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
            # Initialize biases
            self.user_bias = np.zeros(n_users)
            self.item_bias = np.zeros(n_items)
            
            # Convert to sparse matrix for efficiency
            from scipy.sparse import csr_matrix
            full = user_item_matrix.values.astype(float)
            # Compute global bias on observed ratings only (>0)
            observed = full > 0
            if observed.sum() > 0:
                self.global_bias = full[observed].mean()
            else:
                self.global_bias = 0.0
            # Compute user and item biases from observed ratings
            for u in range(n_users):
                mask = observed[u]
                if mask.any():
                    self.user_bias[u] = full[u, mask].mean() - self.global_bias
            for i in range(n_items):
                mask = observed[:, i]
                if mask.any():
                    self.item_bias[i] = full[mask, i].mean() - self.global_bias
            # Create residuals matrix R' = R - (mu + bu + bi) for observed entries
            residual = full.copy()
            for u in range(n_users):
                for i in range(n_items):
                    if observed[u, i]:
                        residual[u, i] = full[u, i] - (self.global_bias + self.user_bias[u] + self.item_bias[i])
                    else:
                        residual[u, i] = 0.0
            sparse_matrix = csr_matrix(residual)
            
            # ALS iterations
            for iteration in range(self.iterations):
                # Update user factors
                for u in range(n_users):
                    # Get items rated by user u
                    user_ratings = sparse_matrix[u].toarray().flatten()
                    rated_items = np.where(user_ratings > 0)[0]
                    
                    if len(rated_items) > 0:
                        # Solve: (item_factors^T * item_factors + λI) * user_factors[u] = item_factors^T * ratings
                        item_subset = self.item_factors[rated_items]
                        ratings_subset = user_ratings[rated_items]
                        
                        # Regularized least squares
                        A = item_subset.T @ item_subset + self.regularization * np.eye(self.n_factors)
                        b = item_subset.T @ ratings_subset
                        
                        try:
                            self.user_factors[u] = np.linalg.solve(A, b)
                        except np.linalg.LinAlgError:
                            # Fallback to pseudo-inverse
                            self.user_factors[u] = np.linalg.pinv(A) @ b
                
                # Update item factors
                for i in range(n_items):
                    # Get users who rated item i
                    item_ratings = sparse_matrix[:, i].toarray().flatten()
                    rating_users = np.where(item_ratings > 0)[0]
                    
                    if len(rating_users) > 0:
                        # Solve: (user_factors^T * user_factors + λI) * item_factors[i] = user_factors^T * ratings
                        user_subset = self.user_factors[rating_users]
                        ratings_subset = item_ratings[rating_users]
                        
                        # Regularized least squares
                        A = user_subset.T @ user_subset + self.regularization * np.eye(self.n_factors)
                        b = user_subset.T @ ratings_subset
                        
                        try:
                            self.item_factors[i] = np.linalg.solve(A, b)
                        except np.linalg.LinAlgError:
                            # Fallback to pseudo-inverse
                            self.item_factors[i] = np.linalg.pinv(A) @ b
                
                if iteration % 10 == 0:
                    print(f"ALS iteration {iteration}/{self.iterations}")
            
            print(f"ALS training completed with {self.n_factors} factors")
            
        def predict(self, user_item_matrix):
            """Predict ratings using learned factors"""
            base = self.user_factors @ self.item_factors.T
            # Add biases back
            n_users, n_items = user_item_matrix.shape
            bias_matrix = (self.global_bias
                           + self.user_bias.reshape(n_users, 1)
                           + self.item_bias.reshape(1, n_items))
            return base + bias_matrix
            
        def get_item_factors(self):
            """Get item factors for similarity calculation"""
            return self.item_factors
    
    # Use ALS with stronger configuration for sparse data
    als_model = ALSRecommender(
        n_factors=min(50, user_item_matrix.shape[1] - 1, user_item_matrix.shape[0] - 1),
        regularization=0.01,
        iterations=200,
        random_state=42
    )
    
    try:
        als_model.fit(user_item_matrix)
        item_factors = als_model.get_item_factors()
        
        # Calculate item-item similarity using cosine similarity on factors
        item_similarity_matrix = cosine_similarity(item_factors)
        
        # Save the ALS model for later use
        import joblib
        os.makedirs('models', exist_ok=True)
        
        # Create a serializable version of the model
        als_data = {
            'user_factors': als_model.user_factors,
            'item_factors': als_model.item_factors,
            'global_bias': als_model.global_bias,
            'user_bias': als_model.user_bias,
            'item_bias': als_model.item_bias,
            'n_factors': als_model.n_factors,
            'regularization': als_model.regularization,
            'iterations': als_model.iterations,
            'random_state': als_model.random_state
        }
        joblib.dump(als_data, 'models/als_model.pkl')
        
        print(f"ALS model saved with {als_model.n_factors} factors")
        
    except Exception as e:
        print(f"ALS failed, falling back to enhanced SVD: {e}")
        # Enhanced SVD with bias terms
        from sklearn.decomposition import TruncatedSVD
        
        # Calculate global bias
        global_bias = user_item_matrix.values[user_item_matrix.values > 0].mean()
        
        # Calculate user and item biases
        user_bias = {}
        item_bias = {}
        
        for i, user_id in enumerate(user_item_matrix.index):
            user_ratings = user_item_matrix.iloc[i].values
            rated_items = user_ratings[user_ratings > 0]
            if len(rated_items) > 0:
                user_bias[user_id] = rated_items.mean() - global_bias
            else:
                user_bias[user_id] = 0
        
        for j, item_id in enumerate(user_item_matrix.columns):
            item_ratings = user_item_matrix.iloc[:, j].values
            rated_by_users = item_ratings[item_ratings > 0]
            if len(rated_by_users) > 0:
                item_bias[item_id] = rated_by_users.mean() - global_bias
            else:
                item_bias[item_id] = 0
        
        # Apply bias correction
        bias_corrected_matrix = user_item_matrix.copy()
        for i, user_id in enumerate(user_item_matrix.index):
            for j, item_id in enumerate(user_item_matrix.columns):
                if user_item_matrix.iloc[i, j] > 0:
                    bias_corrected_matrix.iloc[i, j] = user_item_matrix.iloc[i, j] - user_bias[user_id] - item_bias[item_id]
        
        # Fit SVD on bias-corrected matrix
        n_components = min(20, user_item_matrix.shape[1] - 1, user_item_matrix.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42, algorithm='arpack')
        svd.fit(bias_corrected_matrix)
        item_factors = svd.components_.T
        item_similarity_matrix = cosine_similarity(item_factors)
        
        # Save enhanced model with bias terms
        enhanced_model = {
            'svd': svd,
            'global_bias': global_bias,
            'user_bias': user_bias,
            'item_bias': item_bias
        }
        joblib.dump(enhanced_model, 'models/enhanced_svd_model.pkl')
        print("Enhanced SVD model with bias terms saved")
    
    print(f"Item-item similarity matrix shape: {item_similarity_matrix.shape}")
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
                'rating': books_df.iloc[idx].get('average_rating', 0),
                'method': 'collaborative'
            })
        
        return recommendations
    
    except (IndexError, KeyError):
        # If book not found, return empty list
        return []

def get_user_based_recommendations(user_id, user_item_matrix, books_df, top_n=10, min_co_ratings=2, top_k_neighbors=20, shrinkage=10.0):
    """
    Get user-based CF recommendations using Pearson similarity weighted neighbors.
    Only consider neighbors with at least min_co_ratings on co-rated items.
    """
    try:
        if user_id not in user_item_matrix.index:
            return []

        # Target user ratings
        target = user_item_matrix.loc[user_id]
        rated_by_target = target[target > 0]

        # Compute Pearson similarity with other users on co-rated items (with shrinkage)
        sims = []
        for other_id in user_item_matrix.index:
            if other_id == user_id:
                continue
            other = user_item_matrix.loc[other_id]
            co_mask = (target > 0) & (other > 0)
            if co_mask.sum() < min_co_ratings:
                continue
            corr = np.corrcoef(target[co_mask], other[co_mask])[0, 1]
            # Apply shrinkage toward 0 based on number of co-ratings
            n = co_mask.sum()
            if not np.isnan(corr):
                corr_shrunk = (n / (n + shrinkage)) * corr
                if corr_shrunk > 0:
                    sims.append((other_id, corr_shrunk))

        if not sims:
            return []

        # Top-K similar users
        sims.sort(key=lambda x: x[1], reverse=True)
        neighbors = sims[:top_k_neighbors]

        # Score unrated items by weighted average of neighbor ratings
        unrated_books = target[target == 0].index
        scores = {}
        weights = {}
        for other_id, sim in neighbors:
            other = user_item_matrix.loc[other_id]
            for book_id in unrated_books:
                r = other.get(book_id, 0)
                if r > 0:
                    scores[book_id] = scores.get(book_id, 0.0) + sim * r
                    weights[book_id] = weights.get(book_id, 0.0) + abs(sim)

        ranked = []
        for book_id, num in scores.items():
            denom = weights.get(book_id, 0.0)
            if denom > 0:
                ranked.append((book_id, num / denom))

        ranked.sort(key=lambda x: x[1], reverse=True)

        recommendations = []
        for book_id, score in ranked[:top_n]:
            row = books_df[books_df['book_id'] == book_id]
            if row.empty:
                continue
            info = row.iloc[0]
            recommendations.append({
                'book_id': book_id,
                'title': info['title'],
                'authors': info['authors'],
                'genre': info.get('publisher', 'Unknown'),
                'predicted_rating': float(score),
                'rating': info.get('average_rating', 0)
            })

        return recommendations

    except Exception:
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
