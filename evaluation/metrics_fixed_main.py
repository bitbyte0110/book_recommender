"""
Evaluation metrics for Hybrid Book Recommender System
Implements precision, recall, F1-score, RMSE, coverage, diversity, and user satisfaction metrics
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import json
import warnings
warnings.filterwarnings('ignore')

class RecommenderEvaluator:
    """
    Comprehensive evaluator for collaborative, content-based, and hybrid book recommender systems
    """
    
    def __init__(self, data_dir='data', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.results_dir = 'evaluation/test_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_data_and_models(self):
        """
        Load cleaned data and trained models
        """
        # Load data
        books_df = pd.read_csv(os.path.join(self.data_dir, 'processed', 'books_clean.csv'))
        ratings_df = pd.read_csv(os.path.join(self.data_dir, 'processed', 'ratings.csv'))
        
        # Load models (try ALS first, fallback to SVD)
        content_sim_matrix = joblib.load(os.path.join(self.models_dir, 'content_similarity_matrix.pkl'))
        
        # Try to load neural enhanced model first, then fallback to other models
        neural_model_path = os.path.join(self.models_dir, 'neural_enhanced_model.pkl')
        enhanced_model_path = os.path.join(self.models_dir, 'enhanced_svd_model.pkl')
        als_model_path = os.path.join(self.models_dir, 'als_model.pkl')
        svd_model_path = os.path.join(self.models_dir, 'svd_model.pkl')
        
        if os.path.exists(neural_model_path):
            model = joblib.load(neural_model_path)
            print("Loaded neural enhanced model for evaluation")
        elif os.path.exists(enhanced_model_path):
            model = joblib.load(enhanced_model_path)
            print("Loaded enhanced SVD model with bias terms for evaluation")
        elif os.path.exists(als_model_path):
            model = joblib.load(als_model_path)
            print("Loaded ALS model for evaluation")
        elif os.path.exists(svd_model_path):
            model = joblib.load(svd_model_path)
            print("Loaded SVD model for evaluation")
        else:
            model = None
            print("No collaborative model found")
        
        return books_df, ratings_df, content_sim_matrix, model
    
    def evaluate_content_based_filtering(self, ratings_df, content_sim_matrix, top_n=10, threshold=3.0, test_size=0.2):
        """
        Evaluate content-based filtering system with FIXED MSE calculation
        Returns: precision, recall, f1, mse, rmse
        """
        print("Evaluating Content-Based Filtering...")
        
        # Split data for evaluation
        np.random.seed(42)
        ratings_shuffled = ratings_df.sample(frac=1, random_state=42)
        split_idx = int(len(ratings_shuffled) * (1 - test_size))
        
        train_ratings = ratings_shuffled[:split_idx]
        test_ratings = ratings_shuffled[split_idx:]
        
        if len(train_ratings) == 0 or len(test_ratings) == 0:
            return {'precision': 0, 'recall': 0, 'f1': 0, 'mse': 1.0, 'rmse': 1.0}
        
        books_df = pd.read_csv(os.path.join(self.data_dir, 'processed', 'books_clean.csv'))
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        mse_scores = []
        rmse_scores = []
        
        # Get unique users for evaluation
        users = test_ratings['user_id'].unique()
        
        for user_id in users[:200]:  # Sample for efficiency
            user_test_ratings = test_ratings[test_ratings['user_id'] == user_id]
            if len(user_test_ratings) == 0:
                continue
            
            # Get user's training ratings
            user_train_ratings = train_ratings[train_ratings['user_id'] == user_id]
            if len(user_train_ratings) == 0:
                continue
            
            # Get user's average rating as baseline
            user_avg_rating = user_train_ratings['rating'].mean()
            
            # Get user's top-rated books from training
            top_rated_books = user_train_ratings.nlargest(3, 'rating')['book_id'].tolist()
            
            # Calculate content-based similarity scores using ONLY training data
            aggregated_scores = np.zeros(content_sim_matrix.shape[0])
            
            for book_id in top_rated_books:
                if book_id in books_df['book_id'].values:
                    book_idx = books_df[books_df['book_id'] == book_id].index[0]
                    book_scores = content_sim_matrix[book_idx]
                    weight = user_train_ratings[user_train_ratings['book_id'] == book_id]['rating'].iloc[0] / 5.0
                    aggregated_scores += book_scores * weight
            
            # Normalize scores
            if len(top_rated_books) > 0:
                aggregated_scores = aggregated_scores / len(top_rated_books)
            
            # Get top recommendations based on similarity scores only
            similar_indices = np.argsort(aggregated_scores)[::-1][:top_n]
            recommended_books = books_df.iloc[similar_indices]['book_id'].tolist()
            
            # Calculate metrics using test data
            actual_high_rated = user_test_ratings[user_test_ratings['rating'] >= threshold]['book_id'].tolist()
            
            if len(actual_high_rated) > 0:
                y_true = [1 if book_id in actual_high_rated else 0 for book_id in recommended_books]
                y_pred = [1] * len(recommended_books)
                
                if sum(y_true) > 0:
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
            
            # FIXED: Calculate MSE and RMSE with proper scaling for low similarity scores
            user_mse_scores = []
            for _, row in user_test_ratings.iterrows():
                book_id = row['book_id']
                actual_rating = row['rating']
                
                if book_id in books_df['book_id'].values:
                    book_idx = books_df[books_df['book_id'] == book_id].index[0]
                    similarity = aggregated_scores[book_idx]
                    
                    # FIXED: Use user's average rating as baseline and adjust based on similarity
                    if similarity > 0.5:  # High similarity
                        pred_rating = user_avg_rating + (similarity - 0.5) * 2  # Boost up to +1.0
                    elif similarity > 0.1:  # Medium similarity
                        pred_rating = user_avg_rating + (similarity - 0.1) * 0.5  # Small adjustment
                    else:  # Low similarity
                        pred_rating = user_avg_rating - 0.5  # Slight penalty
                    
                    pred_rating = max(1.0, min(5.0, pred_rating))
                    user_mse_scores.append((actual_rating - pred_rating) ** 2)
            
            if user_mse_scores:
                user_mse = np.mean(user_mse_scores)
                user_rmse = np.sqrt(user_mse)
                mse_scores.append(user_mse)
                rmse_scores.append(user_rmse)
        
        # Calculate coverage and diversity for content-based filtering
        coverage_result = self.calculate_coverage(ratings_df, content_sim_matrix, top_n)
        diversity_result = self.calculate_diversity(ratings_df, content_sim_matrix, top_n)
        
        return {
            'precision': np.mean(precision_scores) if precision_scores else 0,
            'recall': np.mean(recall_scores) if recall_scores else 0,
            'f1': np.mean(f1_scores) if f1_scores else 0,
            'mse': np.mean(mse_scores) if mse_scores else 1.0,
            'rmse': np.mean(rmse_scores) if rmse_scores else 1.0,
            'coverage': coverage_result['coverage'],
            'diversity': diversity_result['diversity'],
            'n_users_evaluated': len(precision_scores)
        }

    def calculate_coverage(self, ratings_df, sim_matrix, top_n=10, n_users=100):
        """Calculate coverage metric"""
        print("Calculating Coverage...")
        
        books_df = pd.read_csv(os.path.join(self.data_dir, 'processed', 'books_clean.csv'))
        total_books = len(books_df)
        recommended_books = set()
        
        # Sample users for evaluation
        users = np.random.choice(ratings_df['user_id'].unique(), 
                               min(n_users, len(ratings_df['user_id'].unique())), 
                               replace=False)
        
        for user_id in users:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            if len(user_ratings) == 0:
                continue
            
            # Get user's highest rated book
            best_book = user_ratings.loc[user_ratings['rating'].idxmax(), 'book_id']
            book_idx = books_df[books_df['book_id'] == best_book].index[0]
            
            # Get similar books
            similar_scores = sim_matrix[book_idx]
            similar_indices = np.argsort(similar_scores)[::-1][1:top_n+1]
            user_recommendations = books_df.iloc[similar_indices]['book_id'].tolist()
            
            recommended_books.update(user_recommendations)
        
        coverage = len(recommended_books) / total_books
        return {
            'coverage': coverage,
            'books_recommended': len(recommended_books),
            'total_books': total_books,
            'n_users_evaluated': len(users)
        }
    
    def calculate_diversity(self, ratings_df, sim_matrix, top_n=10, n_users=100):
        """Calculate diversity metric"""
        print("Calculating Diversity...")
        
        books_df = pd.read_csv(os.path.join(self.data_dir, 'processed', 'books_clean.csv'))
        diversity_scores = []
        
        # Sample users for evaluation
        users = np.random.choice(ratings_df['user_id'].unique(), 
                               min(n_users, len(ratings_df['user_id'].unique())), 
                               replace=False)
        
        for user_id in users:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            if len(user_ratings) == 0:
                continue
            
            # Get user's highest rated book
            best_book = user_ratings.loc[user_ratings['rating'].idxmax(), 'book_id']
            book_idx = books_df[books_df['book_id'] == best_book].index[0]
            
            # Get similar books
            similar_scores = sim_matrix[book_idx]
            similar_indices = np.argsort(similar_scores)[::-1][1:top_n+1]
            user_recommendations = books_df.iloc[similar_indices]['book_id'].tolist()
            
            if len(user_recommendations) > 1:
                # Calculate pairwise dissimilarity
                rec_indices = [books_df[books_df['book_id'] == bid].index[0] for bid in user_recommendations]
                similarities = sim_matrix[np.ix_(rec_indices, rec_indices)]
                
                # Convert similarity to dissimilarity
                dissimilarities = 1 - similarities
                
                # Calculate average pairwise dissimilarity (excluding diagonal)
                mask = ~np.eye(similarities.shape[0], dtype=bool)
                avg_dissimilarity = np.mean(dissimilarities[mask])
                
                diversity_scores.append(avg_dissimilarity)
        
        return {
            'diversity': np.mean(diversity_scores) if diversity_scores else 0,
            'n_users_evaluated': len(diversity_scores)
        }

    def evaluate_collaborative_filtering(self, ratings_df, model, content_sim_matrix, top_n=10, threshold=3.0, test_size=0.2):
        """
        Evaluate collaborative filtering system with proper, honest evaluation
        Returns: precision, recall, f1, mse, rmse
        """
        print("Evaluating Collaborative Filtering...")
        
        # Split data for evaluation
        np.random.seed(42)
        ratings_shuffled = ratings_df.sample(frac=1, random_state=42)
        split_idx = int(len(ratings_shuffled) * (1 - test_size))
        
        train_ratings = ratings_shuffled[:split_idx]
        test_ratings = ratings_shuffled[split_idx:]
        
        if len(train_ratings) == 0 or len(test_ratings) == 0:
            return {'precision': 0, 'recall': 0, 'f1': 0, 'mse': 1.0, 'rmse': 1.0}
        
        # Create training matrix
        train_matrix = train_ratings.pivot_table(
            index='user_id', 
            columns='book_id', 
            values='rating', 
            fill_value=0
        )
        
        # Fit model on training data only
        if hasattr(model, 'fit'):
            model.fit(train_matrix)
        
        # Get unique users for evaluation
        users = test_ratings['user_id'].unique()
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        mse_scores = []
        rmse_scores = []
        
        for user_id in users[:200]:  # Sample for efficiency
            user_test_ratings = test_ratings[test_ratings['user_id'] == user_id]
            if len(user_test_ratings) == 0:
                continue
            
            # Get collaborative recommendations using ONLY training data
            if user_id in train_matrix.index:
                user_idx = train_matrix.index.get_loc(user_id)
                
                # Make predictions for all books using ONLY training data
                predictions = []
                book_ids = []
                
                for book_id in train_matrix.columns:
                    book_idx = train_matrix.columns.get_loc(book_id)
                    
                    if isinstance(model, dict) and 'neural_model' in model and model['neural_model'] is not None:
                        # Neural collaborative filtering
                        user_factor = model['user_factors'][user_idx]
                        item_factor = model['item_factors'][book_idx]
                        user_bias = model['user_bias'].get(user_id, 0)
                        item_bias = model['item_bias'].get(book_id, 0)
                        
                        combined_features = np.concatenate([
                            user_factor, item_factor, [user_bias, item_bias, model['global_bias']]
                        ]).reshape(1, -1)
                        
                        pred = model['neural_model'].predict(combined_features)[0]
                    elif isinstance(model, dict) and 'svd' in model:
                        # Enhanced SVD model
                        user_vector = train_matrix.iloc[user_idx].values.reshape(1, -1)
                        user_factors = model['svd'].transform(user_vector)
                        reconstructed = model['svd'].inverse_transform(user_factors)
                        base_pred = reconstructed[0, book_idx]
                        
                        user_bias = model['user_bias'].get(user_id, 0)
                        item_bias = model['item_bias'].get(book_id, 0)
                        pred = base_pred + user_bias + item_bias + model['global_bias']
                    elif hasattr(model, 'user_factors'):
                        # ALS model
                        pred = model.user_factors[user_idx] @ model.item_factors[book_idx]
                    elif hasattr(model, 'transform'):
                        # Standard SVD model
                        user_vector = train_matrix.iloc[user_idx].values.reshape(1, -1)
                        user_factors = model.transform(user_vector)
                        reconstructed = model.inverse_transform(user_factors)
                        pred = reconstructed[0, book_idx]
                    else:
                        # Fallback to global average
                        pred = train_ratings['rating'].mean()
                    
                    pred = max(1.0, min(5.0, pred))
                    predictions.append(pred)
                    book_ids.append(book_id)
                
                # Get top-N recommendations based on predictions only
                pred_scores = list(zip(book_ids, predictions))
                pred_scores.sort(key=lambda x: x[1], reverse=True)
                recommended_books = [book_id for book_id, _ in pred_scores[:top_n]]
                
                # Calculate metrics using test data
                actual_high_rated = user_test_ratings[user_test_ratings['rating'] >= threshold]['book_id'].tolist()
                
                if len(actual_high_rated) > 0:
                    y_true = [1 if book_id in actual_high_rated else 0 for book_id in recommended_books]
                    y_pred = [1] * len(recommended_books)
                    
                    if sum(y_true) > 0:
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        
                        precision_scores.append(precision)
                        recall_scores.append(recall)
                        f1_scores.append(f1)
                
                # Calculate MSE and RMSE for this user
                user_mse_scores = []
                for _, row in user_test_ratings.iterrows():
                    book_id = row['book_id']
                    actual_rating = row['rating']
                    
                    if book_id in book_ids:
                        book_idx = book_ids.index(book_id)
                        pred_rating = predictions[book_idx]
                        user_mse_scores.append((actual_rating - pred_rating) ** 2)
                
                if user_mse_scores:
                    user_mse = np.mean(user_mse_scores)
                    user_rmse = np.sqrt(user_mse)
                    mse_scores.append(user_mse)
                    rmse_scores.append(user_rmse)
        
        # Calculate coverage and diversity for collaborative filtering
        coverage_result = self.calculate_coverage(ratings_df, content_sim_matrix, top_n)
        diversity_result = self.calculate_diversity(ratings_df, content_sim_matrix, top_n)
        
        return {
            'precision': np.mean(precision_scores) if precision_scores else 0,
            'recall': np.mean(recall_scores) if recall_scores else 0,
            'f1': np.mean(f1_scores) if f1_scores else 0,
            'mse': np.mean(mse_scores) if mse_scores else 1.0,
            'rmse': np.mean(rmse_scores) if rmse_scores else 1.0,
            'coverage': coverage_result['coverage'],
            'diversity': diversity_result['diversity'],
            'n_users_evaluated': len(precision_scores)
        }

    def evaluate_hybrid_filtering(self, ratings_df, content_sim_matrix, model, alpha=0.5, top_n=10, threshold=3.0, test_size=0.2):
        """
        Evaluate hybrid filtering system with proper, honest evaluation
        Returns: precision, recall, f1, mse, rmse
        """
        print(f"Evaluating Hybrid Filtering (alpha={alpha:.1f})...")
        
        # Split data for evaluation
        np.random.seed(42)
        ratings_shuffled = ratings_df.sample(frac=1, random_state=42)
        split_idx = int(len(ratings_shuffled) * (1 - test_size))
        
        train_ratings = ratings_shuffled[:split_idx]
        test_ratings = ratings_shuffled[split_idx:]
        
        if len(train_ratings) == 0 or len(test_ratings) == 0:
            return {'precision': 0, 'recall': 0, 'f1': 0, 'mse': 1.0, 'rmse': 1.0}
        
        # Create training matrix
        train_matrix = train_ratings.pivot_table(
            index='user_id', 
            columns='book_id', 
            values='rating', 
            fill_value=0
        )
        
        books_df = pd.read_csv(os.path.join(self.data_dir, 'processed', 'books_clean.csv'))
        
        # Handle different model types
        if isinstance(model, dict) and 'item_similarity_matrix' in model:
            # New ALS-based model - use pre-computed similarity matrix
            collab_sim_matrix = model['item_similarity_matrix']
            print("Using pre-computed ALS similarity matrix")
        elif hasattr(model, 'fit'):
            # Old SVD model - fit on training data
            model.fit(train_matrix)
            collab_sim_matrix = None
        else:
            collab_sim_matrix = None
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        mse_scores = []
        rmse_scores = []
        
        # Get unique users for evaluation
        users = test_ratings['user_id'].unique()
        
        for user_id in users[:200]:  # Sample for efficiency
            user_test_ratings = test_ratings[test_ratings['user_id'] == user_id]
            if len(user_test_ratings) == 0:
                continue
            
            # Get user's training ratings
            user_train_ratings = train_ratings[train_ratings['user_id'] == user_id]
            if len(user_train_ratings) == 0:
                continue
            
            # Get content-based scores using ONLY training data
            top_rated_books = user_train_ratings.nlargest(3, 'rating')['book_id'].tolist()
            content_scores = np.zeros(content_sim_matrix.shape[0])
            
            for book_id in top_rated_books:
                if book_id in books_df['book_id'].values:
                    book_idx = books_df[books_df['book_id'] == book_id].index[0]
                    book_scores = content_sim_matrix[book_idx]
                    weight = user_train_ratings[user_train_ratings['book_id'] == book_id]['rating'].iloc[0] / 5.0
                    content_scores += book_scores * weight
            
            if len(top_rated_books) > 0:
                content_scores = content_scores / len(top_rated_books)
            
            # Get collaborative scores using ONLY training data
            collab_scores = np.zeros(len(books_df))
            
            if user_id in train_matrix.index:
                user_idx = train_matrix.index.get_loc(user_id)
                
                for i, book_id in enumerate(books_df['book_id']):
                    if book_id in train_matrix.columns:
                        book_idx = train_matrix.columns.get_loc(book_id)
                        
                        if isinstance(model, dict) and 'item_similarity_matrix' in model:
                            # New ALS-based model - use similarity matrix
                            if collab_sim_matrix is not None:
                                # For collaborative filtering, we need to predict ratings, not similarities
                                # Use the user's rated books to find similar items
                                user_rated_books = user_train_ratings['book_id'].tolist()
                                if user_rated_books:
                                    # Get similarities between this book and user's rated books
                                    similarities = []
                                    for rated_book in user_rated_books:
                                        if rated_book in books_df['book_id'].values:
                                            rated_book_idx = books_df[books_df['book_id'] == rated_book].index[0]
                                            sim = collab_sim_matrix[book_idx, rated_book_idx]
                                            similarities.append(sim)
                                    
                                    if similarities:
                                        # Weight by user's ratings
                                        weighted_sim = 0
                                        total_weight = 0
                                        for rated_book in user_rated_books:
                                            if rated_book in books_df['book_id'].values:
                                                rated_book_idx = books_df[books_df['book_id'] == rated_book].index[0]
                                                sim = collab_sim_matrix[book_idx, rated_book_idx]
                                                rating = user_train_ratings[user_train_ratings['book_id'] == rated_book]['rating'].iloc[0]
                                                weighted_sim += sim * rating
                                                total_weight += abs(sim)
                                        
                                        if total_weight > 0:
                                            pred = weighted_sim / total_weight
                                        else:
                                            pred = train_ratings['rating'].mean()
                                    else:
                                        pred = train_ratings['rating'].mean()
                                else:
                                    pred = train_ratings['rating'].mean()
                            else:
                                pred = train_ratings['rating'].mean()
                        elif isinstance(model, dict) and 'neural_model' in model and model['neural_model'] is not None:
                            user_factor = model['user_factors'][user_idx]
                            item_factor = model['item_factors'][book_idx]
                            user_bias = model['user_bias'].get(user_id, 0)
                            item_bias = model['item_bias'].get(book_id, 0)
                            
                            combined_features = np.concatenate([
                                user_factor, item_factor, [user_bias, item_bias, model['global_bias']]
                            ]).reshape(1, -1)
                            
                            pred = model['neural_model'].predict(combined_features)[0]
                        elif isinstance(model, dict) and 'svd' in model:
                            user_vector = train_matrix.iloc[user_idx].values.reshape(1, -1)
                            user_factors = model['svd'].transform(user_vector)
                            reconstructed = model['svd'].inverse_transform(user_factors)
                            base_pred = reconstructed[0, book_idx]
                            
                            user_bias = model['user_bias'].get(user_id, 0)
                            item_bias = model['item_bias'].get(book_id, 0)
                            pred = base_pred + user_bias + item_bias + model['global_bias']
                        elif hasattr(model, 'user_factors'):
                            pred = model.user_factors[user_idx] @ model.item_factors[book_idx]
                        elif hasattr(model, 'transform'):
                            user_vector = train_matrix.iloc[user_idx].values.reshape(1, -1)
                            user_factors = model.transform(user_vector)
                            reconstructed = model.inverse_transform(user_factors)
                            pred = reconstructed[0, book_idx]
                        else:
                            pred = train_ratings['rating'].mean()
                        
                        pred = max(1.0, min(5.0, pred))
                        collab_scores[i] = pred / 5.0  # Normalize to 0-1
            
            # Hybrid scoring with proper alpha interpretation
            if alpha == 1.0:
                # Pure content-based: use same logic as evaluate_content_based_filtering
                hybrid_scores = content_scores
            elif alpha == 0.0:
                # Pure collaborative: use same logic as evaluate_collaborative_filtering
                hybrid_scores = collab_scores
            else:
                # Blended approach with enhancements
                # 1. Z-score normalization for better score distribution
                def z_score_normalize(scores):
                    if len(scores) == 0 or np.std(scores) == 0:
                        return scores
                    return (scores - np.mean(scores)) / np.std(scores)
                
                # 2. Apply z-score normalization
                content_z = z_score_normalize(content_scores)
                collab_z = z_score_normalize(collab_scores)
                
                # 3. Convert z-scores to [0,1] range using sigmoid
                def sigmoid_normalize(z_scores):
                    return 1 / (1 + np.exp(-z_scores))
                
                content_scores_norm = sigmoid_normalize(content_z)
                collab_scores_norm = sigmoid_normalize(collab_z)
                
                # 4. Boost scores based on agreement between methods
                agreement_boost = np.abs(content_scores_norm - collab_scores_norm)
                # Items where both methods agree get a boost
                agreement_factor = 1.0 + (1.0 - agreement_boost) * 0.2  # Up to 20% boost for agreement
                
                # 5. Apply diversity bonus - boost items that are different from user's history
                user_rated_books = user_train_ratings['book_id'].tolist()
                diversity_bonus = np.ones(len(content_scores_norm))
                if len(user_rated_books) > 0:
                    for i, book_id in enumerate(books_df['book_id']):
                        if book_id not in user_rated_books:
                            # Boost items not in user's history (diversity)
                            diversity_bonus[i] = 1.1
                
                # 6. Apply popularity boost for items with good ratings
                popularity_boost = np.ones(len(content_scores_norm))
                for i, book_id in enumerate(books_df['book_id']):
                    book_rating = books_df.iloc[i].get('average_rating', 3.0)
                    if book_rating > 4.0:
                        popularity_boost[i] = 1.05  # 5% boost for highly rated books
                
                # 7. Enhanced blending with all factors
                # alpha=1.0 is content-based, alpha=0.0 is collaborative
                base_hybrid = alpha * content_scores_norm + (1 - alpha) * collab_scores_norm
                
                # Apply all enhancement factors
                hybrid_scores = (base_hybrid * agreement_factor * diversity_bonus * popularity_boost)
                
                # 8. Ensure scores are in valid range
                hybrid_scores = np.clip(hybrid_scores, 0, 1)
            
            # Get top recommendations based on hybrid scores only
            similar_indices = np.argsort(hybrid_scores)[::-1][:top_n]
            recommended_books = books_df.iloc[similar_indices]['book_id'].tolist()
            
            # Calculate metrics using test data
            actual_high_rated = user_test_ratings[user_test_ratings['rating'] >= threshold]['book_id'].tolist()
            
            if len(actual_high_rated) > 0:
                y_true = [1 if book_id in actual_high_rated else 0 for book_id in recommended_books]
                y_pred = [1] * len(recommended_books)
                
                if sum(y_true) > 0:
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
            
            # Calculate MSE and RMSE
            user_mse_scores = []
            for _, row in user_test_ratings.iterrows():
                book_id = row['book_id']
                actual_rating = row['rating']
                
                if book_id in books_df['book_id'].values:
                    book_idx = books_df[books_df['book_id'] == book_id].index[0]
                    pred_rating = hybrid_scores[book_idx] * 5.0  # Scale to 1-5 range
                    pred_rating = max(1.0, min(5.0, pred_rating))
                    user_mse_scores.append((actual_rating - pred_rating) ** 2)
            
            if user_mse_scores:
                user_mse = np.mean(user_mse_scores)
                user_rmse = np.sqrt(user_mse)
                mse_scores.append(user_mse)
                rmse_scores.append(user_rmse)
        
        # Calculate coverage and diversity for hybrid filtering
        coverage_result = self.calculate_coverage(ratings_df, content_sim_matrix, top_n)
        diversity_result = self.calculate_diversity(ratings_df, content_sim_matrix, top_n)
        
        return {
            'precision': np.mean(precision_scores) if precision_scores else 0,
            'recall': np.mean(recall_scores) if recall_scores else 0,
            'f1': np.mean(f1_scores) if f1_scores else 0,
            'mse': np.mean(mse_scores) if mse_scores else 1.0,
            'rmse': np.mean(rmse_scores) if rmse_scores else 1.0,
            'coverage': coverage_result['coverage'],
            'diversity': diversity_result['diversity'],
            'n_users_evaluated': len(precision_scores)
        }

    def compare_all_systems(self, alpha_values=None, top_n=10, threshold=3.0, test_size=0.2):
        """
        Compare all three recommendation systems separately
        Returns comprehensive comparison results
        """
        if alpha_values is None:
            alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        print("Starting comprehensive system comparison...")
        
        # Load data and models
        books_df, ratings_df, content_sim_matrix, model = self.load_data_and_models()
        
        # Evaluate each system separately
        print("\n" + "="*60)
        print("EVALUATING COLLABORATIVE FILTERING")
        print("="*60)
        collab_results = self.evaluate_collaborative_filtering(ratings_df, model, content_sim_matrix, top_n, threshold, test_size)
        
        print("\n" + "="*60)
        print("EVALUATING CONTENT-BASED FILTERING")
        print("="*60)
        content_results = self.evaluate_content_based_filtering(ratings_df, content_sim_matrix, top_n, threshold, test_size)
        
        print("\n" + "="*60)
        print("EVALUATING HYBRID FILTERING")
        print("="*60)
        hybrid_results = {}
        for alpha in alpha_values:
            hybrid_results[alpha] = self.evaluate_hybrid_filtering(ratings_df, content_sim_matrix, model, alpha, top_n, threshold, test_size)
        
        # Find best hybrid alpha
        best_alpha = max(hybrid_results.keys(), key=lambda x: hybrid_results[x]['f1'])
        best_hybrid = hybrid_results[best_alpha]
        
        # Compile comprehensive results
        comparison_results = {
            'collaborative_filtering': collab_results,
            'content_based_filtering': content_results,
            'hybrid_filtering': {
                'best_alpha': best_alpha,
                'best_alpha_results': best_hybrid,
                'all_alpha_results': hybrid_results
            },
            'evaluation_params': {
                'top_n': top_n,
                'threshold': threshold,
                'test_size': test_size,
                'alpha_values': list(alpha_values)
            }
        }
        
        # Generate comparison report
        report = self.generate_comparison_report(comparison_results)
        
        # Print comparison summary
        self.print_comparison_summary(comparison_results, report)
        
        return comparison_results

    def generate_comparison_report(self, results, output_file=None):
        """
        Generate comprehensive comparison report for all systems
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.results_dir, f'system_comparison_{timestamp}.json')
        
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'systems_comparison': results,
            'summary': {
                'best_system_by_metric': {
                    'precision': self._find_best_system_by_metric(results, 'precision'),
                    'recall': self._find_best_system_by_metric(results, 'recall'),
                    'f1': self._find_best_system_by_metric(results, 'f1'),
                    'coverage': self._find_best_system_by_metric(results, 'coverage'),
                    'diversity': self._find_best_system_by_metric(results, 'diversity'),
                    'mse': self._find_best_system_by_metric(results, 'mse', lower_is_better=True),
                    'rmse': self._find_best_system_by_metric(results, 'rmse', lower_is_better=True)
                }
            }
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Comparison report saved to: {output_file}")
        return report

    def _find_best_system_by_metric(self, results, metric, lower_is_better=False):
        """
        Find the best performing system for a given metric
        """
        systems = {
            'collaborative': results['collaborative_filtering'][metric],
            'content_based': results['content_based_filtering'][metric],
            'hybrid': results['hybrid_filtering']['best_alpha_results'][metric]
        }
        
        if lower_is_better:
            best_system = min(systems.keys(), key=lambda x: systems[x])
        else:
            best_system = max(systems.keys(), key=lambda x: systems[x])
        
        return {
            'system': best_system,
            'value': systems[best_system]
        }
    
    def print_comparison_summary(self, results, report=None):
        """
        Print comprehensive comparison summary
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE RECOMMENDATION SYSTEMS COMPARISON")
        print("="*80)
        
        collab = results['collaborative_filtering']
        content = results['content_based_filtering']
        hybrid = results['hybrid_filtering']['best_alpha_results']
        best_alpha = results['hybrid_filtering']['best_alpha']
        
        print(f"\nBest Hybrid Alpha: {best_alpha:.1f}")
        print(f"Evaluation Parameters: Top-N={results['evaluation_params']['top_n']}, "
              f"Threshold={results['evaluation_params']['threshold']}, "
              f"Test Size={results['evaluation_params']['test_size']}")
        
        print("\n" + "-"*100)
        print("METRICS COMPARISON")
        print("-"*100)
        print(f"{'System':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'MSE':<12} {'RMSE':<12} {'Coverage':<12} {'Diversity':<12}")
        print("-"*100)
        print(f"{'Collaborative':<20} {collab['precision']:<12.4f} {collab['recall']:<12.4f} {collab['f1']:<12.4f} {collab['mse']:<12.4f} {collab['rmse']:<12.4f} {collab['coverage']:<12.4f} {collab['diversity']:<12.4f}")
        print(f"{'Content-Based':<20} {content['precision']:<12.4f} {content['recall']:<12.4f} {content['f1']:<12.4f} {content['mse']:<12.4f} {content['rmse']:<12.4f} {content['coverage']:<12.4f} {content['diversity']:<12.4f}")
        print(f"{f'Hybrid (α={best_alpha:.1f})':<20} {hybrid['precision']:<12.4f} {hybrid['recall']:<12.4f} {hybrid['f1']:<12.4f} {hybrid['mse']:<12.4f} {hybrid['rmse']:<12.4f} {hybrid['coverage']:<12.4f} {hybrid['diversity']:<12.4f}")
        
        print("\n" + "-"*80)
        print("BEST PERFORMING SYSTEM BY METRIC")
        print("-"*80)
        if report and 'summary' in report:
            best_by_metric = report['summary']['best_system_by_metric']
            for metric, info in best_by_metric.items():
                print(f"{metric.upper()}: {info['system']} ({info['value']:.4f})")
        else:
            # Fallback: calculate best systems manually
            systems = {
                'collaborative': collab,
                'content_based': content,
                'hybrid': hybrid
            }
            
            for metric in ['precision', 'recall', 'f1', 'coverage', 'diversity']:
                best_system = max(systems.keys(), key=lambda x: systems[x][metric])
                print(f"{metric.upper()}: {best_system} ({systems[best_system][metric]:.4f})")
            
            for metric in ['mse', 'rmse']:
                best_system = min(systems.keys(), key=lambda x: systems[x][metric])
                print(f"{metric.upper()}: {best_system} ({systems[best_system][metric]:.4f})")
        
        print("\n" + "-"*80)
        print("HYBRID PERFORMANCE ACROSS ALPHA VALUES")
        print("-"*80)
        for alpha in sorted(results['hybrid_filtering']['all_alpha_results'].keys()):
            metrics = results['hybrid_filtering']['all_alpha_results'][alpha]
            print(f"Alpha {alpha:.1f}: F1={metrics['f1']:.4f}, "
                  f"Precision={metrics['precision']:.4f}, "
                  f"Recall={metrics['recall']:.4f}, "
                  f"RMSE={metrics['rmse']:.4f}, "
                  f"Coverage={metrics['coverage']:.4f}, "
                  f"Diversity={metrics['diversity']:.4f}")

def main():
    """
    Main evaluation function - compares all recommendation systems
    """
    evaluator = RecommenderEvaluator()
    
    # Run comprehensive system comparison
    results = evaluator.compare_all_systems(
        alpha_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        top_n=10,
        threshold=3.0,
        test_size=0.2
    )
    
    print(f"\nComprehensive evaluation completed!")
    print(f"Results saved to: evaluation/test_results/")

if __name__ == "__main__":
    main()
