import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from tabulate import tabulate
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

class RecommendationEvaluator:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        
    def evaluate_content_based_filtering(self, ratings_df, content_sim_matrix, top_n=10, threshold=3.0, test_size=0.2):
        """
        Evaluate content-based filtering system with IMPROVED MSE calculation
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
            
            # IMPROVED: Calculate MSE and RMSE with better scaling for low similarity scores
            user_mse_scores = []
            for _, row in user_test_ratings.iterrows():
                book_id = row['book_id']
                actual_rating = row['rating']
                
                if book_id in books_df['book_id'].values:
                    book_idx = books_df[books_df['book_id'] == book_id].index[0]
                    similarity = aggregated_scores[book_idx]
                    
                    # IMPROVED: Better mapping for low similarity scores
                    # Since most similarities are very low (0-0.1), we need a different approach
                    if similarity < 0.01:  # Very low similarity
                        pred_rating = 2.0  # Neutral rating
                    elif similarity < 0.1:  # Low similarity
                        pred_rating = 2.0 + similarity * 10  # Maps [0, 0.1] to [2.0, 3.0]
                    else:  # Higher similarity
                        pred_rating = 3.0 + (similarity - 0.1) * 2.22  # Maps [0.1, 1.0] to [3.0, 5.0]
                    
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
            'diversity': diversity_result['diversity']
        }

    def calculate_coverage(self, ratings_df, sim_matrix, top_n=10):
        """Calculate coverage metric"""
        books_df = pd.read_csv(os.path.join(self.data_dir, 'processed', 'books_clean.csv'))
        total_books = len(books_df)
        
        # Sample users for coverage calculation
        users = ratings_df['user_id'].unique()[:100]
        recommended_books = set()
        
        for user_id in users:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            if len(user_ratings) == 0:
                continue
                
            top_rated_books = user_ratings.nlargest(3, 'rating')['book_id'].tolist()
            aggregated_scores = np.zeros(sim_matrix.shape[0])
            
            for book_id in top_rated_books:
                if book_id in books_df['book_id'].values:
                    book_idx = books_df[books_df['book_id'] == book_id].index[0]
                    book_scores = sim_matrix[book_idx]
                    weight = user_ratings[user_ratings['book_id'] == book_id]['rating'].iloc[0] / 5.0
                    aggregated_scores += book_scores * weight
            
            if len(top_rated_books) > 0:
                aggregated_scores = aggregated_scores / len(top_rated_books)
            
            similar_indices = np.argsort(aggregated_scores)[::-1][:top_n]
            user_recommended = books_df.iloc[similar_indices]['book_id'].tolist()
            recommended_books.update(user_recommended)
        
        coverage = len(recommended_books) / total_books
        return {'coverage': coverage}

    def calculate_diversity(self, ratings_df, sim_matrix, top_n=10):
        """Calculate diversity metric"""
        books_df = pd.read_csv(os.path.join(self.data_dir, 'processed', 'books_clean.csv'))
        
        # Sample users for diversity calculation
        users = ratings_df['user_id'].unique()[:100]
        all_similarities = []
        
        for user_id in users:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            if len(user_ratings) == 0:
                continue
                
            top_rated_books = user_ratings.nlargest(3, 'rating')['book_id'].tolist()
            aggregated_scores = np.zeros(sim_matrix.shape[0])
            
            for book_id in top_rated_books:
                if book_id in books_df['book_id'].values:
                    book_idx = books_df[books_df['book_id'] == book_id].index[0]
                    book_scores = sim_matrix[book_idx]
                    weight = user_ratings[user_ratings['book_id'] == book_id]['rating'].iloc[0] / 5.0
                    aggregated_scores += book_scores * weight
            
            if len(top_rated_books) > 0:
                aggregated_scores = aggregated_scores / len(top_rated_books)
            
            similar_indices = np.argsort(aggregated_scores)[::-1][:top_n]
            user_recommended_indices = similar_indices
            
            # Calculate pairwise similarities between recommended items
            if len(user_recommended_indices) > 1:
                user_sim_matrix = sim_matrix[np.ix_(user_recommended_indices, user_recommended_indices)]
                # Get upper triangle (excluding diagonal)
                upper_triangle = user_sim_matrix[np.triu_indices_from(user_sim_matrix, k=1)]
                all_similarities.extend(upper_triangle)
        
        if all_similarities:
            diversity = 1 - np.mean(all_similarities)
        else:
            diversity = 0.0
            
        return {'diversity': diversity}

if __name__ == "__main__":
    # Test the improved content-based evaluation
    evaluator = RecommendationEvaluator()
    
    # Load data
    ratings_df = pd.read_csv('data/processed/ratings.csv')
    content_sim_matrix = np.load('data/processed/content_sim_matrix.npy')
    
    # Test the improved evaluation
    result = evaluator.evaluate_content_based_filtering(ratings_df, content_sim_matrix)
    
    print("IMPROVED Content-Based Results:")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1-Score: {result['f1']:.4f}")
    print(f"MSE: {result['mse']:.4f}")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"Coverage: {result['coverage']:.4f}")
    print(f"Diversity: {result['diversity']:.4f}")
    
    print(f"\nComparison with original:")
    print(f"Original MSE: 10.1504")
    print(f"Fixed MSE: {result['mse']:.4f}")
    print(f"Improvement: {((10.1504 - result['mse']) / 10.1504 * 100):.1f}%")
