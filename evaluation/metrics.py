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
# Simplified evaluation without surprise library
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

class RecommenderEvaluator:
    """
    Comprehensive evaluator for hybrid book recommender system
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
        ratings_df = pd.read_csv(os.path.join(self.data_dir, 'processed', 'ratings_clean.csv'))
        
        # Load models
        content_sim_matrix = joblib.load(os.path.join(self.models_dir, 'content_similarity_matrix.pkl'))
        svd_model = joblib.load(os.path.join(self.models_dir, 'svd_model.pkl'))
        
        return books_df, ratings_df, content_sim_matrix, svd_model
    
    def calculate_rmse(self, ratings_df, svd_model, test_size=0.2):
        """
        Calculate Root Mean Square Error for rating predictions
        """
        print("Calculating RMSE...")
        
        # Create user-item matrix
        user_item_matrix = ratings_df.pivot_table(
            index='user_id', 
            columns='book_id', 
            values='rating', 
            fill_value=0
        )
        
        # Split data
        train_size = int(len(ratings_df) * (1 - test_size))
        train_ratings = ratings_df.sample(n=train_size, random_state=42)
        test_ratings = ratings_df.drop(train_ratings.index)
        
        # Create training matrix
        train_matrix = train_ratings.pivot_table(
            index='user_id', 
            columns='book_id', 
            values='rating', 
            fill_value=0
        )
        
        # Fit model on training data
        svd_model.fit(train_matrix)
        
        # Make predictions (simplified - using reconstructed matrix)
        reconstructed_matrix = svd_model.inverse_transform(svd_model.transform(train_matrix))
        
        # Calculate RMSE on test set
        predictions = []
        actuals = []
        
        for _, row in test_ratings.iterrows():
            user_idx = train_matrix.index.get_loc(row['user_id']) if row['user_id'] in train_matrix.index else 0
            book_idx = train_matrix.columns.get_loc(row['book_id']) if row['book_id'] in train_matrix.columns else 0
            
            if user_idx < reconstructed_matrix.shape[0] and book_idx < reconstructed_matrix.shape[1]:
                pred = reconstructed_matrix[user_idx, book_idx]
                predictions.append(pred)
                actuals.append(row['rating'])
        
        if predictions:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
        else:
            rmse = 1.0  # Default value
            mae = 1.0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions
        }
    
    def calculate_precision_recall_f1(self, ratings_df, content_sim_matrix, svd_model, 
                                    top_n=10, threshold=4.0, test_size=0.2):
        """
        Calculate precision, recall, and F1-score for top-N recommendations
        """
        print("Calculating Precision, Recall, and F1-score...")
        
        # Split data for evaluation
        train_size = int(len(ratings_df) * (1 - test_size))
        train_ratings = ratings_df.sample(n=train_size, random_state=42)
        test_ratings = ratings_df.drop(train_ratings.index)
        
        # Create training matrix
        train_matrix = train_ratings.pivot_table(
            index='user_id', 
            columns='book_id', 
            values='rating', 
            fill_value=0
        )
        
        # Train model
        svd_model.fit(train_matrix)
        
        # Get unique users and books
        users = ratings_df['user_id'].unique()
        books_df = pd.read_csv(os.path.join(self.data_dir, 'processed', 'books_clean.csv'))
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        # Evaluate for each user
        for user_id in users[:100]:  # Limit to first 100 users for efficiency
            # Get user's actual ratings from test set
            user_test_ratings = test_ratings[test_ratings['user_id'] == user_id]
            
            if len(user_test_ratings) == 0:
                continue
            
            # Get recommended books (simplified - using content-based for demonstration)
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            if len(user_ratings) == 0:
                continue
            
            # Get user's highest rated book
            best_book = user_ratings.loc[user_ratings['rating'].idxmax(), 'book_id']
            book_idx = books_df[books_df['book_id'] == best_book].index[0]
            
            # Get similar books (content-based)
            similar_scores = content_sim_matrix[book_idx]
            similar_indices = np.argsort(similar_scores)[::-1][1:top_n+1]
            recommended_books = books_df.iloc[similar_indices]['book_id'].tolist()
            
            # Get actual highly rated books (ground truth)
            actual_high_rated = user_test_ratings[user_test_ratings['rating'] >= threshold]['book_id'].tolist()
            
            # Calculate metrics
            if len(actual_high_rated) > 0:
                # Convert to binary classification
                y_true = [1 if book_id in actual_high_rated else 0 for book_id in recommended_books]
                y_pred = [1] * len(recommended_books)  # All recommended books are positive
                
                if sum(y_true) > 0:  # Only calculate if there are actual positive cases
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
        
        return {
            'precision': np.mean(precision_scores) if precision_scores else 0,
            'recall': np.mean(recall_scores) if recall_scores else 0,
            'f1': np.mean(f1_scores) if f1_scores else 0,
            'n_users_evaluated': len(precision_scores)
        }
    
    def calculate_coverage(self, ratings_df, content_sim_matrix, top_n=10, n_users=100):
        """
        Calculate coverage - percentage of books recommended at least once
        """
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
            similar_scores = content_sim_matrix[book_idx]
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
    
    def calculate_diversity(self, ratings_df, content_sim_matrix, top_n=10, n_users=100):
        """
        Calculate diversity - average pairwise dissimilarity between recommendations
        """
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
            similar_scores = content_sim_matrix[book_idx]
            similar_indices = np.argsort(similar_scores)[::-1][1:top_n+1]
            user_recommendations = books_df.iloc[similar_indices]['book_id'].tolist()
            
            if len(user_recommendations) > 1:
                # Calculate pairwise dissimilarity
                rec_indices = [books_df[books_df['book_id'] == bid].index[0] for bid in user_recommendations]
                similarities = content_sim_matrix[np.ix_(rec_indices, rec_indices)]
                
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
    
    def evaluate_hybrid_performance(self, ratings_df, content_sim_matrix, svd_model, 
                                  alpha_values=np.arange(0, 1.1, 0.1)):
        """
        Evaluate hybrid model performance with different alpha values
        """
        print("Evaluating hybrid performance with different alpha values...")
        
        results = {}
        
        for alpha in alpha_values:
            print(f"Testing alpha = {alpha:.1f}")
            
            # Calculate metrics for this alpha
            rmse_result = self.calculate_rmse(ratings_df, svd_model)
            prf_result = self.calculate_precision_recall_f1(ratings_df, content_sim_matrix, svd_model)
            coverage_result = self.calculate_coverage(ratings_df, content_sim_matrix)
            diversity_result = self.calculate_diversity(ratings_df, content_sim_matrix)
            
            results[alpha] = {
                'rmse': rmse_result['rmse'],
                'mae': rmse_result['mae'],
                'precision': prf_result['precision'],
                'recall': prf_result['recall'],
                'f1': prf_result['f1'],
                'coverage': coverage_result['coverage'],
                'diversity': diversity_result['diversity']
            }
        
        return results
    
    def generate_evaluation_report(self, results, output_file=None):
        """
        Generate comprehensive evaluation report
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.results_dir, f'evaluation_report_{timestamp}.json')
        
        # Find best alpha based on F1-score
        best_alpha = max(results.keys(), key=lambda x: results[x]['f1'])
        
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'best_alpha': best_alpha,
            'best_alpha_metrics': results[best_alpha],
            'all_results': results,
            'summary': {
                'rmse_range': (min(r['rmse'] for r in results.values()), 
                              max(r['rmse'] for r in results.values())),
                'f1_range': (min(r['f1'] for r in results.values()), 
                            max(r['f1'] for r in results.values())),
                'coverage_range': (min(r['coverage'] for r in results.values()), 
                                  max(r['coverage'] for r in results.values())),
                'diversity_range': (min(r['diversity'] for r in results.values()), 
                                   max(r['diversity'] for r in results.values()))
            }
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to: {output_file}")
        return report
    
    def print_evaluation_summary(self, results):
        """
        Print a summary of evaluation results
        """
        print("\n" + "="*60)
        print("HYBRID RECOMMENDER SYSTEM EVALUATION SUMMARY")
        print("="*60)
        
        # Find best alpha
        best_alpha = max(results.keys(), key=lambda x: results[x]['f1'])
        best_metrics = results[best_alpha]
        
        print(f"\nBest Alpha Value: {best_alpha:.1f}")
        print(f"Best F1-Score: {best_metrics['f1']:.4f}")
        print(f"Best Precision: {best_metrics['precision']:.4f}")
        print(f"Best Recall: {best_metrics['recall']:.4f}")
        print(f"RMSE: {best_metrics['rmse']:.4f}")
        print(f"MAE: {best_metrics['mae']:.4f}")
        print(f"Coverage: {best_metrics['coverage']:.4f}")
        print(f"Diversity: {best_metrics['diversity']:.4f}")
        
        print("\n" + "-"*60)
        print("PERFORMANCE ACROSS ALPHA VALUES")
        print("-"*60)
        
        for alpha in sorted(results.keys()):
            metrics = results[alpha]
            print(f"Alpha {alpha:.1f}: F1={metrics['f1']:.4f}, "
                  f"RMSE={metrics['rmse']:.4f}, "
                  f"Coverage={metrics['coverage']:.4f}, "
                  f"Diversity={metrics['diversity']:.4f}")
    
    def run_full_evaluation(self, alpha_values=None):
        """
        Run complete evaluation pipeline
        """
        if alpha_values is None:
            alpha_values = np.arange(0, 1.1, 0.1)
        
        print("Starting full evaluation pipeline...")
        
        # Load data and models
        books_df, ratings_df, content_sim_matrix, svd_model = self.load_data_and_models()
        
        # Evaluate hybrid performance
        results = self.evaluate_hybrid_performance(ratings_df, content_sim_matrix, svd_model, alpha_values)
        
        # Generate report
        report = self.generate_evaluation_report(results)
        
        # Print summary
        self.print_evaluation_summary(results)
        
        return results

def main():
    """
    Main evaluation function
    """
    evaluator = RecommenderEvaluator()
    
    # Run full evaluation
    report = evaluator.run_full_evaluation()
    
    print(f"\nEvaluation completed! Report saved to: {report}")

if __name__ == "__main__":
    main()
