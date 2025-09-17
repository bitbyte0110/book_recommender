"""
Retraining script for Hybrid Book Recommender System
Handles scheduled retraining, model drift detection, and incremental updates
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import json
import joblib
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')
sys.path.append('.')

from src.training import HybridRecommenderTrainer
from evaluation.metrics import RecommenderEvaluator

# Ensure logs directory exists before configuring logging
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/retrain.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelRetrainer:
    """
    Handles model retraining and drift detection
    """
    
    def __init__(self, data_dir='data', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.metadata_file = os.path.join(models_dir, 'training_metadata.pkl')
        self.retrain_log_file = 'logs/retrain_history.json'
        
        # Initialize retrain history
        self._initialize_retrain_history()
    
    def _initialize_retrain_history(self):
        """
        Initialize retrain history file
        """
        if not os.path.exists(self.retrain_log_file):
            history = {
                'retrain_events': [],
                'total_retrains': 0,
                'last_retrain': None,
                'performance_trends': []
            }
            with open(self.retrain_log_file, 'w') as f:
                json.dump(history, f, indent=2)
    
    def load_retrain_history(self):
        """
        Load retrain history
        """
        with open(self.retrain_log_file, 'r') as f:
            return json.load(f)
    
    def save_retrain_event(self, event_data):
        """
        Save retrain event to history
        """
        history = self.load_retrain_history()
        
        event_data['timestamp'] = datetime.now().isoformat()
        event_data['event_id'] = len(history['retrain_events']) + 1
        
        history['retrain_events'].append(event_data)
        history['total_retrains'] = len(history['retrain_events'])
        history['last_retrain'] = event_data['timestamp']
        
        with open(self.retrain_log_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def check_model_age(self, max_age_days=30):
        """
        Check if model is older than specified days
        """
        if not os.path.exists(self.metadata_file):
            logger.info("No existing model found. Retraining needed.")
            return True
        
        try:
            metadata = joblib.load(self.metadata_file)
            training_date = datetime.fromisoformat(metadata['training_date'])
            age_days = (datetime.now() - training_date).days
            
            logger.info(f"Model age: {age_days} days")
            
            if age_days > max_age_days:
                logger.info(f"Model is {age_days} days old. Retraining needed.")
                return True
            else:
                logger.info(f"Model is {age_days} days old. No retraining needed.")
                return False
                
        except Exception as e:
            logger.error(f"Error checking model age: {e}")
            return True
    
    def detect_model_drift(self, threshold=0.05):
        """
        Detect model drift by comparing current performance with historical performance
        """
        history = self.load_retrain_history()
        
        if len(history['performance_trends']) < 2:
            logger.info("Insufficient performance history for drift detection.")
            return False
        
        # Get recent performance metrics
        recent_performance = history['performance_trends'][-1]
        previous_performance = history['performance_trends'][-2]
        
        # Calculate performance degradation
        rmse_change = (recent_performance['rmse'] - previous_performance['rmse']) / previous_performance['rmse']
        f1_change = (previous_performance['f1'] - recent_performance['f1']) / previous_performance['f1']
        
        logger.info(f"RMSE change: {rmse_change:.4f}, F1 change: {f1_change:.4f}")
        
        # Check if degradation exceeds threshold
        if rmse_change > threshold or f1_change > threshold:
            logger.info(f"Model drift detected. RMSE change: {rmse_change:.4f}, F1 change: {f1_change:.4f}")
            return True
        
        logger.info("No significant model drift detected.")
        return False
    
    def evaluate_current_performance(self):
        """
        Evaluate current model performance
        """
        try:
            evaluator = RecommenderEvaluator(self.data_dir, self.models_dir)
            comp = evaluator.compare_all_systems(
                alpha_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                top_n=10,
                threshold=3.0,
                test_size=0.2
            )

            best = comp['hybrid_filtering']['best_alpha_results']
            return {
                'rmse': best.get('rmse', 0.0),
                'mae': 0.0,
                'precision': best.get('precision', 0.0),
                'recall': best.get('recall', 0.0),
                'f1': best.get('f1', 0.0),
                'coverage': best.get('coverage', 0.0),
                'diversity': best.get('diversity', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating current performance: {e}")
            return None
    
    def retrain_models(self, force=False, incremental=False):
        """
        Retrain the models
        """
        logger.info("Starting model retraining...")
        
        # Check if retraining is needed
        if not force:
            if not self.check_model_age():
                logger.info("Model is recent. Use --force to retrain anyway.")
                return False
        
        try:
            # Initialize trainer
            trainer = HybridRecommenderTrainer(self.data_dir)
            
            # Retrain models
            if incremental:
                logger.info("Performing incremental retraining...")
                # For incremental training, we would load existing models and update them
                # This is a simplified version - in practice, you'd implement incremental SVD
                books_df, ratings_df = trainer.train_full_pipeline(
                    use_hyperparameter_tuning=False,
                    use_cross_validation=False
                )
            else:
                logger.info("Performing full retraining...")
                books_df, ratings_df = trainer.train_full_pipeline(
                    use_hyperparameter_tuning=True,
                    use_cross_validation=True
                )
            
            # Evaluate new model performance
            new_performance = self.evaluate_current_performance()
            
            if new_performance:
                # Save performance trend
                history = self.load_retrain_history()
                history['performance_trends'].append(new_performance)
                
                with open(self.retrain_log_file, 'w') as f:
                    json.dump(history, f, indent=2)
                
                # Save retrain event
                event_data = {
                    'type': 'incremental' if incremental else 'full',
                    'books_count': len(books_df),
                    'ratings_count': len(ratings_df),
                    'performance': new_performance,
                    'success': True
                }
                self.save_retrain_event(event_data)
                
                logger.info("Model retraining completed successfully!")
                logger.info(f"New performance - RMSE: {new_performance['rmse']:.4f}, F1: {new_performance['f1']:.4f}")
                
                return True
            else:
                logger.error("Failed to evaluate new model performance.")
                return False
                
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            
            # Save failed retrain event
            event_data = {
                'type': 'failed',
                'error': str(e),
                'success': False
            }
            self.save_retrain_event(event_data)
            
            return False
    
    def get_retrain_recommendations(self):
        """
        Get recommendations for when to retrain
        """
        recommendations = []
        
        # Check model age
        if self.check_model_age():
            recommendations.append("Model is older than 30 days")
        
        # Check for model drift
        if self.detect_model_drift():
            recommendations.append("Model drift detected")
        
        # Check data freshness
        try:
            ratings_file = os.path.join(self.data_dir, 'processed', 'ratings.csv')
            if os.path.exists(ratings_file):
                ratings_df = pd.read_csv(ratings_file)
                if len(ratings_df) > 0:
                    # Check if there are new ratings (simplified check)
                    history = self.load_retrain_history()
                    if history['last_retrain']:
                        last_retrain = datetime.fromisoformat(history['last_retrain'])
                        # If more than 7 days since last retrain and new data available
                        if (datetime.now() - last_retrain).days > 7:
                            recommendations.append("New data available for incremental training")
        except Exception as e:
            logger.warning(f"Could not check data freshness: {e}")
        
        return recommendations
    
    def print_retrain_status(self):
        """
        Print current retrain status and recommendations
        """
        print("\n" + "="*60)
        print("MODEL RETRAINING STATUS")
        print("="*60)
        
        # Model age
        age_check = self.check_model_age()
        print(f"Model Age Check: {'Retraining needed' if age_check else 'Model is recent'}")
        
        # Drift detection
        drift_detected = self.detect_model_drift()
        print(f"Drift Detection: {'Drift detected' if drift_detected else 'No drift detected'}")
        
        # Current performance
        performance = self.evaluate_current_performance()
        if performance:
            print(f"Current RMSE: {performance['rmse']:.4f}")
            print(f"Current F1-Score: {performance['f1']:.4f}")
        
        # Recommendations
        recommendations = self.get_retrain_recommendations()
        if recommendations:
            print("\nRetraining Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("\nNo retraining recommendations at this time.")
        
        # Retrain history
        history = self.load_retrain_history()
        print(f"\nTotal Retrains: {history['total_retrains']}")
        if history['last_retrain']:
            last_retrain = datetime.fromisoformat(history['last_retrain'])
            print(f"Last Retrain: {last_retrain.strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """
    Main function for retraining script
    """
    parser = argparse.ArgumentParser(description='Retrain Hybrid Book Recommender Models')
    parser.add_argument('--force', action='store_true', 
                       help='Force retraining even if model is recent')
    parser.add_argument('--incremental', action='store_true',
                       help='Perform incremental retraining instead of full retraining')
    parser.add_argument('--check', action='store_true',
                       help='Check retraining status without retraining')
    parser.add_argument('--data-dir', default='data',
                       help='Directory containing data files')
    parser.add_argument('--models-dir', default='models',
                       help='Directory for model files')
    
    args = parser.parse_args()
    
    # Initialize retrainer
    retrainer = ModelRetrainer(args.data_dir, args.models_dir)
    
    if args.check:
        # Just check status
        retrainer.print_retrain_status()
    else:
        # Perform retraining
        success = retrainer.retrain_models(
            force=args.force,
            incremental=args.incremental
        )
        
        if success:
            print("\n✅ Retraining completed successfully!")
            retrainer.print_retrain_status()
        else:
            print("\n❌ Retraining failed. Check logs for details.")
            sys.exit(1)

if __name__ == "__main__":
    main()
