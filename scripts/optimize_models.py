"""
Model Optimization Script for Hybrid Book Recommender System
Iteratively trains models with different configurations to achieve best performance
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import shutil

# Add src to path
sys.path.append('src')
sys.path.append('.')

from src.training import HybridRecommenderTrainer
from evaluation.metrics import RecommenderEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """
    Optimizes model performance through iterative training and evaluation
    """
    
    def __init__(self, data_dir='data', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.optimization_log = 'logs/optimization_history.json'
        self.best_performance = None
        self.best_config = None
        self.iteration = 0
        
        # Initialize optimization history
        self._initialize_optimization_history()
    
    def _initialize_optimization_history(self):
        """Initialize optimization history file"""
        if not os.path.exists(self.optimization_log):
            history = {
                'optimization_runs': [],
                'best_performance': None,
                'best_config': None,
                'total_iterations': 0,
                'convergence_history': []
            }
            with open(self.optimization_log, 'w') as f:
                json.dump(history, f, indent=2)
    
    def load_optimization_history(self):
        """Load optimization history"""
        with open(self.optimization_log, 'r') as f:
            return json.load(f)
    
    def save_optimization_result(self, config, performance, improvement):
        """Save optimization result"""
        history = self.load_optimization_history()
        
        result = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'performance': performance,
            'improvement': float(improvement),
            'is_best': bool(improvement > 0)
        }
        
        history['optimization_runs'].append(result)
        history['total_iterations'] = len(history['optimization_runs'])
        
        if improvement > 0:
            history['best_performance'] = performance
            history['best_config'] = config
        
        with open(self.optimization_log, 'w') as f:
            json.dump(history, f, indent=2)
    
    def generate_config_variations(self, base_config=None):
        """Generate different configuration variations for optimization"""
        if base_config is None:
            base_config = {
                'content_max_features': [1000, 2000, 3000, 5000],
                'content_ngram_range': [(1, 1), (1, 2), (2, 2)],
                'content_min_df': [1, 2, 3],
                'collab_n_factors': [10, 20, 30, 40, 50],
                'alpha_values': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            }
        
        configs = []
        
        # Generate combinations of hyperparameters
        for max_features in base_config['content_max_features']:
            for ngram_range in base_config['content_ngram_range']:
                for min_df in base_config['content_min_df']:
                    for n_factors in base_config['collab_n_factors']:
                        config = {
                            'content_max_features': max_features,
                            'content_ngram_range': ngram_range,
                            'content_min_df': min_df,
                            'collab_n_factors': n_factors,
                            'alpha_values': base_config['alpha_values']
                        }
                        configs.append(config)
        
        return configs
    
    def train_with_config(self, config):
        """Train models with specific configuration"""
        logger.info(f"Training with config: {config}")
        
        try:
            # Initialize trainer
            trainer = HybridRecommenderTrainer(self.data_dir)
            
            # Train with custom hyperparameters
            books_df, ratings_df = trainer.train_full_pipeline(
                use_hyperparameter_tuning=False,  # We're doing our own tuning
                use_cross_validation=False,
                custom_params=config
            )
            
            return True, books_df, ratings_df
            
        except Exception as e:
            logger.error(f"Training failed with config {config}: {e}")
            return False, None, None
    
    def evaluate_performance(self, config):
        """Evaluate model performance with specific configuration"""
        try:
            evaluator = RecommenderEvaluator(self.data_dir, self.models_dir)
            
            # Test with different alpha values
            alpha_values = config.get('alpha_values', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            results = evaluator.run_full_evaluation(alpha_values=alpha_values)
            
            # Find best alpha and performance
            best_alpha = max(results.keys(), key=lambda x: results[x]['f1'])
            best_metrics = results[best_alpha]
            
            # Calculate composite score (weighted combination of metrics)
            composite_score = (
                0.3 * best_metrics['f1'] +           # 30% weight on F1
                0.2 * (1 - best_metrics['rmse']/5) + # 20% weight on RMSE (inverted, normalized)
                0.2 * best_metrics['precision'] +    # 20% weight on precision
                0.15 * best_metrics['coverage'] +    # 15% weight on coverage
                0.15 * best_metrics['diversity']     # 15% weight on diversity
            )
            
            performance = {
                'alpha': best_alpha,
                'f1': best_metrics['f1'],
                'precision': best_metrics['precision'],
                'recall': best_metrics['recall'],
                'rmse': best_metrics['rmse'],
                'mae': best_metrics['mae'],
                'coverage': best_metrics['coverage'],
                'diversity': best_metrics['diversity'],
                'composite_score': composite_score,
                'all_results': results
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None
    
    def optimize_models(self, max_iterations=50, improvement_threshold=0.001):
        """
        Optimize models through iterative training and evaluation
        """
        logger.info("Starting model optimization...")
        
        start_time = time.time()
        configs = self.generate_config_variations()
        
        logger.info(f"Generated {len(configs)} configuration variations")
        
        best_composite_score = 0
        no_improvement_count = 0
        max_no_improvement = 10
        
        for i, config in enumerate(configs):
            if i >= max_iterations:
                logger.info(f"Reached maximum iterations ({max_iterations})")
                break
            
            self.iteration = i + 1
            logger.info(f"\n--- Iteration {self.iteration}/{len(configs)} ---")
            
            # Train with current configuration
            success, books_df, ratings_df = self.train_with_config(config)
            
            if not success:
                logger.warning(f"Skipping failed configuration")
                continue
            
            # Evaluate performance
            performance = self.evaluate_performance(config)
            
            if performance is None:
                logger.warning(f"Skipping failed evaluation")
                continue
            
            # Check for improvement
            current_score = performance['composite_score']
            improvement = current_score - best_composite_score
            
            logger.info(f"Performance - F1: {performance['f1']:.4f}, "
                       f"RMSE: {performance['rmse']:.4f}, "
                       f"Composite: {current_score:.4f}")
            
            if improvement > improvement_threshold:
                best_composite_score = current_score
                self.best_performance = performance
                self.best_config = config
                no_improvement_count = 0
                
                logger.info(f"NEW BEST PERFORMANCE! Composite Score: {current_score:.4f}")
                logger.info(f"Best Alpha: {performance['alpha']}")
                logger.info(f"F1: {performance['f1']:.4f}, RMSE: {performance['rmse']:.4f}")
            else:
                no_improvement_count += 1
                logger.info(f"No significant improvement (count: {no_improvement_count})")
            
            # Save result
            self.save_optimization_result(config, performance, improvement)
            
            # Early stopping if no improvement for too long
            if no_improvement_count >= max_no_improvement:
                logger.info(f"No improvement for {max_no_improvement} iterations. Stopping optimization.")
                break
        
        total_time = time.time() - start_time
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION COMPLETED")
        logger.info("="*80)
        logger.info(f"Total iterations: {self.iteration}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        
        if self.best_performance:
            logger.info(f"\n🏆 BEST PERFORMANCE ACHIEVED:")
            logger.info(f"Composite Score: {self.best_performance['composite_score']:.4f}")
            logger.info(f"F1-Score: {self.best_performance['f1']:.4f}")
            logger.info(f"Precision: {self.best_performance['precision']:.4f}")
            logger.info(f"Recall: {self.best_performance['recall']:.4f}")
            logger.info(f"RMSE: {self.best_performance['rmse']:.4f}")
            logger.info(f"Coverage: {self.best_performance['coverage']:.4f}")
            logger.info(f"Diversity: {self.best_performance['diversity']:.4f}")
            logger.info(f"Best Alpha: {self.best_performance['alpha']}")
            
            logger.info(f"\n🔧 BEST CONFIGURATION:")
            for key, value in self.best_config.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.warning("No successful optimization runs completed.")
        
        return self.best_performance, self.best_config

def cleanup_pycache():
    """Clean up __pycache__ directories"""
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs[:]:  # Use slice to avoid modifying list while iterating
            if dir_name == '__pycache__':
                pycache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(pycache_path)
                    print(f"Cleaned up: {pycache_path}")
                except Exception as e:
                    print(f"Could not remove {pycache_path}: {e}")

def main():
    """Main optimization function"""
    print("🚀 Starting Model Optimization for Best Performance")
    print("="*60)
    
    optimizer = ModelOptimizer()
    
    # Run optimization
    best_performance, best_config = optimizer.optimize_models(
        max_iterations=100,  # Test up to 100 configurations
        improvement_threshold=0.001  # Stop if improvement is less than 0.1%
    )
    
    # Clean up __pycache__ directories
    cleanup_pycache()
    
    if best_performance:
        print(f"\n✅ Optimization completed successfully!")
        print(f"Best composite score: {best_performance['composite_score']:.4f}")
        print(f"Check logs/optimization_history.json for detailed results.")
    else:
        print(f"\n❌ Optimization failed. Check logs for details.")

if __name__ == "__main__":
    main()
