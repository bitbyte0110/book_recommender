"""
Comprehensive Evaluation Script for Hybrid Book Recommender System
Runs complete training, evaluation, and user satisfaction assessment
"""

import os
import sys
import time
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from src.training import HybridRecommenderTrainer
from evaluation.metrics import RecommenderEvaluator
from evaluation.user_survey import UserSatisfactionSurvey

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_section(title):
    """Print formatted section"""
    print(f"\n--- {title} ---")

def main():
    """
    Main evaluation pipeline
    """
    print_header("HYBRID BOOK RECOMMENDER SYSTEM - COMPREHENSIVE EVALUATION")
    
    start_time = time.time()
    
    try:
        # Step 1: Training
        print_section("STEP 1: MODEL TRAINING")
        print("Training hybrid recommender models...")
    
        trainer = HybridRecommenderTrainer()
        
        # Check if models exist
        if trainer.load_models():
            print("✅ Pre-trained models found and loaded successfully!")
        else:
            print("🔄 No pre-trained models found. Starting training pipeline...")
            
            books_df, ratings_df = trainer.train_full_pipeline(
                use_hyperparameter_tuning=True,
                use_cross_validation=True
            )
            print(f"✅ Training completed successfully!")
            print(f"   - Books: {len(books_df)}")
            print(f"   - Ratings: {len(ratings_df)}")
            print(f"   - Users: {ratings_df['user_id'].nunique()}")
        
        # Step 2: Model Evaluation
        print_section("STEP 2: MODEL EVALUATION")
        print("Evaluating model performance...")
        
        evaluator = RecommenderEvaluator()
        
        # Run comprehensive evaluation
        results = evaluator.run_full_evaluation()
        
        print("✅ Evaluation completed successfully!")
        
        # Find best alpha
        best_alpha = max(results.keys(), key=lambda x: results[x]['f1'])
        best_metrics = results[best_alpha]
        
        print(f"\n📊 BEST PERFORMANCE (Alpha = {best_alpha:.1f}):")
        print(f"   - F1-Score: {best_metrics['f1']:.4f}")
        print(f"   - Precision: {best_metrics['precision']:.4f}")
        print(f"   - Recall: {best_metrics['recall']:.4f}")
        print(f"   - RMSE: {best_metrics['rmse']:.4f}")
        print(f"   - MAE: {best_metrics['mae']:.4f}")
        print(f"   - Coverage: {best_metrics['coverage']:.4f}")
        print(f"   - Diversity: {best_metrics['diversity']:.4f}")
        
        # Step 3: User Satisfaction Survey
        print_section("STEP 3: USER SATISFACTION ASSESSMENT")
        print("Setting up user satisfaction survey system...")
        
        survey = UserSatisfactionSurvey()
        
        # Get current satisfaction metrics
        try:
            satisfaction_metrics = survey.get_satisfaction_metrics()
            
            print(f"📈 CURRENT SATISFACTION METRICS:")
            print(f"   - Total Responses: {satisfaction_metrics['total_responses']}")
            print(f"   - Average Satisfaction: {satisfaction_metrics['average_satisfaction']:.2f}/5")
            
            if satisfaction_metrics['total_responses'] > 0 and 'component_scores' in satisfaction_metrics:
                print(f"   - Component Scores:")
                for component, score in satisfaction_metrics['component_scores'].items():
                    print(f"     * {component.capitalize()}: {score:.2f}/5")
        except Exception as e:
            print(f"⚠️  Warning: Could not load satisfaction metrics: {e}")
            satisfaction_metrics = {
                'total_responses': 0,
                'average_satisfaction': 0.0,
                'component_scores': {}
            }
        
        # Step 4: Performance Analysis
        print_section("STEP 4: PERFORMANCE ANALYSIS")
        
        # Analyze performance across alpha values
        alpha_performance = {}
        for alpha in sorted(results.keys()):
            metrics = results[alpha]
            alpha_performance[alpha] = {
                'f1': metrics['f1'],
                'rmse': metrics['rmse'],
                'coverage': metrics['coverage'],
                'diversity': metrics['diversity']
            }
        
        # Find optimal alpha for different criteria
        best_f1_alpha = max(alpha_performance.keys(), key=lambda x: alpha_performance[x]['f1'])
        best_rmse_alpha = min(alpha_performance.keys(), key=lambda x: alpha_performance[x]['rmse'])
        best_coverage_alpha = max(alpha_performance.keys(), key=lambda x: alpha_performance[x]['coverage'])
        best_diversity_alpha = max(alpha_performance.keys(), key=lambda x: alpha_performance[x]['diversity'])
        
        print("🎯 OPTIMAL ALPHA VALUES:")
        print(f"   - Best F1-Score: Alpha = {best_f1_alpha:.1f} (F1 = {alpha_performance[best_f1_alpha]['f1']:.4f})")
        print(f"   - Best RMSE: Alpha = {best_rmse_alpha:.1f} (RMSE = {alpha_performance[best_rmse_alpha]['rmse']:.4f})")
        print(f"   - Best Coverage: Alpha = {best_coverage_alpha:.1f} (Coverage = {alpha_performance[best_coverage_alpha]['coverage']:.4f})")
        print(f"   - Best Diversity: Alpha = {best_diversity_alpha:.1f} (Diversity = {alpha_performance[best_diversity_alpha]['diversity']:.4f})")
        
        # Step 5: System Recommendations
        print_section("STEP 5: SYSTEM RECOMMENDATIONS")
        
        recommendations = []
        
        # Alpha recommendations
        if best_f1_alpha != 0.6:
            recommendations.append(f"Consider using Alpha = {best_f1_alpha:.1f} for optimal F1-score")
        
        # Performance thresholds
        if best_metrics['f1'] < 0.3:
            recommendations.append("F1-score is low - consider improving data quality or model parameters")
        
        if best_metrics['rmse'] > 1.0:
            recommendations.append("RMSE is high - consider feature engineering or model tuning")
        
        if best_metrics['coverage'] < 0.1:
            recommendations.append("Coverage is low - consider increasing recommendation diversity")
        
        # User satisfaction
        if satisfaction_metrics['total_responses'] > 0:
            if satisfaction_metrics['average_satisfaction'] < 3.0:
                recommendations.append("User satisfaction is low - gather more feedback and improve recommendations")
        
        if recommendations:
            print("💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print("✅ No specific recommendations at this time.")
        
        # Step 6: Generate Final Report
        print_section("STEP 6: GENERATING FINAL REPORT")
        
        # Create comprehensive report
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'execution_time_seconds': time.time() - start_time,
            'best_alpha': best_alpha,
            'best_metrics': best_metrics,
            'alpha_performance': alpha_performance,
            'optimal_alphas': {
                'best_f1': best_f1_alpha,
                'best_rmse': best_rmse_alpha,
                'best_coverage': best_coverage_alpha,
                'best_diversity': best_diversity_alpha
            },
            'satisfaction_metrics': satisfaction_metrics,
            'recommendations': recommendations,
            'all_results': results
        }
        
        # Save report
        report_file = f"evaluation/test_results/comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Comprehensive report saved to: {report_file}")
        
        # Step 7: Summary
        print_section("EVALUATION SUMMARY")
        
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total execution time: {time.time() - start_time:.2f} seconds")
        print(f"📊 Best F1-Score: {best_metrics['f1']:.4f} (Alpha = {best_alpha:.1f})")
        print(f"📈 Best RMSE: {best_metrics['rmse']:.4f}")
        print(f"🎯 Coverage: {best_metrics['coverage']:.4f}")
        print(f"🌈 Diversity: {best_metrics['diversity']:.4f}")
        
        if satisfaction_metrics['total_responses'] > 0:
            print(f"😊 User Satisfaction: {satisfaction_metrics['average_satisfaction']:.2f}/5")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Review the comprehensive evaluation report")
        print("2. Implement system recommendations")
        print("3. Run user satisfaction surveys")
        print("4. Set up automated retraining pipeline")
        print("5. Monitor system performance over time")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
