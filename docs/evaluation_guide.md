# Evaluation Framework for Hybrid Book Recommender System

This directory contains the comprehensive evaluation framework for the hybrid book recommender system, including training, evaluation metrics, user satisfaction surveys, and automated retraining capabilities.

## 📁 Directory Structure

```
evaluation/
├── README.md                 # This file
├── metrics.py               # Evaluation metrics implementation
├── user_survey.py           # User satisfaction survey system
├── test_results/            # Evaluation outputs and reports
└── user_survey/             # Survey responses and analytics
```

## 🎯 Evaluation Requirements Met

This framework addresses all the evaluation requirements specified:

### i. Precision, Recall, and F1-Score
- **Implementation**: `evaluation/metrics.py`
- **Metrics**: Calculated for top-N recommendations
- **Method**: Binary classification approach with threshold-based evaluation
- **Usage**: `RecommenderEvaluator.calculate_precision_recall_f1()`

### ii. Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
- **Implementation**: `evaluation/metrics.py`
- **Metrics**: RMSE and MAE for rating predictions
- **Method**: Surprise library integration with train/test splits
- **Usage**: `RecommenderEvaluator.calculate_rmse()`

### iii. User Satisfaction through Questionnaire
- **Implementation**: `evaluation/user_survey.py`
- **Features**: Interactive Streamlit survey with analytics
- **Metrics**: Relevance, accuracy, diversity, and overall satisfaction
- **Usage**: `UserSatisfactionSurvey.show_satisfaction_survey()`

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Evaluation
```bash
python evaluate_system.py
```

### 3. Run Individual Components

#### Training Only
```bash
python src/training.py
```

#### Evaluation Only
```bash
python evaluation/metrics.py
```

#### User Survey
```bash
streamlit run evaluation/user_survey.py
```

#### Retraining Check
```bash
python retrain.py --check
```

## 📊 Evaluation Metrics

### Content-Based Metrics
- **TF-IDF Vectorization**: Text-based similarity using book features
- **Cosine Similarity**: Content similarity matrix calculation
- **Feature Engineering**: Combined title, author, and publisher features

### Collaborative Filtering Metrics
- **SVD (Singular Value Decomposition)**: Matrix factorization for user-item ratings
- **Cross-Validation**: 5-fold cross-validation for robust evaluation
- **Hyperparameter Tuning**: Grid search for optimal parameters

### Hybrid Metrics
- **Alpha Optimization**: Weight tuning between content and collaborative approaches
- **Performance Comparison**: Side-by-side evaluation of different alpha values
- **Coverage Analysis**: Percentage of books recommended at least once
- **Diversity Measurement**: Average pairwise dissimilarity between recommendations

## 🔧 Training Framework

### `src/training.py`
Comprehensive training pipeline with:

- **Data Preprocessing**: Automatic cleaning and feature engineering
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Cross-Validation**: Robust model evaluation
- **Model Persistence**: Save/load trained models
- **Fallback Mechanisms**: Sample data generation when needed

### Key Features:
```python
trainer = HybridRecommenderTrainer()
books_df, ratings_df = trainer.train_full_pipeline(
    use_hyperparameter_tuning=True,
    use_cross_validation=True
)
```

## 📈 Evaluation Framework

### `evaluation/metrics.py`
Comprehensive evaluation system with:

- **RMSE Calculation**: Rating prediction accuracy
- **Precision/Recall/F1**: Top-N recommendation quality
- **Coverage Analysis**: Recommendation diversity
- **Diversity Measurement**: Recommendation variety
- **Alpha Optimization**: Best hybrid weight selection

### Key Features:
```python
evaluator = RecommenderEvaluator()
results = evaluator.run_full_evaluation()
```

## 📝 User Satisfaction Survey

### `evaluation/user_survey.py`
Interactive survey system with:

- **Multi-dimensional Assessment**: Relevance, accuracy, diversity
- **Real-time Analytics**: Live satisfaction metrics
- **Data Export**: CSV/JSON export capabilities
- **Trend Analysis**: Historical satisfaction tracking

### Survey Components:
- Relevance assessment (Very/Somewhat/Not relevant)
- Rating accuracy (1-5 scale)
- Diversity evaluation (Very/Somewhat/Not diverse)
- Book preference selection
- Additional feedback collection
- Usage frequency analysis

## 🔄 Automated Retraining

### `retrain.py`
Intelligent retraining system with:

- **Model Age Detection**: Automatic retraining triggers
- **Drift Detection**: Performance degradation monitoring
- **Incremental Training**: Efficient model updates
- **Performance Tracking**: Historical performance trends

### Usage:
```bash
# Check retraining status
python retrain.py --check

# Force retraining
python retrain.py --force

# Incremental retraining
python retrain.py --incremental
```

## 📊 Performance Thresholds

### Recommended Thresholds:
- **F1-Score**: > 0.3 (Good), > 0.5 (Excellent)
- **RMSE**: < 1.0 (Good), < 0.8 (Excellent)
- **Coverage**: > 0.1 (Good), > 0.2 (Excellent)
- **Diversity**: > 0.3 (Good), > 0.5 (Excellent)
- **User Satisfaction**: > 3.0/5 (Good), > 4.0/5 (Excellent)

## 📋 Evaluation Workflow

### 1. Data Preparation
```python
# Load and preprocess data
trainer = HybridRecommenderTrainer()
books_df, ratings_df = trainer.load_and_preprocess_data()
```

### 2. Model Training
```python
# Train with hyperparameter tuning
trainer.train_full_pipeline(
    use_hyperparameter_tuning=True,
    use_cross_validation=True
)
```

### 3. Performance Evaluation
```python
# Comprehensive evaluation
evaluator = RecommenderEvaluator()
results = evaluator.run_full_evaluation()
```

### 4. User Satisfaction Assessment
```python
# Survey and analytics
survey = UserSatisfactionSurvey()
metrics = survey.get_satisfaction_metrics()
```

### 5. Report Generation
```python
# Generate comprehensive report
report = evaluator.generate_evaluation_report(results)
```

## 📈 Monitoring and Maintenance

### Regular Evaluation Schedule:
- **Daily**: User satisfaction monitoring
- **Weekly**: Performance metrics review
- **Monthly**: Full system evaluation
- **Quarterly**: Model retraining

### Automated Alerts:
- Performance degradation detection
- User satisfaction drops
- Model drift identification
- Data quality issues

## 🔍 Troubleshooting

### Common Issues:

1. **Memory Issues with Large Datasets**
   - Use data sampling for evaluation
   - Implement chunked processing
   - Reduce feature dimensionality

2. **Slow Training**
   - Use incremental training
   - Reduce hyperparameter search space
   - Implement early stopping

3. **Low Performance Metrics**
   - Check data quality
   - Increase training data
   - Tune hyperparameters
   - Feature engineering

### Performance Optimization:
```python
# For large datasets, use sampling
evaluator.calculate_precision_recall_f1(
    ratings_df.sample(n=10000),  # Sample for efficiency
    content_sim_matrix,
    svd_model
)
```

## 📚 Additional Resources

### Documentation:
- [Training Documentation](src/README.md)
- [Evaluation Metrics Guide](metrics_guide.md)
- [User Survey Manual](survey_manual.md)

### Examples:
- [Basic Evaluation Example](examples/basic_evaluation.py)
- [Advanced Metrics Example](examples/advanced_metrics.py)
- [Custom Survey Example](examples/custom_survey.py)

## 🤝 Contributing

To contribute to the evaluation framework:

1. Follow the existing code structure
2. Add comprehensive tests for new metrics
3. Update documentation
4. Ensure backward compatibility
5. Submit pull requests with detailed descriptions

## 📄 License

This evaluation framework is part of the Hybrid Book Recommender System and is subject to the same licensing terms as the main project.

---

**Note**: This evaluation framework is designed to be comprehensive and production-ready. It includes all the metrics and evaluation methods required for a robust recommender system assessment.
