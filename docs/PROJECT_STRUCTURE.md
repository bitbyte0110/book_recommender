# Hybrid Book Recommender System - Project Structure

## 📁 Directory Organization

```
book_recommender/
├── 📄 app.py                    # Main Streamlit application entry point
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Project overview and setup instructions
├── 📄 LICENSE                   # Project license
├── 📄 .gitignore               # Git ignore rules
│
├── 📄 install.bat               # Windows auto-installer (detects PowerShell)
├── 📄 install_dependencies.bat  # Windows batch installer
├── 📄 install_dependencies.ps1  # Windows PowerShell installer
├── 📄 install_dependencies.sh   # Linux/macOS bash installer
│
├── 📁 src/                     # Core recommendation system modules
│   ├── 📄 data_processing.py   # Data loading and preprocessing
│   ├── 📄 content_based.py     # Content-based filtering algorithms
│   ├── 📄 collaborative.py     # Collaborative filtering algorithms
│   ├── 📄 hybrid.py           # Hybrid recommendation logic
│   ├── 📄 training.py         # Model training pipeline
│   └── 📄 utils.py            # Utility functions and helpers
│
├── 📁 frontend/                # Streamlit UI components
│   ├── 📄 home.py             # Main search and selection interface
│   ├── 📄 results.py          # Results display and visualization
│   └── 📁 assets/             # Static assets (CSS, images, etc.)
│
├── 📁 data/                    # Data storage
│   ├── 📁 raw/                # Original datasets
│   │   ├── 📄 books.csv       # Book metadata
│   │   └── 📄 Books_rating.csv # User ratings
│   └── 📁 processed/          # Cleaned and processed data
│       ├── 📄 books_clean.csv # Cleaned book data
│       ├── 📄 ratings.csv     # Processed ratings
│       ├── 📄 content_sim_matrix.npy # Content similarity matrix
│       └── 📄 collab_sim_matrix.npy  # Collaborative similarity matrix
│
├── 📁 models/                  # Trained models and metadata
│   ├── 📄 tfidf_vectorizer.pkl # TF-IDF vectorizer
│   ├── 📄 svd_model.pkl       # SVD model for collaborative filtering
│   ├── 📄 nmf_model.pkl       # NMF model (alternative)
│   ├── 📄 user_item_matrix.pkl # User-item rating matrix
│   ├── 📄 content_similarity_matrix.pkl # Content similarity matrix
│   └── 📄 training_metadata.pkl # Training metadata and parameters
│
├── 📁 evaluation/              # Evaluation framework
│   ├── 📄 metrics.py          # Evaluation metrics implementation
│   ├── 📄 user_survey.py      # User satisfaction survey
│   ├── 📁 test_results/       # Evaluation results storage
│   └── 📁 user_survey/        # Survey responses
│
├── 📁 scripts/                 # Utility scripts
│   ├── 📄 evaluate_system.py  # System evaluation script
│   └── 📄 retrain.py          # Model retraining script
│
└── 📁 docs/                    # Documentation
    ├── 📄 PROJECT_STRUCTURE.md # This file
    ├── 📄 QUICK_START.md       # Quick start guide
    ├── 📄 INSTALLATION.md      # Detailed installation guide
    └── 📄 evaluation_guide.md  # Evaluation framework guide
```

## 🔧 Core Components

### 1. **Main Application (`app.py`)**
- Streamlit web application entry point
- Orchestrates data loading, UI rendering, and recommendation generation
- Handles user interactions and displays results

### 2. **Core Modules (`src/`)**
- **`data_processing.py`**: Handles data loading, cleaning, and preprocessing
- **`content_based.py`**: Implements TF-IDF and cosine similarity for content-based filtering
- **`collaborative.py`**: Implements user-item and item-item collaborative filtering
- **`hybrid.py`**: Combines content-based and collaborative filtering with weighted fusion
- **`training.py`**: Model training pipeline with hyperparameter optimization
- **`utils.py`**: Helper functions for data validation, formatting, and system utilities

### 3. **Frontend (`frontend/`)**
- **`home.py`**: Main search interface with book selection and parameter controls
- **`results.py`**: Results display with visualizations and detailed analysis
- **`assets/`**: Static files for styling and images

### 4. **Data Management (`data/`)**
- **`raw/`**: Original datasets from external sources
- **`processed/`**: Cleaned and preprocessed data ready for model training
- Similarity matrices stored as NumPy arrays for fast access

### 5. **Models (`models/`)**
- Trained machine learning models saved as pickle files
- Includes vectorizers, decomposition models, and metadata
- Supports model versioning and retraining

### 6. **Evaluation (`evaluation/`)**
- **`metrics.py`**: Implementation of evaluation metrics (RMSE, Precision, Recall, etc.)
- **`user_survey.py`**: User satisfaction survey and analytics
- Results storage for performance tracking

### 7. **Scripts (`scripts/`)**
- **`evaluate_system.py`**: Comprehensive system evaluation pipeline
- **`retrain.py`**: Automated model retraining with drift detection

## 🚀 Key Features

### Hybrid Recommendation System
- **Content-based filtering**: Uses TF-IDF and cosine similarity
- **Collaborative filtering**: User-item and item-item approaches
- **Hybrid fusion**: Weighted combination with adjustable alpha parameter
- **Fallback mechanisms**: Graceful degradation when data is insufficient

### Advanced Features
- **Duplicate handling**: Robust title matching with duplicate resolution
- **Data validation**: Comprehensive data integrity checks
- **Performance optimization**: Caching and efficient similarity matrix operations
- **User interface**: Intuitive Streamlit interface with real-time feedback

### Evaluation Framework
- **Multiple metrics**: RMSE, Precision, Recall, F1-score, Coverage, Diversity
- **User surveys**: Satisfaction assessment and feedback collection
- **Performance tracking**: Historical performance monitoring
- **A/B testing**: Alpha optimization for hybrid weights

## 📊 Data Flow

1. **Data Loading**: Raw data → Cleaning → Processing → Similarity matrices
2. **Model Training**: Processed data → Feature extraction → Model training → Model saving
3. **Recommendation**: User input → Book matching → Similarity calculation → Hybrid fusion → Results
4. **Evaluation**: Test data → Metric calculation → Performance analysis → Reporting

## 🔄 Maintenance

### Regular Tasks
- **Model retraining**: Automated retraining based on data drift detection
- **Performance monitoring**: Continuous evaluation and metric tracking
- **Data updates**: Handling new books and ratings
- **User feedback**: Survey analysis and system improvements

### Best Practices
- **Version control**: All code and configuration tracked in Git
- **Documentation**: Comprehensive documentation for all components
- **Testing**: Regular evaluation and validation of system performance
- **Monitoring**: Continuous performance and user satisfaction tracking
