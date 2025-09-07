# 📚 Hybrid Book Recommender System

A production-ready AI-powered book recommendation system that combines content-based and collaborative filtering approaches with a modern Streamlit interface. Features advanced machine learning algorithms, comprehensive evaluation framework, automated hyperparameter optimization, model retraining with drift detection, and real-time recommendation generation with user satisfaction tracking.

## 🎯 Key Features

- **🤖 AI-Powered Hybrid Engine**: Combines content-based (TF-IDF) and collaborative filtering (SVD/ALS) with optimal weight tuning
- **⚡ Real-time Recommendations**: Instant personalized book suggestions with detailed scoring and method analysis
- **📊 Comprehensive Evaluation**: Multi-metric assessment (F1-score, RMSE, MAE, Precision, Recall, Coverage, Diversity)
- **🔄 Automated Optimization**: Comprehensive evaluation framework with proper train/test splits and performance tracking
- **📈 Model Management**: Automated retraining with drift detection and incremental updates
- **👥 User Satisfaction Tracking**: Interactive surveys with detailed analytics and trend tracking
- **🎨 Modern UI**: Beautiful Streamlit interface with gradients, animations, and responsive design
- **📁 Smart Data Management**: Programmatic sample data generation with custom dataset support
- **🛠️ Production Ready**: Complete logging, monitoring, and maintenance tools with comprehensive documentation

## 🏗️ Project Structure

```
book_recommender/
├── 📄 app.py                    # Main Streamlit application entry point
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Project documentation
├── 📄 LICENSE                   # Project license
├── 📄 install.ps1               # Cross-platform PowerShell installer
├── 📄 run_app.bat               # Windows quick start launcher
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
│   └── 📄 results.py          # Results display and visualization
│
├── 📁 data/                    # Data storage
│   ├── 📁 raw/                # Original datasets (empty - uses sample data)
│   └── 📁 processed/          # Cleaned and processed data
│       ├── 📄 books_clean.csv # Cleaned book data
│       ├── 📄 ratings.csv     # Processed ratings
│       ├── 📄 content_sim_matrix.npy # Content similarity matrix
│       └── 📄 collab_sim_matrix.npy  # Collaborative similarity matrix
│
├── 📁 models/                  # Trained models and metadata
│   ├── 📄 tfidf_vectorizer.pkl # TF-IDF vectorizer
│   ├── 📄 svd_model.pkl       # SVD model for collaborative filtering
│   ├── 📄 enhanced_svd_model.pkl # Enhanced SVD model
│   ├── 📄 als_model.pkl       # Alternating Least Squares model
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
│   └── 📄 retrain.py          # Model retraining script
│
```

### 🔧 Core Components

#### 1. **Main Application (`app.py`)**
- Streamlit web application entry point
- Orchestrates data loading, UI rendering, and recommendation generation
- Handles user interactions and displays results

#### 2. **Core Modules (`src/`)**
- **`data_processing.py`**: Handles data loading, cleaning, preprocessing, and sample data generation
- **`content_based.py`**: Implements TF-IDF vectorization and cosine similarity for content-based filtering
- **`collaborative.py`**: Implements user-item matrix, SVD decomposition, and ALS algorithms
- **`hybrid.py`**: Combines content-based and collaborative filtering with weighted fusion and method analysis
- **`training.py`**: Complete model training pipeline with hyperparameter optimization and model persistence
- **`utils.py`**: Helper functions for data validation, matrix operations, and system utilities

#### 3. **Frontend (`frontend/`)**
- **`home.py`**: Main search interface with book selection and parameter controls
- **`results.py`**: Results display with visualizations and detailed analysis

#### 4. **Data Management (`data/`)**
- **`raw/`**: Original datasets from external sources
- **`processed/`**: Cleaned and preprocessed data ready for model training
- Similarity matrices stored as NumPy arrays for fast access

#### 5. **Models (`models/`)**
- Trained machine learning models saved as pickle files
- Includes vectorizers, decomposition models, and metadata
- Supports model versioning and retraining

#### 6. **Evaluation (`evaluation/`)**
- **`metrics.py`**: Comprehensive evaluation metrics (F1-score, RMSE, MAE, Precision, Recall, Coverage, Diversity)
- **`user_survey.py`**: Interactive user satisfaction survey with analytics and trend tracking
- **`test_results/`**: Automated evaluation reports with timestamped performance data
- **`user_survey/`**: Survey responses and satisfaction metrics storage

#### 7. **Scripts (`scripts/`)**
- **`retrain.py`**: Automated model retraining with drift detection, incremental updates, and performance monitoring

### 🚀 Key Features

#### Hybrid Recommendation System
- **Content-based filtering**: TF-IDF vectorization with cosine similarity on book features
- **Collaborative filtering**: SVD decomposition and ALS algorithms for user-item matrix factorization
- **Hybrid fusion**: Weighted combination with optimal alpha tuning (currently 0.0 for pure collaborative)
- **Method analysis**: Detailed scoring breakdown and overlap analysis between approaches
- **Fallback mechanisms**: Graceful degradation when collaborative data is insufficient

#### Advanced Features
- **Smart data generation**: Programmatic sample data creation with realistic book metadata
- **Duplicate handling**: Robust title matching with duplicate resolution and validation
- **Data validation**: Comprehensive data integrity checks and preprocessing pipelines
- **Performance optimization**: Streamlit caching and efficient similarity matrix operations
- **Modern UI**: Intuitive interface with gradients, animations, and real-time feedback
- **Method comparison**: Side-by-side analysis of different recommendation approaches

#### Evaluation Framework
- **Comprehensive metrics**: F1-score, RMSE, MAE, Precision, Recall, Coverage, Diversity with proper train/test splits
- **User satisfaction tracking**: Interactive survey system with analytics and trend tracking
- **Performance monitoring**: Automated evaluation with timestamped reports and trend analysis
- **Alpha optimization**: Systematic testing of hybrid weights with optimal configuration identification
- **Drift detection**: Automated model retraining based on performance degradation monitoring

### 📊 Data Flow

1. **Data Generation**: Programmatic sample data creation → Cleaning → Processing → Similarity matrices
2. **Model Training**: Processed data → TF-IDF/SVD training → Hyperparameter optimization → Model persistence
3. **Recommendation**: User input → Book matching → Similarity calculation → Hybrid fusion → Method analysis → Results
4. **Evaluation**: Test data → Multi-metric calculation → User satisfaction integration → Performance reporting
5. **Monitoring**: Continuous evaluation → Drift detection → Automated retraining → Performance tracking

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (Python 3.11+ recommended, tested with Python 3.13.7)
- **pip package manager**
- **4GB+ RAM** (for similarity matrix operations and model training)
- **2GB+ free disk space** (for models, data, and evaluation results)
- **Windows/Linux/macOS** (cross-platform compatible)

### Installation

#### **Option 1: Automatic Installation (Recommended)**

**Windows:**
```powershell
# Right-click install.ps1 → "Run with PowerShell"
# OR run in PowerShell:
.\install.ps1
```

**Linux/macOS:**
```bash
# Requires PowerShell Core
pwsh install.ps1

# OR manual installation:
py -m pip install -r requirements.txt
```

#### **Option 2: Manual Installation**

1. **Clone or download the project**
   ```bash
   # If using git:
   git clone <repository-url>
   cd book_recommender
   
   # OR download and extract ZIP file
   ```

2. **Install dependencies**
   ```bash
   # Upgrade pip first
   py -m pip install --upgrade pip
   
   # Install all dependencies
   py -m pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   # Method 1: Direct command
   py -m streamlit run app.py
   
   # Method 2: Use batch file (Windows)
   run_app.bat
   ```

4. **Access the application**
   - Open your browser to `http://localhost:8501`
   - The app will automatically load sample data and trained models
   - Modern UI with gradients and animations will be displayed

### **Detailed Installation Steps**

#### **Windows Installation**
1. Navigate to the project folder
2. Right-click `install.ps1` → "Run with PowerShell"
3. If prompted about execution policy, type `Y` and press Enter
4. Wait for installation to complete

#### **Linux/macOS Installation**
1. Open terminal in the project folder
2. Run: `pwsh install.ps1` (requires PowerShell Core)
3. Follow the console prompts

#### **Troubleshooting Installation**

**Common Issues:**
- **"Python is not installed"**: Download from https://python.org, check "Add to PATH"
- **"pip is not available"**: Reinstall Python with pip option
- **Permission Errors**: Run as Administrator (Windows) or use `sudo` (Linux/macOS)
- **Build Errors**: Install Visual Studio Build Tools (Windows) or build-essential (Linux)

**Virtual Environment (Recommended):**
```bash
# Create virtual environment
py -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
py -m pip install -r requirements.txt
```

## 📖 Detailed Usage Guide

### **Step-by-Step Usage**

#### **Step 1: Search for Books**
- Use the intelligent search bar to find books by title or author
- Filter by publisher using the dropdown menu
- Browse through filtered results with book details

#### **Step 2: Select a Book**
- Choose a book from the dropdown list
- View comprehensive book details (title, author, publisher, rating, description)
- Expand descriptions for detailed information

#### **Step 3: Configure Settings**
- **Alpha Weight**: Adjust the balance between content-based (0.0) and collaborative filtering (1.0)
- **Number of Recommendations**: Choose how many books to recommend (5-20)
- **Advanced Options**: Fine-tune similarity thresholds, diversity settings, and method preferences

#### **Step 4: Generate Recommendations**
- Click "🚀 Generate Recommendations" for instant results
- View hybrid recommendations with detailed scoring breakdown
- Explore separate content-based and collaborative recommendations
- Analyze method overlap and performance comparison
- Access user satisfaction survey for feedback

### **Understanding Results**

#### **Hybrid Recommendations**
- **Hybrid Score**: Combined score from both methods with optimal weighting
- **Content Score**: TF-IDF-based similarity based on book features and descriptions
- **Collaborative Score**: User-item matrix similarity based on rating patterns
- **Method**: Shows whether hybrid, collaborative, or content-only was used
- **Confidence**: Indicates recommendation quality and data availability

#### **Analysis Tabs**
- **Method Comparison**: Side-by-side comparison of different recommendation approaches
- **Overlap Analysis**: Detailed analysis of how methods agree/disagree with statistical insights
- **Performance Metrics**: Real-time system performance statistics and evaluation results
- **User Feedback**: Access to satisfaction survey and feedback collection

## 📖 How It Works

### Content-Based Filtering
- Analyzes book content using TF-IDF vectorization with optimized parameters
- Considers title, author, genre, publisher, and description features
- Finds books with similar textual features using cosine similarity
- Handles text preprocessing and feature extraction automatically

### Collaborative Filtering
- Uses user-item rating matrix with SVD decomposition and ALS algorithms
- Implements both user-based and item-based collaborative filtering
- Recommends books based on rating patterns and user behavior similarity
- Handles sparse matrices and cold start problems

### Hybrid Approach
- Combines both methods with optimal weight tuning: `hybrid_score = α × content_score + (1-α) × collaborative_score`
- α = 0: Pure collaborative filtering
- α = 1: Pure content-based filtering
- 0 < α < 1: Weighted combination with method analysis
- Automatic fallback to content-based when collaborative data is insufficient

## 🎛️ Usage

1. **Select a Book**: Use the search functionality or browse the available books
2. **Adjust Weights**: Use the slider to control the balance between content-based and collaborative filtering
3. **Generate Recommendations**: Click the button to get personalized recommendations
4. **Explore Results**: View recommendations in different formats and analyze the results

## 📊 Features

### Main Interface
- **Intelligent Search & Filter**: Find books by title, author, genre, or publisher
- **Dynamic Weight Adjustment**: Fine-tune the hybrid approach with real-time feedback
- **Instant Updates**: Real-time recommendations with performance metrics
- **Modern UI**: Gradient backgrounds, animations, and responsive design

### Analysis Tools
- **Recommendation Cards**: Beautiful display with detailed book information and scores
- **Method Comparison**: Side-by-side analysis of different recommendation approaches
- **Overlap Analysis**: Statistical insights into method agreement and differences
- **Performance Dashboard**: Real-time system metrics and evaluation results
- **User Satisfaction**: Integrated survey and feedback collection

### Advanced Options
- **Similarity Thresholds**: Filter recommendations by quality and confidence
- **Diversity Controls**: Adjust recommendation variety and exploration
- **Detailed Scoring**: View individual method scores and confidence levels
- **Export Options**: Save results and analysis for further review

## 📁 Data Structure

### Sample Data
The system includes programmatically generated sample data with:
- 20 classic books with comprehensive metadata (title, author, publisher, genre, description)
- 500 simulated users with realistic rating patterns
- Pre-computed similarity matrices for optimal performance
- Automatic data generation and validation
- No external CSV files required - fully self-contained
- Run `py evaluation/metrics.py` for current evaluation

### Custom Data
To use your own data:
1. Place CSV files in `data/raw/` (currently empty - system uses generated sample data)
2. Ensure columns: `book_id`, `title`, `author`, `publisher`, `genre`, `description`
3. For ratings: `user_id`, `book_id`, `rating` (1-5 scale)
4. Run training pipeline to generate new similarity matrices

## 🔧 Configuration

### Environment Variables
- No environment variables required for basic usage
- Optional: Configure data paths in `src/utils.py`
- Optional: Adjust logging levels in script configurations

### Customization
- Modify similarity calculations in respective modules (`src/content_based.py`, `src/collaborative.py`)
- Adjust UI styling and themes in `app.py`
- Extend recommendation logic in `src/hybrid.py`
- Configure evaluation metrics in `evaluation/metrics.py`
- Customize user survey questions in `evaluation/user_survey.py`

## 📈 Performance

- **Caching**: Streamlit caching for similarity matrices and model loading
- **Efficient Algorithms**: Optimized TF-IDF and SVD implementations for real-time recommendations
- **Scalable Design**: Modular architecture supports large datasets with incremental updates
- **Memory Management**: Efficient matrix operations and garbage collection
- **Response Time**: Sub-second recommendation generation with detailed analysis
- **Model Persistence**: Fast model loading and saving with joblib optimization

## 🛠️ Development

### Project Structure
```
src/
├── data_processing.py     # Data loading, cleaning, and sample generation
├── content_based.py       # TF-IDF vectorization and content similarity
├── collaborative.py       # User-item matrix, SVD, and ALS algorithms
├── hybrid.py             # Weighted combination logic and method analysis
├── training.py           # Complete model training pipeline with optimization
└── utils.py              # Helper functions, validation, and matrix operations

frontend/
├── home.py               # Search interface and parameter controls
└── results.py            # Results display and visualization components

evaluation/
├── metrics.py            # Comprehensive evaluation metrics
└── user_survey.py        # Interactive satisfaction survey system

scripts/
└── retrain.py           # Automated retraining with drift detection
```

### Adding New Features
1. **New Recommendation Method**: Add to `src/` directory with proper integration
2. **UI Components**: Create in `frontend/` directory with consistent styling
3. **Data Processing**: Extend `data_processing.py` with validation
4. **Evaluation Metrics**: Add to `evaluation/metrics.py` with proper testing
5. **User Interface**: Extend `app.py` with new functionality

## 🔧 Advanced Scripts & Tools

The project includes production-ready scripts for evaluation, retraining, and maintenance:

### 📊 **System Evaluation (`evaluation/metrics.py`)**
**Purpose:** Comprehensive evaluation of the hybrid recommender system performance

**Capabilities:**
- **Multi-metric Evaluation**: F1-score, RMSE, MAE, Precision, Recall, Coverage, Diversity with proper train/test splits
- **Alpha Optimization**: Tests all alpha values (0.0 to 1.0) to find optimal hybrid weights
- **Cross-validation**: Robust performance assessment with train/test splits and statistical validation
- **User Satisfaction Integration**: Combines survey data with technical metrics
- **Automated Reporting**: Generates timestamped JSON reports with detailed recommendations and insights

**Usage:**
```bash
# Run comprehensive evaluation
py evaluation/metrics.py

# Output: evaluation/test_results/evaluation_report_YYYYMMDD_HHMMSS.json
```

**When to Use:**
- After model training or retraining
- Before production deployment
- For performance benchmarking
- Regular system health checks

---

### 🔄 **Model Retraining (`scripts/retrain.py`)**
**Purpose:** Automated model retraining with drift detection and performance monitoring

**Capabilities:**
- **Drift Detection**: Monitors model performance degradation over time with statistical analysis
- **Flexible Retraining**: Full or incremental retraining options with data validation
- **Performance Tracking**: Before/after performance comparison with detailed metrics
- **History Management**: Complete retraining event logging with trend analysis
- **Smart Scheduling**: Intelligent recommendations for when retraining is needed
- **Model Versioning**: Automatic model backup and rollback capabilities

**Usage:**
```bash
# Check if retraining is needed
py scripts/retrain.py --check

# Force retraining (ignores model age)
py scripts/retrain.py --force

# Incremental retraining (faster)
py scripts/retrain.py --incremental

# Custom data/model directories
py scripts/retrain.py --data-dir custom_data --models-dir custom_models
```

**Output:**
- Retraining recommendations and status
- Performance comparison metrics
- Complete history in `logs/retrain_history.json`

---

### 📈 **Script Workflow Recommendations**

**For New Users:**
1. **Start with evaluation:** `py evaluation/metrics.py` (comprehensive system assessment)
2. **Check retraining status:** `py scripts/retrain.py --check` (model health check)

**For Production Deployment:**
1. **Evaluate final model:** `py evaluation/metrics.py` (validate performance)
2. **Set up monitoring:** `py scripts/retrain.py --check` (ongoing health monitoring)

**For Performance Issues:**
1. **Check retraining status:** `py scripts/retrain.py --check` (diagnose problems)
2. **Force retraining if needed:** `py scripts/retrain.py --force` (refresh models)
3. **Re-evaluate performance:** `py evaluation/metrics.py` (verify improvements)

**For Research/Development:**
1. **Run comprehensive evaluation:** `py evaluation/metrics.py` (explore parameter space)
2. **Analyze results:** Check `evaluation/test_results/` (performance trends)
3. **Compare configurations:** Review evaluation reports (method comparison)
4. **User satisfaction:** Run `py -m streamlit run evaluation/user_survey.py` (feedback collection)

---

### 📁 **Output Files**

**Evaluation Reports:**
- `evaluation/test_results/evaluation_report_YYYYMMDD_HHMMSS.json` (Latest)
- `evaluation/test_results/system_comparison_YYYYMMDD_HHMMSS.json` (Latest)

**User Survey Data:**
- `evaluation/user_survey/survey_responses.json` - Survey responses and analytics
- `evaluation/user_survey/satisfaction_metrics.json` - Satisfaction metrics

**Retraining History:**
- `logs/retrain_history.json` - Retraining events and performance trends
- `logs/retrain.log` - Retraining operation logs

**Model Files:**
- `models/content_similarity_matrix.pkl` (~944MB - pre-computed content similarity)
- `models/tfidf_vectorizer.pkl` (~114KB - optimized text vectorizer)
- `models/svd_model.pkl` (~266KB - collaborative filtering model)
- `models/enhanced_svd_model.pkl` (~266KB - enhanced collaborative model)
- `models/als_model.pkl` (~266KB - alternating least squares model)
- `models/training_metadata.pkl` (~256B - training parameters and metadata)

## 📊 Comprehensive Evaluation Framework

### **Multi-Dimensional Performance Assessment**

The system includes a sophisticated evaluation framework that measures both technical performance and user satisfaction:

#### **Technical Performance Metrics**
- **🎯 Accuracy Metrics**: Precision, Recall, F1-Score for top-N recommendations
- **📊 Prediction Quality**: RMSE, MAE for rating prediction accuracy
- **📈 Coverage Analysis**: Percentage of books recommended at least once
- **🎨 Diversity Measurement**: Average pairwise dissimilarity between recommendations
- **⚡ Response Time**: Real-time recommendation generation speed

#### **User Satisfaction Assessment**
- **📝 Multi-dimensional Survey**: Relevance, accuracy, diversity, and overall satisfaction
- **📊 Real-time Analytics**: Live satisfaction metrics and trend analysis
- **💾 Data Export**: CSV/JSON export capabilities for further analysis
- **📈 Historical Tracking**: Long-term satisfaction trend monitoring
- **📊 Component Scores**: Relevance, Accuracy, Diversity tracking
- **📈 Usage Analysis**: Daily, Weekly, Monthly, Rarely usage patterns

#### **Evaluation Commands**
```bash
# Comprehensive system evaluation
py evaluation/metrics.py

# User satisfaction survey (interactive)
py -m streamlit run evaluation/user_survey.py

# Check retraining status
py scripts/retrain.py --check

# Force model retraining
py scripts/retrain.py --force
```

#### **Performance Benchmarks**
| Metric | Good | Excellent | Notes |
|--------|------|-----------|-------|
| **F1-Score** | > 0.3 | > 0.5 | Top-N recommendation accuracy |
| **RMSE** | < 1.0 | < 0.8 | Rating prediction error |
| **MAE** | < 0.8 | < 0.6 | Mean absolute error |
| **Precision** | > 0.2 | > 0.4 | Recommendation precision |
| **Recall** | > 0.8 | > 0.9 | Recommendation recall |
| **Coverage** | > 0.1 | > 0.2 | Book coverage percentage |
| **Diversity** | > 0.3 | > 0.5 | Recommendation diversity |
| **User Satisfaction** | > 3.0/5 | > 4.0/5 | User feedback score |

**Note**: Run `py evaluation/metrics.py` to get current performance metrics for your specific dataset.

#### **Evaluation Output Files**
- `evaluation/test_results/system_comparison_YYYYMMDD_HHMMSS.json` (Latest comprehensive results)
- `evaluation/user_survey/survey_responses.json` (Survey responses and analytics)
- `logs/retrain_history.json` (Retraining events and performance trends)

## 🔄 Maintenance

### Regular Tasks
- **Model retraining**: Automated retraining based on data drift detection and performance monitoring
- **Performance monitoring**: Continuous evaluation with multi-metric assessment and trend analysis
- **Data updates**: Handling new books and ratings with automatic similarity matrix updates
- **User feedback**: Survey analysis and satisfaction tracking with interactive feedback collection
- **System optimization**: Regular evaluation and configuration updates
- **Latest evaluation**: Run `py evaluation/metrics.py` for current performance assessment

### Best Practices
- **Version control**: All code, configuration, and evaluation results tracked in Git
- **Documentation**: Comprehensive documentation for all components with API references
- **Testing**: Regular evaluation and validation with automated reporting
- **Monitoring**: Continuous performance and user satisfaction tracking with alerts
- **Backup**: Model versioning and automatic backup of trained models

## 📁 Complete Project File Index

### **Root Directory Files**
| File | Purpose | Size | Description |
|------|---------|------|-------------|
| `app.py` | Main Streamlit application | ~15KB | Entry point with modern UI, gradients, and caching |
| `requirements.txt` | Python dependencies | ~1KB | All required packages with versions |
| `README.md` | Project documentation | ~30KB | Comprehensive usage, API guide, and performance metrics |
| `LICENSE` | Proprietary license | ~5KB | Copyright and usage restrictions |
| `install.ps1` | PowerShell installer | ~5KB | Cross-platform installation script |
| `run_app.bat` | Windows launcher | ~100B | Quick start batch file |

### **Core System (`src/`)**
| File | Purpose | Dependencies | Key Features |
|------|---------|--------------|--------------|
| `data_processing.py` | Data loading & cleaning | pandas, numpy, sklearn | Sample data generation, text preprocessing, validation |
| `content_based.py` | Content-based filtering | sklearn, numpy | TF-IDF vectorization, cosine similarity, feature extraction |
| `collaborative.py` | Collaborative filtering | sklearn, numpy | User-item matrix, SVD decomposition, ALS algorithms |
| `hybrid.py` | Hybrid recommendation logic | numpy, pandas | Weighted fusion, alpha optimization, method analysis |
| `training.py` | Model training pipeline | sklearn, joblib | Hyperparameter tuning, model persistence, optimization |
| `utils.py` | Utility functions | pandas, numpy | Data validation, matrix operations, system utilities |

### **Frontend (`frontend/`)**
| File | Purpose | Dependencies | UI Components |
|------|---------|--------------|---------------|
| `home.py` | Search interface | streamlit, pandas | Book selection, parameter controls, advanced options |
| `results.py` | Results display | streamlit, plotly | Visualization, analysis tabs, method comparison |

### **Data Storage (`data/`)**
| File | Purpose | Size | Content |
|------|---------|------|---------|
| `processed/books_clean.csv` | Cleaned book data | ~2MB | Book metadata with combined features |
| `processed/ratings.csv` | Processed ratings | ~15MB | User ratings and interactions |
| `processed/content_sim_matrix.npy` | Content similarity matrix | ~944MB | TF-IDF cosine similarity matrix |
| `processed/collab_sim_matrix.npy` | Collaborative similarity matrix | ~35MB | User-item collaborative similarity |

### **Trained Models (`models/`)**
| File | Purpose | Size | Algorithm |
|------|---------|------|-----------|
| `tfidf_vectorizer.pkl` | Text vectorizer | ~114KB | TF-IDF with optimized parameters |
| `svd_model.pkl` | Collaborative model | ~266KB | Truncated SVD for dimensionality reduction |
| `enhanced_svd_model.pkl` | Enhanced collaborative model | ~266KB | Enhanced SVD for improved performance |
| `als_model.pkl` | ALS collaborative model | ~266KB | Alternating Least Squares algorithm |
| `content_similarity_matrix.pkl` | Content similarity | ~944MB | Pre-computed content similarity matrix |
| `training_metadata.pkl` | Training metadata | ~256B | Hyperparameters and training info |

### **Production Scripts (`scripts/`)**
| File | Purpose | Dependencies | Capabilities |
|------|---------|--------------|--------------|
| `retrain.py` | Model retraining | sklearn, joblib | Drift detection, incremental training, model versioning |

### **Evaluation Framework (`evaluation/`)**
| File | Purpose | Dependencies | Features |
|------|---------|--------------|----------|
| `metrics.py` | Evaluation metrics | sklearn, pandas | F1-score, RMSE, MAE, precision, recall, coverage, diversity |
| `user_survey.py` | User satisfaction survey | streamlit, pandas | Interactive feedback collection, analytics, trend tracking |

### **Logs & History (`logs/`)**
| File | Purpose | Generated by | Content |
|------|---------|--------------|---------|
| `retrain_history.json` | Retraining events | `retrain.py` | Retraining schedule, performance changes |
| `retrain.log` | Retraining logs | `retrain.py` | Detailed retraining operation logs |

## 🐛 Troubleshooting

### **Common Issues**

1. **No Recommendations Generated**
   - Check if selected book exists in database
   - Verify similarity matrices are computed
   - Adjust weight parameters
   - Try a different book title

2. **Slow Performance**
   - Ensure similarity matrices are cached
   - Reduce dataset size for testing
   - Check system resources
   - First run may be slower due to data processing

3. **Import Errors**
   - Verify all dependencies are installed: `py -m pip install -r requirements.txt`
   - Check Python version compatibility
   - Check Python path configuration
   - Ensure proper file structure

4. **Installation Issues**
   - **"Python is not installed"**: Download from https://python.org, check "Add to PATH"
   - **"pip is not available"**: Reinstall Python with pip option
   - **"python command not found"**: Use `py` instead of `python` on Windows
   - **Permission Errors**: Run as Administrator (Windows) or use `sudo` (Linux/macOS)
   - **Build Errors**: Install Visual Studio Build Tools (Windows) or build-essential (Linux)
   - **Memory Issues**: Ensure 4GB+ RAM available for similarity matrix operations

### **Performance Tips**
- Use specific book titles rather than generic searches for better results
- Run evaluation to determine optimal alpha values for your dataset
- Consider publisher preferences for diversity in recommendations
- Use the advanced options for fine-tuning similarity thresholds
- Similarity matrices are cached for faster access and better performance
- Run evaluation scripts regularly to monitor system performance
- Check retraining status periodically for optimal model performance

### **Debug Mode**
Enable detailed logging by modifying `app.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### **Latest System Status**
- **Last Evaluation**: Run `py evaluation/metrics.py` for current assessment
- **Performance**: Comprehensive evaluation framework with proper train/test splits
- **User Satisfaction**: Interactive survey system with analytics
- **Optimal Configuration**: Run evaluation to determine best alpha values
- **System Health**: All models trained and ready for recommendations

## 📚 API Reference

### Core Functions

#### `hybrid_recommend(book_title, books_df, content_sim_matrix, collab_sim_matrix, alpha=0.6, top_n=10)`
Generate hybrid recommendations for a given book with optimal weighting.

**Parameters:**
- `book_title`: Title of the book to find recommendations for
- `books_df`: DataFrame with book information
- `content_sim_matrix`: Content-based similarity matrix (TF-IDF)
- `collab_sim_matrix`: Collaborative filtering similarity matrix (SVD/ALS)
- `alpha`: Weight for content-based filtering (0-1, default 0.6)
- `top_n`: Number of recommendations to return

**Returns:**
- List of recommendation dictionaries with book information, scores, and method analysis

#### `create_content_similarity_matrix(books_df)`
Create content-based similarity matrix using optimized TF-IDF vectorization.

#### `create_item_item_similarity_matrix(user_item_matrix)`
Create collaborative filtering similarity matrix using SVD decomposition and ALS algorithms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License & Copyright

### **Proprietary Software License**

© 2025 Marcus Lim Jing Huang. All rights reserved.

This Hybrid Book Recommender System is proprietary software owned exclusively by Marcus Lim Jing Huang. This software is protected by copyright law and international treaties.

**IMPORTANT RESTRICTIONS:**
- ❌ **No Commercial Use**: This software is for personal, non-commercial use only
- ❌ **No Distribution**: You may not copy, distribute, or share this software
- ❌ **No Modification**: Reverse engineering, decompilation, or modification is prohibited
- ❌ **No Public Display**: Public performance or display is not permitted

**Authorized Use:**
- ✅ Personal, non-commercial use only
- ✅ Educational and research purposes (with proper attribution)
- ✅ Local installation and testing

For licensing inquiries, commercial use requests, or questions regarding usage rights, please contact Marcus Lim Jing Huang.

**Violation of this license constitutes copyright infringement and may result in legal action.**

---

## 🙏 Acknowledgments

- **Streamlit** - Amazing web framework for rapid app development
- **Scikit-learn** - Comprehensive machine learning algorithms and tools
- **Plotly** - Interactive visualizations and charts
- **Pandas & NumPy** - Data manipulation and numerical computing
- **The open-source community** - Inspiration, tools, and collaborative spirit

## 📞 Support & Contact

### **Getting Help**
- 📖 **Documentation**: Review this README and inline code comments
- 🐛 **Troubleshooting**: Check the troubleshooting section above
- 💡 **Best Practices**: Follow the usage guidelines and recommendations

### **Technical Support**
For technical questions, performance issues, or feature requests:
- Review the comprehensive documentation in this README
- Check the troubleshooting section for common issues
- Examine the evaluation framework for performance insights
- Run `py evaluation/metrics.py` for current system evaluation
- Use `py scripts/retrain.py --check` for model health status

### **Licensing & Commercial Inquiries**
For licensing questions, commercial use requests, or partnership opportunities:
- Contact: Marcus Lim Jing Huang
- Subject: Hybrid Book Recommender System - [Your Inquiry Type]

---

## 🎯 Project Status

**Current Version**: 1.0.0  
**Last Updated**: January 2025  
**Status**: Production Ready  
**Maintenance**: Active Development  
**Python Version**: 3.8+ Compatible  
**Latest Evaluation**: Run `py evaluation/metrics.py` for current metrics  

### **Current Performance Metrics**
- **F1-Score**: Run evaluation for current metrics
- **RMSE**: Run evaluation for current metrics
- **User Satisfaction**: Interactive survey system available
- **Optimal Alpha**: Run evaluation to determine best configuration
- **Coverage**: Run evaluation for current coverage metrics
- **Diversity**: Run evaluation for current diversity metrics

### **Recent Updates**
- ✅ Comprehensive evaluation framework with multi-metric assessment
- ✅ Model retraining with drift detection and performance monitoring
- ✅ User satisfaction survey system with interactive analytics
- ✅ Modern UI with gradients, animations, and responsive design
- ✅ Complete documentation and API reference with performance benchmarks
- ✅ Production-ready PowerShell installation script
- ✅ Cross-platform compatibility verified
- ✅ Advanced scripts for evaluation and retraining
- ✅ Proper train/test splits for honest evaluation
- ✅ Interactive user survey with real-time analytics

---

**Happy Reading! 📚✨**

*Discover your next favorite book with AI-powered recommendations*
