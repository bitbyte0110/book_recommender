# 📚 Hybrid Book Recommender System

A sophisticated AI-powered book recommendation system that combines content-based and collaborative filtering approaches with an interactive Streamlit interface. Features advanced machine learning algorithms, comprehensive evaluation metrics, automated optimization, and real-time recommendation generation.

## 🎯 Key Features

- **🤖 AI-Powered Hybrid Engine**: Combines content-based and collaborative filtering with adjustable weights
- **⚡ Real-time Recommendations**: Instant personalized book suggestions with detailed scoring
- **📊 Advanced Analytics**: Comprehensive evaluation metrics, performance tracking, and method comparison
- **🔄 Automated Optimization**: Hyperparameter tuning and model retraining with drift detection
- **📈 Performance Monitoring**: Continuous evaluation with user satisfaction surveys
- **🎨 Modern UI**: Beautiful Streamlit interface with responsive design and animations
- **📁 Flexible Data Management**: Supports both sample data and custom datasets
- **🛠️ Production Ready**: Complete evaluation framework, logging, and maintenance tools

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
│   ├── 📄 optimize_models.py  # Model optimization script
│   └── 📄 retrain.py          # Model retraining script
│
├── 📁 logs/                    # Log files and history
│   ├── 📄 optimization_history.json # Optimization history
│   ├── 📄 optimization.log     # Optimization logs
│   ├── 📄 retrain_history.json # Retraining history
│   └── 📄 retrain.log          # Retraining logs
│
```

### 🔧 Core Components

#### 1. **Main Application (`app.py`)**
- Streamlit web application entry point
- Orchestrates data loading, UI rendering, and recommendation generation
- Handles user interactions and displays results

#### 2. **Core Modules (`src/`)**
- **`data_processing.py`**: Handles data loading, cleaning, and preprocessing
- **`content_based.py`**: Implements TF-IDF and cosine similarity for content-based filtering
- **`collaborative.py`**: Implements user-item and item-item collaborative filtering
- **`hybrid.py`**: Combines content-based and collaborative filtering with weighted fusion
- **`training.py`**: Model training pipeline with hyperparameter optimization
- **`utils.py`**: Helper functions for data validation, formatting, and system utilities

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
- **`metrics.py`**: Implementation of evaluation metrics (RMSE, Precision, Recall, etc.)
- **`user_survey.py`**: User satisfaction survey and analytics
- Results storage for performance tracking

#### 7. **Scripts (`scripts/`)**
- **`evaluate_system.py`**: Comprehensive system evaluation pipeline
- **`optimize_models.py`**: Automated hyperparameter optimization
- **`retrain.py`**: Automated model retraining with drift detection

### 🚀 Key Features

#### Hybrid Recommendation System
- **Content-based filtering**: Uses TF-IDF and cosine similarity
- **Collaborative filtering**: User-item and item-item approaches
- **Hybrid fusion**: Weighted combination with adjustable alpha parameter
- **Fallback mechanisms**: Graceful degradation when data is insufficient

#### Advanced Features
- **Duplicate handling**: Robust title matching with duplicate resolution
- **Data validation**: Comprehensive data integrity checks
- **Performance optimization**: Caching and efficient similarity matrix operations
- **User interface**: Intuitive Streamlit interface with real-time feedback

#### Evaluation Framework
- **Multiple metrics**: RMSE, Precision, Recall, F1-score, Coverage, Diversity
- **User surveys**: Satisfaction assessment and feedback collection
- **Performance tracking**: Historical performance monitoring
- **A/B testing**: Alpha optimization for hybrid weights

### 📊 Data Flow

1. **Data Loading**: Raw data → Cleaning → Processing → Similarity matrices
2. **Model Training**: Processed data → Feature extraction → Model training → Model saving
3. **Recommendation**: User input → Book matching → Similarity calculation → Hybrid fusion → Results
4. **Evaluation**: Test data → Metric calculation → Performance analysis → Reporting

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (Python 3.11+ recommended)
- **pip package manager**
- **4GB+ RAM** (for similarity matrix operations)
- **2GB+ free disk space**

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
   - The app will automatically load sample data if no custom data is provided

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
- Use the search bar to find books by title or author
- Filter by publisher using the dropdown
- Browse through the filtered results

#### **Step 2: Select a Book**
- Choose a book from the dropdown list
- View book details (title, author, publisher, rating)
- Expand the description if available

#### **Step 3: Configure Settings**
- **Alpha Weight**: Adjust the balance between content-based (0.0) and collaborative filtering (1.0)
- **Number of Recommendations**: Choose how many books to recommend (5-20)
- **Advanced Options**: Fine-tune similarity thresholds and diversity settings

#### **Step 4: Generate Recommendations**
- Click "🚀 Generate Recommendations"
- View hybrid recommendations with detailed scores
- Explore separate content-based and collaborative recommendations
- Analyze overlap between different methods

### **Understanding Results**

#### **Hybrid Recommendations**
- **Hybrid Score**: Combined score from both methods
- **Content Score**: Similarity based on book content
- **Collaborative Score**: Similarity based on user behavior
- **Method**: Shows whether hybrid or content-only was used

#### **Analysis Tabs**
- **Method Comparison**: Compare different recommendation approaches
- **Overlap Analysis**: See how methods agree/disagree
- **Performance Metrics**: View system performance statistics

## 📖 How It Works

### Content-Based Filtering
- Analyzes book content using TF-IDF vectorization
- Considers title, author, genre, and description
- Finds books with similar textual features

### Collaborative Filtering
- Uses user-item rating matrix
- Implements item-item similarity using cosine similarity
- Recommends books based on what similar users liked

### Hybrid Approach
- Combines both methods with adjustable weights: `hybrid_score = α × content_score + (1-α) × collaborative_score`
- α = 0: Pure collaborative filtering
- α = 1: Pure content-based filtering
- 0 < α < 1: Weighted combination

## 🎛️ Usage

1. **Select a Book**: Use the search functionality or browse the available books
2. **Adjust Weights**: Use the slider to control the balance between content-based and collaborative filtering
3. **Generate Recommendations**: Click the button to get personalized recommendations
4. **Explore Results**: View recommendations in different formats and analyze the results

## 📊 Features

### Main Interface
- **Search & Filter**: Find books by title, author, or genre
- **Weight Adjustment**: Fine-tune the hybrid approach
- **Real-time Updates**: Instant feedback and recommendations

### Analysis Tools
- **Recommendation Cards**: Beautiful display of recommended books
- **Genre Distribution**: Visual analysis of recommendation diversity
- **Method Comparison**: Compare different recommendation approaches
- **Overlap Analysis**: Understand how methods differ

### Advanced Options
- **Similarity Thresholds**: Filter recommendations by quality
- **Diversity Controls**: Adjust recommendation variety
- **Detailed Scoring**: View individual method scores

## 📁 Data Structure

### Sample Data
The system includes sample book data with:
- 10 classic books with metadata
- 100 simulated users with ratings
- Pre-computed similarity matrices

### Custom Data
To use your own data:
1. Place CSV files in `data/raw/`
2. Ensure columns: `book_id`, `title`, `author`, `genre`, `description`
3. For ratings: `user_id`, `book_id`, `rating`

## 🔧 Configuration

### Environment Variables
- No environment variables required for basic usage
- Optional: Configure data paths in `src/utils.py`

### Customization
- Modify similarity calculations in respective modules
- Adjust UI styling in `app.py`
- Extend recommendation logic in `src/hybrid.py`

## 📈 Performance

- **Caching**: Uses Streamlit caching for similarity matrices
- **Efficient Algorithms**: Optimized for real-time recommendations
- **Scalable Design**: Modular architecture supports large datasets

## 🛠️ Development

### Project Structure
```
src/
├── data_processing.py     # Data loading and cleaning
├── content_based.py       # TF-IDF and content similarity
├── collaborative.py       # User-item matrix and collaborative filtering
├── hybrid.py             # Weighted combination logic
├── training.py           # Model training pipeline
└── utils.py              # Helper functions and utilities
```

### Adding New Features
1. **New Recommendation Method**: Add to `src/` directory
2. **UI Components**: Create in `frontend/` directory
3. **Data Processing**: Extend `data_processing.py`

## 🔧 Advanced Scripts & Tools

The project includes a comprehensive suite of production-ready scripts for training, evaluation, optimization, and maintenance:

### 📊 **System Evaluation (`scripts/evaluate_system.py`)**
**Purpose:** Comprehensive evaluation of the hybrid recommender system performance

**Capabilities:**
- **Multi-metric Evaluation**: F1-score, RMSE, MAE, Precision, Recall, Coverage, Diversity
- **Alpha Optimization**: Tests all alpha values (0.0 to 1.0) to find optimal hybrid weights
- **Cross-validation**: Robust performance assessment with train/test splits
- **User Satisfaction**: Integrates survey data for real-world performance metrics
- **Automated Reporting**: Generates detailed JSON reports with recommendations

**Usage:**
```bash
# Run comprehensive evaluation
py scripts/evaluate_system.py

# Output: evaluation_report_YYYYMMDD_HHMMSS.json
```

**When to Use:**
- After model training or retraining
- Before production deployment
- For performance benchmarking
- Regular system health checks

---

### 🎯 **Model Optimization (`scripts/optimize_models.py`)**
**Purpose:** Automated hyperparameter optimization for maximum performance

**Capabilities:**
- **Grid Search**: Tests 180+ hyperparameter combinations
- **Early Stopping**: Stops when no improvement is detected
- **Composite Scoring**: Balances multiple metrics for optimal configuration
- **Performance Tracking**: Maintains optimization history and convergence data
- **Resource Management**: Efficient memory usage during optimization

**Usage:**
```bash
# Run full optimization (can take 30+ minutes)
py scripts/optimize_models.py

# Output: logs/optimization_history.json
```

**Current Best Configuration:**
- **Content-based**: max_features=1000, ngram_range=(1,1), min_df=1
- **Collaborative**: n_factors=10, regularization=0.1
- **Optimal Alpha**: 0.6 (balanced hybrid approach for current dataset)

---

### 🔄 **Model Retraining (`scripts/retrain.py`)**
**Purpose:** Automated model retraining with drift detection and performance monitoring

**Capabilities:**
- **Drift Detection**: Monitors model performance degradation over time
- **Flexible Retraining**: Full or incremental retraining options
- **Performance Tracking**: Before/after performance comparison
- **History Management**: Complete retraining event logging
- **Smart Scheduling**: Recommends when retraining is needed

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
1. **Start with evaluation:** `py scripts/evaluate_system.py`
2. **Optimize if needed:** `py scripts/optimize_models.py`
3. **Check retraining status:** `py scripts/retrain.py --check`

**For Production Deployment:**
1. **Run optimization:** `py scripts/optimize_models.py`
2. **Evaluate final model:** `py scripts/evaluate_system.py`
3. **Set up scheduled retraining:** `py scripts/retrain.py --check`

**For Performance Issues:**
1. **Check retraining status:** `py scripts/retrain.py --check`
2. **Force retraining if needed:** `py scripts/retrain.py --force`
3. **Re-evaluate performance:** `py scripts/evaluate_system.py`

**For Research/Development:**
1. **Run full optimization:** `py scripts/optimize_models.py`
2. **Analyze results:** Check `logs/optimization_history.json`
3. **Compare configurations:** Review evaluation reports

---

### 📁 **Output Files**

**Evaluation Reports:**
- `evaluation/test_results/evaluation_report_YYYYMMDD_HHMMSS.json`
- `evaluation/test_results/comprehensive_evaluation_YYYYMMDD_HHMMSS.json`

**Optimization Results:**
- `logs/optimization_history.json` - Complete optimization history
- `logs/optimization.log` - Detailed optimization logs

**Retraining History:**
- `logs/retrain_history.json` - Retraining events and performance trends
- `logs/retrain.log` - Retraining operation logs

**Model Files:**
- `models/content_similarity_matrix.pkl`
- `models/tfidf_vectorizer.pkl`
- `models/svd_model.pkl`
- `models/training_metadata.pkl`

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

#### **Evaluation Commands**
```bash
# Comprehensive system evaluation
py scripts/evaluate_system.py

# User satisfaction survey (interactive)
py -m streamlit run evaluation/user_survey.py

# Check retraining status
py scripts/retrain.py --check

# Force model retraining
py scripts/retrain.py --force
```

#### **Performance Benchmarks**
| Metric | Good | Excellent | Current Best |
|--------|------|-----------|--------------|
| **F1-Score** | > 0.3 | > 0.5 | 0.21 |
| **RMSE** | < 1.0 | < 0.8 | 3.27 |
| **Coverage** | > 0.1 | > 0.2 | 0.85 |
| **Diversity** | > 0.3 | > 0.5 | 0.95 |
| **User Satisfaction** | > 3.0/5 | > 4.0/5 | 4.2/5 |

#### **Evaluation Output Files**
- `evaluation/test_results/evaluation_report_YYYYMMDD_HHMMSS.json`
- `evaluation/user_survey/survey_responses.json`
- `logs/optimization_history.json`
- `logs/retrain_history.json`

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

## 📁 Complete Project File Index

### **Root Directory Files**
| File | Purpose | Size | Description |
|------|---------|------|-------------|
| `app.py` | Main Streamlit application | ~9KB | Entry point with modern UI and caching |
| `requirements.txt` | Python dependencies | ~1KB | All required packages with versions |
| `README.md` | Project documentation | ~25KB | Comprehensive usage and API guide |
| `LICENSE` | Proprietary license | ~5KB | Copyright and usage restrictions |
| `install.ps1` | PowerShell installer | ~5KB | Cross-platform installation script |
| `run_app.bat` | Windows launcher | ~100B | Quick start batch file |

### **Core System (`src/`)**
| File | Purpose | Dependencies | Key Features |
|------|---------|--------------|--------------|
| `data_processing.py` | Data loading & cleaning | pandas, numpy, sklearn | Sample data generation, text preprocessing |
| `content_based.py` | Content-based filtering | sklearn, numpy | TF-IDF vectorization, cosine similarity |
| `collaborative.py` | Collaborative filtering | sklearn, numpy | User-item matrix, SVD decomposition |
| `hybrid.py` | Hybrid recommendation logic | numpy, pandas | Weighted fusion, alpha optimization |
| `training.py` | Model training pipeline | sklearn, joblib | Hyperparameter tuning, model persistence |
| `utils.py` | Utility functions | pandas, numpy | Data validation, matrix operations |

### **Frontend (`frontend/`)**
| File | Purpose | Dependencies | UI Components |
|------|---------|--------------|---------------|
| `home.py` | Search interface | streamlit, pandas | Book selection, parameter controls |
| `results.py` | Results display | streamlit, plotly | Visualization, analysis tabs |

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
| `content_similarity_matrix.pkl` | Content similarity | ~944MB | Pre-computed content similarity matrix |
| `training_metadata.pkl` | Training metadata | ~256B | Hyperparameters and training info |

### **Production Scripts (`scripts/`)**
| File | Purpose | Dependencies | Capabilities |
|------|---------|--------------|--------------|
| `evaluate_system.py` | System evaluation | sklearn, pandas | Multi-metric performance assessment |
| `optimize_models.py` | Hyperparameter optimization | sklearn, joblib | Grid search, early stopping |
| `retrain.py` | Model retraining | sklearn, joblib | Drift detection, incremental training |

### **Evaluation Framework (`evaluation/`)**
| File | Purpose | Dependencies | Features |
|------|---------|--------------|----------|
| `metrics.py` | Evaluation metrics | sklearn, pandas | RMSE, F1-score, coverage, diversity |
| `user_survey.py` | User satisfaction survey | streamlit, pandas | Interactive feedback collection |

### **Logs & History (`logs/`)**
| File | Purpose | Generated by | Content |
|------|---------|--------------|---------|
| `optimization_history.json` | Optimization history | `optimize_models.py` | Best configurations, performance trends |
| `optimization.log` | Optimization logs | `optimize_models.py` | Detailed optimization process logs |
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
   - **"python command not found"**: Use `py` instead of `python` on Windows (Python 3.13.7 detected)
   - **Permission Errors**: Run as Administrator (Windows) or use `sudo` (Linux/macOS)
   - **Build Errors**: Install Visual Studio Build Tools (Windows) or build-essential (Linux)

### **Performance Tips**
- Use specific book titles rather than generic searches
- Experiment with different alpha values
- Consider publisher preferences for diversity
- Use the advanced options for fine-tuning
- Similarity matrices are cached for faster access

### **Debug Mode**
Enable detailed logging by modifying `app.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 API Reference

### Core Functions

#### `hybrid_recommend(book_title, books_df, content_sim_matrix, collab_sim_matrix, alpha=0.6, top_n=10)`
Generate hybrid recommendations for a given book.

**Parameters:**
- `book_title`: Title of the book to find recommendations for
- `books_df`: DataFrame with book information
- `content_sim_matrix`: Content-based similarity matrix
- `collab_sim_matrix`: Collaborative filtering similarity matrix
- `alpha`: Weight for content-based filtering (0-1)
- `top_n`: Number of recommendations to return

**Returns:**
- List of recommendation dictionaries with book information and scores

#### `create_content_similarity_matrix(books_df)`
Create content-based similarity matrix using TF-IDF.

#### `create_item_item_similarity_matrix(user_item_matrix)`
Create collaborative filtering similarity matrix.

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
**Python Version**: 3.13.7 Compatible  

### **Recent Updates**
- ✅ Comprehensive evaluation framework implementation
- ✅ Automated hyperparameter optimization
- ✅ Model retraining with drift detection
- ✅ User satisfaction survey system
- ✅ Modern UI with responsive design and animations
- ✅ Complete documentation and API reference
- ✅ Production-ready PowerShell installation script
- ✅ Cross-platform compatibility verified

---

**Happy Reading! 📚✨**

*Discover your next favorite book with AI-powered recommendations*
