# 📚 Hybrid Book Recommender System

A sophisticated book recommendation system that combines content-based and collaborative filtering approaches with an interactive Streamlit interface.

## 🎯 Features

- **Hybrid Recommendation Engine**: Combines content-based and collaborative filtering with adjustable weights
- **Interactive Web Interface**: Beautiful Streamlit UI with real-time recommendations
- **Advanced Analytics**: Detailed analysis of recommendation quality and method comparison
- **Flexible Data Management**: Supports both sample data and custom datasets
- **Real-time Adjustments**: Dynamic weight adjustment between recommendation methods

## 🏗️ Architecture

```
book_recommender/
├── 📄 app.py                    # Main Streamlit application
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Project documentation
├── 📄 LICENSE                   # Project license
├── 📄 .gitignore               # Git ignore rules
│
├── 📁 src/                     # Core recommendation system
│   ├── 📄 data_processing.py   # Data loading and preprocessing
│   ├── 📄 content_based.py     # Content-based filtering
│   ├── 📄 collaborative.py     # Collaborative filtering
│   ├── 📄 hybrid.py           # Hybrid recommendation logic
│   ├── 📄 training.py         # Model training pipeline
│   └── 📄 utils.py            # Utility functions
│
├── 📁 frontend/                # Streamlit UI components
│   ├── 📄 home.py             # Main search interface
│   ├── 📄 results.py          # Results display
│   └── 📁 assets/             # Static assets
│
├── 📁 data/                    # Data storage
│   ├── 📁 raw/                # Original datasets
│   └── 📁 processed/          # Cleaned data and matrices
│
├── 📁 models/                  # Trained models
├── 📁 evaluation/              # Evaluation framework
├── 📁 scripts/                 # Utility scripts
└── 📁 docs/                    # Documentation
```

📖 **Detailed structure**: See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

#### **Option 1: Automatic Installation (Recommended)**

**Windows Users:**
- Double-click `install.bat` in the project folder
- Or right-click `install_dependencies.ps1` → "Run with PowerShell"

**Linux/macOS Users:**
```bash
chmod +x install_dependencies.sh && ./install_dependencies.sh
```

#### **Option 2: Manual Installation**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd book_recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

📖 **Detailed guides**: 
- See [`docs/QUICK_START.md`](docs/QUICK_START.md) for usage instructions
- See [`docs/INSTALLATION.md`](docs/INSTALLATION.md) for detailed installation guide
- See [`docs/FILE_INDEX.md`](docs/FILE_INDEX.md) for comprehensive file index

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
└── utils.py              # Helper functions and utilities
```

### Adding New Features
1. **New Recommendation Method**: Add to `src/` directory
2. **UI Components**: Create in `frontend/` directory
3. **Data Processing**: Extend `data_processing.py`

## 🐛 Troubleshooting

### Common Issues

1. **No Recommendations Generated**
   - Check if selected book exists in database
   - Verify similarity matrices are computed
   - Adjust weight parameters

2. **Slow Performance**
   - Ensure similarity matrices are cached
   - Reduce dataset size for testing
   - Check system resources

3. **Import Errors**
   - Verify all dependencies are installed
   - Check Python path configuration
   - Ensure proper file structure

### Debug Mode
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

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit for the amazing web framework
- Scikit-learn for machine learning algorithms
- Plotly for interactive visualizations
- The open-source community for inspiration and tools

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Happy Reading! 📚✨**

---

## 📄 Copyright

© 2025 Marcus Lim Jing Huang. All rights reserved.

This project and its associated documentation, including but not limited to source code, algorithms, user interface designs, and technical specifications, are the intellectual property of Marcus Lim Jing Huang.

**Unauthorized copying, distribution, modification, public display, or public performance of this copyrighted work is strictly prohibited and constitutes copyright infringement.**

For licensing inquiries, please refer to the LICENSE file in this repository.
