# 🚀 Quick Start Guide - Hybrid Book Recommender

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for version control)

## ⚡ Quick Setup

### 1. **Clone and Install**
```bash
git clone <repository-url>
cd book_recommender
pip install -r requirements.txt
```

### 2. **Run the Application**
```bash
streamlit run app.py
```

### 3. **Access the App**
Open your browser and go to: `http://localhost:8501`

## 🎯 How to Use

### **Step 1: Search for Books**
- Use the search bar to find books by title or author
- Filter by publisher using the dropdown
- Browse through the filtered results

### **Step 2: Select a Book**
- Choose a book from the dropdown list
- View book details (title, author, publisher, rating)
- Expand the description if available

### **Step 3: Configure Settings**
- **Alpha Weight**: Adjust the balance between content-based (0.0) and collaborative filtering (1.0)
- **Number of Recommendations**: Choose how many books to recommend (5-20)
- **Advanced Options**: Fine-tune similarity thresholds and diversity settings

### **Step 4: Generate Recommendations**
- Click "🚀 Generate Recommendations"
- View hybrid recommendations with detailed scores
- Explore separate content-based and collaborative recommendations
- Analyze overlap between different methods

## 🔧 Advanced Usage

### **System Evaluation**
```bash
python scripts/evaluate_system.py
```

### **Model Retraining**
```bash
python scripts/retrain.py
```

### **User Survey**
```bash
streamlit run evaluation/user_survey.py
```

## 📊 Understanding Results

### **Hybrid Recommendations**
- **Hybrid Score**: Combined score from both methods
- **Content Score**: Similarity based on book content
- **Collaborative Score**: Similarity based on user behavior
- **Method**: Shows whether hybrid or content-only was used

### **Analysis Tabs**
- **Method Comparison**: Compare different recommendation approaches
- **Overlap Analysis**: See how methods agree/disagree
- **Performance Metrics**: View system performance statistics

## 🛠️ Troubleshooting

### **Common Issues**

1. **"No recommendations found"**
   - Try a different book title
   - Check if the book exists in the database
   - Adjust alpha weight settings

2. **Slow loading**
   - First run may be slower due to data processing
   - Subsequent runs use cached data

3. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

### **Data Issues**
- Verify data files exist in `data/raw/` and `data/processed/`
- Check file permissions and paths
- Ensure similarity matrices are properly generated

## 📈 Performance Tips

### **For Best Results**
- Use specific book titles rather than generic searches
- Experiment with different alpha values
- Consider publisher preferences for diversity
- Use the advanced options for fine-tuning

### **System Optimization**
- Similarity matrices are cached for faster access
- Data is preprocessed and optimized
- Fallback mechanisms ensure recommendations even with limited data

## 🔍 Exploring the Code

### **Key Files to Understand**
- `app.py`: Main application entry point
- `src/hybrid.py`: Core hybrid recommendation logic
- `frontend/home.py`: User interface components
- `src/data_processing.py`: Data handling and preprocessing

### **Customization**
- Modify alpha weights in the UI
- Adjust similarity thresholds in advanced options
- Customize evaluation metrics in `evaluation/metrics.py`
- Add new recommendation methods in `src/`

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the project documentation in `docs/`
3. Examine the evaluation results for system performance
4. Use the user survey to provide feedback

## 🎉 Success Indicators

You'll know the system is working correctly when:
- ✅ Recommendations are generated quickly
- ✅ Results include both content and collaborative scores
- ✅ Different alpha values produce varied recommendations
- ✅ Analysis tabs show meaningful comparisons
- ✅ System evaluation reports good performance metrics
