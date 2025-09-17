import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import altair as alt

# Add src directory to path
sys.path.append('src')

# Import our modules
from src.data_processing import load_and_clean_data, create_sample_ratings
from src.content_based import create_content_similarity_matrix, save_content_similarity_matrix
from src.collaborative import create_user_item_matrix, create_item_item_similarity_matrix, save_collaborative_similarity_matrix
from src.hybrid import hybrid_recommend, get_separate_recommendations, analyze_recommendation_overlap
from src.utils import load_sim_matrix, load_books_data, load_ratings_data, validate_data_integrity
from frontend.home import show_search_ui, show_quick_stats, show_advanced_options, show_recommendation_controls
from frontend.results import (
    display_recommendations, display_separate_recommendations, 
    display_overlap_analysis, display_error_message, display_loading_message
)

# Page configuration
st.set_page_config(
    page_title="Hybrid Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern UI CSS with gradients, animations, and improved styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        animation: fadeInDown 1s ease-out;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
        animation: fadeInUp 1s ease-out 0.3s both;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Card Styles */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Input Styles */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.9);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    /* Selectbox Styles */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        background: rgba(255, 255, 255, 0.9);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Slider Styles */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Container Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Recommendation Cards */
    .recommendation-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .recommendation-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Genre Badge */
    .genre-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    /* Score Display */
    .score-display {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Expander Styles */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        font-weight: 600;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 12px;
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border-radius: 12px;
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border-radius: 12px;
        border: none;
    }
    
    /* Info Box */
    .stInfo {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border-radius: 12px;
        border: none;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem !important;
        }
        
        .subtitle {
            font-size: 1.1rem !important;
        }
        
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        /* Stack feature cards on mobile */
        .feature-cards {
            flex-direction: column !important;
            gap: 1rem !important;
        }
        
        .feature-cards > div {
            width: 100% !important;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            font-size: 2rem !important;
        }
        
        .subtitle {
            font-size: 1rem !important;
        }
        
        /* Make recommendation cards stack on very small screens */
        .recommendation-card {
            margin-bottom: 1rem;
        }
    }
    
    /* Dark mode support (if user prefers dark mode) */
    @media (prefers-color-scheme: dark) {
        .main {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            color: #e5e7eb;
        }
        
        .recommendation-card {
            background: rgba(255, 255, 255, 0.1);
            color: #e5e7eb;
        }
    }
    
    /* Loading states */
    .loading-shimmer {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% {
            background-position: -200% 0;
        }
        100% {
            background-position: 200% 0;
        }
    }
    
    /* Focus states for accessibility */
    .stButton > button:focus,
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within {
        outline: 2px solid #667eea;
        outline-offset: 2px;
    }
    
    /* Smooth transitions for all interactive elements */
    * {
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data(cache_key=None):
    """
    Load and cache all necessary data for the application.
    """
    try:
        # Load books data
        books_df = load_books_data()
        if books_df.empty:
            # Create sample data if no data exists
            books_df = load_and_clean_data()
        
        # Load or create ratings data
        ratings_df = load_ratings_data()
        if ratings_df.empty:
            # Create sample ratings
            ratings_df = create_sample_ratings(books_df)
        
        # Load or create similarity matrices
        content_sim_matrix = load_sim_matrix('content')
        if content_sim_matrix is None:
            # Create content similarity matrix
            content_sim_matrix, _ = create_content_similarity_matrix(books_df)
            save_content_similarity_matrix(content_sim_matrix)
        
        collab_sim_matrix = load_sim_matrix('collab')
        if collab_sim_matrix is None:
            # Create collaborative similarity matrix
            user_item_matrix = create_user_item_matrix(ratings_df, books_df)
            collab_sim_matrix = create_item_item_similarity_matrix(user_item_matrix)
            save_collaborative_similarity_matrix(collab_sim_matrix)
        
        return books_df, ratings_df, content_sim_matrix, collab_sim_matrix
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def main():
    """
    Main application function.
    """
    # Modern Header with gradient background
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; 
                border-radius: 16px; 
                margin-bottom: 3rem;
                text-align: center;
                box-shadow: 0 4px 16px rgba(0,0,0,0.1);'>
        <h1 class="main-header">📚 Hybrid Book Recommender</h1>
        <p class="subtitle">
            Discover your next favorite book using advanced AI-powered recommendations
        </p>
        <div class="feature-cards" style='display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; flex-wrap: wrap;'>
            <div style='background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 16px; backdrop-filter: blur(10px); transition: transform 0.3s ease;'>
                <div style='font-size: 1.5rem; font-weight: 700; color: white;'>🤖 AI-Powered</div>
                <div style='font-size: 0.9rem; color: rgba(255,255,255,0.8);'>Smart Recommendations</div>
            </div>
            <div style='background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 16px; backdrop-filter: blur(10px); transition: transform 0.3s ease;'>
                <div style='font-size: 1.5rem; font-weight: 700; color: white;'>📊 Hybrid</div>
                <div style='font-size: 0.9rem; color: rgba(255,255,255,0.8);'>Content + Collaborative</div>
            </div>
            <div style='background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 16px; backdrop-filter: blur(10px); transition: transform 0.3s ease;'>
                <div style='font-size: 1.5rem; font-weight: 700; color: white;'>⚡ Real-time</div>
                <div style='font-size: 0.9rem; color: rgba(255,255,255,0.8);'>Instant Results</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data with cache key based on file modification time
    import os
    import time
    try:
        # Use file modification time as cache key to refresh when matrices are updated
        content_matrix_path = 'data/processed/content_sim_matrix.npy'
        collab_matrix_path = 'data/processed/collab_sim_matrix.npy'
        ratings_path = 'data/processed/ratings.csv'
        
        # Create cache key based on multiple files
        cache_key = 0
        if os.path.exists(content_matrix_path):
            cache_key += int(os.path.getmtime(content_matrix_path))
        if os.path.exists(collab_matrix_path):
            cache_key += int(os.path.getmtime(collab_matrix_path))
        if os.path.exists(ratings_path):
            cache_key += int(os.path.getmtime(ratings_path))
    except:
        cache_key = 0
    
    with st.spinner("Loading book database and similarity matrices..."):
        books_df, ratings_df, content_sim_matrix, collab_sim_matrix = load_data(cache_key)
    
    if books_df is None:
        st.error("Failed to load data. Please check your data files and try again.")
        return
    
    # Sidebar with stats
    show_quick_stats(books_df)
    
    # Data validation info
    with st.sidebar.expander("🔍 Data Validation"):
        validation_results = validate_data_integrity(books_df, ratings_df)
        st.write(f"**Books:** {validation_results['books_total']}")
        st.write(f"**Users:** {validation_results['unique_users']}")
        st.write(f"**Ratings:** {validation_results['ratings_total']}")
        st.write(f"**Avg Rating:** {validation_results['avg_rating']:.2f}")
    
    # Main content
    try:
        # Show search UI
        selected_book, alpha, selected_genres, num_recommendations = show_search_ui(books_df)
        
        # Show advanced options
        advanced_options = show_advanced_options()
        
        # Show recommendation controls
        generate_button = show_recommendation_controls()
        
        # Generate recommendations when button is clicked
        if generate_button and selected_book:
            # Show loading message
            with st.spinner("🤖 Generating your personalized recommendations..."):
                # Get hybrid recommendations with improved blending
                recommendations = hybrid_recommend(
                    selected_book, 
                    books_df, 
                    content_sim_matrix, 
                    collab_sim_matrix, 
                    alpha=alpha, 
                    top_n=num_recommendations,
                    min_similarity=advanced_options.get('min_similarity', 0.1),
                    diversity_weight=advanced_options.get('diversity_weight', 0.3),
                    use_candidate_union=True,  # Enable candidate union strategy
                    candidate_size=100,       # Top 100 from each method
                    use_rank_fusion=False    # Use score blending instead of RRF
                )
                
                # Get separate recommendations for comparison
                separate_recs = get_separate_recommendations(
                    selected_book, 
                    books_df, 
                    content_sim_matrix, 
                    collab_sim_matrix, 
                    top_n=num_recommendations
                )
                
                # Calculate overlap analysis
                overlap_data = analyze_recommendation_overlap(
                    separate_recs['content_based'],
                    separate_recs['collaborative'],
                    recommendations
                )
            
            # Display results
            if recommendations:
                # Main recommendations
                display_recommendations(
                    recommendations, 
                    selected_book, 
                    alpha, 
                    show_detailed_scores=advanced_options.get('show_detailed_scores', False)
                )
                
                # Additional analysis tabs
                with st.expander("📊 Detailed Analysis"):
                    tab1, tab2 = st.tabs(["Method Comparison", "Overlap Analysis"])
                    
                    with tab1:
                        display_separate_recommendations(separate_recs)
                    
                    with tab2:
                        display_overlap_analysis(overlap_data)
                
                # Compute and store Pearson correlations for current recommendations
                try:
                    # Build ratings matrix and compute correlation only for current recs
                    user_item_matrix_curr = create_user_item_matrix(ratings_df, books_df)
                    ratings_nan = user_item_matrix_curr.replace(0, np.nan)

                    # Resolve selected book_id from title
                    import re
                    sel_matches = books_df[books_df['title'].str.lower().str.contains(re.escape(str(selected_book).lower()), regex=True, na=False)]
                    if len(sel_matches) > 1:
                        sel_matches = sel_matches.sort_values('book_id')
                    selected_book_id = int(sel_matches.iloc[0]['book_id']) if len(sel_matches) > 0 else None

                    # Collect rec book_ids
                    rec_ids = [int(r.get('book_id')) for r in recommendations if isinstance(r.get('book_id'), (int, np.integer))]

                    corr_values = []
                    if selected_book_id is not None and selected_book_id in ratings_nan.columns:
                        s_selected = ratings_nan[selected_book_id]
                        for rid in rec_ids:
                            if rid in ratings_nan.columns:
                                val = s_selected.corr(ratings_nan[rid], min_periods=2)
                                if pd.notna(val):
                                    corr_values.append(float(val))

                    # Store in session state for the frontend to use
                    st.session_state.current_pearson_correlations = corr_values
                except Exception as e:
                    st.session_state.current_pearson_correlations = []
                
                # Global similarity distributions
                with st.expander("📈 Global Similarity Distributions"):
                    try:
                        col_a, col_b = st.columns(2)

                        # Cosine similarity distribution from collaborative matrix
                        if collab_sim_matrix is not None and isinstance(collab_sim_matrix, np.ndarray):
                            tri_upper_idx = np.triu_indices_from(collab_sim_matrix, k=1)
                            cosine_values = collab_sim_matrix[tri_upper_idx].astype(float)
                            cosine_df = pd.DataFrame({"score": cosine_values})
                            cosine_chart = (
                                alt.Chart(cosine_df)
                                .mark_bar()
                                .encode(
                                    x=alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Similarity Score"),
                                    y=alt.Y("count()", title="count")
                                )
                                .properties(title="Global Cosine Similarity Distribution")
                            )
                            col_a.altair_chart(cosine_chart, use_container_width=True)

                        # Pearson correlation distribution computed from ratings
                        try:
                            user_item_matrix = create_user_item_matrix(ratings_df, books_df)
                            # Replace zeros (unrated) with NaN so Pearson uses only co-rated pairs
                            ratings_for_corr = user_item_matrix.replace(0, np.nan)
                            pearson_corr = ratings_for_corr.corr(method="pearson", min_periods=2)
                            tri_upper_idx_p = np.triu_indices_from(pearson_corr.values, k=1)
                            pearson_values = pearson_corr.values[tri_upper_idx_p]
                            pearson_values = pearson_values[~np.isnan(pearson_values)]
                            pearson_df = pd.DataFrame({"correlation": pearson_values.astype(float)})
                            pearson_chart = (
                                alt.Chart(pearson_df)
                                .mark_bar()
                                .encode(
                                    x=alt.X("correlation:Q", bin=alt.Bin(maxbins=30), title="Pearson Correlation"),
                                    y=alt.Y("count()", title="count")
                                )
                                .properties(title="Global Pearson Correlation Distribution")
                            )
                            col_b.altair_chart(pearson_chart, use_container_width=True)
                        except Exception as e:
                            st.info(f"Could not compute global Pearson correlation distribution: {e}")
                    except Exception as e:
                        st.info(f"Could not render global similarity distributions: {e}")

                # Success message
                st.success("✅ Recommendations generated successfully!")
                
            else:
                st.warning("No recommendations found. Try selecting a different book or adjusting the parameters.")
        
        elif generate_button and not selected_book:
            st.warning("Please select a book to get recommendations.")
        
        # Show sample recommendations for demonstration
        if not generate_button:
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.9); 
                        backdrop-filter: blur(10px); 
                        border-radius: 16px; 
                        padding: 1.5rem; 
                        margin: 3rem 0;
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        min-height: 120px;'>
                <h2 style='text-align: center; 
                           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           -webkit-background-clip: text; 
                           -webkit-text-fill-color: transparent; 
                           background-clip: text; 
                           margin: 0;
                           font-size: 2.5rem;
                           font-weight: 700;'>💡 How It Works</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div style='background: rgba(255, 255, 255, 0.9); 
                            backdrop-filter: blur(10px); 
                            border-radius: 16px; 
                            padding: 1.5rem; 
                            margin: 1rem 0;
                            border: 1px solid rgba(255, 255, 255, 0.2);
                            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
                            text-align: center;
                            transition: transform 0.3s ease;
                            height: 280px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>📖</div>
                    <h3 style='color: #1f2937; margin-bottom: 1rem; font-weight: 700;'>Content-Based Filtering</h3>
                    <p style='color: #6b7280; line-height: 1.6;'>Analyzes book content, genres, and descriptions to find similar books based on their features.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style='background: rgba(255, 255, 255, 0.9); 
                            backdrop-filter: blur(10px); 
                            border-radius: 16px; 
                            padding: 1.5rem; 
                            margin: 1rem 0;
                            border: 1px solid rgba(255, 255, 255, 0.2);
                            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
                            text-align: center;
                            transition: transform 0.3s ease;
                            height: 280px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>🎯</div>
                    <h3 style='color: #1f2937; margin-bottom: 1rem; font-weight: 700;'>Collaborative Filtering</h3>
                    <p style='color: #6b7280; line-height: 1.6;'>Uses user ratings and preferences to find books that similar users have enjoyed.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div style='background: rgba(255, 255, 255, 0.9); 
                            backdrop-filter: blur(10px); 
                            border-radius: 16px; 
                            padding: 1.5rem; 
                            margin: 1rem 0;
                            border: 1px solid rgba(255, 255, 255, 0.2);
                            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
                            text-align: center;
                            transition: transform 0.3s ease;
                            height: 280px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>🔄</div>
                    <h3 style='color: #1f2937; margin-bottom: 1rem; font-weight: 700;'>Hybrid Approach</h3>
                    <p style='color: #6b7280; line-height: 1.6;'>Combines both methods with adjustable weights to provide the best of both worlds.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show sample books with modern styling
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.9); 
                        backdrop-filter: blur(10px); 
                        border-radius: 16px; 
                        padding: 1.5rem; 
                        margin: 3rem 0;
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        min-height: 120px;'>
                <h2 style='text-align: center; 
                           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           -webkit-background-clip: text; 
                           -webkit-text-fill-color: transparent; 
                           background-clip: text; 
                           margin: 0;
                           font-size: 2.5rem;
                           font-weight: 700;'>📚 Sample Books in Our Database</h2>
            </div>
            """, unsafe_allow_html=True)
            
            sample_books = books_df.head(20)
            cols = st.columns(4)
            
            for i, (_, book) in enumerate(sample_books.iterrows()):
                with cols[i % 4]:
                    st.markdown(f"""
                    <div style='background: rgba(255, 255, 255, 0.9); 
                                backdrop-filter: blur(10px); 
                                border-radius: 16px; 
                                padding: 1.5rem; 
                                margin: 1rem 0;
                                border: 1px solid rgba(255, 255, 255, 0.2);
                                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
                                transition: transform 0.3s ease;'>
                        <h4 style='color: #1f2937; margin-bottom: 0.5rem; font-weight: 700;'>{book['title']}</h4>
                        <p style='color: #6b7280; margin-bottom: 0.5rem; font-weight: 500;'>by {book['authors']}</p>
                        <p style='color: #9ca3af; margin-bottom: 0.5rem; font-size: 0.9rem;'>Publisher: {book['publisher']}</p>
                        {f'<div style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); color: white; padding: 0.5rem 1rem; border-radius: 12px; text-align: center; margin-top: 1rem;"><strong>⭐ {book["average_rating"]:.1f}/5.0</strong></div>' if 'average_rating' in book and book['average_rating'] > 0 else ''}
                    </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        display_error_message(str(e))

if __name__ == "__main__":
    main()
