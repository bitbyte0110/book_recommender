import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

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
    page_title="📚 Hybrid Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
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
    # Header
    st.markdown('<h1 class="main-header">📚 Hybrid Book Recommender</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Discover your next favorite book using advanced AI-powered recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading book database and similarity matrices..."):
        books_df, ratings_df, content_sim_matrix, collab_sim_matrix = load_data()
    
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
                # Get hybrid recommendations
                recommendations = hybrid_recommend(
                    selected_book, 
                    books_df, 
                    content_sim_matrix, 
                    collab_sim_matrix, 
                    alpha=alpha, 
                    top_n=num_recommendations
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
                
                # Success message
                st.success("✅ Recommendations generated successfully!")
                
            else:
                st.warning("No recommendations found. Try selecting a different book or adjusting the parameters.")
        
        elif generate_button and not selected_book:
            st.warning("Please select a book to get recommendations.")
        
        # Show sample recommendations for demonstration
        if not generate_button:
            st.markdown("---")
            st.subheader("💡 How It Works")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                ### 📖 Content-Based Filtering
                Analyzes book content, genres, and descriptions to find similar books based on their features.
                """)
            
            with col2:
                st.markdown("""
                ### 🎯 Collaborative Filtering
                Uses user ratings and preferences to find books that similar users have enjoyed.
                """)
            
            with col3:
                st.markdown("""
                ### 🔄 Hybrid Approach
                Combines both methods with adjustable weights to provide the best of both worlds.
                """)
            
            # Show sample books
            st.subheader("📚 Sample Books in Our Database")
            sample_books = books_df.head(6)
            
            cols = st.columns(3)
            for i, (_, book) in enumerate(sample_books.iterrows()):
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f"**{book['title']}**")
                        st.write(f"by {book['authors']}")
                        st.write(f"Publisher: {book['publisher']}")
                        if 'average_rating' in book and book['average_rating'] > 0:
                            st.write(f"Rating: {book['average_rating']:.1f} ⭐")
                        st.markdown("---")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        display_error_message(str(e))

if __name__ == "__main__":
    main()
