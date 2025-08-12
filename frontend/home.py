import streamlit as st
import pandas as pd
from src.utils import get_book_titles, get_genres, search_books, filter_books_by_genre

def show_search_ui(books_df):
    """
    Display the main search and selection interface.
    
    Args:
        books_df: DataFrame with book information
    
    Returns:
        tuple: (selected_book, alpha_weight, selected_genres, num_recommendations)
    """
    st.header("📚 Find Your Next Great Read")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Search & Select")
        
        # Search functionality
        search_term = st.text_input(
            "Search books by title or author:",
            placeholder="Enter book title or author name..."
        )
        
        # Filter by genre
        all_genres = get_genres(books_df)
        selected_genres = st.multiselect(
            "Filter by genre (optional):",
            options=all_genres,
            default=[]
        )
        
        # Filter books based on search and genre
        filtered_books = books_df.copy()
        if search_term:
            filtered_books = search_books(filtered_books, search_term)
        if selected_genres:
            filtered_books = filter_books_by_genre(filtered_books, selected_genres)
        
        # Display filtered books count
        st.info(f"Found {len(filtered_books)} books matching your criteria")
        
        # Book selection
        if len(filtered_books) > 0:
            # Create a more user-friendly display for book selection
            book_options = []
            for _, book in filtered_books.iterrows():
                display_text = f"{book['title']} by {book['author']} ({book['genre']})"
                book_options.append((display_text, book['title']))
            
            selected_display = st.selectbox(
                "Choose a book to get recommendations:",
                options=[opt[0] for opt in book_options],
                index=0 if book_options else None
            )
            
            # Extract the actual book title
            selected_book = None
            if selected_display and book_options:
                for display_text, title in book_options:
                    if display_text == selected_display:
                        selected_book = title
                        break
        else:
            selected_book = None
            st.warning("No books found matching your criteria. Try adjusting your search or genre filters.")
    
    with col2:
        st.subheader("⚙️ Settings")
        
        # Hybrid weight adjustment
        st.write("**Hybrid Weight Adjustment**")
        st.write("Adjust the balance between content-based and collaborative filtering:")
        
        alpha = st.slider(
            "Content vs Collaborative Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="0.0 = Pure collaborative filtering, 1.0 = Pure content-based filtering"
        )
        
        # Display weight explanation
        if alpha == 0.0:
            st.info("🎯 **Pure Collaborative Filtering**: Recommendations based on what similar users liked")
        elif alpha == 1.0:
            st.info("📖 **Pure Content-Based Filtering**: Recommendations based on book content similarity")
        else:
            st.info(f"🔄 **Hybrid Approach**: {alpha*100:.0f}% content-based + {(1-alpha)*100:.0f}% collaborative")
        
        # Number of recommendations
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
        
        # Show selected book info if available
        if selected_book:
            st.subheader("📖 Selected Book")
            book_info = books_df[books_df['title'] == selected_book].iloc[0]
            
            st.write(f"**Title:** {book_info['title']}")
            st.write(f"**Author:** {book_info['author']}")
            st.write(f"**Genre:** {book_info['genre']}")
            if 'rating' in book_info and book_info['rating'] > 0:
                st.write(f"**Rating:** {book_info['rating']:.1f} ⭐")
            if 'description' in book_info:
                with st.expander("📝 Description"):
                    st.write(book_info['description'])
    
    return selected_book, alpha, selected_genres, num_recommendations

def show_quick_stats(books_df):
    """
    Display quick statistics about the book database.
    
    Args:
        books_df: DataFrame with book information
    """
    st.sidebar.header("📊 Database Stats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Books", len(books_df))
        st.metric("Unique Authors", books_df['author'].nunique())
    
    with col2:
        st.metric("Genres", books_df['genre'].nunique())
        avg_rating = books_df['rating'].mean() if 'rating' in books_df.columns else 0
        st.metric("Avg Rating", f"{avg_rating:.1f} ⭐" if avg_rating > 0 else "N/A")
    
    # Genre distribution
    st.sidebar.subheader("📚 Genre Distribution")
    genre_counts = books_df['genre'].value_counts()
    
    for genre, count in genre_counts.head(5).items():
        st.sidebar.write(f"• {genre}: {count}")
    
    if len(genre_counts) > 5:
        st.sidebar.write(f"... and {len(genre_counts) - 5} more genres")

def show_advanced_options():
    """
    Display advanced options for power users.
    
    Returns:
        dict: Dictionary with advanced settings
    """
    with st.expander("🔧 Advanced Options"):
        st.write("Fine-tune your recommendation experience:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Minimum similarity threshold
            min_similarity = st.slider(
                "Minimum Similarity Score:",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Only show recommendations above this similarity threshold"
            )
            
            # Include ratings in scoring
            include_ratings = st.checkbox(
                "Include ratings in scoring",
                value=True,
                help="Factor in book ratings when calculating recommendations"
            )
        
        with col2:
            # Diversity penalty
            diversity_weight = st.slider(
                "Diversity Weight:",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Penalize similar recommendations to increase diversity"
            )
            
            # Show detailed scores
            show_detailed_scores = st.checkbox(
                "Show detailed scores",
                value=False,
                help="Display individual content and collaborative scores"
            )
        
        return {
            'min_similarity': min_similarity,
            'include_ratings': include_ratings,
            'diversity_weight': diversity_weight,
            'show_detailed_scores': show_detailed_scores
        }
    
    return {}

def show_recommendation_controls():
    """
    Display controls for recommendation generation.
    
    Returns:
        bool: Whether to generate recommendations
    """
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        generate_button = st.button(
            "🚀 Generate Recommendations",
            type="primary",
            use_container_width=True
        )
    
    return generate_button
