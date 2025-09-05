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
    # Modern search section with glassmorphism effect
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.9); 
                backdrop-filter: blur(10px); 
                border-radius: 16px; 
                padding: 1.5rem; 
                margin-bottom: 2rem;
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100px;'>
        <h2 style='text-align: center; 
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent; 
                   background-clip: text; 
                   margin: 0;
                   font-size: 2rem;
                   font-weight: 700;'>📚 Find Your Next Great Read</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Modern search section
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.9); 
                    backdrop-filter: blur(10px); 
                    border-radius: 16px; 
                    padding: 1.5rem; 
                    margin-bottom: 1rem;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);'>
            <h3 style='color: #374151; margin-bottom: 1rem; font-weight: 600;'>🔍 Search & Select</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Search functionality with modern styling
        search_term = st.text_input(
            "🔎 Search books by title or author:",
            placeholder="Enter book title or author name...",
            help="Type to search through our book database"
        )
        
        # Filter by publisher with modern styling
        all_publishers = get_genres(books_df)  # This now returns publishers
        selected_publishers = st.multiselect(
            "📚 Filter by publisher (optional):",
            options=all_publishers,
            default=[],
            help="Select publishers to narrow down your search"
        )
        
        # Filter books based on search and publisher
        filtered_books = books_df.copy()
        if search_term:
            filtered_books = search_books(filtered_books, search_term)
        if selected_publishers:
            filtered_books = filter_books_by_genre(filtered_books, selected_publishers)  # Function name unchanged but filters by publisher
        
        # Display filtered books count with modern styling
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); 
                    color: white; 
                    padding: 1rem; 
                    border-radius: 12px; 
                    text-align: center; 
                    margin: 1rem 0;
                    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);'>
            <strong>📊 Found {len(filtered_books)} books matching your criteria</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Book selection with modern styling
        if len(filtered_books) > 0:
            # Create a more user-friendly display for book selection
            book_options = []
            for _, book in filtered_books.iterrows():
                display_text = f"{book['title']} by {book['authors']} ({book['publisher']})"
                book_options.append((display_text, book['title']))
            
            selected_display = st.selectbox(
                "📖 Choose a book to get recommendations:",
                options=[opt[0] for opt in book_options],
                index=0 if book_options else None,
                help="Select a book to discover similar recommendations"
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
        # Modern settings panel
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.9); 
                    backdrop-filter: blur(10px); 
                    border-radius: 16px; 
                    padding: 1.5rem; 
                    margin-bottom: 1rem;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);'>
            <h3 style='color: #374151; margin-bottom: 1rem; font-weight: 600;'>⚙️ Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Hybrid weight adjustment with modern styling
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.9); 
                    border-radius: 16px; 
                    padding: 1.5rem; 
                    margin-bottom: 1rem;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);'>
            <h4 style='color: #374151; margin-bottom: 0.5rem; font-weight: 600;'>🎛️ Hybrid Weight Adjustment</h4>
            <p style='color: #6b7280; font-size: 0.9rem; margin-bottom: 1rem;'>Adjust the balance between content-based and collaborative filtering</p>
        </div>
        """, unsafe_allow_html=True)
        
        alpha = st.slider(
            "Content vs Collaborative Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="0.0 = Pure collaborative filtering, 1.0 = Pure content-based filtering"
        )
        
        # Display weight explanation with modern styling
        if alpha == 0.0:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                        color: white; 
                        padding: 1rem; 
                        border-radius: 12px; 
                        margin: 1rem 0;
                        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);'>
                <strong>🎯 Pure Collaborative Filtering</strong><br>
                Recommendations based on what similar users liked
            </div>
            """, unsafe_allow_html=True)
        elif alpha == 1.0:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        color: white; 
                        padding: 1rem; 
                        border-radius: 12px; 
                        margin: 1rem 0;
                        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);'>
                <strong>📖 Pure Content-Based Filtering</strong><br>
                Recommendations based on book content similarity
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; 
                        padding: 1rem; 
                        border-radius: 12px; 
                        margin: 1rem 0;
                        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);'>
                <strong>🔄 Hybrid Approach</strong><br>
                {alpha*100:.0f}% content-based + {(1-alpha)*100:.0f}% collaborative
            </div>
            """, unsafe_allow_html=True)
        
        # Number of recommendations with modern styling
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.9); 
                    border-radius: 16px; 
                    padding: 1.5rem; 
                    margin-bottom: 1rem;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);'>
            <h4 style='color: #374151; margin-bottom: 0.5rem; font-weight: 600;'>📊 Number of Recommendations</h4>
        </div>
        """, unsafe_allow_html=True)
        
        num_recommendations = st.slider(
            "How many recommendations would you like?",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="Choose between 5-20 recommendations"
        )
        
        # Show selected book info if available with modern styling
        if selected_book:
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.8); 
                        backdrop-filter: blur(10px); 
                        border-radius: 16px; 
                        padding: 1.5rem; 
                        margin-top: 1rem;
                        border: 1px solid rgba(255, 255, 255, 0.3);
                        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);'>
                <h3 style='color: #374151; margin-bottom: 1rem; font-weight: 600;'>📖 Selected Book</h3>
            </div>
            """, unsafe_allow_html=True)
            
            import re
            escaped_title = re.escape(selected_book)
            matches = books_df[books_df['title'].str.contains(escaped_title, regex=True, na=False)]
            
            if len(matches) == 0:
                st.error("Book not found in database")
                return selected_book, alpha, selected_publishers, num_recommendations
            
            # If multiple matches, prefer the one with the lowest book_id (usually the first one)
            if len(matches) > 1:
                # Sort by book_id and take the first one
                matches = matches.sort_values('book_id')
            
            book_info = matches.iloc[0]
            
            # Modern book info display
            st.markdown(f"""
            <div style='background: rgba(255, 255, 255, 0.9); 
                        border-radius: 16px; 
                        padding: 1.5rem; 
                        margin: 1rem 0;
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);'>
                <div style='margin-bottom: 0.5rem;'>
                    <strong style='color: #374151;'>📚 Title:</strong> 
                    <span style='color: #1f2937; font-weight: 500;'>{book_info['title']}</span>
                </div>
                <div style='margin-bottom: 0.5rem;'>
                    <strong style='color: #374151;'>✍️ Author:</strong> 
                    <span style='color: #1f2937; font-weight: 500;'>{book_info['authors']}</span>
                </div>
                <div style='margin-bottom: 0.5rem;'>
                    <strong style='color: #374151;'>🏢 Publisher:</strong> 
                    <span style='color: #1f2937; font-weight: 500;'>{book_info['publisher']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if 'average_rating' in book_info and book_info['average_rating'] > 0:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); 
                            color: white; 
                            padding: 0.75rem 1rem; 
                            border-radius: 12px; 
                            text-align: center;
                            margin: 1rem 0;
                            box-shadow: 0 4px 15px rgba(251, 191, 36, 0.3);'>
                    <strong>⭐ Rating: {book_info['average_rating']:.1f}/5.0</strong>
                </div>
                """, unsafe_allow_html=True)
            
            if 'description' in book_info and book_info['description'] and book_info['description'] != 'No description available':
                with st.expander("📝 Book Description", expanded=False):
                    st.write(book_info['description'])
            elif 'description' in book_info and book_info['description'] == 'No description available':
                st.markdown("""
                <div style='background: rgba(107, 114, 128, 0.1); 
                            color: #6b7280; 
                            padding: 1rem; 
                            border-radius: 12px; 
                            text-align: center;
                            margin: 1rem 0;'>
                    📝 No description available for this book
                </div>
                """, unsafe_allow_html=True)
    
    return selected_book, alpha, selected_publishers, num_recommendations

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
        st.metric("Unique Authors", books_df['authors'].nunique())
    
    with col2:
        st.metric("Publishers", books_df['publisher'].nunique())
        avg_rating = books_df['average_rating'].mean() if 'average_rating' in books_df.columns else 0
        st.metric("Avg Rating", f"{avg_rating:.1f} ⭐" if avg_rating > 0 else "N/A")
    
    # Publisher distribution
    st.sidebar.subheader("📚 Publisher Distribution")
    publisher_counts = books_df['publisher'].value_counts()
    
    for publisher, count in publisher_counts.head(5).items():
        st.sidebar.write(f"• {publisher}: {count}")
    
    if len(publisher_counts) > 5:
        st.sidebar.write(f"... and {len(publisher_counts) - 5} more publishers")

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
    st.markdown("""
    <div style='margin: 3rem 0; text-align: center;'>
        <div style='background: rgba(255, 255, 255, 0.9); 
                    backdrop-filter: blur(10px); 
                    border-radius: 16px; 
                    padding: 1.5rem; 
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);'>
            <h3 style='color: #374151; margin-bottom: 1rem; font-weight: 600;'>Ready to discover your next favorite book?</h3>
            <p style='color: #6b7280; margin-bottom: 2rem;'>Click the button below to generate personalized recommendations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        generate_button = st.button(
            "🚀 Generate Recommendations",
            type="primary",
            use_container_width=True,
            help="Generate personalized book recommendations based on your selection"
        )
    
    return generate_button
