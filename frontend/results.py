import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils import format_rating, format_similarity_score, calculate_recommendation_metrics

def display_recommendations(recommendations, selected_book, alpha, show_detailed_scores=False):
    """
    Display the main recommendations with a beautiful UI.
    
    Args:
        recommendations: List of recommendation dictionaries
        selected_book: Title of the selected book
        alpha: Hybrid weight used
        show_detailed_scores: Whether to show detailed scoring breakdown
    """
    if not recommendations:
        st.error("No recommendations found. Try selecting a different book or adjusting the parameters.")
        return
    
    st.header("🎯 Your Personalized Recommendations")
    
    # Show recommendation summary
    st.subheader(f"Based on: **{selected_book}**")
    
    # Display weight information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Recommendations", len(recommendations))
    with col2:
        st.metric("Content Weight", f"{alpha*100:.0f}%")
    with col3:
        st.metric("Collaborative Weight", f"{(1-alpha)*100:.0f}%")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📚 Recommendations", "📊 Analysis", "🔄 Comparison"])
    
    with tab1:
        display_recommendation_cards(recommendations, show_detailed_scores)
    
    with tab2:
        display_recommendation_analysis(recommendations)
    
    with tab3:
        display_comparison_view(recommendations, alpha)

def display_recommendation_cards(recommendations, show_detailed_scores=False):
    """
    Display recommendations as beautiful cards.
    
    Args:
        recommendations: List of recommendation dictionaries
        show_detailed_scores: Whether to show detailed scoring
    """
    # Create a grid layout for recommendations
    cols = st.columns(2)
    
    for i, rec in enumerate(recommendations):
        col_idx = i % 2
        with cols[col_idx]:
            # Create a modern card using Streamlit components
            with st.container():
                # Card header with title and score
                col_title, col_score = st.columns([3, 1])
                
                with col_title:
                    st.markdown(f"### {rec['title']}")
                    st.markdown(f"**by {rec['authors']}**")
                
                with col_score:
                    score_key = 'hybrid_score' if 'hybrid_score' in rec else 'similarity_score'
                    score_value = rec.get(score_key, 0)
                    st.metric("Score", f"{format_similarity_score(score_value)}")
                
                # Genre and rating badges
                col_genre, col_rating = st.columns(2)
                
                with col_genre:
                    st.markdown(f"**Publisher:** {rec['genre']}")
                
                with col_rating:
                    if rec.get("rating", 0) > 0:
                        st.markdown(f"**Rating:** ⭐ {rec['rating']:.1f}")
                    else:
                        st.markdown("**Rating:** N/A")
                
                # Method indicator
                method = rec.get('method', 'hybrid')
                method_emojis = {
                    'hybrid': '🔄',
                    'content_based': '📖',
                    'collaborative': '👥'
                }
                method_labels = {
                    'hybrid': 'Hybrid',
                    'content_based': 'Content-Based',
                    'collaborative': 'Collaborative'
                }
                
                emoji = method_emojis.get(method, '📖')
                label = method_labels.get(method, method.replace('_', ' ').title())
                
                st.markdown(f"**Method:** {emoji} {label}")
                st.markdown(f"**Rank:** #{i+1}")
                
                # Detailed scores if requested
                if show_detailed_scores and 'content_score' in rec and 'collab_score' in rec:
                    with st.expander("📊 Detailed Scores", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Content Score", f"{format_similarity_score(rec['content_score'])}")
                        with col2:
                            st.metric("Collaborative Score", f"{format_similarity_score(rec['collab_score'])}")
                
                st.markdown("---")

def display_recommendation_analysis(recommendations):
    """
    Display analysis and metrics for the recommendations.
    
    Args:
        recommendations: List of recommendation dictionaries
    """
    if not recommendations:
        return
    
    # Calculate metrics
    metrics = calculate_recommendation_metrics(recommendations)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Recommendations", metrics['total_recommendations'])
    
    with col2:
        st.metric("Unique Genres", metrics['unique_genres'])
    
    with col3:
        st.metric("Genre Diversity", f"{metrics['genre_diversity']:.2f}")
    
    with col4:
        avg_rating = metrics['avg_rating']
        st.metric("Avg Rating", f"{avg_rating:.1f} ⭐" if avg_rating > 0 else "N/A")
    
    # Genre distribution chart
    st.subheader("📚 Genre Distribution")
    if metrics['genre_distribution']:
        genre_df = pd.DataFrame(list(metrics['genre_distribution'].items()), 
                              columns=['Genre', 'Count'])
        
        fig = px.bar(genre_df, x='Genre', y='Count', 
                    title="Recommendations by Genre",
                    color='Count',
                    color_continuous_scale='Blues')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Similarity score distribution
    st.subheader("📊 Similarity Score Distribution")
    scores = [rec.get('hybrid_score', 0) for rec in recommendations]
    if scores:
        fig = px.histogram(x=scores, nbins=10, 
                          title="Distribution of Similarity Scores",
                          labels={'x': 'Similarity Score', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)

def display_comparison_view(recommendations, alpha):
    """
    Display comparison between different recommendation approaches.
    
    Args:
        recommendations: List of recommendation dictionaries
        alpha: Hybrid weight used
    """
    st.subheader("🔄 Recommendation Method Comparison")
    
    # Create comparison metrics
    if 'content_score' in recommendations[0] and 'collab_score' in recommendations[0]:
        content_scores = [rec['content_score'] for rec in recommendations]
        collab_scores = [rec['collab_score'] for rec in recommendations]
        hybrid_scores = [rec['hybrid_score'] for rec in recommendations]
        
        # Calculate averages
        avg_content = sum(content_scores) / len(content_scores)
        avg_collab = sum(collab_scores) / len(collab_scores)
        avg_hybrid = sum(hybrid_scores) / len(hybrid_scores)
        
        # Display comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Content-Based Avg", f"{avg_content:.3f}")
        
        with col2:
            st.metric("Collaborative Avg", f"{avg_collab:.3f}")
        
        with col3:
            st.metric("Hybrid Avg", f"{avg_hybrid:.3f}")
        
        # Create comparison chart
        methods = ['Content-Based', 'Collaborative', 'Hybrid']
        scores = [avg_content, avg_collab, avg_hybrid]
        
        fig = go.Figure(data=[
            go.Bar(x=methods, y=scores, 
                  marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ])
        fig.update_layout(
            title="Average Similarity Scores by Method",
            yaxis_title="Similarity Score",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Weight explanation
        st.info(f"**Hybrid Weight Used**: {alpha*100:.0f}% content-based + {(1-alpha)*100:.0f}% collaborative filtering")
    else:
        st.info("Detailed comparison not available for these recommendations.")

def display_separate_recommendations(separate_recs):
    """
    Display separate content-based and collaborative recommendations.
    
    Args:
        separate_recs: Dictionary with separate recommendations
    """
    st.subheader("📊 Method Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📖 Content-Based Recommendations")
        if separate_recs['content_based']:
            for i, rec in enumerate(separate_recs['content_based'][:5]):
                st.write(f"{i+1}. **{rec['title']}** by {rec['authors']}")
                st.write(f"   Genre: {rec['genre']} | Score: {format_similarity_score(rec['similarity_score'])}")
        else:
            st.write("No content-based recommendations available.")
    
    with col2:
        st.markdown("### 🎯 Collaborative Recommendations")
        if separate_recs['collaborative']:
            for i, rec in enumerate(separate_recs['collaborative'][:5]):
                st.write(f"{i+1}. **{rec['title']}** by {rec['authors']}")
                st.write(f"   Genre: {rec['genre']} | Score: {format_similarity_score(rec['similarity_score'])}")
        else:
            st.write("No collaborative recommendations available.")

def display_overlap_analysis(overlap_data):
    """
    Display overlap analysis between different recommendation methods.
    
    Args:
        overlap_data: Dictionary with overlap statistics
    """
    st.subheader("🔄 Recommendation Overlap Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Content-Collab Overlap", overlap_data['content_collab_overlap'])
        st.metric("Content-Hybrid Overlap", overlap_data['content_hybrid_overlap'])
        st.metric("Collab-Hybrid Overlap", overlap_data['collab_hybrid_overlap'])
    
    with col2:
        st.metric("Jaccard Content-Collab", f"{overlap_data['jaccard_content_collab']:.3f}")
        st.metric("Jaccard Content-Hybrid", f"{overlap_data['jaccard_content_hybrid']:.3f}")
        st.metric("Jaccard Collab-Hybrid", f"{overlap_data['jaccard_collab_hybrid']:.3f}")
    
    st.metric("Total Unique Recommendations", overlap_data['total_unique_recommendations'])

def get_score_color(score):
    """
    Get color for similarity score display.
    
    Args:
        score: Similarity score
    
    Returns:
        color: Hex color code
    """
    if score >= 0.8:
        return "#10b981"  # Green
    elif score >= 0.6:
        return "#f59e0b"  # Yellow
    elif score >= 0.4:
        return "#f97316"  # Orange
    else:
        return "#ef4444"  # Red

def get_method_indicator(method):
    """
    Get method indicator with emoji and styling.
    
    Args:
        method: Recommendation method
    
    Returns:
        HTML string with method indicator
    """
    method_emojis = {
        'hybrid': '🔄',
        'content_based': '📖',
        'collaborative': '👥'
    }
    method_labels = {
        'hybrid': 'Hybrid',
        'content_based': 'Content-Based',
        'collaborative': 'Collaborative'
    }
    method_colors = {
        'hybrid': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'content_based': 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
        'collaborative': 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)'
    }
    
    emoji = method_emojis.get(method, '📖')
    label = method_labels.get(method, method.replace('_', ' ').title())
    color = method_colors.get(method, 'linear-gradient(135deg, #6b7280 0%, #4b5563 100%)')
    
    return f"""
    <span style='background: {color}; 
                 color: white; 
                 padding: 0.25rem 0.75rem; 
                 border-radius: 20px; 
                 font-size: 0.8rem; 
                 font-weight: 500;
                 display: inline-flex;
                 align-items: center;
                 gap: 0.25rem;'>
        {emoji} {label}
    </span>
    """

def display_error_message(error_msg):
    """
    Display error message with helpful suggestions.
    
    Args:
        error_msg: Error message to display
    """
    st.error("❌ Error Generating Recommendations")
    st.write(f"**Error:** {error_msg}")
    
    st.info("**Troubleshooting Tips:**")
    st.write("• Make sure you've selected a book from the dropdown")
    st.write("• Try adjusting the hybrid weight slider")
    st.write("• Check if the book exists in our database")
    st.write("• Try a different book or search term")

def display_loading_message():
    """
    Display loading message while generating recommendations.
    """
    with st.spinner("🤖 Generating your personalized recommendations..."):
        st.write("This may take a few seconds as we analyze book similarities and user preferences.")
        st.progress(0)
        
        # Simulate progress
        for i in range(100):
            st.progress(i + 1)
            st.empty()
