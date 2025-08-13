"""
User Satisfaction Survey for Hybrid Book Recommender System
Implements interactive surveys and satisfaction metrics collection
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class UserSatisfactionSurvey:
    """
    User satisfaction survey and feedback collection system
    """
    
    def __init__(self, survey_dir='evaluation/user_survey'):
        self.survey_dir = survey_dir
        self.responses_file = os.path.join(survey_dir, 'survey_responses.json')
        self.satisfaction_file = os.path.join(survey_dir, 'satisfaction_metrics.json')
        
        # Create survey directory if it doesn't exist
        os.makedirs(survey_dir, exist_ok=True)
        
        # Initialize responses file if it doesn't exist
        if not os.path.exists(self.responses_file):
            self._initialize_responses_file()
    
    def _initialize_responses_file(self):
        """
        Initialize the responses file with empty structure
        """
        initial_data = {
            'responses': [],
            'total_responses': 0,
            'average_satisfaction': 0.0,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.responses_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
    
    def load_responses(self):
        """
        Load existing survey responses
        """
        try:
            with open(self.responses_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self._initialize_responses_file()
            return self.load_responses()
    
    def save_response(self, response_data):
        """
        Save a new survey response
        """
        data = self.load_responses()
        
        # Add timestamp and response ID
        response_data['timestamp'] = datetime.now().isoformat()
        response_data['response_id'] = len(data['responses']) + 1
        
        # Add to responses
        data['responses'].append(response_data)
        data['total_responses'] = len(data['responses'])
        
        # Calculate average satisfaction
        if data['responses']:
            satisfaction_scores = [r.get('satisfaction_score', 0) for r in data['responses']]
            data['average_satisfaction'] = sum(satisfaction_scores) / len(satisfaction_scores)
        
        data['last_updated'] = datetime.now().isoformat()
        
        # Save updated data
        with open(self.responses_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return response_data['response_id']
    
    def calculate_satisfaction_score(self, relevance_score, accuracy_score, diversity_score):
        """
        Calculate overall satisfaction score
        """
        # Weighted combination of different aspects
        satisfaction = (0.4 * relevance_score + 
                       0.3 * accuracy_score + 
                       0.3 * diversity_score)
        return round(satisfaction, 2)
    
    def show_satisfaction_survey(self, recommendations=None):
        """
        Display the satisfaction survey in Streamlit
        """
        st.subheader("📊 Feedback Survey")
        st.write("Please help us improve our recommendations by providing your feedback!")
        
        with st.form(key='satisfaction_survey'):
            # Relevance assessment
            st.write("**How relevant were the recommendations?**")
            relevance = st.radio(
                "Relevance",
                options=["Very relevant", "Somewhat relevant", "Not relevant"],
                key="relevance_radio"
            )
            relevance_score = {"Very relevant": 5, "Somewhat relevant": 3, "Not relevant": 1}[relevance]
            
            # Rating accuracy
            st.write("**How accurate were the predicted ratings?**")
            accuracy = st.slider(
                "Rating Accuracy (1-5)",
                min_value=1,
                max_value=5,
                value=3,
                key="accuracy_slider"
            )
            
            # Diversity assessment
            st.write("**How diverse were the recommendations?**")
            diversity = st.radio(
                "Diversity",
                options=["Very diverse", "Somewhat diverse", "Not diverse"],
                key="diversity_radio"
            )
            diversity_score = {"Very diverse": 5, "Somewhat diverse": 3, "Not diverse": 1}[diversity]
            
            # Book preferences
            if recommendations:
                st.write("**Which books did you like from the recommendations?**")
                liked_books = st.multiselect(
                    "Liked Books",
                    options=[f"{rec['title']} by {rec['authors']}" for rec in recommendations],
                    key="liked_books_select"
                )
            else:
                liked_books = []
            
            # Additional feedback
            st.write("**Additional Comments**")
            additional_feedback = st.text_area(
                "Please share any additional thoughts about the recommendations:",
                placeholder="Your feedback helps us improve the system...",
                key="feedback_textarea"
            )
            
            # System usage
            st.write("**System Usage**")
            usage_frequency = st.selectbox(
                "How often do you use book recommendation systems?",
                options=["Daily", "Weekly", "Monthly", "Rarely", "First time"],
                key="usage_frequency"
            )
            
            # Submit button
            submitted = st.form_submit_button("Submit Feedback")
            
            if submitted:
                # Calculate satisfaction score
                satisfaction_score = self.calculate_satisfaction_score(
                    relevance_score, accuracy, diversity_score
                )
                
                # Prepare response data
                response_data = {
                    'relevance': relevance,
                    'relevance_score': relevance_score,
                    'accuracy': accuracy,
                    'diversity': diversity,
                    'diversity_score': diversity_score,
                    'satisfaction_score': satisfaction_score,
                    'liked_books': liked_books,
                    'additional_feedback': additional_feedback,
                    'usage_frequency': usage_frequency
                }
                
                # Save response
                response_id = self.save_response(response_data)
                
                st.success(f"Thank you for your feedback! Response ID: {response_id}")
                st.info(f"Your satisfaction score: {satisfaction_score}/5")
                
                return response_data
        
        return None
    
    def show_survey_analytics(self):
        """
        Display survey analytics and insights
        """
        st.subheader("📈 Survey Analytics")
        
        data = self.load_responses()
        
        if not data['responses']:
            st.warning("No survey responses available yet.")
            return
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(data['responses'])
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Responses", data['total_responses'])
        
        with col2:
            st.metric("Average Satisfaction", f"{data['average_satisfaction']:.2f}/5")
        
        with col3:
            avg_relevance = df['relevance_score'].mean()
            st.metric("Average Relevance", f"{avg_relevance:.2f}/5")
        
        with col4:
            avg_accuracy = df['accuracy'].mean()
            st.metric("Average Accuracy", f"{avg_accuracy:.2f}/5")
        
        # Satisfaction distribution
        st.write("**Satisfaction Score Distribution**")
        fig_satisfaction = px.histogram(
            df, 
            x='satisfaction_score',
            nbins=10,
            title="Distribution of Satisfaction Scores",
            labels={'satisfaction_score': 'Satisfaction Score', 'count': 'Number of Responses'}
        )
        fig_satisfaction.update_layout(showlegend=False)
        st.plotly_chart(fig_satisfaction, use_container_width=True)
        
        # Component scores comparison
        st.write("**Component Scores Comparison**")
        component_scores = {
            'Relevance': df['relevance_score'].mean(),
            'Accuracy': df['accuracy'].mean(),
            'Diversity': df['diversity_score'].mean()
        }
        
        fig_components = px.bar(
            x=list(component_scores.keys()),
            y=list(component_scores.values()),
            title="Average Scores by Component",
            labels={'x': 'Component', 'y': 'Average Score'}
        )
        fig_components.update_layout(showlegend=False)
        st.plotly_chart(fig_components, use_container_width=True)
        
        # Usage frequency analysis
        st.write("**User Usage Frequency**")
        usage_counts = df['usage_frequency'].value_counts()
        fig_usage = px.pie(
            values=usage_counts.values,
            names=usage_counts.index,
            title="Distribution of Usage Frequency"
        )
        st.plotly_chart(fig_usage, use_container_width=True)
        
        # Recent feedback
        st.write("**Recent Feedback Comments**")
        recent_feedback = df[df['additional_feedback'].notna() & 
                           (df['additional_feedback'] != '')].tail(5)
        
        if not recent_feedback.empty:
            for _, row in recent_feedback.iterrows():
                with st.expander(f"Feedback #{row['response_id']} (Score: {row['satisfaction_score']}/5)"):
                    st.write(f"**Relevance:** {row['relevance']}")
                    st.write(f"**Accuracy:** {row['accuracy']}/5")
                    st.write(f"**Diversity:** {row['diversity']}")
                    st.write(f"**Comment:** {row['additional_feedback']}")
        else:
            st.info("No text feedback available yet.")
    
    def export_survey_data(self, format='csv'):
        """
        Export survey data for analysis
        """
        data = self.load_responses()
        
        if not data['responses']:
            st.warning("No survey data to export.")
            return None
        
        df = pd.DataFrame(data['responses'])
        
        if format == 'csv':
            csv = df.to_csv(index=False)
            return csv
        elif format == 'json':
            return json.dumps(data, indent=2)
        else:
            return df
    
    def get_satisfaction_metrics(self):
        """
        Get comprehensive satisfaction metrics
        """
        try:
            data = self.load_responses()
            
            if not data['responses']:
                return {
                    'total_responses': 0,
                    'average_satisfaction': 0.0,
                    'satisfaction_breakdown': {},
                    'component_scores': {},
                    'trends': {}
                }
            
            df = pd.DataFrame(data['responses'])
            
            # Check if required columns exist
            required_columns = ['satisfaction_score', 'relevance_score', 'accuracy_score', 'diversity_score', 'usage_frequency']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: Missing columns in survey data: {missing_columns}")
                return {
                    'total_responses': data['total_responses'],
                    'average_satisfaction': data['average_satisfaction'],
                    'satisfaction_breakdown': {},
                    'component_scores': {},
                    'usage_analysis': {},
                    'last_updated': data['last_updated']
                }
            
            # Satisfaction breakdown
            satisfaction_breakdown = {
                'high_satisfaction': len(df[df['satisfaction_score'] >= 4]),
                'medium_satisfaction': len(df[(df['satisfaction_score'] >= 2) & (df['satisfaction_score'] < 4)]),
                'low_satisfaction': len(df[df['satisfaction_score'] < 2])
            }
            
            # Component scores
            component_scores = {
                'relevance': df['relevance_score'].mean(),
                'accuracy': df['accuracy_score'].mean(),
                'diversity': df['diversity_score'].mean()
            }
            
            # Usage frequency analysis
            usage_analysis = df['usage_frequency'].value_counts().to_dict()
            
            return {
                'total_responses': data['total_responses'],
                'average_satisfaction': data['average_satisfaction'],
                'satisfaction_breakdown': satisfaction_breakdown,
                'component_scores': component_scores,
                'usage_analysis': usage_analysis,
                'last_updated': data['last_updated']
            }
        except Exception as e:
            print(f"Error in get_satisfaction_metrics: {e}")
            return {
                'total_responses': 0,
                'average_satisfaction': 0.0,
                'satisfaction_breakdown': {},
                'component_scores': {},
                'usage_analysis': {},
                'last_updated': datetime.now().isoformat()
            }

def main():
    """
    Main function to run the survey system
    """
    st.title("User Satisfaction Survey System")
    
    survey = UserSatisfactionSurvey()
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Take Survey", "View Analytics", "Export Data"]
    )
    
    if page == "Take Survey":
        st.header("📝 Book Recommendation Feedback Survey")
        
        # Sample recommendations for demonstration
        sample_recommendations = [
            {'title': 'The Great Gatsby', 'authors': 'F. Scott Fitzgerald'},
            {'title': 'To Kill a Mockingbird', 'authors': 'Harper Lee'},
            {'title': '1984', 'authors': 'George Orwell'},
            {'title': 'Pride and Prejudice', 'authors': 'Jane Austen'},
            {'title': 'The Hobbit', 'authors': 'J.R.R. Tolkien'}
        ]
        
        response = survey.show_satisfaction_survey(sample_recommendations)
        
        if response:
            st.balloons()
    
    elif page == "View Analytics":
        st.header("📊 Survey Analytics Dashboard")
        survey.show_survey_analytics()
    
    elif page == "Export Data":
        st.header("📤 Export Survey Data")
        
        export_format = st.selectbox("Export Format", ["CSV", "JSON"])
        
        if st.button("Export Data"):
            data = survey.export_survey_data(export_format.lower())
            
            if data is not None:
                if export_format == "CSV":
                    st.download_button(
                        label="Download CSV",
                        data=data,
                        file_name=f"survey_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.download_button(
                        label="Download JSON",
                        data=data,
                        file_name=f"survey_data_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )

if __name__ == "__main__":
    main()
