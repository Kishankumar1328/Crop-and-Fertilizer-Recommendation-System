import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import warnings
import plotly.express as px
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings('ignore')

# STEP 2: Create a class for our RAG-based crop bot
class CropRecommendationRAG:
    def __init__(self):
        # Load pre-trained embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.crop_data = None
        self.embeddings = None

    # STEP 3: Load data and prepare combined text
    def load_data(self):
        try:
            csv_path = r"E:\Intern\project\Crop_recommendation (1).csv"
            self.crop_data = pd.read_csv(csv_path)

            # Combine all relevant columns into one text field
            self.crop_data['combined_text'] = self.crop_data.apply(
                lambda row: f"N: {row['N']}, P: {row['P']}, K: {row['K']}, Temp: {row['temperature']}, "
                            f"Humidity: {row['humidity']}, pH: {row['ph']}, Rainfall: {row['rainfall']}, "
                            f"Recommended Crop: {row['label']}",
                axis=1
            )

            self.create_embeddings()
        except Exception as e:
            st.error(f"Error loading crop data: {str(e)}")

    # STEP 4: Generate or load embeddings
    def create_embeddings(self):
        embedding_file = "crop_embeddings.npy"

        if os.path.exists(embedding_file):
            self.embeddings = torch.tensor(np.load(embedding_file))
        else:
            texts = self.crop_data['combined_text'].tolist()
            self.embeddings = self.model.encode(texts, convert_to_tensor=True)
            np.save(embedding_file, self.embeddings.numpy())

    # STEP 5: Get top matching crops from user query
    def find_best_crop(self, query, top_n=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(cosine_scores, k=top_n)

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            data = self.crop_data.iloc[idx.item()].to_dict()
            results.append({
                "recommended_crop": data['label'],
                "similarity_score": float(score.item()),
                "parameters": {
                    "N": data['N'], "P": data['P'], "K": data['K'],
                    "temperature": data['temperature'],
                    "humidity": data['humidity'],
                    "ph": data['ph'],
                    "rainfall": data['rainfall']
                }
            })
        return results
    
    # Get crop distribution data for visualization
    def get_crop_distribution(self):
        return self.crop_data['label'].value_counts().reset_index()

# STEP 6: Cache the bot so it's not reloaded every time
@st.cache_resource
def load_crop_bot():
    bot = CropRecommendationRAG()
    bot.load_data()
    return bot

# Custom CSS for better styling
def apply_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388E3C;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .crop-card {
        background-color: #F1F8E9;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #7CB342;
    }
    .crop-name {
        font-size: 1.3rem;
        font-weight: 600;
        color: #33691E;
    }
    .score-badge {
        background-color: #C8E6C9;
        padding: 0.3rem 0.6rem;
        border-radius: 20px;
        font-weight: 500;
        color: #1B5E20;
    }
    .parameter-section {
        margin-top: 0.8rem;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
    }
    .parameter {
        display: flex;
        align-items: center;
    }
    .parameter-icon {
        margin-right: 0.5rem;
        color: #558B2F;
    }
    .info-box {
        background-color: #E8F5E9;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border: 1px solid #A5D6A7;
    }
    </style>
    """, unsafe_allow_html=True)

# STEP 7: Build the Streamlit UI
def main():
    st.set_page_config(
        page_title="🌾 AI Crop Recommendation",
        page_icon="🌱",
        layout="wide"
    )
    
    apply_custom_css()

    st.markdown('<div class="main-header">🌱 AI-Powered Crop Recommendation System</div>', unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Crop Recommendation", "Data Insights", "About"])
    
    chatbot = load_crop_bot()
    
    with tab1:
        st.markdown('<div class="info-box">Enter details about your soil conditions, weather, or nutrient levels to get personalized crop recommendations.</div>', unsafe_allow_html=True)
        
        # Create two columns for input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_area("🧪 Describe your conditions:", 
                               placeholder="Example: high nitrogen, low pH, moderate rainfall, warm temperature",
                               height=100)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            top_n = st.slider("Number of recommendations:", min_value=1, max_value=5, value=3)
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.button("Get Recommendations", type="primary", use_container_width=True)
        
        if submit_button and query:
            with st.spinner("Analyzing your conditions..."):
                recommendations = chatbot.find_best_crop(query, top_n=top_n)
            
            st.markdown('<div class="sub-header">✅ Best Matching Crops</div>', unsafe_allow_html=True)
            
            for i, rec in enumerate(recommendations):
                crop_name = rec['recommended_crop'].title()
                score = rec['similarity_score']
                params = rec['parameters']
                
                st.markdown(f"""
                <div class="crop-card">
                    <div class="crop-name">🌿 {i+1}. {crop_name} <span class="score-badge">Match: {score:.2f}</span></div>
                    <div class="parameter-section">
                        <div class="parameter"><span class="parameter-icon">🧪</span> N: {params['N']} kg/ha</div>
                        <div class="parameter"><span class="parameter-icon">🌡️</span> Temp: {params['temperature']}°C</div>
                        <div class="parameter"><span class="parameter-icon">🧪</span> P: {params['P']} kg/ha</div>
                        <div class="parameter"><span class="parameter-icon">💧</span> Humidity: {params['humidity']}%</div>
                        <div class="parameter"><span class="parameter-icon">🧪</span> K: {params['K']} kg/ha</div>
                        <div class="parameter"><span class="parameter-icon">🌧️</span> Rainfall: {params['rainfall']} mm</div>
                        <div class="parameter"><span class="parameter-icon">⚗️</span> pH: {params['ph']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="sub-header">📊 Crop Distribution in Dataset</div>', unsafe_allow_html=True)
        
        # Create a bar chart of crop distribution
        crop_counts = chatbot.get_crop_distribution()
        crop_counts.columns = ['Crop', 'Count']
        
        fig = px.bar(
            crop_counts, 
            x='Crop', 
            y='Count',
            title='Distribution of Crops in Dataset',
            color='Count',
            color_continuous_scale='Greens'
        )
        fig.update_layout(xaxis_title='Crop Type', yaxis_title='Number of Samples')
        st.plotly_chart(fig, use_container_width=True)
        
        # Add some insights
        st.markdown("""
        **Insights:**
        - The dataset contains a balanced distribution of different crops
        - This ensures the model is trained without bias towards any particular crop
        - Each crop has sufficient samples for the model to learn its specific requirements
        """)
        
        # Add NPK visualization
        st.markdown('<div class="sub-header">🧪 NPK Requirements by Crop</div>', unsafe_allow_html=True)
        
        # Get average NPK values by crop
        npk_data = chatbot.crop_data.groupby('label')[['N', 'P', 'K']].mean().reset_index()
        npk_melted = pd.melt(npk_data, id_vars=['label'], value_vars=['N', 'P', 'K'], 
                             var_name='Nutrient', value_name='Value')
        
        fig2 = px.bar(
            npk_melted, 
            x='label', 
            y='Value', 
            color='Nutrient',
            barmode='group',
            title='Average NPK Requirements by Crop',
            labels={'label': 'Crop', 'Value': 'Amount (kg/ha)'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="sub-header">About This System</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🌱 AI-Powered Crop Recommendation System
        
        This intelligent system helps farmers and gardeners make informed decisions about which crops to plant based on soil conditions and environmental factors. Using advanced AI techniques:
        
        - **Semantic Search**: The system uses a pre-trained language model to understand the meaning behind your query
        - **Embedding Vectors**: Your input is converted into a mathematical representation that captures its meaning
        - **Similarity Matching**: The system finds the closest matches in our agricultural database
        
        ### How It Works
        
        1. The system uses a Sentence Transformer model to understand your input
        2. It compares your description with thousands of agricultural data points
        3. It identifies the crops that best match your specific conditions
        4. It provides detailed recommendations with confidence scores
        
        ### Benefits
        
        - **Personalized Recommendations**: Get crop suggestions tailored to your specific conditions
        - **Data-Driven Decisions**: Make farming choices based on scientific data
        - **Improved Yield**: Plant crops that are most likely to thrive in your environment
        - **Sustainable Farming**: Optimize resource usage by planting suitable crops
        
        ### Data Source
        
        The system is trained on a comprehensive agricultural dataset containing information about various crops and their optimal growing conditions, including soil nutrients, pH levels, temperature, humidity, and rainfall requirements.
        """)

# STEP 8: Run the app
if __name__ == "__main__":
    main()