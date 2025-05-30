import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Fertilizer Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        color: #2E7D32;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        color: #388E3C;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #C8E6C9;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    .feature-card {
        background-color: #F1F8E9;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        border-top: 1px solid #C8E6C9;
        color: #757575;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_model():
    try:
        # Try to load the saved model
        model = joblib.load('e:/Intern/Final Submission/model 2/fertilizer_model.pkl')
        return model
    except:
        # If model doesn't exist, train a new one
        # Load the dataset
        df = pd.read_csv(r'E:\Intern\Final Submission\model 2\Fertilizer Prediction.csv')
        
        # Prepare features and target
        X = df.drop('Fertilizer Name', axis=1)
        y = df['Fertilizer Name']
        
        # Encode categorical variables
        soil_type_mapping = {"Sandy": 0, "Loamy": 1, "Black": 2, "Red": 3, "Clayey": 4}
        crop_type_mapping = {"Wheat": 0, "Maize": 1, "Cotton": 2, "Rice": 3, "Sugarcane": 4, 
                            "Tobacco": 5, "Millets": 6, "Oil seeds": 7, "Pulses": 8, "Ground Nuts": 9}
        
        X['Soil Type'] = X['Soil Type'].map(soil_type_mapping)
        X['Crop Type'] = X['Crop Type'].map(crop_type_mapping)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(model, 'e:/Intern/Final Submission/model 2/fertilizer_model.pkl')
        
        return model

# Function to load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r'E:\Intern\Final Submission\model 2\Fertilizer Prediction.csv')
        return df
    except:
        st.error("Dataset not found. Please make sure 'Fertilizer.csv' is in the correct location.")
        return None

# Home page function
def home():
    st.markdown("<h1 class='main-header'>Fertilizer Recommendation System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Welcome to the Fertilizer Recommendation System! This application helps farmers and gardeners 
    determine the most suitable fertilizer based on soil conditions and crop requirements.
    </div>
    """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("<h2 class='sub-header'>Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
        <h3>Soil Analysis</h3>
        <p>Input your soil parameters to get personalized fertilizer recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
        <h3>Data Visualization</h3>
        <p>Explore the relationships between soil parameters and fertilizer types.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
        <h3>Expert Insights</h3>
        <p>Get detailed information about recommended fertilizers and their benefits.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works section
    st.markdown("<h2 class='sub-header'>How It Works</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <ol>
        <li>Input your soil parameters (Nitrogen, Phosphorus, Potassium, etc.)</li>
        <li>Our machine learning model analyzes the data</li>
        <li>Receive personalized fertilizer recommendations</li>
        <li>View detailed information about the recommended fertilizers</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional information section
    st.markdown("""
    <div class='info-box'>
    <h3>Why Fertilizer Selection Matters</h3>
    <p>Choosing the right fertilizer is crucial for optimal plant growth and crop yield. 
    The right balance of nutrients can significantly improve soil health and plant productivity.</p>
    </div>
    """, unsafe_allow_html=True)

# Recommendation page function
def recommendation():
    st.markdown("<h1 class='main-header'>Get Fertilizer Recommendations</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Enter your soil parameters below to receive personalized fertilizer recommendations.
    </div>
    """, unsafe_allow_html=True)
    
    # Load the model
    model = load_model()
    
    # Create input form
    with st.form("soil_parameters_form"):
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
            moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=30.0)
            soil_type = st.selectbox("Soil Type", 
                                    ["Sandy", "Loamy", "Black", "Red", "Clayey"])
        
        with col2:
            nitrogen = st.number_input("Nitrogen (kg/ha)", min_value=0, max_value=200, value=50)
            phosphorus = st.number_input("Phosphorus (kg/ha)", min_value=0, max_value=200, value=50)
            potassium = st.number_input("Potassium (kg/ha)", min_value=0, max_value=200, value=50)
            crop_type = st.selectbox("Crop Type", 
                                    ["Wheat", "Maize", "Cotton", "Rice", "Sugarcane", "Tobacco", "Millets", "Oil seeds", "Pulses", "Ground Nuts"])
        
        # Submit button
        submitted = st.form_submit_button("Get Recommendation")
    
    # Process form submission
    if submitted:
        # Encode categorical variables
        soil_type_mapping = {"Sandy": 0, "Loamy": 1, "Black": 2, "Red": 3, "Clayey": 4}
        crop_type_mapping = {"Wheat": 0, "Maize": 1, "Cotton": 2, "Rice": 3, "Sugarcane": 4, 
                            "Tobacco": 5, "Millets": 6, "Oil seeds": 7, "Pulses": 8, "Ground Nuts": 9}
        
        soil_type_encoded = soil_type_mapping[soil_type]
        crop_type_encoded = crop_type_mapping[crop_type]
        
        # Create input array for prediction
        input_data = np.array([[temperature, humidity, moisture, soil_type_encoded, 
                               crop_type_encoded, nitrogen, phosphorus, potassium]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display recommendation
        st.markdown("<h2 class='sub-header'>Recommendation Result</h2>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='feature-card'>
        <h3>Recommended Fertilizer:</h3>
        <p style='font-size: 1.5rem; font-weight: 600; color: #2E7D32;'>{prediction}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Fertilizer information
        fertilizer_info = {
            "Urea": {
                "description": "Urea is a high-nitrogen fertilizer that's commonly used for crops that need a lot of nitrogen.",
                "npk": "46-0-0",
                "best_for": "Leafy vegetables, corn, rice",
                "application": "Apply 2-3 weeks before planting or during active growth"
            },
            "DAP": {
                "description": "Diammonium Phosphate (DAP) is high in phosphorus and also contains nitrogen.",
                "npk": "18-46-0",
                "best_for": "Root vegetables, flowering plants",
                "application": "Apply at planting time or early in the growing season"
            },
            "14-35-14": {
                "description": "A balanced NPK fertilizer with higher phosphorus content.",
                "npk": "14-35-14",
                "best_for": "General purpose, good for flowering and fruiting",
                "application": "Apply during planting and throughout the growing season"
            },
            "28-28": {
                "description": "A balanced nitrogen and phosphorus fertilizer.",
                "npk": "28-28-0",
                "best_for": "Early growth stages of most crops",
                "application": "Apply during planting and early growth stages"
            },
            "17-17-17": {
                "description": "A perfectly balanced NPK fertilizer for general use.",
                "npk": "17-17-17",
                "best_for": "All-purpose fertilizer for most crops",
                "application": "Apply throughout the growing season"
            },
            "20-20": {
                "description": "A balanced nitrogen and phosphorus fertilizer.",
                "npk": "20-20-0",
                "best_for": "Vegetative growth and root development",
                "application": "Apply during planting and early growth stages"
            },
            "10-26-26": {
                "description": "Low nitrogen, high phosphorus and potassium fertilizer.",
                "npk": "10-26-26",
                "best_for": "Flowering and fruiting stages",
                "application": "Apply during flowering and fruiting stages"
            }
        }
        
        # Display fertilizer information if available
        if prediction in fertilizer_info:
            info = fertilizer_info[prediction]
            
            st.markdown("<h2 class='sub-header'>Fertilizer Information</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class='feature-card'>
                <h3>Description</h3>
                <p>{info['description']}</p>
                <h3>NPK Ratio</h3>
                <p>{info['npk']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='feature-card'>
                <h3>Best For</h3>
                <p>{info['best_for']}</p>
                <h3>Application Timing</h3>
                <p>{info['application']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Application tips
        st.markdown("<h2 class='sub-header'>Application Tips</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        <h3>General Tips for Fertilizer Application:</h3>
        <ul>
            <li>Always follow the manufacturer's recommended application rates</li>
            <li>Apply fertilizers in the early morning or late evening to reduce evaporation</li>
            <li>Water the soil after applying fertilizer to help nutrients reach the roots</li>
            <li>Avoid applying fertilizers before heavy rain to prevent runoff</li>
            <li>Use protective gear when handling chemical fertilizers</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Data visualization page function
def data_visualization():
    st.markdown("<h1 class='main-header'>Data Visualization</h1>", unsafe_allow_html=True)
    
    # Load the dataset
    df = load_data()
    
    if df is not None:
        st.markdown("""
        <div class='info-box'>
        Explore the relationships between soil parameters and fertilizer recommendations through interactive visualizations.
        </div>
        """, unsafe_allow_html=True)
        
        # Display dataset overview
        st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
        
        # Show basic statistics
        st.dataframe(df.describe())
        
        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Fertilizer Distribution", "Parameter Relationships", "Soil & Crop Analysis"])
        
        with viz_tab1:
            st.markdown("<h3>Fertilizer Distribution</h3>", unsafe_allow_html=True)
            
            # Create fertilizer count plot
            fig = px.histogram(df, x='Fertilizer Name', title='Distribution of Recommended Fertilizers',
                              color='Fertilizer Name', color_discrete_sequence=px.colors.qualitative.Set3)
            
            fig.update_layout(xaxis_title='Fertilizer Type', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
            
            # NPK distribution by fertilizer
            st.markdown("<h3>NPK Distribution by Fertilizer Type</h3>", unsafe_allow_html=True)
            
            # Calculate average NPK values for each fertilizer
            npk_by_fertilizer = df.groupby('Fertilizer Name')[['Nitrogen', 'Phosphorous', 'Potassium']].mean().reset_index()
            
            # Create grouped bar chart
            fig = px.bar(npk_by_fertilizer, x='Fertilizer Name', y=['Nitrogen', 'Phosphorous', 'Potassium'],
                        title='Average NPK Values by Fertilizer Type',
                        barmode='group', color_discrete_sequence=['#4CAF50', '#2196F3', '#FFC107'])
            
            fig.update_layout(xaxis_title='Fertilizer Type', yaxis_title='Average Value (kg/ha)')
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            st.markdown("<h3>Parameter Relationships</h3>", unsafe_allow_html=True)
            
            # Select parameters to visualize
            col1, col2 = st.columns(2)
            
            with col1:
                x_param = st.selectbox("Select X-axis parameter:", 
                                      df.columns.drop('Fertilizer Name'), index=0)
            
            with col2:
                y_param = st.selectbox("Select Y-axis parameter:", 
                                      df.columns.drop('Fertilizer Name'), index=1)
            
            # Create scatter plot
            fig = px.scatter(df, x=x_param, y=y_param, color='Fertilizer Name',
                            title=f'Relationship between {x_param} and {y_param}',
                            color_discrete_sequence=px.colors.qualitative.Set3)
            
            fig.update_layout(xaxis_title=x_param, yaxis_title=y_param)
            # Correlation heatmap
            st.markdown("<h3>Parameter Correlation</h3>", unsafe_allow_html=True)
            
            # Calculate correlation matrix
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            
            # Create heatmap
            fig = px.imshow(corr, text_auto=True, aspect="auto",
                           title="Parameter Correlation Heatmap",
                           color_continuous_scale='Greens')
            
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab3:
            st.markdown("<h3>Soil & Crop Analysis</h3>", unsafe_allow_html=True)
            
            # Soil type distribution
            soil_counts = df['Soil Type'].value_counts().reset_index()
            soil_counts.columns = ['Soil Type', 'Count']
            
            fig = px.pie(soil_counts, values='Count', names='Soil Type',
                        title='Distribution of Soil Types',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Crop type distribution
            crop_counts = df['Crop Type'].value_counts().reset_index()
            crop_counts.columns = ['Crop Type', 'Count']
            
            fig = px.pie(crop_counts, values='Count', names='Crop Type',
                        title='Distribution of Crop Types',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Fertilizer recommendation by soil and crop type
            st.markdown("<h3>Fertilizer Recommendations by Soil and Crop Type</h3>", unsafe_allow_html=True)
            
            # Create a crosstab of soil type, crop type, and fertilizer
            cross_tab = pd.crosstab([df['Soil Type'], df['Crop Type']], df['Fertilizer Name'])
            
            # Display the crosstab
            st.dataframe(cross_tab)

# About page function
def about():
    st.markdown("<h1 class='main-header'>About This Project</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    The Fertilizer Recommendation System is an intelligent application designed to help farmers 
    make informed decisions about which fertilizers to use based on soil conditions and crop requirements.
    By leveraging machine learning, this system analyzes various factors such as soil nutrient content (N, P, K),
    soil type, crop type, and environmental conditions to recommend the most suitable fertilizer.
    </div>
    """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("<h2 class='sub-header'>How It Works</h2>", unsafe_allow_html=True)
    
    # Display information in a single column for better readability
    st.markdown("""
        <div class='feature-card'>
        <h4>Data Collection</h4>
        <p>The system is trained on a dataset containing information about various soil parameters, crop types, and suitable fertilizers.</p>
        
        <h4>Machine Learning Model</h4>
        <p>A Decision Tree classifier is used to learn patterns and relationships between soil/crop parameters and suitable fertilizers.</p>
        
        <h4>Prediction</h4>
        <p>When users input their specific soil and crop parameters, the model predicts the most suitable fertilizer for those conditions.</p>
        
        <h4>Visualization</h4>
        <p>The system provides detailed visualizations and information to help users understand the recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Technologies used
    st.markdown("<h2 class='sub-header'>Technologies Used</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
        <h4>Data Science & ML</h4>
        <ul>
            <li>Python</li>
            <li>Pandas & NumPy</li>
            <li>Scikit-learn</li>
            <li>Joblib</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
        <h4>Web Application</h4>
        <ul>
            <li>Streamlit</li>
            <li>HTML/CSS</li>
            <li>Plotly & Matplotlib</li>
            <li>Seaborn</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
        <h4>Development Tools</h4>
        <ul>
            <li>Git & GitHub</li>
            <li>VS Code</li>
            <li>Jupyter Notebook</li>
        </ul>
        </div>
""", unsafe_allow_html=True)

def main():
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    pages = {
        "Home": "🏠",
        "Get Recommendation": "🌱",
        "Data Visualization": "📊",
        "About": "ℹ️"
    }
    
    # Create sidebar navigation with icons
    selection = st.sidebar.radio(
        "Go to",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]} {x}"
    )
    
    # Display the selected page
    if selection == "Home":
        home()
    elif selection == "Get Recommendation":
        recommendation()
    elif selection == "Data Visualization":
        data_visualization()
    elif selection == "About":
        about()

    # Footer
    st.markdown("""
    <div class='footer'>
        <p>&copy; 2023 Fertilizer Recommendation System. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()