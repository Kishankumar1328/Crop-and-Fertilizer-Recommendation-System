import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388E3C;
    }
    .description {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>This application helps farmers determine the most suitable crop based on soil and climate conditions.</p>", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('e:/Intern/Final Submission/model/Crop/Crop_recommendation.csv')
    return data

# Load or train the model
@st.cache_resource
def load_model(data):
    # Prepare the data
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, scaler, accuracy, report, conf_matrix, X_test, y_test, y_pred

# Main function
def main():
    # Sidebar
    st.sidebar.markdown("<h2 class='sub-header'>Navigation</h2>", unsafe_allow_html=True)
    page = st.sidebar.radio("Go to", ["Home", "Predict", "Model Performance", "Data Exploration"])
    
    # Load data
    data = load_data()
    model, scaler, accuracy, report, conf_matrix, X_test, y_test, y_pred = load_model(data)
    
    if page == "Home":
        display_home(data)
    elif page == "Predict":
        predict_crop(model, scaler, data)
    elif page == "Model Performance":
        display_model_performance(accuracy, report, conf_matrix, y_test, y_pred)
    elif page == "Data Exploration":
        explore_data(data)

def display_home(data):
    st.markdown("<h2 class='sub-header'>Welcome to the Crop Recommendation System</h2>", unsafe_allow_html=True)
    
    # Display dataset overview
    st.markdown("<h3>Dataset Overview</h3>", unsafe_allow_html=True)
    st.write(f"Number of records: {data.shape[0]}")
    st.write(f"Number of features: {data.shape[1] - 1}")
    
    # Display unique crops
    unique_crops = data['label'].unique()
    st.markdown("<h3>Available Crops for Recommendation</h3>", unsafe_allow_html=True)
    
    # Create a more visually appealing display of crops
    cols = st.columns(3)
    for i, crop in enumerate(sorted(unique_crops)):
        cols[i % 3].markdown(f"- {crop}")
    
    # Show a sample of the dataset
    st.markdown("<h3>Sample Data</h3>", unsafe_allow_html=True)
    st.dataframe(data.sample(5))
    
    # Show a pie chart of crop distribution
    st.markdown("<h3>Crop Distribution</h3>", unsafe_allow_html=True)
    crop_counts = data['label'].value_counts()
    fig = px.pie(values=crop_counts.values, names=crop_counts.index, title='Distribution of Crops in Dataset')
    st.plotly_chart(fig)

def predict_crop(model, scaler, data):
    st.markdown("<h2 class='sub-header'>Crop Prediction</h2>", unsafe_allow_html=True)
    st.write("Enter the soil and climate parameters to get crop recommendations:")
    
    # Create columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        n = st.number_input("Nitrogen (N) content in soil", min_value=0, max_value=150, value=50)
        p = st.number_input("Phosphorus (P) content in soil", min_value=0, max_value=150, value=50)
        k = st.number_input("Potassium (K) content in soil", min_value=0, max_value=150, value=50)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    
    with col2:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
        ph = st.number_input("pH value of soil", min_value=0.0, max_value=14.0, value=7.0)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)
    
    # Prediction button
    if st.button("Predict Suitable Crop"):
        # Create input array for prediction
        input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Get top 3 predictions
        top_indices = probabilities.argsort()[-3:][::-1]
        top_crops = [model.classes_[i] for i in top_indices]
        top_probs = [probabilities[i] for i in top_indices]
        
        # Display the result
        st.success(f"The recommended crop is: **{prediction}**")
        
        # Display top 3 recommendations with probabilities
        st.markdown("<h3>Top 3 Recommendations</h3>", unsafe_allow_html=True)
        
        # Create a bar chart for top recommendations
        fig = go.Figure(data=[
            go.Bar(
                x=top_crops,
                y=top_probs,
                marker_color=['#2E7D32', '#388E3C', '#4CAF50']
            )
        ])
        fig.update_layout(
            title="Top Crop Recommendations",
            xaxis_title="Crop",
            yaxis_title="Confidence Score",
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig)
        
        # Display crop information
        st.markdown("<h3>Crop Information</h3>", unsafe_allow_html=True)
        
        # You can add more detailed information about the predicted crop here
        crop_info = {
            "rice": "Rice is a cereal grain that is the most widely consumed staple food for a large part of the world's human population, especially in Asia.",
            "maize": "Maize (corn) is a cereal grain that was domesticated in Mesoamerica. It is now the most widely grown grain crop throughout the Americas.",
            "chickpea": "Chickpea is a nutrient-dense legume that's high in protein, fiber, and various essential nutrients.",
            "kidneybeans": "Kidney beans are a variety of the common bean, named for their visual resemblance to kidneys. They are high in protein and fiber.",
            "pigeonpeas": "Pigeon peas are a legume crop grown in tropical and semi-tropical regions. They are high in protein and important in many diets.",
            "mothbeans": "Moth beans are small, drought-resistant legumes commonly grown in arid regions. They are high in protein and minerals.",
            "mungbean": "Mung beans are small, green legumes that are commonly used in Asian cuisine. They are high in protein, fiber, and antioxidants.",
            "blackgram": "Black gram is a bean grown in the Indian subcontinent and is highly prized for its nutritional value.",
            "lentil": "Lentils are edible legumes known for their lens-shaped seeds. They are high in protein and fiber.",
            "pomegranate": "Pomegranate is a fruit-bearing deciduous shrub. The fruit is rich in antioxidants and has numerous health benefits.",
            "banana": "Bananas are one of the most popular fruits worldwide. They are rich in potassium and provide good energy.",
            "mango": "Mango is a juicy stone fruit that is rich in vitamins, minerals, and antioxidants.",
            "grapes": "Grapes are a non-climacteric type of fruit, specifically a berry, that grow on the perennial and deciduous woody vines.",
            "watermelon": "Watermelon is a sweet and refreshing fruit that is high in water content and provides hydration.",
            "muskmelon": "Muskmelon is a species of melon that has a sweet and musky aroma. It is high in vitamins A and C.",
            "apple": "Apples are one of the most popular and versatile fruits. They are rich in fiber and antioxidants.",
            "orange": "Oranges are a citrus fruit known for their high vitamin C content and refreshing taste.",
            "papaya": "Papaya is a tropical fruit known for its sweet taste and soft, butter-like consistency.",
            "coconut": "Coconut is a versatile fruit that provides food, drink, and oil. It is rich in fiber and MCTs.",
            "cotton": "Cotton is a soft, fluffy staple fiber that grows in a boll around the seeds of the cotton plant.",
            "jute": "Jute is a long, soft, shiny vegetable fiber that can be spun into coarse, strong threads.",
            "coffee": "Coffee is a brewed drink prepared from roasted coffee beans, the seeds of berries from the Coffea plant."
        }
        
        if prediction.lower() in crop_info:
            st.info(crop_info[prediction.lower()])
        else:
            st.info("This crop is suitable for the given soil and climate conditions.")

def display_model_performance(accuracy, report, conf_matrix, y_test, y_pred):
    st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
    
    # Display accuracy
    st.markdown(f"<h3>Model Accuracy: {accuracy:.2%}</h3>", unsafe_allow_html=True)
    
    # Create columns for metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Display classification report
        st.markdown("<h3>Classification Report</h3>", unsafe_allow_html=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        # Create a bar chart for precision, recall, and f1-score
        classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
        precision = [report[cls]['precision'] for cls in classes]
        recall = [report[cls]['recall'] for cls in classes]
        f1 = [report[cls]['f1-score'] for cls in classes]
        
        # Only show top 10 classes for better visualization
        if len(classes) > 10:
            top_indices = np.argsort([report[cls]['f1-score'] for cls in classes])[-10:]
            classes = [classes[i] for i in top_indices]
            precision = [precision[i] for i in top_indices]
            recall = [recall[i] for i in top_indices]
            f1 = [f1[i] for i in top_indices]
        
        metrics_df = pd.DataFrame({
            'Class': classes,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        fig = px.bar(metrics_df, x='Class', y=['Precision', 'Recall', 'F1-Score'], 
                    title='Model Metrics by Class', barmode='group')
        st.plotly_chart(fig)
    
    with col2:
        # Display confusion matrix
        st.markdown("<h3>Confusion Matrix</h3>", unsafe_allow_html=True)
        
        # Create a heatmap for the confusion matrix
        unique_labels = sorted(set(y_test) | set(y_pred))
        if len(unique_labels) <= 10:  # Only show full matrix if it's not too large
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=unique_labels, yticklabels=unique_labels)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            st.pyplot(fig)
        else:
            st.write("Confusion matrix is too large to display fully. Showing a sample visualization.")
            # Create a simplified visualization
            fig = px.imshow(conf_matrix, labels=dict(x="Predicted", y="Actual"), 
                           title="Confusion Matrix Heatmap")
            st.plotly_chart(fig)

def explore_data(data):
    st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)
    
    # Display basic statistics
    st.markdown("<h3>Statistical Summary</h3>", unsafe_allow_html=True)
    st.dataframe(data.describe())
    
    # Feature correlation
    st.markdown("<h3>Feature Correlation</h3>", unsafe_allow_html=True)
    
    # Create correlation matrix
    corr = data.drop('label', axis=1).corr()
    
    # Plot correlation heatmap
    fig = px.imshow(corr, text_auto=True, aspect="auto", 
                   title="Feature Correlation Heatmap")
    st.plotly_chart(fig)
    
    # Feature distributions
    st.markdown("<h3>Feature Distributions</h3>", unsafe_allow_html=True)
    
    # Select feature to visualize
    feature = st.selectbox("Select a feature to visualize:", 
                          data.drop('label', axis=1).columns.tolist())
    
    # Create distribution plot
    fig = px.histogram(data, x=feature, color='label', 
                      title=f"Distribution of {feature} by Crop",
                      opacity=0.7)
    st.plotly_chart(fig)
    
    # Feature comparison
    st.markdown("<h3>Feature Comparison</h3>", unsafe_allow_html=True)
    
    # Select features to compare
    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Select X-axis feature:", 
                               data.drop('label', axis=1).columns.tolist(), index=0)
    with col2:
        y_feature = st.selectbox("Select Y-axis feature:", 
                               data.drop('label', axis=1).columns.tolist(), index=1)
    
    # Create scatter plot
    fig = px.scatter(data, x=x_feature, y=y_feature, color='label', 
                    title=f"{x_feature} vs {y_feature} by Crop",
                    opacity=0.7)
    st.plotly_chart(fig)
    
    # Crop-specific analysis
    st.markdown("<h3>Crop-Specific Analysis</h3>", unsafe_allow_html=True)
    
    # Select crop to analyze
    selected_crop = st.selectbox("Select a crop to analyze:", 
                               sorted(data['label'].unique()))
    
    # Filter data for selected crop
    crop_data = data[data['label'] == selected_crop]
    
    # Display crop statistics
    st.write(f"Statistics for {selected_crop}:")
    st.dataframe(crop_data.describe())
    
    # Create radar chart for crop requirements
    features = data.drop('label', axis=1).columns.tolist()
    
    # Calculate mean values for the selected crop
    crop_means = crop_data[features].mean()
    
    # Calculate overall means for comparison
    overall_means = data[features].mean()
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=crop_means.values,
        theta=features,
        fill='toself',
        name=f'{selected_crop}'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=overall_means.values,
        theta=features,
        fill='toself',
        name='Average of All Crops'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
            )),
        showlegend=True,
        title=f"Requirements for {selected_crop} vs Average"
    )
    
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()