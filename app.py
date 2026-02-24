import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="User Retention Predictor", page_icon="🌎", layout="wide")

@st.cache_resource
def load_model_file():
    try:
        with open('retention_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model_data = load_model_file()

st.sidebar.title("Menu")
choice = st.sidebar.radio("Navigation", 
    ["Introduction", "Data Loading", "Data Exploration", "Visualization", "Machine Learning"])

if 'df' not in st.session_state:
    st.session_state['df'] = None

if choice == "Introduction":
    st.title("Retention Prediction Model: Digital Stickiness Auditor")
    
    st.markdown(f"""
    This project provides a cross-sectional snapshot of the world's most influential websites as of **February 2026**. 
    By merging traditional traffic rankings (Tranco/Majestic) with behavioral engagement metrics, this ML Model provides a 
    holistic view of **"Digital Stickiness."**
    """)
    
    st.divider()

    st.header(" Data Dictionary")
    st.markdown("Understanding the metrics used to train our Machine Learning model:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        * **Domain:** The website URL.
        * **Global Rank:** Position based on total traffic .

        * **Monthly Visits:** Total 30-day traffic .
        * **Avg Session Duration (s):** Time spent per visit.
        * **Bounce Rate (%):** Percentage of users who leave after one page.
        """)

    with col2:
        st.markdown("""
        * **Category:** Industry type (e.g., Streaming, Tech, Finance).
        * **Primary Market:** Top geographic region for traffic.

        * **Stickiness Index:** Our custom score for user engagement.
        * **Retention Status:** Classified as **High** or **Low** based on the global median.
        """)

    st.divider()

    tab1, tab2 = st.tabs(["Cliff", "ML Logic"])
    
    with tab1:
        st.write("Our analysis discovered a 'cliff' in user behavior: websites that keep their **Bounce Rate under 30%** see a massive jump in Stickiness, especially in the Streaming sector.")
    
    with tab2:
        st.write("We use a **Random Forest Classifier** with 97.67% accuracy. It doesn't just look at one number; it looks at how Category, Duration, and Bounce Rate interact to predict if a site is 'Sticky'.")

    st.info(" Use the sidebar to upload your data and start the analysis!")

if choice == "Data Loading":
    st.title("Dataset Insertion")
    st.markdown("Upload your `globalwebtraffic.csv` file to start the Retention AI model.")
    data_file = st.file_uploader("Choose CSV file", type=["csv"])
    
    if data_file:
        df = pd.read_csv(data_file)
        st.session_state['df'] = df
        st.success("Data loaded successfully!")
        st.dataframe(df.head(10))

elif choice == "Data Exploration":
    st.title(" Data Exploration")
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Missing Values")
            st.write(df.isnull().sum())
        with col2:
            st.subheader("Data Types")
            st.write(df.dtypes)
        st.subheader("Statistical Summary")
        st.write(df.describe())
    else:
        st.warning("Please upload a dataset first.")

elif choice == "Visualization":
    st.title(" Retention Insights")
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        tab1, tab2, tab3, tab4 = st.tabs(["Feature Heatmap", "Retention Analysis", "Model Accuracy", "Model Evaluation Curves"])
        
        with tab1:
            numeric_df = df.select_dtypes(include=[np.number])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='RdYlGn', ax=ax)
            st.pyplot(fig)
            
        with tab2:
            # Show how Bounce Rate affects Stickiness
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='bounce_rate_pct', y='stickiness_index', hue='category', ax=ax)
            st.subheader("Correlation: Bounce Rate vs Stickiness")
            st.pyplot(fig)
            
        with tab3: 
            st.subheader("Confusion Matrix")
            try:
                st.image("model_performance.png", caption="AI accuracy in identifying user retention patterns.")
            except FileNotFoundError:
                st.warning("Performance plot not found. Run train_model.py first.")
            
        with tab4: 
            st.subheader("Model Evaluation Curves")
            try:
                st.image("learning_curves.png", caption="Analysis of Model Learning vs. Overfitting")
                st.markdown("""
                - **Accuracy Curve:** Shows how well the model identifies retention as it gets more 'experience'.
                - **Error Curve:** We want to see this dropping. If the test error starts rising again, the model is overfitting.
                """)
            except FileNotFoundError:
                st.warning("Learning curves not found. Run the updated train_model.py.")

elif choice == "Machine Learning":
    st.title("User Retention Auditor")
    
    if st.session_state['df'] is None:
        st.warning("Access Denied: Please upload a dataset in 'Data Loading' first.")
    elif model_data is None:
        st.error("File 'retention_model.pkl' not found.")
    else:
        st.markdown("### Input Website Metrics to Predict User Loyalty")
        
        col1, col2 = st.columns(2)
        with col1:
            bounce = st.slider("Bounce Rate (%)", 0.0, 100.0, 50.0)
            duration = st.number_input("Avg Session Duration (s)", 0.0, 5000.0, 300.0)
            visits = st.number_input("Monthly Visits", 0, 10000000, 50000)
            
        with col2:
            cat = st.selectbox("Category", model_data['categories'])
            market = st.selectbox("Primary Market", model_data['markets'])
            
        st.divider()

        if st.button("Analyse Retention Potential", type="primary"):
            log_dur = np.log1p(duration)
            log_visits = np.log1p(visits)
            
            input_row = pd.DataFrame([[bounce, log_dur, log_visits, cat, market]], 
                columns=['bounce_rate_pct', 'log_avg_session_duration_s', 'log_monthly_visits', 'category', 'primary_market'])
            
            prediction = model_data['pipeline'].predict(input_row)[0]
            probs = model_data['pipeline'].predict_proba(input_row)[0]
            confidence = max(probs) * 100

            st.subheader("Analysis Results:")
            if prediction == "High Retention":
                st.success(f" Result: {prediction}")
                st.balloons()
            else:
                st.error(f" Result: {prediction}")
                
            st.write(f"**AI model Confidence Score:** {confidence:.1f}%")
            
            if bounce > 70:
                st.warning("the high bounce rate is likely the  reason for low retention.")
            elif duration < 60:
                st.warning(" Users aren't staying long enough to engage in the websites. Improve your content depth.")