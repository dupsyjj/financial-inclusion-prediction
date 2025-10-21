import pandas as pd
import numpy as np
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data for context
data = pd.read_csv('Financial_inclusion_dataset (2).csv')

# ----------------- HEADER -----------------
st.markdown("<h1 style='color:#114232; text-align:center; font-size:60px; font-family:Monospace'>FINANCIAL INCLUSION APP</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='margin:-30px; color:#87A922; text-align:center; font-family:cursive'>Built by Modupe Oshinjirin</h4>", unsafe_allow_html=True)
st.image('pngwing.com (4).png')

# ----------------- BACKGROUND -----------------
st.markdown("<h2 style='color:#FF9800; text-align:center; font-family:montserrat'>Background Of Study</h2>", unsafe_allow_html=True)
st.markdown("""
Financial inclusion aims to ensure that individuals and businesses have access to useful and affordable financial products and services.
This project seeks to predict whether a person is likely to have a bank account, based on socioeconomic and demographic factors such as income level,
education, gender, and marital status. By predicting financial inclusion levels, stakeholders can design better policies to improve access to finance.
""")

st.sidebar.image('pngwing.com (3).png')

# ----------------- DATA DISPLAY -----------------
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width=True)

# ----------------- SIDEBAR INPUT -----------------
st.sidebar.header("🔹 Customer Information")

Country = st.sidebar.selectbox('Country', data['country'].unique())
Year = st.sidebar.number_input('Year', int(data['year'].min()), int(data['year'].max()))
Location_Type = st.sidebar.selectbox('Location Type', data['location_type'].unique())
Cellphone_access = st.sidebar.selectbox('Cellphone Access', data['cellphone_access'].unique())
Household_Size = st.sidebar.number_input('Household Size', int(data['household_size'].min()), int(data['household_size'].max()))
Age = st.sidebar.number_input('Age of Respondent', int(data['age_of_respondent'].min()), int(data['age_of_respondent'].max()))
Gender = st.sidebar.selectbox('Gender of Respondent', data['gender_of_respondent'].unique())
Relationship_With_Head = st.sidebar.selectbox('Relationship with Head', data['relationship_with_head'].unique())
Marital_Status = st.sidebar.selectbox('Marital Status', data['marital_status'].unique())
Education_Level = st.sidebar.selectbox('Education Level', data['education_level'].unique())
Job_Type = st.sidebar.selectbox('Job Type', data['job_type'].unique())

# ----------------- USER INPUT DATAFRAME -----------------
input_var = pd.DataFrame({
    'country': [Country],
    'year': [Year],
    'location_type': [Location_Type],
    'cellphone_access': [Cellphone_access],
    'household_size': [Household_Size],
    'age_of_respondent': [Age],
    'gender_of_respondent': [Gender],
    'relationship_with_head': [Relationship_With_Head],
    'marital_status': [Marital_Status],
    'education_level': [Education_Level],
    'job_type': [Job_Type]
})

st.divider()
st.subheader('User Input')
st.dataframe(input_var, use_container_width=True)

# ----------------- LOAD ENCODERS -----------------
country_enc = joblib.load('country_encoder.pkl')
location_enc = joblib.load('location_type_encoder.pkl')
cellphone_enc = joblib.load('cellphone_access_encoder.pkl')
gender_enc = joblib.load('gender_of_respondent_encoder.pkl')
relation_enc = joblib.load('relationship_with_head_encoder.pkl')
marital_enc = joblib.load('marital_status_encoder.pkl')
education_enc = joblib.load('education_level_encoder.pkl')
job_enc = joblib.load('job_type_encoder.pkl')

# ----------------- LOAD MODEL -----------------
model = joblib.load('financialmodel.pkl')  # replace with your actual model filename

# ----------------- BUTTON AND PREDICTION -----------------
if st.button('🔍 Check Bank Account Status'):
    # Encode categorical values safely
    df = input_var.copy()
    
    try:
        df['country'] = country_enc.transform(df['country'])
        df['location_type'] = location_enc.transform(df['location_type'])
        df['cellphone_access'] = cellphone_enc.transform(df['cellphone_access'])
        df['gender_of_respondent'] = gender_enc.transform(df['gender_of_respondent'])
        df['relationship_with_head'] = relation_enc.transform(df['relationship_with_head'])
        df['marital_status'] = marital_enc.transform(df['marital_status'])
        df['education_level'] = education_enc.transform(df['education_level'])
        df['job_type'] = job_enc.transform(df['job_type'])
    except Exception as e:
        st.error(f"Encoding Error: {e}")
    
    # Make prediction
    try:
        prediction = model.predict(df)[0]
        if prediction == 0:
            st.error("This customer is **not likely** to have a bank account.")
        else:
            st.success("This customer is **likely** to have a bank account.")
    except Exception as e:
        st.error(f"Prediction Error: {e}")