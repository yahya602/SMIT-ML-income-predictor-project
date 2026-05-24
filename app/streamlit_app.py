import streamlit as st
import requests

st.set_page_config(page_title="Adult Income Predictor", page_icon="💰")
st.title("💰 Adult Income Predictor Application")
st.write("Provide demographics data to check if individual income exceeds **$50k/year**.")


# FIXED: Appended /predict to the endpoint URL to prevent KeyError
API_URL = "http://localhost:8000/predict"

 
# Categorical options extracted from the Adult dataset
workclass_opts = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
edu_opts = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
marital_opts = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occup_opts = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
relation_opts = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
race_opts = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
country_opts = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']

# Create columns for nice layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=17, max_value=90, value=30)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, value=150000)
    edu_num = st.slider("Years of Education", 1, 16, 10)
    cap_gain = st.number_input("Capital Gain ($)", value=0)
    cap_loss = st.number_input("Capital Loss ($)", value=0)
    hours = st.slider("Hours per Week", 1, 99, 40)
    gender = st.selectbox("Gender", ['Male', 'Female'])

with col2:
    workclass = st.selectbox("Workclass", workclass_opts)
    education = st.selectbox("Education Level", edu_opts)
    marital = st.selectbox("Marital Status", marital_opts)
    occupation = st.selectbox("Occupation", occup_opts)
    relationship = st.selectbox("Relationship status", relation_opts)
    race = st.selectbox("Race", race_opts)
    country = st.selectbox("Native Country", country_opts)

if st.button("Predict Classification"):
    # Match the FastAPI Pydantic Model keys exactly
    payload = {
        "age": age, "fnlwgt": fnlwgt, "educational_num": edu_num,
        "capital_gain": cap_gain, "capital_loss": cap_loss, "hours_per_week": hours,
        "workclass": workclass, "education": education, "marital_status": marital,
        "occupation": occupation, "relationship": relationship, "race": race,
        "gender": gender, "native_country": country
    }
    
    try:
        with st.spinner("Calling Backend Engine..."):
            response = requests.post(API_URL, json=payload)
            res_data = response.json()
            
        st.subheader("Result Dashboard")
        if res_data["income_class"] == ">50K":
            st.success(f"Income Prediction: **{res_data['income_class']}** (High Earner)")
        else:
            st.warning(f"Income Prediction: **{res_data['income_class']}** (Standard Earner)")
            
        st.info(f"Model Certainty Index: {res_data.get('confidence', 'N/A')}")
        
    except Exception as e:
        st.error(f"Cannot sync with API backend. Check URL config. Error: {e}")
