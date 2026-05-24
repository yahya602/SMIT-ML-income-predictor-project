import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Income Predictor", page_icon="💲")
st.title("💲 Smart Income Predictor")
st.write(
    "Enter basic demographic profile details below to check if an individual's "
    "annual income is estimated to exceed **$50,000 per year**."
)
# Direct Pipeline Model Loading (No FastAPI required in cloud)
@st.cache_resource
def load_model():
    # Looks for model.pkl in root or app directory automatically
    model_path = "model.pkl" if os.path.exists("model.pkl") else os.path.join("app", "model.pkl")
    with open(model_path, "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading machine learning architecture file (model.pkl): {e}")
    st.stop()

# Categorical options extracted from the Adult dataset
workclass_opts = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
edu_opts = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
marital_opts = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occup_opts = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
relation_opts = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
race_opts = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
country_opts = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']

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
    formatted_data = {
        'age': age,
        'fnlwgt': fnlwgt,
        'educational-num': edu_num,
        'capital-gain': cap_gain,
        'capital-loss': cap_loss,
        'hours-per-week': hours,
        'workclass': workclass,
        'education': education,
        'marital-status': marital,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'native-country': country
    }
    
    df = pd.DataFrame([formatted_data])
    
    with st.spinner("Processing local prediction matrices..."):
        try:
            prediction = model.predict(df)[0]
            
            try:
                probability = model.predict_proba(df)[0][prediction] * 100
            except:
                probability = None

            result = ">50K" if prediction == 1 else "<=50K"
            
            st.subheader("Result Dashboard")
            if result == ">50K":
                st.success(f"Income Prediction: **{result}** (High Earner)")
            else:
                st.warning(f"Income Prediction: **{result}** (Standard Earner)")
                
            st.info(f"Model Certainty Index: {f'{probability:.2f}%' if probability else 'N/A'}")
        except Exception as err:
            st.error(f"Inference pipeline execution failed: {err}")
