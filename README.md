# 💲Smart Income Predictor - ML Classification Project

## 📌 Project Overview
This project is an end-to-end Machine Learning Classification system built using the classic **Adult Census Income Dataset**. The goal is to predict whether an individual's annual income exceeds **$50,000** based on demographic data (such as age, education, occupation, and hours worked per week). 

The architecture features an end-to-end data preprocessing pipeline (`ColumnTransformer`), a robust machine learning backbone, a decoupled **FastAPI** backend REST API engine, and an interactive **Streamlit** frontend interface.

---

## 📊 Model Evaluation & Accuracies
Four distinct machine learning classification models were trained, evaluated, and compared inside a single Jupyter Notebook:

* **Logistic Regression**: `87.07%` 🏆 *(Best Performing Model)*
* **Support Vector Classifier (SVC)**: `87.07%`
* **K-Nearest Neighbors (KNN)**: `84.90%`
* **Decision Tree Classifier**: `81.27%`

### 🥇 Winning Model Details
Both **Logistic Regression** and **SVC** tied for the highest accuracy score of **87.07%**. Due to optimal inference speeds and computational efficiency, the **Logistic Regression** pipeline was selected as the operational model and serialized cleanly into `model.pkl`. 

The saved file contains both the structural data preprocessors (imputers, scalers, and encoders) and the trained classifier, preventing any downstream data leakage during live predictions.

---

## 🌐 Live Demo Links
* **Frontend Web Application (Streamlit Cloud)**: https://smart-income-predictor.streamlit.app/
* **Backend API Engine (Hugging Face Spaces)**: https://huggingface.co/spaces/yahya602/smart-income-API
---

## 📁 Repository Structure
```text
ml-classification-project/
├── data/
│   └── adult.csv        # Census dataset
├── app/
│   ├── main.py          # FastAPI backend server
│   └── streamlit_app.py # Streamlit frontend client
├── notebook.ipynb       # Model training & comparison notebook
├── model.pkl            # Serialized winning pipeline
├── requirements.txt     # Environment dependencies
└── README.md            # Documentation
```

---

## 💻 Steps to Run the Project Locally

Follow these instructions to spin up the development environment on your local machine:

### 1. Clone the Repository
```bash
git clone <your-github-repository-url>
cd ml-classification-project
```

### 2. Install Project Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Launch the FastAPI Backend Server
Open your terminal and boot up the REST API:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
*The API interface documentation will be accessible at: `http://127.0.0`*

### 4. Launch the Streamlit Frontend Client
Open a **second, separate terminal window** and run the user interface:
```bash
streamlit run app/streamlit_app.py
```
*The interactive prediction panel will automatically launch in your browser at: `http://localhost:8501`*
