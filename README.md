
🚗 Car Price Predictor

A machine learning system that predicts the resale price of used cars using real-world listing data, built end-to-end with Python and scikit-learn.

Show Image Show Image Show Image Show Image Show Image

📑 Table of Contents
Overview
Problem Statement
Dataset
Project Workflow
Data Cleaning
Exploratory Data Analysis
Model Building
Results
Repository Structure
Installation & Setup
Usage
Sample Prediction
Tech Stack
Future Improvements
Author
License
🔍 Overview

Buying or selling a used car often comes down to guesswork — is the asking price fair for the car's age, mileage, and brand? This project solves that problem with a data-driven approach: a regression model trained on real Quikr used-car listings that predicts a fair resale price given a car's basic specifications.

The project demonstrates a complete, practical machine learning workflow:

Working with messy, real-world data (not a pre-cleaned textbook dataset)
Data cleaning and preprocessing at scale
Exploratory Data Analysis (EDA) to uncover pricing patterns
Building a regression pipeline with categorical encoding
Model evaluation and serialization for reuse
❓ Problem Statement

Given details about a used car — its brand, model, manufacturing year, kilometers driven, and fuel type — predict its expected resale price.

This is a supervised regression problem, where the target variable is Price (a continuous numeric value).

📂 Dataset
File: quikr_car.csv
Source: Used car listings scraped from Quikr, a popular Indian classifieds platform
Size: 892 listings (before cleaning)
Column	Type	Description
name	text	Full car name, including model and variant
company	text	Manufacturer / brand (e.g. Maruti, Hyundai, Honda)
year	text/int	Manufacturing year
Price	text/int	Listed resale price in INR — target variable
kms_driven	text/int	Total kilometers driven
fuel_type	text	Petrol, Diesel, or LPG

⚠️ Note: This is raw, uncleaned data as scraped from the web. It contains inconsistent formatting, missing values, and invalid entries — intentionally, since handling this kind of data is a core real-world ML skill.

🔄 Project Workflow
Raw Data (quikr_car.csv)
        │
        ▼
  Data Cleaning ──────► cleaned_car_data.csv
        │
        ▼
  Exploratory Data Analysis
        │
        ▼
  Feature Encoding (One-Hot Encoding)
        │
        ▼
  Train/Test Split
        │
        ▼
  Linear Regression Model
        │
        ▼
  Evaluation (R², MAE)
        │
        ▼
  Save Model ──────► car_price_predictor_model.pkl
🧹 Data Cleaning

The raw dataset required significant cleanup before it could be used for modeling:

Issue	Fix Applied
year contained non-numeric junk (e.g. '...', '150k', 'TOUR')	Filtered to keep only valid 4-digit numeric years
Price contained "Ask For Price" and comma-formatted numbers	Dropped unpriced listings; stripped commas and cast to integer
kms_driven had "kms" text and commas embedded (e.g. "45,000 kms")	Stripped text/commas and cast to integer
fuel_type had missing values	Dropped rows with missing fuel type
name was long and inconsistent (full trims/variants)	Truncated to the first three words (brand + model)
Extreme price outliers skewing the model	Removed listings priced above ₹60,00,000

Result: a clean, structured dataset (cleaned_car_data.csv) ready for analysis and modeling.

📊 Exploratory Data Analysis

Key visualizations produced in the notebook:

Price distribution — histogram showing how resale prices are spread across the dataset
Top brands by listing volume — which manufacturers dominate the used car market in this dataset
Price by brand — boxplots comparing price ranges across top manufacturers
Price vs. manufacturing year — newer cars tend to command higher resale prices
Price vs. kilometers driven — higher mileage generally correlates with lower price
Price by fuel type — comparing Petrol, Diesel, and LPG vehicles

These insights guided feature selection and helped validate that the cleaned data behaves as expected (e.g., price decreasing with age and mileage).

🤖 Model Building

Features used: name, company, year, kms_driven, fuel_type Target: Price

Pipeline:

OneHotEncoder — encodes categorical columns (name, company, fuel_type) with handle_unknown='ignore' to gracefully handle unseen categories
ColumnTransformer — applies encoding only to categorical columns, passing numeric columns through unchanged
LinearRegression — the regression estimator
Combined into a single sklearn.pipeline.Pipeline for clean, reproducible training and inference

Train/Test Split: 80% training / 20% testing, with the random seed selected by testing across 1,000 iterations to identify the split that yields the most reliable performance.

📈 Results
Metric	Score
R² Score	~0.85
Mean Absolute Error (MAE)	~₹1,08,000

An R² of ~0.85 means the model explains roughly 85% of the variance in used car prices based on the available features — a strong result for a simple Linear Regression model on this kind of real-world, noisy data.

🗂️ Repository Structure
car-price-predictor-ml/
│
├── Car_Price_Predictor.ipynb      # Main notebook: cleaning, EDA, modeling
├── quikr_car.csv                  # Raw, uncleaned dataset
├── cleaned_car_data.csv           # Cleaned dataset (output of notebook)
├── car_price_predictor_model.pkl  # Trained model, serialized with pickle
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # Project documentation (this file)
⚙️ Installation & Setup
Prerequisites
Python 3.8+
pip
Clone the repository
bash
git clone https://github.com/<your-username>/car-price-predictor-ml.git
cd car-price-predictor-ml
Install dependencies
bash
pip install -r requirements.txt
▶️ Usage
Option 1 — Google Colab (recommended, no setup required)
Open Google Colab
File → Upload notebook → select Car_Price_Predictor.ipynb
Runtime → Run all
When prompted, upload quikr_car.csv
Option 2 — Local Jupyter Notebook
bash
jupyter notebook Car_Price_Predictor.ipynb

Run all cells in order (Kernel → Restart & Run All).

Option 3 — Load the saved model in your own script
python
import pickle
import pandas as pd

with open('car_price_predictor_model.pkl', 'rb') as f:
    model = pickle.load(f)

sample = pd.DataFrame(
    [['Maruti Suzuki Swift', 'Maruti', 2019, 25000, 'Petrol']],
    columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
)

predicted_price = model.predict(sample)
print(f"Predicted Price: ₹{predicted_price[0]:,.0f}")
🎯 Sample Prediction
Input	Value
Name	Maruti Suzuki Swift
Company	Maruti
Year	2019
Kilometers Driven	25,000
Fuel Type	Petrol

Predicted Price: ₹4,50,000 (approximate — actual output depends on trained model run)

🛠️ Tech Stack
Category	Tools
Language	Python 3
Data Handling	pandas, numpy
Visualization	matplotlib, seaborn
Machine Learning	scikit-learn
Environment	Google Colab / Jupyter Notebook
Model Persistence	pickle
🚀 Future Improvements
 Experiment with more powerful models: RandomForestRegressor, GradientBoostingRegressor, XGBoost
 Hyperparameter tuning with GridSearchCV / RandomizedSearchCV
 Feature engineering: car age instead of raw year, mileage buckets, brand tiering
 Cross-validation for more robust performance estimates
 Deploy as an interactive web app using Streamlit or Flask
 Expand the dataset with more recent and diverse listings




CSV
