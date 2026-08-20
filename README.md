# ChurnPredict AI

ChurnPredict AI is a full-stack data science and business analytics application for customer churn prediction. It uses the IBM Telco Customer Churn dataset to clean customer data, analyze churn behavior, train machine learning models, compare performance, and deliver an interactive Streamlit dashboard for business users.

## Problem Statement

Customer churn directly affects revenue and growth. Businesses need a practical way to understand why customers leave and identify high-risk customers early enough to act. This project predicts whether a customer is likely to churn and summarizes the business factors most associated with churn.

## Features

- Dataset loading, cleaning, missing-value handling, and type conversion
- Exploratory data analysis with business insights
- Interactive BI dashboard with KPIs and Plotly charts
- Dynamic sidebar filters for contract, internet service, payment method, tenure, and monthly charges
- Custom modern dashboard styling instead of default Streamlit metric blocks
- Machine learning comparison across three classifiers
- Automatic best-model selection using ROC-AUC and F1 score
- Saved reusable preprocessing and prediction pipeline
- Feature importance analysis with top churn factors
- Interactive churn prediction form with probability and risk level
- Data-driven business recommendations

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- CSV dataset storage

## Project Structure

```text
customer-churn-prediction/
  data/
    customer_churn.csv
  notebooks/
    churn_analysis.ipynb
  models/
    churn_model.pkl
  src/
    data_preprocessing.py
    train_model.py
    evaluate_model.py
  app.py
  requirements.txt
  README.md
```

## Dataset Description

The project uses the IBM Telco Customer Churn dataset as the original source. The working CSV has been augmented to 12,000 customer records for a richer demo experience:

- 7,043 original IBM Telco customer records
- 4,957 synthetic Telco-style records generated from realistic distributions and churn-risk rules
- 21 columns covering demographics, account information, subscribed services, charges, and the target variable `Churn`

Important fields include:

- `customerID`
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn`

Dataset source: IBM Telco Customer Churn sample dataset from the archived IBM GitHub repository.

The original dataset backup is stored at `data/customer_churn_original.csv`.

## Machine Learning Workflow

1. Load the customer churn dataset.
2. Inspect shape, columns, data types, missing values, and duplicates.
3. Clean the data and convert `TotalCharges` to numeric.
4. Handle missing values.
5. Split features and target variable.
6. Encode categorical features with one-hot encoding.
7. Scale numerical features.
8. Train Logistic Regression, Random Forest, and Gradient Boosting models.
9. Evaluate each model using Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrix.
10. Select the best model using ROC-AUC and F1 score.
11. Save the best model pipeline as `models/churn_model.pkl`.

## Installation

From the `customer-churn-prediction` folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## How to Run

Train the machine learning model:

```bash
python src/train_model.py
```

Regenerate the augmented dataset:

```bash
python src/augment_dataset.py --target-size 12000 --seed 42
```

Evaluate the saved model:

```bash
python src/evaluate_model.py
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

## Dashboard Pages

- Home: project overview, objective, technologies, dataset summary, and KPIs
- Data Overview: preview, dimensions, column information, missing values, and descriptive statistics
- Customer Analytics: filtered KPIs, interactive churn charts, and dynamic business insights
- Model Performance: model comparison, metrics, confusion matrix, and ROC curve
- Feature Importance: top churn factors and recommendations
- Churn Prediction: customer form, churn prediction, probability, risk level, and explanation

## Model Performance

After training, model results are saved to `models/model_metrics.json` and displayed in the Streamlit dashboard. The final selected model is saved as `models/churn_model.pkl`.

## Business Insights

The dashboard automatically generates recommendations from observed churn patterns and model feature importance. Typical analysis focuses on contract type, tenure, monthly charges, payment method, technical support, and online security.

## Screenshots

Add screenshots here after running the Streamlit app:

- Home dashboard
- Customer analytics charts
- Model performance page
- Churn prediction form

## Future Improvements

- Add customer segmentation for retention campaigns
- Add batch CSV upload for scoring multiple customers
- Add SHAP-based model explainability
- Add threshold tuning based on business cost
- Add exportable business reports
