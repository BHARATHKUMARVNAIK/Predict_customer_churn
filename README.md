# Customer Churn Prediction and Retention Intelligence

> An end-to-end churn intelligence project that predicts who will leave, explains why, and recommends what the business should do next.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Status](https://img.shields.io/badge/Project-End%20to%20End-success)
![Use%20Case](https://img.shields.io/badge/Use%20Case-Customer%20Churn%20Prediction-0A66C2)

This project started as a Kaggle competition solution and was expanded into a full end-to-end machine learning product:

- predict customer churn probability
- explain why churn happens
- identify which customers the business should target first
- power a Streamlit app that returns:
  - churn probability
  - risk level
  - action recommendation

If a recruiter, hiring manager, or teammate opens this repository, they should be able to see not just a model, but a complete business-ready workflow from raw data to deployable decision support.

## Resume-Ready Highlights

- Built an end-to-end churn prediction pipeline on `594K+` training rows, from cleaning and EDA to modeling, deployment, and Kaggle submission.
- Trained and compared machine learning models, with `HistGradientBoostingClassifier` reaching `0.9158` ROC-AUC on validation data.
- Identified the main churn drivers, including `Contract`, `tenure`, `TotalCharges`, `MonthlyCharges`, and `PaymentMethod`.
- Converted model output into business-friendly risk bands and action recommendations for retention teams.
- Developed a Streamlit app that accepts customer details and returns churn probability, risk level, and next-best action.

## App Preview

The Streamlit app is designed for quick business use, not just notebook experimentation.

What a user can do:

- enter customer service, billing, contract, and pricing information
- get an instant churn probability
- view a risk label: `Low`, `Medium`, or `High`
- see an action recommendation for customer retention

Suggested screenshots or GIFs to add on GitHub:

- `assets/app-home.png` for the main app landing screen
- `assets/app-form.png` for the customer input form
- `assets/app-prediction.png` for the churn probability and recommendation output
- `assets/app-demo.gif` for a short end-to-end app walkthrough

Example markdown once screenshots are added:

```md
![App Home](assets/app-home.png)
![Prediction Output](assets/app-prediction.png)
```

## Why This Project Matters

Customer churn is one of the most valuable prediction problems in business. A strong churn project shows the ability to:

- clean and validate large tabular datasets
- perform focused EDA instead of random plotting
- build and compare machine learning models
- translate model outputs into business actions
- create a usable application layer for non-technical users

This project demonstrates all five.

## Project Outcomes

By the end of this project, the pipeline produces:

- a trained churn prediction model
- feature importance to explain the key churn drivers
- business risk segmentation
- action recommendations for retention
- a Streamlit app for interactive scoring
- a Kaggle-ready `submission.csv`

## Dataset Snapshot

- Train data: `594,194` rows and `21` columns
- Test data: `254,655` rows and `20` columns
- Target: `Churn`
- Churn rate in training data: `22.52%`

The data includes customer profile information, service usage, contract details, billing behavior, and charges.

## What Was Built

### 1. Data Cleaning

Cleaning was done to make the dataset reliable and reusable for later stages.

Completed work:

- checked and validated missing values
- checked duplicate rows and duplicate `id`s
- enforced numeric types for numeric columns
- mapped the training target `Churn` from `Yes/No` to `1/0`
- validated service consistency rules
- saved reusable cleaned files

Generated files:

- [cleaned_train.csv](data/cleaned_train.csv)
- [cleaned_test.csv](data/cleaned_test.csv)

Notebook:

- [1_cleaning.ipynb](notebooks/1_cleaning.ipynb)

Script:

- [clean_data.py](scripts/clean_data.py)

### 2. Exploratory Data Analysis

The EDA was built to answer the business question, not just to create charts.

Key EDA goals:

- understand the customer base
- measure churn distribution
- find the strongest churn patterns
- identify the most useful features for modeling
- translate data patterns into retention strategy

Important churn insights found:

- `Month-to-month` contracts are much riskier than longer contracts
- low-tenure customers churn much more than long-tenure customers
- higher `MonthlyCharges` are associated with higher churn
- customers with `Electronic check` have elevated churn
- `Fiber optic` customers show higher churn than other internet groups
- customers without `OnlineSecurity` or `TechSupport` are more likely to churn

Notebook:

- [2_eda.ipynb](notebooks/2_eda.ipynb)

### 3. Modeling

Two models were built and compared:

- Logistic Regression
- HistGradientBoostingClassifier

Why compare both:

- Logistic Regression provides a strong interpretable baseline
- HistGradientBoosting captures more complex non-linear churn behavior

Best model selected:

- `HistGradientBoostingClassifier`

Validation results:

- HistGradientBoosting ROC-AUC: `0.9158`
- Logistic Regression ROC-AUC: `0.9083`

This means the final model ranks churn risk very well and is strong enough to support both Kaggle submission and business-facing prediction.

Notebook:

- [3_modeling.ipynb](notebooks/3_modeling.ipynb)

Saved model artifacts:

- [best_churn_model.joblib](models/best_churn_model.joblib)
- [streamlit_churn_model.joblib](models/streamlit_churn_model.joblib)
- [streamlit_model_metadata.joblib](models/streamlit_model_metadata.joblib)

## Top Feature Importance

The most important churn drivers from the final workflow were:

1. `Contract`
2. `tenure`
3. `TotalCharges`
4. `MonthlyCharges`
5. `PaymentMethod`
6. `InternetService`
7. `OnlineSecurity`
8. `PaperlessBilling`

This is strong because the model is not acting like a black box. It lines up with the EDA and gives a clear business explanation for why churn risk rises.

Report:

- [feature_importance.csv](reports/feature_importance.csv)

## Business Risk Segmentation

The project does not stop at “predict churn.” It also translates predictions into action.

Current validation risk grouping:

- Low risk: `82,245` customers, average predicted risk `0.0566`
- Medium risk: `24,348` customers, average predicted risk `0.4972`
- High risk: `12,246` customers, average predicted risk `0.8190`

Why this matters:

- low-risk customers can be maintained with lightweight engagement
- medium-risk customers can receive targeted retention messaging
- high-risk customers should receive immediate retention offers or direct outreach

Report:

- [risk_summary.csv](reports/risk_summary.csv)

## Streamlit App

The app makes the project usable for a non-technical stakeholder.

Input:

- customer details such as contract, tenure, service usage, billing type, and charges

Output:

- churn probability
- risk level
- action recommendation

This turns the project into a product-like experience instead of a notebook-only exercise.

Main app:

- [app.py](app.py)

Support utility:

- [model_utils.py](scripts/model_utils.py)

Run locally:

```bash
cd /Users/bharathkumarvnaik/Downloads/programing/Data_Scientist_projects/playground-series-s6e3
streamlit run app.py
```

## Kaggle Submission

This repository also generates a valid Kaggle competition submission file with:

- `id`
- predicted churn probability in `Churn`

Submission file:

- [submission.csv](submission.csv)

Generation script:

- [generate_submission.py](scripts/generate_submission.py)

Regenerate submission:

```bash
python3 scripts/generate_submission.py
```

## Project Structure

```text
playground-series-s6e3/
├── app.py
├── README.md
├── requirements.txt
├── submission.csv
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── cleaned_train.csv
│   ├── cleaned_test.csv
│   └── sample_submission.csv
├── models/
│   ├── best_churn_model.joblib
│   ├── streamlit_churn_model.joblib
│   └── streamlit_model_metadata.joblib
├── notebooks/
│   ├── 1_cleaning.ipynb
│   ├── 2_eda.ipynb
│   └── 3_modeling.ipynb
├── reports/
│   ├── feature_importance.csv
│   ├── risk_summary.csv
│   ├── submission.csv
│   └── validation_risk_preview.csv
└── scripts/
    ├── clean_data.py
    ├── generate_submission.py
    └── model_utils.py
```

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## How To Run This Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Generate a Kaggle submission:

```bash
python3 scripts/generate_submission.py
```

## Why This Project Stands Out

Many portfolio projects stop at one notebook and a model score.

This one goes further:

- it cleans and validates the data properly
- it uses EDA to tell a business story
- it compares models instead of training only one
- it explains predictions using feature importance
- it translates model outputs into business action
- it includes a usable Streamlit interface
- it produces a competition-ready submission file

That combination shows not only data science skill, but also product thinking and business understanding.

## Final Takeaway

This project answers three important real-world questions:

1. Which customers are likely to churn?
2. Why are they likely to churn?
3. What should the business do next?

That is the core reason this project is valuable. It is not just a model. It is a complete churn intelligence workflow.
