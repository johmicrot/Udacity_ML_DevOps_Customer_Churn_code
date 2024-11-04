# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This project aims to predict customer churn for a bank by analyzing customer data and building machine learning models to identify customers who are likely to churn. The project involves:

- Performing exploratory data analysis (EDA)
- Feature engineering and encoding of categorical variables
- Training machine learning models (Logistic Regression and Random Forest) using GridSearchCV
- Generating predictions and evaluating model performance
- Implementing best coding practices, including modular code, logging, and unit testing

## Files and Data Description

- **`data/bank_data.csv`**: The dataset containing customer information.
- **`churn_library.py`**: Contains all the functions required for data processing, modeling, and evaluation.
- **`churn_script_logging_and_tests.py`**: Includes unit tests for the functions in `churn_library.py` and sets up logging.
- **`README.md`**: Provides project overview and instructions to use the code.
- **`logs/`**: Directory where log files are stored.
- **`models/`**: Directory where trained models are saved.
- **`images/`**: Contains subdirectories:
  - `eda/`: Stores EDA plots.
  - `results/`: Stores model evaluation plots.

## Running the Files

### 1. Setting up enviornment

Install the requirements first with  `python -m pip install -r requirements_py3.6.txt`

### 2. Running the Main Script

Execute the `churn_script_logging_and_tests.py` script to perform the full data science pipeline:

``` ipython churn_script_logging_and_tests.py ``` 
