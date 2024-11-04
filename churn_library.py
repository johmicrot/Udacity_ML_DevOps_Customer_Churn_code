"""
Library of functions to predict customer churn.

Author: John Rothman
Date: 2024-11-04
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    Returns dataframe for the CSV found at pth.

    input:
            pth: a path to the CSV
    output:
            data_frame: pandas dataframe
    '''
    data_frame = pd.read_csv(pth)
    return data_frame


def perform_eda(data_frame):
    '''
    Perform EDA on data_frame and save figures to images folder.

    input:
            data_frame: pandas dataframe
    output:
            None
    '''
    if not os.path.exists('./images/eda'):
        os.makedirs('./images/eda')

    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))
    data_frame['Churn'].hist()
    plt.savefig('./images/eda/churn_histogram.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    data_frame['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_histogram.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_bar_plot.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/total_trans_ct_distribution.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap.png')
    plt.close()


def encoder_helper(data_frame, category_lst, response):
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category.

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
    output:
            data_frame: pandas dataframe with new columns
    '''
    for cat_col in category_lst:
        cat_groups = data_frame.groupby(cat_col).mean()[response]
        data_frame[f'{cat_col}_Churn'] = data_frame[cat_col].map(cat_groups)

    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    Split the data into training and testing sets.

    input:
            data_frame: pandas dataframe
            response: string of response name
    output:
            x_train, x_test, y_train, y_test: split data
    '''
    labels = data_frame[response]
    features = data_frame.drop(columns=['Attrition_Flag', response])

    # Keep specific columns as per the notebook
    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]
    features = features[keep_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_rf,
                                y_test_preds_rf,
                                y_train_preds_lr,
                                y_test_preds_lr):
    '''
    Produces classification report for training and testing results and stores report as image.

    input:
            y_train: training response values
            y_test: test response values
            y_train_preds_rf: training predictions from random forest
            y_test_preds_rf: test predictions from random forest
            y_train_preds_lr: training predictions from logistic regression
            y_test_preds_lr: test predictions from logistic regression
    output:
            None
    '''
    if not os.path.exists('./images/results'):
        os.makedirs('./images/results')

    # Random Forest Classification Report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, 'Random Forest Train',
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, 'Random Forest Test',
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_classification_report.png')
    plt.close()

    # Logistic Regression Classification Report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, 'Logistic Regression Train',
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, 'Logistic Regression Test',
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/logistic_regression_classification_report.png')
    plt.close()


def feature_importance_plot(model, features, output_pth):
    '''
    Creates and stores the feature importances.

    input:
            model: model object containing feature_importances_
            features: pandas dataframe of X values
            output_pth: path to store the figure
    output:
            None
    '''
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [features.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(features.shape[1]), importances[indices])
    plt.xticks(range(features.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    Train, store model results: images + scores, and store models.

    input:
            x_train, x_test, y_train, y_test: split data
    output:
            None
    '''
    # Grid Search for Random Forest
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    # Logistic Regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(x_train, y_train)

    # Predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Classification Report
    classification_report_image(
        y_train, y_test, y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr)

    # ROC Curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, ax=axis, alpha=0.8)
    plot_roc_curve(lrc, x_test, y_test, ax=axis, alpha=0.8)
    plt.savefig('./images/results/roc_curve.png')
    plt.close()

    # Feature Importance Plot
    feature_importance_plot(
        cv_rfc.best_estimator_, x_train, './images/results/feature_importance.png')

    # Save Models
    if not os.path.exists('./models'):
        os.makedirs('./models')

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


def main():
    '''
    Main function to run the churn prediction pipeline.
    '''
    # File path
    DATA_PATH = "./data/bank_data.csv"
    data_frame = import_data(DATA_PATH)

    # EDA
    perform_eda(data_frame)

    # Encode categorical variables with  one-hot
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    data_frame = encoder_helper(data_frame, cat_columns, 'Churn')

    # Feature Engineering
    x_train, x_test, y_train, y_test = perform_feature_engineering(data_frame, 'Churn')

    # Train models
    train_models(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
