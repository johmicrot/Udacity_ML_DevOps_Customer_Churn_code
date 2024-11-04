# churn_script_logging_and_tests.py
"""
Script to test functions in churn_library.py and log the results.

Author: John Rothman
Date: 2024-11-04
"""

import os
import logging
import churn_library as cl

if not os.path.exists('./logs'):
    os.makedirs('./logs')

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    Test data import.
    '''
    try:
        data_frame = cl.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
        logging.info("Testing import_data: DataFrame has rows and columns")
    except AssertionError as err:
        logging.error("Testing import_data: The DataFrame is empty")
        raise err


def test_eda():
    '''
    Test perform_eda function.
    '''
    try:
        data_frame = cl.import_data("./data/bank_data.csv")
        cl.perform_eda(data_frame)
        assert os.path.exists('./images/eda/churn_histogram.png')
        assert os.path.exists('./images/eda/customer_age_histogram.png')
        assert os.path.exists('./images/eda/marital_status_bar_plot.png')
        assert os.path.exists('./images/eda/total_trans_ct_distribution.png')
        assert os.path.exists('./images/eda/heatmap.png')
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: EDA plots not found")
        raise err
    except Exception as err:
        logging.error("Testing perform_eda: An error occurred: %s", err)
        raise err


def test_encoder_helper():
    '''
    Test encoder_helper function.
    '''
    try:
        data_frame = cl.import_data("./data/bank_data.csv")
        data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        data_frame = cl.encoder_helper(data_frame, cat_columns, 'Churn')
        for col in cat_columns:
            assert f'{col}_Churn' in data_frame.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: Encoded columns not found")
        raise err
    except Exception as err:
        logging.error("Testing encoder_helper: An error occurred: %s", err)
        raise err


def test_perform_feature_engineering():
    '''
    Test perform_feature_engineering function.
    '''
    try:
        data_frame = cl.import_data("./data/bank_data.csv")
        data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        data_frame = cl.encoder_helper(data_frame, cat_columns, 'Churn')
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            data_frame, 'Churn')
        assert len(x_train) > 0
        assert len(x_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The output datasets are empty")
        raise err
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering: An error occurred: %s", err)
        raise err


def test_train_models():
    '''
    Test train_models function.
    '''
    try:
        data_frame = cl.import_data("./data/bank_data.csv")
        data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        data_frame = cl.encoder_helper(data_frame, cat_columns, 'Churn')
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            data_frame, 'Churn')
        cl.train_models(x_train, x_test, y_train, y_test)
        assert os.path.exists('./models/rfc_model.pkl')
        assert os.path.exists('./models/logistic_model.pkl')
        assert os.path.exists('./images/results/roc_curve.png')
        assert os.path.exists('./images/results/rf_classification_report.png')
        assert os.path.exists(
            './images/results/logistic_regression_classification_report.png')
        assert os.path.exists('./images/results/feature_importance.png')
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: Models or result images are missing")
        raise err
    except Exception as err:
        logging.error("Testing train_models: An error occurred: %s", err)
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
    