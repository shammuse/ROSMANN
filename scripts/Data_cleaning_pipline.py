#detect and handle outliers and missing data
# Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import os


### Data Loading ###

def load_data(train_path, test_path, store_path):
    """
    Load train, test, and store datasets.
    Args:
        train_path (str): Path to the train dataset.
        test_path (str): Path to the test dataset.
        store_path (str): Path to the store dataset.
    Returns:
        train (DataFrame): Training dataset merged with store information.
        test (DataFrame): Test dataset merged with store information.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    store = pd.read_csv(store_path)

    logging.info("Datasets loaded successfully")

    # Merge store information into train and test datasets
    train = train.merge(store, on='Store', how='left')
    test = test.merge(store, on='Store', how='left')

    logging.info("Datasets merged successfully")
    return train, test

### Data Cleaning ###

def clean_data(df):
    """
    Handle missing values and outliers in the dataset.
    Args:
        df (DataFrame): Dataset to clean.
    Returns:
        df_cleaned (DataFrame): Cleaned dataset.
    """
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna('None', inplace=True)
    
    logging.info("Missing values handled in the dataset")
    
    # Handle outliers in Sales using z-scores
    z_scores = np.abs(stats.zscore(df['Sales']))
    df_cleaned = df[(z_scores < 3)]
    
    logging.info("Outliers handled for Sales using z-scores")
    
    return df_cleaned

def clean_test_data(df):
    """
    Handle missing values in the test dataset.
    Args:
        df (DataFrame): Test dataset to clean.
    Returns:
        df_cleaned (DataFrame): Cleaned test dataset.
    """
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna('None', inplace=True)
    
    logging.info("Missing values handled in the test dataset")
    return df
