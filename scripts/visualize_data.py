# Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import os

### Exploratory Data Analysis (EDA) ###

def promo_distribution(df, title):
    """
    Visualize the promo distribution in the dataset.
    Args:
        df (DataFrame): Dataset to analyze.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Promo')
    plt.title(f"Promo Distribution in {title} Set")
    plt.show()
    logging.info(f"Promo distribution in {title} set visualized")

def sales_holiday_behavior(df):
    """
    Visualize sales behavior before, during, and after holidays.
    Args:
        df (DataFrame): Dataset to analyze.
    """
    df['Holiday'] = np.where((df['StateHoliday'] != '0') | (df['SchoolHoliday'] == 1), 'Holiday', 'No Holiday')
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Holiday', y='Sales')
    plt.title("Sales Behavior Before, During, and After Holidays")
    plt.show()
    logging.info("Sales behavior before, during, and after holidays visualized")

def seasonal_sales_behavior(df):
    """
    Visualize seasonal sales behavior based on the month.
    Args:
        df (DataFrame): Dataset to analyze.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Month', y='Sales')
    plt.title("Seasonal Purchase Behavior (Monthly Sales Trend)")
    plt.show()
    logging.info("Seasonal purchase behavior visualized")

def sales_customers_correlation(df):
    """
    Visualize the correlation between sales and number of customers.
    Args:
        df (DataFrame): Dataset to analyze.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Customers', y='Sales')
    plt.title("Correlation between Sales and Number of Customers")
    plt.show()

    correlation = df['Sales'].corr(df['Customers'])
    logging.info(f"Correlation between Sales and Customers: {correlation}")

def promo_effect(df):
    """
    Visualize the effect of promo on sales and customer count.
    Args:
        df (DataFrame): Dataset to analyze.
    """
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Promo', y='Sales')
    plt.title("Effect of Promo on Sales")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Promo', y='Customers')
    plt.title("Effect of Promo on Number of Customers")
    plt.show()
    logging.info("Promo effects on sales and customer count visualized")

def store_type_promo_effectiveness(df):
    """
    Visualize the effectiveness of promo by StoreType.
    Args:
        df (DataFrame): Dataset to analyze.
    """
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='StoreType', y='Sales', hue='Promo')
    plt.title("Promo Effectiveness by StoreType")
    plt.show()
    logging.info("Promo effectiveness by StoreType visualized")

def sales_trend_open(df):
    """
    Visualize sales trend by store open/close status.
    Args:
        df (DataFrame): Dataset to analyze.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='DayOfWeek', y='Sales', hue='Open')
    plt.title("Sales Trend by Day of the Week")
    plt.show()
    logging.info("Sales trend by day of the week visualized")

def weekend_sales_comparison(df):
    """
    Compare weekend sales for stores open all week vs not open all week.
    Args:
        df (DataFrame): Dataset to analyze.
    """
    stores_open_all_weekdays = df[df['Open'] == 1].groupby('Store')['DayOfWeek'].nunique()
    weekday_open_stores = stores_open_all_weekdays[stores_open_all_weekdays == 7].index

    weekend_sales = df[df['DayOfWeek'] >= 5].groupby('Store')['Sales'].mean()
    open_weekend_sales = weekend_sales[weekend_sales.index.isin(weekday_open_stores)]
    non_open_weekend_sales = weekend_sales[~weekend_sales.index.isin(weekday_open_stores)]

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=[open_weekend_sales, non_open_weekend_sales], notch=True)
    plt.title("Sales on Weekends: Stores Open All Week vs Not Open All Week")
    plt.xticks([0, 1], ['Open All Week', 'Not Open All Week'])
    plt.ylabel("Average Sales")
    plt.show()
    logging.info("Weekend sales comparison visualized")

def assortment_sales_effect(df):
    """
    Visualize sales by assortment type.
    Args:
        df (DataFrame): Dataset to analyze.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Assortment', y='Sales')
    plt.title("Sales by Assortment Type")
    plt.show()
    logging.info("Assortment type effect on sales visualized")

def competitor_distance_effect(df):
    """
    Visualize the effect of competitor distance on sales.
    Args:
        df (DataFrame): Dataset to analyze.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='CompetitionDistance', y='Sales')
    plt.title("Sales vs Competitor Distance")
    plt.show()
    logging.info("Competitor distance effect on sales visualized")

def sales_competition_trend(df):
    """
    Visualize sales trends with respect to new and old competitors.
    Args:
        df (DataFrame): The dataset.
    """
    df['CompetitionOpened'] = np.where(df['CompetitionDistance'].isna(), 'No Competitor', 'Old Competitor')
    df['CompetitionOpened'] = np.where(df['CompetitionDistance'].notna() & df['CompetitionDistance'].shift(1).isna(), 'New Competitor', df['CompetitionOpened'])

    plt.figure(figsize=(15, 6))
    sns.lineplot(data=df, x='Date', y='Sales', hue='CompetitionOpened')
    plt.title("Sales Trend with New and Old Competitors")
    plt.show()
    logging.info("Sales trend with new and old competitors visualized")

### Correlation Heatmap ###

def correlation_heatmap(df):
    """
    Creates a correlation heatmap of numerical features in the dataset.
    Args:
        df (DataFrame): The dataset.
    """
    df.replace('None', np.nan, inplace=True)  # Replace 'None' with NaN
    df.fillna(0, inplace=True)  # Fill NaN values with 0
    
    # Convert categorical columns into numeric
    label_encoder = LabelEncoder()
    df['StateHoliday'] = label_encoder.fit_transform(df['StateHoliday'].astype(str))
    df['StoreType'] = label_encoder.fit_transform(df['StoreType'].astype(str))
    df['Assortment'] = label_encoder.fit_transform(df['Assortment'].astype(str))

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    # Generate correlation heatmap
    plt.figure(figsize=(16, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()
    logging.info("Correlation heatmap visualized")
