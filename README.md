# Rossmann Pharmaceuticals Sales Forecasting

## Overview
This project focuses on predicting sales for Rossmann Pharmaceuticals across multiple store locations using historical sales data and various external factors. By forecasting sales six weeks ahead, the finance team can make data-driven decisions to optimize operations, promotions, and inventory management.

## Table of Contents
- [Business Need](#business-need)
- [Data Description](#data-description)
- [Methodologies](#methodologies)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Business Need
Rossmann Pharmaceuticals’ finance team currently relies on intuition and past experience for sales forecasting. This method, however, lacks precision and may not account for all relevant factors such as promotions, competition, holidays, or seasonal trends.

This project aims to improve sales forecasts by implementing machine learning models that can predict sales with greater accuracy. By leveraging historical data and a combination of external factors, we provide a solution that helps Rossmann make better financial and operational decisions.

## Data Description
The dataset comprises historical sales data and additional features relevant to store operations. Key features include:

- **Sales**: The sales revenue for each store.
- **Customers**: The number of customers who visited each store on a given day.
- **Promotions**: Details about whether the store was running a promotion.
- **Competitor Distance**: Distance to the nearest competitor store.
- **Store Operational Status**: Whether the store was open or closed on a particular day.
- **Holiday Information**: Information on whether the day was a national or regional holiday.

The data is collected at the store level, with each row representing a daily record of a store's performance and characteristics.

## Methodologies
1. **Data Preprocessing**:
   - Handled missing values (e.g., Competitor Distance).
   - Transformed categorical data into numerical format where necessary.
   - Created new features such as "Competition Status" to distinguish between new and old competitors.
   
2. **Exploratory Data Analysis (EDA)**:
   - Visualized sales trends across different days, regions, and promotional events.
   - Explored relationships between variables using correlation heatmaps, scatter plots, and time series analysis.
   
3. **Model Building**:
   - Applied machine learning models to predict sales six weeks ahead.
   - Key factors included promotions, holidays, customer count, and competitor proximity.
   - Utilized models such as Random Forest, Gradient Boosting, and XGBoost, with hyperparameter tuning to optimize results.

4. **Model Evaluation**:
   - Evaluated the models using RMSE (Root Mean Squared Error) to measure prediction accuracy.
   - Cross-validated results to ensure robustness across different stores and regions.

## Key Findings
- **Day of the Week Impact**: Sales peak on Mondays, dip midweek, and recover on Sundays. Stores closed on weekends experience fewer fluctuations.
- **Customer Count Correlation**: A strong positive correlation (~0.89) between customer counts and sales indicates that customer traffic is the key driver of revenue.
- **Promotions**: Promotions consistently lead to higher sales, especially during the midweek slump.
- **Competitor Impact**: The presence of competitors, including their proximity, shows no clear impact on sales, especially for stores located in densely populated city centers.
- **Assortment Type**: Stores with broader assortments generally outperform those with limited offerings.

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shammuse/ROSMANN.git
   ```

2. **Navigate into the project directory**:
   ## Project Structure

The repository is structured as follows:

```
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   └── __init__.py
├── notebooks/
│   ├── __init__.py
│   ├── EDA.ipynb
│   └── README.md
├── tests/
│   └── __init__.py
├── scripts/
    ├── __init__.py
    ├── Data_cleaning_pipline.py
    ├── visualize_data.py
    └── README.md

```

3. **Set up a virtual environment**
4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the project**
   ```

## Contributing
Contributions to improve the project are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of the changes.

Feel free to open issues for bugs, feature requests, or general questions.

