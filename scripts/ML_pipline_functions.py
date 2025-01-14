import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import logging
import warnings

def load_data(train_path, test_path, store_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    store = pd.read_csv(store_path)
    
    # Merge store information
    train = train.merge(store, on='Store', how='left')
    test = test.merge(store, on='Store', how='left')
    logging.info("Datasets loaded successfully")
    return train, test

def clean_data(train, test):
    # Clean StateHoliday column
    train['StateHoliday'].replace({'0': 0}, inplace=True)
    test['StateHoliday'].replace({'0': 0}, inplace=True)
    
    # Remove rows where store is closed
    reduced_train_df = train[train.Open == 1].copy()
    
    # Convert Date column to datetime
    reduced_train_df['Date'] = pd.to_datetime(reduced_train_df['Date'])
    test['Date'] = pd.to_datetime(test['Date'])
    
    # Create year, month, day columns
    for df in [reduced_train_df, test]:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        
    return reduced_train_df, test

def create_pipeline(input_cols, num_cols, cat_cols):
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    
    return Pipeline(steps=[('preprocessor', preprocessor)])


# Defining loss function (root mean square error)
def rmspe(y_true, y_pred):
    percentage_error = (y_true - y_pred) / y_true
    percentage_error[y_true == 0] = 0
    squared_percentage_error = percentage_error ** 2
    mean_squared_percentage_error = np.mean(squared_percentage_error)
    return np.sqrt(mean_squared_percentage_error)

def try_model(model, train_inputs, train_targets, val_inputs, val_targets):
    model.fit(train_inputs, train_targets)
    train_preds = model.predict(train_inputs)
    val_preds = model.predict(val_inputs)

    train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
    val_rmse = mean_squared_error(val_targets, val_preds, squared=False)

    train_rmspe = rmspe(train_targets, train_preds)
    val_rmspe = rmspe(val_targets, val_preds)

    print(f"Train RMSE: {train_rmse}")
    print(f"Val RMSE: {val_rmse}")
    print(f"Train RMSPE: {train_rmspe}")
    print(f"Val RMSPE: {val_rmspe}")


def main(train_path, test_path, store_path, model):
    # Load and clean data
    train, test = load_data(train_path, test_path, store_path)
    reduced_train_df, test = clean_data(train, test)

    # Define input columns for transformation
    num_cols = ['Sales', 'Customers', 'Year', 'Month', 'Day']  # Example numeric columns
    cat_cols = ['StoreType', 'Assortment', 'StateHoliday']  # Example categorical columns

    # Split into features and targets
    target = 'Sales'  # Example target variable
    train_inputs = reduced_train_df[num_cols + cat_cols]
    train_targets = reduced_train_df[target]
    val_inputs = test[num_cols + cat_cols]
    val_targets = test[target]

    # Create model pipeline
    pipeline = create_pipeline(train_inputs.columns, num_cols, cat_cols)

    # Apply transformations
    train_inputs_transformed = pipeline.fit_transform(train_inputs)
    val_inputs_transformed = pipeline.transform(val_inputs)

    # Fit the model on the training data
    print("Training the model...")
    model.fit(train_inputs_transformed, train_targets)
    print("Model trained successfully.")
    
    # Now make predictions on the validation data
    val_preds = model.predict(val_inputs_transformed)

    # Calculate RMSE and RMSPE
    val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
    val_rmspe = rmspe(val_targets, val_preds)

    print(f"Validation RMSE: {val_rmse}")
    print(f"Validation RMSPE: {val_rmspe}")

    # Predict on the test set
    test_inputs_transformed = pipeline.transform(test_inputs)
    test_preds = model.predict(test_inputs_transformed)

    # Add Store identifier from the test dataset
    test_preds_df_with_store = pd.DataFrame({
        'Store': test_inputs['Store'],  
        'Predicted Sales': test_preds
    })
    print(test_preds_df_with_store)

# Instantiate the model
model = RandomForestRegressor()

# Example paths (replace with your actual file paths)
train_path = 'train.csv'
test_path = 'test.csv'
store_path = 'store.csv'

# Run the main function
main(train_path, test_path, store_path, model)
