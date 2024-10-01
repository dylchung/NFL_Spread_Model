import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import joblib

### Take input data and preprocess it for training/testing
def Prepare_Data(file_path):
    # Read the Excel file into a DataFrame
    data = pd.read_excel(file_path)

    # Save columns for later analysis
    saved_columns = data[['Season', 'HomeTeam', 'AwayTeam','Margin','Vegas_Margin']]
    vegas = data['Vegas_Margin']
    
    # Change every value in 'HomeTeam' to 'Home'
    data['HomeTeam'] = 1
    data['AwayTeam'] = 0

    # DROP UNNEEDED COLUMNS (dropping Season)
    X = data.drop(columns=['Season','Date', 'Day','Winner/tie', 'Margin', 'HPts', 'APts', 'Vegas_Margin','Time','Home_Key','Away_Key','WeekMinus','Vegas_Margin'])
    y = data['Margin']

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=['HomeDiv', 'AwayDiv'], drop_first=True)
    
    # Create interaction term between Week and TO Margin
    X['Week_TO_Margin_Interaction'] = X['Week'] * X['Season_TO_Margin']
    X['Away_TO_Margin_Interaction'] = X['Week'] * X['A_Season_TO_Margin']
    
    # Added in week interaction for win pct
    X['Week_WinInteraction'] = X['Week'] * X['Home_WinPct']
    X['Away_WinInteraction'] = X['Week'] * X['Away_WinPct']
    
    # add in week interactions for penalty YPG and third down conv
    X['Week_Third_Down'] = X['Week'] * X['Third_Down']
    X['Away_Third_Down'] = X['Week'] * X['A_Third_Down']
    X['Penalty_Yards'] = X['Week'] * X['Penalty_Yards']
    X['Away_Penalty_Yards'] = X['Week'] * X['A_Penalty_Yards']
    
    # Drop unused columns
    X.drop(columns=['Home_WinPct', 'Away_WinPct', 'Season_TO_Margin', 'A_Season_TO_Margin','Penalty_Yards','A_Penalty_Yards','Third_Down','A_Third_Down'], inplace=True)
    
    return X, y, saved_columns, vegas

# show distribution of margin
def plot_target_distribution(y):
    """Plots the distribution of the target variable."""
    plt.figure(figsize=(10, 6))
    plt.hist(y, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Target Variable (Margin)')
    plt.xlabel('Margin')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
def export_feature_importance_to_excel(model, X, output_path='feature_importance.xlsx'):
    """Exports feature importance for tree-based models to an Excel file."""
    if hasattr(model, 'feature_importances_'):
        # Get feature importance
        importance = model.feature_importances_
        # Create a DataFrame for better visualization
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Export to Excel
        feature_importance_df.to_excel(output_path, index=False)
        print(f"Feature importance exported to {output_path}")
    else:
        print("The model does not have feature_importances_ attribute.")


## Function to build and evaluate models with GridSearchCV and visualize performance
def build_pipeline(X, y, original_columns, vegas_margin):
    # Split the data into validation, and holdout sets
    X_train_val, X_holdout, y_train_val, y_holdout, vegas_train_val, vegas_holdout = train_test_split(
        X, y, vegas_margin, test_size=0.2, random_state=30)
    # further split into training and validation sets
    X_train, X_val, y_train, y_val, vegas_train, vegas_val = train_test_split(
        X_train_val, y_train_val, vegas_train_val, test_size=0.2, random_state=30)

    # Define pipelines with scaling where necessary
    pipelines = {
        'dummy': Pipeline([
            ('model', DummyRegressor(strategy='mean'))
        ]),
        'Random Forest': Pipeline([
            ('model', RandomForestRegressor(random_state=30))
        ]),
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBRegressor(random_state=30))
        ])
    }

    # Hyperparameter grids for each model
    param_grids = {
        'dummy': {},
        'Random Forest': {
            'model__n_estimators': [100, 300, 500, 1000],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10]
        },
        'Linear Regression': {},  # No hyperparameters to tune for Linear Regression
        'XGBoost': {
            'model__n_estimators': [100, 200, 300, 400, 1000],
            'model__learning_rate': [0.001, 0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        }
    }

    best_model = None
    best_score = 1000
    best_model_name = None

    # Train and evaluate each model using GridSearchCV
    for model_name, pipeline in pipelines.items():
        print(f'Training {model_name}...')

        # Perform GridSearchCV if there are hyperparameters to tune
        if param_grids[model_name]:
            grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f'Best params for {model_name}: {best_params}')
        else:
            # Directly fit pipeline if no hyperparameters to tune
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline

        # Evaluate the best model on the validation set
        y_val_pred = best_pipeline.predict(X_val)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
        r2_val = r2_score(y_val, y_val_pred)

        print(f'{model_name} Validation:')
        print(f'  RMSE: {rmse_val:.4f}')
        print(f'  R^2: {r2_val:.4f}')

        # Track the best model based on RMSE score
        if rmse_val < best_score:
            best_score = rmse_val
            best_model = best_pipeline
            best_model_name = model_name

    print(f'\nBest Model: {best_model_name} with neg RMSE: {best_score:.4f}')

    # Test the best model on the holdout set
    y_holdout_pred = best_model.predict(X_holdout)
    rmse_holdout = np.sqrt(mean_squared_error(y_holdout, y_holdout_pred))
    r2_holdout = r2_score(y_holdout, y_holdout_pred)

    print(f'\n{best_model_name} Holdout Performance:')
    print(f'  RMSE: {rmse_holdout:.4f}')
    print(f'  R^2: {r2_holdout:.4f}')
    
    # Visualize feature importance for tree-based models
    if best_model_name in ['Random Forest', 'XGBoost']:
        export_feature_importance_to_excel(best_model.named_steps['model'], X)

    # Visualize the performance of the best model on the holdout set
    plt.figure(figsize=(10, 6))
    plt.scatter(y_holdout, y_holdout_pred, alpha=0.7)
    plt.plot([min(y_holdout), max(y_holdout)], [min(y_holdout), max(y_holdout)], color='red', linestyle='--')
    plt.xlabel('Actual Margin')
    plt.ylabel('Predicted Margin')
    plt.title(f'Performance of {best_model_name} on Holdout Set')
    plt.grid(True)
    plt.show()

    # RESIDUAL ANALYSIS
    residuals = y_holdout - y_holdout_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_holdout_pred, residuals, color='blue', edgecolor='k', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs. Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()

    # Histogram of Residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, color='skyblue', edgecolor='black')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # After creating the DataFrame with predicted values and residuals
    result_df = X_holdout.copy()
    result_df['Predicted'] = y_holdout_pred
    result_df['Residual'] = residuals

    # Merge the original columns back
    result_df = result_df.merge(original_columns, left_index=True, right_index=True, how='left')

    # Get the top N largest residuals
    top_n = 30
    top_residuals_df = result_df.assign(Residual_Abs=np.abs(result_df['Residual'])).nlargest(top_n, 'Residual_Abs')

    # Export to Excel
    top_residuals_df.to_excel('largest_residuals_v2.xlsx', index=False)
    print("Largest residuals exported to 'largest_residuals_v2.xlsx'")

    # Save the best model for later use
    joblib.dump(best_model, f'{best_model_name}_best_model_v2.pkl')
    print(f'Best model saved as {best_model_name}_best_model_v2.pkl')


# Prep Data
file_path = r"C:\Users\Dylan Chung\Desktop\NFL_Spread_Model\Data\Model_2020-2025v3(PFF).xlsx"
X, margin, saved, vegas = Prepare_Data(file_path)

# Use the original Pipeline function
build_pipeline(X, margin, saved, vegas)