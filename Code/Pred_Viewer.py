import pandas as pd
import numpy as np
import joblib

# First, run the same preprocessing steps as before
def Prepare_Data(file_path):
    # Read the Excel file into a DataFrame
    data = pd.read_excel(file_path)

    # Save columns for later analysis
    saved_columns = data[['Season', 'HomeTeam', 'AwayTeam', 'Margin', 'Vegas_Margin']]
    vegas = data['Vegas_Margin']
    
    # Change every value in 'HomeTeam' to 'Home'
    data['HomeTeam'] = 1
    data['AwayTeam'] = 0

    # DROP UNNEEDED COLUMNS (dropping Season)
    X = data.drop(columns=['Season', 'Date', 'Winner/tie', 'Margin', 'HPts', 'APts', 'Vegas_Margin', 'Time', 'Home_Key', 'Away_Key', 'WeekMinus', 'Vegas_Margin','Day'])
    y = data['Margin']

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=['HomeDiv', 'AwayDiv'], drop_first=True)
    
    # Create interaction term between Week and TO Margin
    X['Week_TO_Margin_Interaction'] = X['Week'] * X['Season_TO_Margin']
    X['Away_TO_Margin_Interaction'] = X['Week'] * X['A_Season_TO_Margin']
    
    # Added in week interaction for win pct
    X['Week_WinInteraction'] = X['Week'] * X['Home_WinPct']
    X['Away_WinInteraction'] = X['Week'] * X['Away_WinPct']
    
    # Add in week interactions for penalty YPG and third down conv
    X['Week_Third_Down'] = X['Week'] * X['Third_Down']
    X['Away_Third_Down'] = X['Week'] * X['A_Third_Down']
    X['Penalty_Yards'] = X['Week'] * X['Penalty_Yards']
    X['Away_Penalty_Yards'] = X['Week'] * X['A_Penalty_Yards']
    
    # Drop unused columns
    X.drop(columns=['Home_WinPct', 'Away_WinPct', 'Season_TO_Margin', 'A_Season_TO_Margin', 'Penalty_Yards', 'A_Penalty_Yards', 'Third_Down', 'A_Third_Down'], inplace=True)
    
    return X, y, saved_columns, vegas

# Function to predict, evaluate and export results to Excel
def predict_and_evaluate(X, y, vegas_margin, best_model):
    # Predict margins on the entire dataset
    y_pred = best_model.predict(X)

    # Calculate absolute errors
    model_error = np.abs(y_pred - y)
    vegas_error = np.abs(vegas_margin - y)

    # Create a DataFrame with predictions and compare with Vegas_Margin
    results = pd.DataFrame({
        'Actual_Margin': y,
        'Predicted_Margin': y_pred,
        'Vegas_Margin': vegas_margin,
        'Model_Error': model_error,
        'Vegas_Error': vegas_error,
        'Closer_Than_Vegas': model_error < vegas_error,
        'Difference': vegas_margin - y_pred,  # Compute difference between Vegas margin and predicted margin
        'Abs_Difference': np.abs(vegas_margin - y_pred)  # Absolute value of the difference
    })

    # Export to Excel
    output_path = 'closertovegas_v2_with_differences.xlsx'
    results.to_excel(output_path, index=False)
    print(f'Results exported to {output_path}')

    return results

# Prepare the entire dataset
file_path = r"C:\Users\Dylan Chung\Desktop\NFL_Spread_Model\Data\Model_2020-2025v3(PFF).xlsx"
X, margin, saved_columns, vegas = Prepare_Data(file_path)

# Load the best model
best_model = joblib.load(r"C:\Users\Dylan Chung\Desktop\NFL_Spread_Model\Random Forest_best_model_v2.pkl")

# Predict and evaluate
results = predict_and_evaluate(X, margin, vegas, best_model)