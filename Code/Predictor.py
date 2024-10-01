import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def prepare_future_data(file_path):
    """
    Prepares future game data for predictions by preprocessing it similarly to training data.
    """
    # Read the Excel file containing future games
    data = pd.read_excel(file_path)

    # Save columns for later analysis (to include them in the final Excel file)
    saved_columns = data[['Season', 'HomeTeam', 'AwayTeam']]

    # Change 'HomeTeam' to 1 and 'AwayTeam' to 0 for consistency
    data['HomeTeam'] = 1
    data['AwayTeam'] = 0

    # Drop unnecessary columns (such as 'Margin' since it's not available for future games)
    X = data.drop(columns=['Season', 'Date', 'Winner/tie', 'Margin', 'HPts', 'APts', 'Vegas_Margin', 'Time', 
                           'Home_Key', 'Away_Key', 'WeekMinus','Day'])

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=['HomeDiv', 'AwayDiv'], drop_first=True)

    # Create interaction terms
    X['Week_TO_Margin_Interaction'] = X['Week'] * X['Season_TO_Margin']
    X['Away_TO_Margin_Interaction'] = X['Week'] * X['A_Season_TO_Margin']
    X['Week_WinInteraction'] = X['Week'] * X['Home_WinPct']
    X['Away_WinInteraction'] = X['Week'] * X['Away_WinPct']
    X['Week_Third_Down'] = X['Week'] * X['Third_Down']
    X['Away_Third_Down'] = X['Week'] * X['A_Third_Down']
    X['Penalty_Yards'] = X['Week'] * X['Penalty_Yards']
    X['Away_Penalty_Yards'] = X['Week'] * X['A_Penalty_Yards']

    # Drop unused columns
    X.drop(columns=['Home_WinPct', 'Away_WinPct', 'Season_TO_Margin', 'A_Season_TO_Margin',
                    'Penalty_Yards', 'A_Penalty_Yards', 'Third_Down', 'A_Third_Down'], inplace=True)

    return X, saved_columns

def predict_future_games(model_path, future_data_path, output_path='future_predictions.xlsx'):
    """
    Loads the best saved model, predicts on future games, and exports results to an Excel file.
    """
    # Load the best model
    best_model = joblib.load(model_path)

    # Prepare the future data for predictions
    X_future, saved_columns = prepare_future_data(future_data_path)

    # Make predictions on the future games
    future_predictions = best_model.predict(X_future)

    # Create a DataFrame with the predictions and the original saved columns
    predictions_df = saved_columns.copy()
    predictions_df['Predicted_Margin'] = future_predictions

    # Export the predictions to an Excel file
    predictions_df.to_excel(output_path, index=False)
    print(f"Predictions exported to {output_path}")

# Usage example
# Replace 'future_games.xlsx' with the path to your future game data
predict_future_games(model_path=r'C:\Users\Dylan Chung\Desktop\NFL_Spread_Model\Random Forest_best_model_v2.pkl', future_data_path=r'C:\Users\Dylan Chung\Desktop\NFL_Spread_Model\Data\Prediction_Input.xlsx')
