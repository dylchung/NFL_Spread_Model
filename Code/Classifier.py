### Pass the Trained Model through the entire dataset, then run a SVM to identify when it beats the spread
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib

# First, run the same preprocessing steps as before
def Prepare_Data(file_path):
    # Read the Excel file into a DataFrame
    data = pd.read_excel(file_path)

    # Save columns for later analysis
    saved_columns = data[['Season', 'HomeTeam', 'AwayTeam','Margin','Vegas_Margin']]
    vegas = data['Vegas_Margin']
    
    # Change every value in 'HomeTeam' to 'Home'
    data['HomeTeam'] = 1
    data['AwayTeam'] = 0

    # DROP UNNEEDED COLUMNS
    X = data.drop(columns=['Date', 'Winner/tie', 'Margin', 'HPts', 'APts', 'Vegas_Margin','Time','Home_Key','Away_Key','WeekMinus','Vegas_Margin'])
    y = data['Margin']

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=['HomeDiv', 'AwayDiv', 'Day','Season'], drop_first=True)
    
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

# Next, predict the entire dataset with the model and create a new column for when predicted margin is closer to real margin than the vegas margin
# Then, pass a SVM (or add different classifiers) to identify which matchups are best predicted and output accuracy scores. Then save the model.

def predict_and_evaluate(X, y, vegas_margin, best_model):
    # Predict margins on the entire dataset
    y_pred = best_model.predict(X)
    print(vegas_margin.head())  # Preview the first few values
    print(vegas_margin.shape)   # Print the shape of the array/Series

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
        'Closer_Than_Vegas': model_error < vegas_error
    })

    # Define features and labels for SVM
    results['Closer_Than_Vegas'] = results['Closer_Than_Vegas'].astype(int)  # Convert boolean to int
    
    # Export to Excel
    output_path = 'closertovegas.xlsx'
    results.to_excel(output_path, index=False)
    X_svm = results[['Predicted_Margin', 'Vegas_Margin']]
    y_svm = results['Closer_Than_Vegas']

    # Split the data for SVM
    X_train, X_test, y_train, y_test = train_test_split(X_svm, y_svm, test_size=0.2, random_state=42)

    # Train the SVM model
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)

    # Predict on the test set
    y_test_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f'SVM Accuracy: {accuracy:.4f}')
    print(classification_report(y_test, y_test_pred))

    # Save the SVM model
    joblib.dump(svm_model, 'svm_model.pkl')
    print('SVM model saved as svm_model.pkl')

    return results


# Prepare the entire dataset
file_path = r"C:\Users\Dylan Chung\Desktop\NFL_Spread_Model\Data\Model_2020-2024v2(PFF).xlsx"
X, margin, saved_columns, vegas = Prepare_Data(file_path)

# Load the best model
best_model = joblib.load(r"C:\Users\Dylan Chung\Desktop\NFL_Spread_Model\XGBoost_best_model.pkl")

# Predict and evaluate
results = predict_and_evaluate(X, margin, vegas, best_model)
