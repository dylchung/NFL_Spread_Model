import pandas as pd

def add_recent_point_differential_feature(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Ensure the data is sorted by season and date
    df = df.sort_values(by=['Season', 'Date']).reset_index(drop=True)

    # Dictionary to store the last 5 games' point differentials by team and season
    team_differentials = {}

    # Iterate through the dataframe to calculate performance
    for index, row in df.iterrows():
        season = row['Season']
        home_team = row['Home Team']
        away_team = row['AwayTeam']
        home_score = row['HPts']
        away_score = row['APts']

        # Initialize the team's records if they are not already in the dictionary or reset for a new season
        if home_team not in team_differentials:
            team_differentials[home_team] = {}
        if away_team not in team_differentials:
            team_differentials[away_team] = {}
        if season not in team_differentials[home_team]:
            team_differentials[home_team][season] = []
        if season not in team_differentials[away_team]:
            team_differentials[away_team][season] = []

        # Update the recent point differential for both teams
        df.at[index, 'Home Team Last 5 Point Differential'] = sum(team_differentials[home_team][season][-5:])
        df.at[index, 'Away Team Last 5 Point Differential'] = sum(team_differentials[away_team][season][-5:])

        # Calculate the point differential for each game
        home_differential = home_score - away_score
        away_differential = away_score - home_score

        # Append the point differential to the lists
        team_differentials[home_team][season].append(home_differential)
        team_differentials[away_team][season].append(away_differential)

        # Keep only the last 5 games' point differentials for the current season
        team_differentials[home_team][season] = team_differentials[home_team][season][-5:]
        team_differentials[away_team][season] = team_differentials[away_team][season][-5:]

    return df

# Usage
file_path = r'C:\Users\Dylan Chung\Desktop\NFL_Spread_Model\Data\Model_2020-2024v2(PFF).xlsx'
df_with_point_differential = add_recent_point_differential_feature(file_path)
print(df_with_point_differential)

# Save the updated dataframe back to an Excel file
df_with_point_differential.to_excel('marginfixed.xlsx', index=False)