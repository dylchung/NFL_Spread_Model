import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta

# Function to fetch penalty yard data for a specific date
def fetch_penalty_data(date):
    url = f"https://www.teamrankings.com/nfl/stat/penalty-yards-per-game?date={date}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    table = soup.find('table', {'class': 'tr-table datatable scrollable'})
    rows = table.find_all('tr')

    data = []
    for row in rows[1:]:
        cells = row.find_all('td')
        if len(cells) >= 4:
            rank = cells[0].text.strip()
            team = cells[1].text.strip()
            penalty_yards = cells[2].text.strip()  # Penalty yards per game
            data.append([rank, team, penalty_yards])

    return data

# Function to fetch third-down conversion data for a specific date
def fetch_third_down_data(date):
    url = f"https://www.teamrankings.com/nfl/stat/third-down-conversion-pct?date={date}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    table = soup.find('table', {'class': 'tr-table datatable scrollable'})
    rows = table.find_all('tr')

    data = []
    for row in rows[1:]:
        cells = row.find_all('td')
        if len(cells) >= 4:
            rank = cells[0].text.strip()
            team = cells[1].text.strip()
            third_down_pct = cells[2].text.strip()  # Third down conversion percentage
            data.append([rank, team, third_down_pct])

    return data

# Initialize lists for storing the data
penalty_data = []
third_down_data = []

# Define the start and end dates
start_date = datetime(2024, 9, 10)
end_date = datetime(2024, 10, 9)
week = 1

# Fetch both datasets in one loop
while start_date <= end_date:
    date_str = start_date.strftime('%Y-%m-%d')
    try:
        # Fetch penalty data
        penalty = fetch_penalty_data(date_str)
        for row in penalty:
            row.append(f"{week}")
        penalty_data.extend(penalty)
        
        # Fetch third-down conversion data
        third_down = fetch_third_down_data(date_str)
        for row in third_down:
            row.append(f"{week}")
        third_down_data.extend(third_down)

        print(f"Data for {date_str} (Week {week}) added.")
        
        # Move to the next week
        start_date += timedelta(days=7)
        week += 1

    except Exception as e:
        print(f"An error occurred for {date_str}: {e}")
        break

# Create DataFrames from the collected data
df_penalty = pd.DataFrame(penalty_data, columns=['Rank', 'Team', 'Penalty_Yards_Per_Game', 'Week'])
df_third_down = pd.DataFrame(third_down_data, columns=['Rank', 'Team', 'Third_Down_Conversion_Pct', 'Week'])

# Write the DataFrames to separate Excel files
df_penalty.to_excel('NFL_Penalty2024.xlsx', sheet_name='All Weeks', index=False)
df_third_down.to_excel('NFL_Third2024.xlsx', sheet_name='All Weeks', index=False)

print("Excel files created successfully.")