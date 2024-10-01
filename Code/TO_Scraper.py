import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta

# Function to fetch data from the website for a specific date
def fetch_data_for_date(date):
    url = f"https://www.teamrankings.com/nfl/stat/turnover-margin-per-game?date={date}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Locate the table containing the turnover margin data
    table = soup.find('table', {'class': 'tr-table datatable scrollable'})
    rows = table.find_all('tr')

    data = []
    for row in rows[1:]:  # Skip the header row
        cells = row.find_all('td')
        if len(cells) >= 4:  # Make sure there are enough cells in the row
            rank = cells[0].text.strip()  # Rank column
            team = cells[1].text.strip()  # Team column
            turnover_margin = cells[2].text.strip()  # "2020" or season column
            data.append([rank, team, turnover_margin])

    return data

# Initialize a list to collect all data
all_data = []

# Define the start date and week counter (GO YEAR BY YEAR)
start_date = datetime(2024, 9, 10)
end_date = datetime(2025, 2, 8)  # Set the end date
week = 1

while start_date <= end_date:  # Continue loop until start_date exceeds end_date
    date_str = start_date.strftime('%Y-%m-%d')
    try:
        # Fetch data for the current date
        data = fetch_data_for_date(date_str)
        if not data:
            break  # Exit the loop if no data is found

        # Add "Week" column data
        for row in data:
            row.append(f"{week}")

        # Append the week's data to the all_data list
        all_data.extend(data)

        print(f"Data for {date_str} (Week {week}) added.")

        # Update to the next week
        start_date += timedelta(days=7)
        week += 1

    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Create a single DataFrame with all the data
df = pd.DataFrame(all_data, columns=['Rank', 'Team', 'Margin', 'Week'])

# Write the DataFrame to the same Excel sheet
df.to_excel('NFL_Turnover_Margin_2024.xlsx', sheet_name='All Weeks', index=False)

print("Excel file created successfully.")
