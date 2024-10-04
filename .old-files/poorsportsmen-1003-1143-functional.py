import pandas as pd
import requests
import numpy as np

# ---------------------------
# Configuration and Constants
# ---------------------------

LEAGUE_ID = "487244852"
SEASON_YEAR = "2024"
TOTAL_WEEKS = 14  # Total number of weeks in the season
PLAYOFF_SPOTS = 6  # Number of playoff spots

SWID = "{656BD96B-8BA1-43D4-A25C-9B6C875EA612}"
ESPNS2 = "AEBCq1QtmqNkCd4ZLNoajnM5wDA2kG%2BAE3RiGNVgjQxs5pdKfAzrJmNC5P%2Bcs3X7iwBupAkh3k74Bpgh29k6NHcIIQW%2BKuaJhY6c3r0iP3XRFGiAMKgs52LkuI7VEwz51YPIbjDOku6rSmtjIv86mE703DGigKALOw%2F4ELtknuvdWoxZQSQ7MSSzSUKzBUKOlcGEILWhtYnz3iqB1hyKlFNUzzsSYUlYSyKJ9ExjbvNqGGdjMKZsVZcVmjFwaXISHomMt75dgW7JgWTDBZ5YBHkuOQT98MRwDfWeVHfTdOmyJLANdKXvtb0ikenj9QPaVdE%3D"

# Cookies for authentication
cookies = {"SWID": SWID, "espn_s2": ESPNS2}

# ---------------------------
# Function Definitions
# ---------------------------

def calculate_team_records(df):
    teams = pd.unique(df[['Home Team', 'Away Team']].values.ravel('K'))
    teams = [team for team in teams if team != "Bye"]

    records = {team: {"Wins": 0, "Losses": 0, "Ties": 0} for team in teams}

    for _, row in df.iterrows():
        home_team = row["Home Team"]
        away_team = row["Away Team"]
        home_score = row["Home Score"]
        away_score = row["Away Score"]

        if away_team == "Bye":
            if home_score > 0:
                records[home_team]["Wins"] += 1
            continue

        if home_score > away_score:
            records[home_team]["Wins"] += 1
            records[away_team]["Losses"] += 1
        elif home_score < away_score:
            records[away_team]["Wins"] += 1
            records[home_team]["Losses"] += 1
        else:
            records[home_team]["Ties"] += 1
            records[away_team]["Ties"] += 1

    records_df = pd.DataFrame([
        {"Team": team, "Wins": data["Wins"], "Losses": data["Losses"], "Ties": data["Ties"]}
        for team, data in records.items()
    ])

    return records_df

def calculate_points(df):
    teams = pd.unique(df[['Home Team', 'Away Team']].values.ravel('K'))
    teams = [team for team in teams if team != "Bye"]

    points_for = {team: 0 for team in teams}
    points_against = {team: 0 for team in teams}

    for _, row in df.iterrows():
        home_team = row["Home Team"]
        away_team = row["Away Team"]
        home_score = row["Home Score"]
        away_score = row["Away Score"]

        if away_team != "Bye":
            points_for[home_team] += home_score
            points_against[home_team] += away_score
            points_for[away_team] += away_score
            points_against[away_team] += home_score
        else:
            points_for[home_team] += home_score

    points_df = pd.DataFrame([
        {"Team": team, "Points For": points_for[team], "Points Against": points_against[team]}
        for team in teams
    ])

    return points_df

def calculate_all_play_records(df):
    teams = pd.unique(df[['Home Team', 'Away Team']].values.ravel('K'))
    teams = [team for team in teams if team != "Bye"]

    all_play_records = []

    weeks = sorted(df['Week'].unique())

    for week in weeks:
        week_data = df[df['Week'] == week]
        team_scores = {}

        for _, row in week_data.iterrows():
            home_team = row["Home Team"]
            away_team = row["Away Team"]
            home_score = row["Home Score"]
            away_score = row["Away Score"]

            team_scores[home_team] = home_score
            if away_team != "Bye":
                team_scores[away_team] = away_score

        all_play_wins = {team: 0 for team in teams}
        all_play_losses = {team: 0 for team in teams}

        for team in teams:
            team_score = team_scores.get(team, 0)
            for opponent in teams:
                if opponent == team:
                    continue
                opponent_score = team_scores.get(opponent, 0)

                if team_score > opponent_score:
                    all_play_wins[team] += 1
                elif team_score < opponent_score:
                    all_play_losses[team] += 1

        for team in teams:
            total_opponents = len(teams) - 1
            if total_opponents > 0:
                win_pct = all_play_wins[team] / total_opponents
            else:
                win_pct = 0
            all_play_records.append({
                "Week": week,
                "Team": team,
                "All-Play Wins": all_play_wins[team],
                "All-Play Losses": all_play_losses[team],
                "All-Play Win Percentage": win_pct
            })

    all_play_df = pd.DataFrame(all_play_records)
    return all_play_df

def calculate_win_pct_vs_median(df_cleaned):
    teams = pd.unique(df_cleaned[['Home Team', 'Away Team']].values.ravel('K'))
    teams = [team for team in teams if team != "Bye"]
    
    win_vs_median = {team: 0 for team in teams}
    total_games = {team: 0 for team in teams}

    for week in df_cleaned['Week'].unique():
        week_data = df_cleaned[df_cleaned['Week'] == week]
        median_score = week_data[['Home Score', 'Away Score']].values.flatten()
        median_score = pd.Series(median_score).median()

        for _, row in week_data.iterrows():
            home_team = row["Home Team"]
            away_team = row["Away Team"]
            home_score = row["Home Score"]
            away_score = row["Away Score"]

            if home_team != "Bye":
                total_games[home_team] += 1
                if home_score > median_score:
                    win_vs_median[home_team] += 1

            if away_team != "Bye":
                total_games[away_team] += 1
                if away_score > median_score:
                    win_vs_median[away_team] += 1

    win_pct_vs_median = {
        team: (win_vs_median[team] / total_games[team]) if total_games[team] > 0 else 0
        for team in teams
    }

    win_vs_median_df = pd.DataFrame([
        {"Team": team, "Win Percentage vs Median": win_pct_vs_median[team]}
        for team in teams
    ])

    return win_vs_median_df

def generate_total_table(records_df, points_df, all_play_df, current_week, df_cleaned):
    total_df = pd.merge(records_df, points_df, on="Team")
    total_df["Net Points"] = total_df["Points For"] - total_df["Points Against"]
    total_df["Total Games"] = total_df["Wins"] + total_df["Losses"] + total_df["Ties"]
    total_df["Win Percentage"] = (total_df["Wins"] + 0.5 * total_df["Ties"]) / total_df["Total Games"]
    total_df["Points Per Game"] = total_df["Points For"] / total_df["Total Games"]

    all_play_agg = all_play_df.groupby('Team').agg({
        "All-Play Wins": "sum",
        "All-Play Losses": "sum",
        "All-Play Win Percentage": "mean"
    }).reset_index()

    total_df = pd.merge(total_df, all_play_agg, on="Team", how='left')
    total_df["Difference (Actual vs. All-Play Win Percentage)"] = total_df["Win Percentage"] - total_df["All-Play Win Percentage"]
    total_df["Tiebreak Score"] = total_df["Win Percentage"] + (total_df["Points For"] / 10000)

    total_df["Remaining Games"] = TOTAL_WEEKS - current_week
    total_df["Max Wins"] = total_df["Wins"] + total_df["Remaining Games"]

    sorted_max_wins = total_df.sort_values(by="Max Wins", ascending=False)
    if len(sorted_max_wins) >= PLAYOFF_SPOTS:
        threshold_max_wins = sorted_max_wins.iloc[PLAYOFF_SPOTS - 1]["Max Wins"]
    else:
        threshold_max_wins = 0

    total_df["Magic Number"] = threshold_max_wins + 1 - total_df["Wins"]
    total_df["Magic Number"] = total_df["Magic Number"].apply(lambda x: max(x, 0))

    win_vs_median_df = calculate_win_pct_vs_median(df_cleaned)
    total_df = pd.merge(total_df, win_vs_median_df, on="Team", how="left")

    total_df["Power Score"] = (total_df["Points For"] * 2) + \
                               (total_df["Points For"] * total_df["Win Percentage"]) + \
                               (total_df["Points For"] * total_df["Win Percentage vs Median"])

    # Calculate Luck Rating based on Diff
    conditions = [
        (total_df["Difference (Actual vs. All-Play Win Percentage)"] >= 0.35),
        (total_df["Difference (Actual vs. All-Play Win Percentage)"] >= 0.25),
        (total_df["Difference (Actual vs. All-Play Win Percentage)"] >= 0.15),
        (total_df["Difference (Actual vs. All-Play Win Percentage)"] == 0),
        (total_df["Difference (Actual vs. All-Play Win Percentage)"] >= -0.1),
        (total_df["Difference (Actual vs. All-Play Win Percentage)"] >= -0.2),
        (total_df["Difference (Actual vs. All-Play Win Percentage)"] >= -0.3),
        (total_df["Difference (Actual vs. All-Play Win Percentage)"] < -0.3)
    ]
    choices = [
        "Luckiest",
        "Very Lucky",
        "Moderately Lucky",
        "Neutral Luck",
        "Slightly Unlucky",
        "Unlucky",
        "Very Unlucky",
        "Unluckiest"
    ]
    total_df["Luck Rating"] = np.select(conditions, choices, default="Unluckiest")

    columns_order = [
        "Team",
        "Wins",
        "Losses",
        "Ties",
        "Points For",
        "Points Against",
        "Net Points",
        "Win Percentage",
        "Points Per Game",
        "All-Play Wins",
        "All-Play Losses",
        "All-Play Win Percentage",
        "Difference (Actual vs. All-Play Win Percentage)",
        "Magic Number",
        "Power Score",
        "Tiebreak Score",
        "Luck Rating",
        "Remaining Games",
        "Max Wins"
    ]

    existing_columns = [col for col in columns_order if col in total_df.columns]
    total_df = total_df[existing_columns]
    total_df = total_df.sort_values(by="Power Score", ascending=False).reset_index(drop=True)

    return total_df

def generate_weekly_total_table(week, df_cleaned, all_play_df):
    df_up_to_week = df_cleaned[df_cleaned['Week'] <= week]
    records_weekly = calculate_team_records(df_up_to_week)
    points_weekly = calculate_points(df_cleaned[df_cleaned['Week'] <= week])
    all_play_weekly = calculate_all_play_records(df_cleaned[df_cleaned['Week'] <= week])
    total_table_weekly = generate_total_table(records_weekly, points_weekly, all_play_weekly, current_week=week, df_cleaned=df_up_to_week)
    return total_table_weekly

def generate_weekly_total_tables(df_cleaned, all_play_df):
    weeks = sorted(df_cleaned['Week'].unique())
    weekly_total_tables = {}

    for week in weeks:
        weekly_total_table = generate_weekly_total_table(week, df_cleaned, all_play_df)
        weekly_total_tables[week] = weekly_total_table

    return weekly_total_tables

def create_data_studio_sheet(total_df, weekly_total_tables):
    data_studio_df = total_df.copy()
    
    weekly_df_list = []
    for week, df in weekly_total_tables.items():
        df_copy = df.copy()
        df_copy['Week'] = week
        weekly_df_list.append(df_copy)
    
    if weekly_df_list:
        weekly_concatenated = pd.concat(weekly_df_list, ignore_index=True)
        weekly_concatenated['Team'] = weekly_concatenated['Team'].astype(str)
        data_studio_df = pd.concat([data_studio_df, weekly_concatenated], ignore_index=True)
    
    return data_studio_df

def apply_conditional_formatting(workbook, worksheet, df, sheet_type="Total Table"):
    format_highlight = workbook.add_format({'bg_color': '#C6EFCE',
                                           'font_color': '#006100'})
    
    if "Power Score" in df.columns and "Tiebreak Score" in df.columns:
        power_col_idx = df.columns.get_loc("Power Score")
        tiebreak_col_idx = df.columns.get_loc("Tiebreak Score")
    else:
        return
    
    def col_idx_to_letter(idx):
        letters = ''
        while idx >= 0:
            letters = chr(idx % 26 + ord('A')) + letters
            idx = idx // 26 - 1
        return letters
    
    power_col_letter = col_idx_to_letter(power_col_idx)
    tiebreak_col_letter = col_idx_to_letter(tiebreak_col_idx)
    
    max_row = len(df) + 1
    worksheet.conditional_format(f'{power_col_letter}2:{power_col_letter}{max_row}', 
                                 {'type': 'top',
                                  'value': 1,
                                  'format': format_highlight})
    
    worksheet.conditional_format(f'{tiebreak_col_letter}2:{tiebreak_col_letter}{max_row}', 
                                 {'type': 'top',
                                  'value': 1,
                                  'format': format_highlight})

    color_scale_format = {'type': '3_color_scale',
                          'min_color': "#FFEB84",
                          'mid_color': "#FFE699",
                          'max_color': "#FFD966"}
    
    worksheet.conditional_format(f'{power_col_letter}2:{power_col_letter}{max_row}', 
                                 color_scale_format)
    
    worksheet.conditional_format(f'{tiebreak_col_letter}2:{tiebreak_col_letter}{max_row}', 
                                 color_scale_format)

def create_luck_scores_sheet(weekly_total_tables, teams):
    luck_scores = {team: [] for team in teams}
    weeks_sorted = sorted(weekly_total_tables.keys())
    
    for week in weeks_sorted:
        weekly_df = weekly_total_tables[week]
        for _, row in weekly_df.iterrows():
            team = row["Team"]
            diff = row["Difference (Actual vs. All-Play Win Percentage)"]
            luck_scores[team].append(diff)
    
    luck_scores_df = pd.DataFrame(luck_scores).T
    luck_scores_df.columns = [str(week) for week in weeks_sorted]
    luck_scores_df.reset_index(inplace=True)
    luck_scores_df.rename(columns={'index': 'Team'}, inplace=True)
    
    return luck_scores_df

def create_power_scores_sheet(weekly_total_tables, teams):
    power_scores = {team: [] for team in teams}
    weeks_sorted = sorted(weekly_total_tables.keys())
    
    for week in weeks_sorted:
        weekly_df = weekly_total_tables[week]
        for _, row in weekly_df.iterrows():
            team = row["Team"]
            power = row["Power Score"]
            power_scores[team].append(power)
    
    power_scores_df = pd.DataFrame(power_scores).T
    power_scores_df.columns = [str(week) for week in weeks_sorted]
    power_scores_df.reset_index(inplace=True)
    power_scores_df.rename(columns={'index': 'Team'}, inplace=True)
    
    return power_scores_df

def create_metric_sheets(weekly_total_tables, metrics, teams):
    metric_dataframes = {}
    weeks_sorted = sorted(weekly_total_tables.keys())
    
    for metric_key, metric_name in metrics.items():
        data = {'Team': teams}
        for week in weeks_sorted:
            weekly_df = weekly_total_tables[week]
            if metric_key in weekly_df.columns:
                metric_value = weekly_df.set_index('Team')[metric_key]
                # Ensure the team order matches
                metric_values = [metric_value.get(team, np.nan) for team in teams]
                data[str(week)] = metric_values
            else:
                data[str(week)] = [np.nan] * len(teams)
        metric_df = pd.DataFrame(data)
        metric_dataframes[metric_name] = metric_df
    
    return metric_dataframes

# ---------------------------
# Additional Function to Create Charts in Excel
# ---------------------------

def create_line_chart(writer, sheet_name, metric_name, teams_sorted, weeks_sorted):
    workbook  = writer.book
    worksheet = workbook.add_worksheet(f"{metric_name} Chart")
    
    # Access the corresponding metric sheet
    metric_sheet = writer.sheets[metric_name]
    
    # Define the range of data
    # Assuming 'Team' is in column A and weeks are in columns B onward
    num_weeks = len(weeks_sorted)
    num_teams = len(teams_sorted)
    
    # Create a line chart object
    chart = workbook.add_chart({'type': 'line'})
    
    # Add a series for each team
    for i, team in enumerate(teams_sorted):
        # Teams start from row 2 (index 1)
        team_row = i + 1
        # Points Against start from column B (which is index 1, 'B')
        # Define the cell range for each team
        chart.add_series({
            'name':       [metric_name, team_row, 0],  # Team name in column A
            'categories': [metric_name, 0, 1, 0, num_weeks],  # Weeks from row 1, columns B onward
            'values':     [metric_name, team_row, 1, team_row, num_weeks],  # Points Against from row 2 onward
            'marker':     {'type': 'circle', 'size': 5},
        })
    
    # Configure the chart axes
    chart.set_x_axis({'name': 'Week', 'position': 'low'})
    chart.set_y_axis({'name': 'Points Against', 'major_gridlines': {'visible': False}})
    
    # Add chart title
    chart.set_title({'name': f'{metric_name} Over Weeks'})
    
    # Set style
    chart.set_style(10)
    
    # Insert the chart into the worksheet
    worksheet.insert_chart('B2', chart, {'x_scale': 2, 'y_scale': 1.5})

# ---------------------------
# Main Script Execution
# ---------------------------

# ---------------------------
# Data Fetching and Parsing
# ---------------------------

url = (
    f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
    f"seasons/{SEASON_YEAR}/segments/0/leagues/{LEAGUE_ID}?view=mMatchup&view=mScoreboard"
)

response = requests.get(url, cookies=cookies)

if response.status_code != 200:
    print(f"Failed to fetch data: {response.status_code}")
    exit()

data = response.json()

# ---------------------------
# Extract Team Information
# ---------------------------

teams = {}
for team in data["teams"]:
    team_id = team["id"]
    team_name = team.get("name") or team.get("nickname") or f"Team {team_id}"
    teams[team_id] = team_name

# ---------------------------
# Parse Matchup Data
# ---------------------------

matchups = []
for matchup in data["schedule"]:
    week = matchup["matchupPeriodId"]

    home_team_id = matchup["home"]["teamId"]
    home_team_name = teams.get(home_team_id, f"Team {home_team_id}")
    home_score = matchup["home"]["totalPoints"]

    if "away" in matchup:
        away_team_id = matchup["away"]["teamId"]
        away_team_name = teams.get(away_team_id, f"Team {away_team_id}")
        away_score = matchup["away"]["totalPoints"]
    else:
        away_team_id = None
        away_team_name = "Bye"
        away_score = 0.0

    matchups.append(
        {
            "Week": week,
            "Home Team": home_team_name,
            "Home Score": home_score,
            "Away Team": away_team_name,
            "Away Score": away_score,
        }
    )

master_df = pd.DataFrame(matchups)

# ---------------------------
# Data Cleaning
# ---------------------------

df_cleaned = master_df[
    ~((master_df["Home Score"] == 0.00) & (master_df["Away Score"] == 0.00))
]

# ---------------------------
# Calculate Team Records
# ---------------------------

records_df = calculate_team_records(df_cleaned)

# ---------------------------
# Calculate Points For and Against
# ---------------------------

points_df = calculate_points(df_cleaned)

# ---------------------------
# Calculate All-Play Records
# ---------------------------

all_play_df = calculate_all_play_records(df_cleaned)

# ---------------------------
# Generate Total Table
# ---------------------------

current_week = df_cleaned['Week'].max()
total_table_df = generate_total_table(records_df, points_df, all_play_df, current_week, df_cleaned)

# ---------------------------
# Generate Weekly Total Tables
# ---------------------------

weekly_total_tables = generate_weekly_total_tables(df_cleaned, all_play_df)

# ---------------------------
# Create Data Studio Sheet
# ---------------------------

data_studio_df = create_data_studio_sheet(total_table_df, weekly_total_tables)

# ---------------------------
# Create Luck Scores Sheet
# ---------------------------

luck_scores_df = create_luck_scores_sheet(weekly_total_tables, teams.values())

# ---------------------------
# Create Power Scores Sheet
# ---------------------------

power_scores_df = create_power_scores_sheet(weekly_total_tables, teams.values())

# ---------------------------
# Create Metric Sheets for Second Spreadsheet
# ---------------------------

metrics = {
    "Wins": "WINS",
    "Losses": "LOSSES",
    "Ties": "TIES",
    "Points For": "POINTS FOR",
    "Points Against": "POINTS AGAINST",
    "Net Points": "NET POINTS",
    "Win Percentage": "WIN PERCENTAGE",
    "Points Per Game": "POINTS PER GAME",
    "All-Play Wins": "ALL PLAY WINS",
    "All-Play Losses": "ALL PLAY LOSSES",
    "All-Play Win Percentage": "ALL PLAY WIN PERCENTAGE",
    "Magic Number": "MAGIC NUMBER",
    "Power Score": "POWER SCORE",
    "Tiebreak Score": "TIEBREAK SCORE",
    "Luck Rating": "LUCK SCORE",
    "Max Wins": "MAXIMUM WINS"
}

teams_sorted = total_table_df['Team'].tolist()
metric_dataframes = create_metric_sheets(weekly_total_tables, metrics, teams_sorted)

# ---------------------------
# Exporting to Excel with Conditional Formatting and Charts
# ---------------------------

filename = "fantasy_matchups.xlsx"
metrics_filename = "fantasy_matchups_metrics.xlsx"

# Define metrics you want to create charts for
metrics_to_chart = {
    "Points Against": "Points Against",
    # Add more metrics here if you want charts for them
}

with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
    master_df.to_excel(writer, sheet_name="Master Data", index=False)
    records_df.to_excel(writer, sheet_name="Records", index=False)
    points_df.to_excel(writer, sheet_name="Points", index=False)
    all_play_df.to_excel(writer, sheet_name="All-Play Records", index=False)
    total_table_df.to_excel(writer, sheet_name="Total Table", index=False)
    
    weekly_groups = df_cleaned.groupby('Week')
    for week, group in weekly_groups:
        sheet_name = f"Week {int(week)}"
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]
        group.to_excel(writer, sheet_name=sheet_name, index=False)
    
    for week, weekly_total_df in weekly_total_tables.items():
        sheet_name = f"Week {int(week)} Total Table"
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]
        weekly_total_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Add Data Studio Sheet
    data_studio_df.to_excel(writer, sheet_name="Data Studio", index=False)
    
    # Add Luck Scores Sheet
    luck_scores_df.to_excel(writer, sheet_name="Luck Scores", index=False)
    
    # Add Power Scores Sheet
    power_scores_df.to_excel(writer, sheet_name="Power Scores", index=False)
    
    workbook = writer.book
    
    # Apply Conditional Formatting to "Total Table"
    worksheet_total = writer.sheets["Total Table"]
    apply_conditional_formatting(workbook, worksheet_total, total_table_df, sheet_type="Total Table")
    
    # Apply Conditional Formatting to Weekly Total Tables
    for week, weekly_total_df in weekly_total_tables.items():
        sheet_name = f"Week {int(week)} Total Table"
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]
        
        worksheet_weekly_total = writer.sheets[sheet_name]
        apply_conditional_formatting(workbook, worksheet_weekly_total, weekly_total_df, sheet_type="Weekly Total Table")
    
    # Apply Conditional Formatting to Data Studio Sheet
    worksheet_data_studio = writer.sheets["Data Studio"]
    apply_conditional_formatting(workbook, worksheet_data_studio, data_studio_df, sheet_type="Data Studio")
    
    # Apply Conditional Formatting to Luck Scores Sheet
    worksheet_luck = writer.sheets["Luck Scores"]
    # Assuming "Week X" columns start from column B
    for i in range(1, len(luck_scores_df.columns)):
        col_letter = chr(66 + i - 1)  # 'B' is 66 in ASCII
        worksheet_luck.conditional_format(f'{col_letter}2:{col_letter}{len(luck_scores_df)+1}', 
                                          {'type': '3_color_scale',
                                           'min_color': "#FFCCCC",
                                           'mid_color': "#FFFFFF",
                                           'max_color': "#CCCCFF"})
    
    # Apply Conditional Formatting to Power Scores Sheet
    worksheet_power = writer.sheets["Power Scores"]
    for i in range(1, len(power_scores_df.columns)):
        col_letter = chr(66 + i - 1)  # 'B' is 66 in ASCII
        worksheet_power.conditional_format(f'{col_letter}2:{col_letter}{len(power_scores_df)+1}', 
                                           {'type': '3_color_scale',
                                            'min_color': "#FFCCCC",
                                            'mid_color': "#FFFFFF",
                                            'max_color': "#CCCCFF"})
    
    # Format Headers and Column Widths for "Total Table"
    header_format = workbook.add_format({'bold': True, 'bg_color': '#F9DA04'})
    worksheet_total.set_row(0, None, header_format)
    worksheet_total.set_column('A:A', 20)
    worksheet_total.set_column('B:S', 15)
    
    # Format Headers and Column Widths for Weekly Total Tables
    for week, weekly_total_df in weekly_total_tables.items():
        sheet_name = f"Week {int(week)} Total Table"
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]
        
        worksheet_weekly_total = writer.sheets[sheet_name]
        worksheet_weekly_total.set_row(0, None, header_format)
        worksheet_weekly_total.set_column('A:A', 20)
        worksheet_weekly_total.set_column('B:T', 15)
    
    # Format Headers and Column Widths for Data Studio Sheet
    worksheet_data_studio.set_row(0, None, header_format)
    worksheet_data_studio.set_column('A:A', 20)
    worksheet_data_studio.set_column('B:Z', 15)
    
    # Format Headers and Column Widths for Luck Scores Sheet
    worksheet_luck.set_row(0, None, header_format)
    worksheet_luck.set_column('A:A', 20)
    for i in range(1, len(luck_scores_df.columns)):
        col_letter = chr(66 + i - 1)  # 'B' is 66 in ASCII
        worksheet_luck.set_column(f'{col_letter}:{col_letter}', 15)
    
    # Format Headers and Column Widths for Power Scores Sheet
    worksheet_power.set_row(0, None, header_format)
    worksheet_power.set_column('A:A', 20)
    for i in range(1, len(power_scores_df.columns)):
        col_letter = chr(66 + i - 1)  # 'B' is 66 in ASCII
        worksheet_power.set_column(f'{col_letter}:{col_letter}', 15)
    
    # ---------------------------
    # Exporting Metrics to Second Excel Spreadsheet with Charts
    # ---------------------------

    with pd.ExcelWriter(metrics_filename, engine='xlsxwriter') as metrics_writer:
        for metric_name, metric_df in metric_dataframes.items():
            metric_df.to_excel(metrics_writer, sheet_name=metric_name, index=False)
        
        # Add Charts
        for metric_key, metric_name in metrics_to_chart.items():
            if metric_name in metric_dataframes:
                create_line_chart(metrics_writer, metric_name, metric_name, teams_sorted, sorted(weekly_total_tables.keys()))
        
        workbook_metrics = metrics_writer.book
        header_format_metrics = workbook_metrics.add_format({'bold': True, 'bg_color': '#F9DA04'})
        
        for metric_name, metric_df in metric_dataframes.items():
            worksheet = metrics_writer.sheets[metric_name]
            worksheet.set_row(0, None, header_format_metrics)
            worksheet.set_column('A:A', 20)
            for i in range(1, len(metric_df.columns)):
                col_letter = chr(66 + i - 1)  # 'B' is 66 in ASCII
                worksheet.set_column(f'{col_letter}:{col_letter}', 15)

    # ---------------------------
    # Run Script
    # ---------------------------

    print(f"DataFrames with 'Total Table', 'Data Studio', 'Luck Scores', 'Power Scores', and per-week 'Total Tables' sorted by 'Power Score' and formatted have been saved to {filename} successfully!")
    print(f"Metrics sheets for 'Wins', 'Losses', 'Ties', etc., along with their charts, have been saved to {metrics_filename} successfully!")