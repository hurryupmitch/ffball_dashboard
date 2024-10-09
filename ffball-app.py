import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.express as px
from io import BytesIO
from PIL import Image

# ---------------------------
# Streamlit App Configuration
# ---------------------------

# Attempt to set the page icon, handle missing favicon gracefully
try:
    st.set_page_config(
        page_title="🏈 Fantasy Football Dashboard",
        layout="wide",  # 'wide' layout is generally better for responsiveness
        initial_sidebar_state="collapsed",  # Sidebar collapsed by default since we're removing filters
        page_icon="assets/favicon.ico"  # Path to your favicon
    )
except FileNotFoundError:
    st.set_page_config(
        page_title="🏈 Fantasy Football Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon=":football:"  # Use a default emoji icon
    )
except Exception as e:
    st.error(f"An error occurred during page configuration: {e}")
    st.stop()

# ---------------------------
# Custom CSS for Mobile Responsiveness
# ---------------------------

def local_css():
    st.markdown("""
    <style>
    /* Make images responsive */
    img {
        max-width: 100%;
        height: auto;
    }

    /* Adjust header font size on mobile */
    @media (max-width: 768px) {
        h1 {
            font-size: 1.8rem;
        }
        h2 {
            font-size: 1.4rem;
        }
    }

    /* Hide certain elements on very small screens if necessary */
    @media (max-width: 480px) {
        .hide-on-mobile {
            display: none;
        }
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ---------------------------
# Configuration and Constants
# ---------------------------

LEAGUE_ID = "487244852"
SEASON_YEAR = "2024"
TOTAL_WEEKS = 14  # Total number of weeks in the season
PLAYOFF_SPOTS = 6  # Number of playoff spots

# Accessing credentials securely from Streamlit secrets with error handling
try:
    SWID = st.secrets["SWID"]
    ESPNS2 = st.secrets["ESPNS2"]
except KeyError as e:
    st.error(f"Missing secret key: {e}. Please set it in your Streamlit secrets.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while accessing secrets: {e}")
    st.stop()

# Cookies for authentication
cookies = {"SWID": SWID, "espn_s2": ESPNS2}

# ---------------------------
# Function Definitions
# ---------------------------

@st.cache_data(ttl=3600)
def fetch_data():
    """
    Fetches data from the ESPN Fantasy Football API.
    """
    url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
        f"seasons/{SEASON_YEAR}/segments/0/leagues/{LEAGUE_ID}?view=mMatchup&view=mScoreboard"
    )

    try:
        response = requests.get(url, cookies=cookies)
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching data: {e}")
        st.stop()

    if response.status_code != 200:
        st.error(f"Failed to fetch data: {response.status_code}")
        st.stop()

    data = response.json()

    # Validate expected keys
    if "teams" not in data or "schedule" not in data:
        st.error("API response is missing required data. Please check your league ID and API access.")
        st.stop()

    return data

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
                "All-Play Win Pct": win_pct
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
        {"Team": team, "Win Pct vs Median": win_pct_vs_median[team]}
        for team in teams
    ])

    return win_vs_median_df

def calculate_strength_of_schedule(df_cleaned, total_table_df):
    """
    Calculates the Strength of Schedule (SoS) for each team based on opponents' Win Percentage.

    Parameters:
    - df_cleaned (pd.DataFrame): Cleaned matchups data up to the current week.
    - total_table_df (pd.DataFrame): Current total standings with Win Pct.

    Returns:
    - sos_df (pd.DataFrame): DataFrame containing SoS for each team.
    """
    teams = total_table_df['Team'].unique()
    sos_dict = {}

    for team in teams:
        # Find all matchups where this team was either Home or Away
        team_matches = df_cleaned[(df_cleaned['Home Team'] == team) | (df_cleaned['Away Team'] == team)]
        
        # Get opponents, excluding "Bye"
        opponents = team_matches.apply(
            lambda row: row['Away Team'] if row['Home Team'] == team else row['Home Team'],
            axis=1
        )
        opponents = opponents[opponents != "Bye"]
        
        # Get opponents' Win Pct from total_table_df
        opponents_stats = total_table_df[total_table_df['Team'].isin(opponents)]['Win Pct']
        
        if len(opponents_stats) > 0:
            sos = opponents_stats.mean()
        else:
            sos = np.nan  # Assign NaN if no opponents have been played
        
        sos_dict[team] = sos

    sos_df = pd.DataFrame({
        "Team": list(sos_dict.keys()),
        "Strength of Schedule": list(sos_dict.values())
    })

    return sos_df

def generate_total_table(records_df, points_df, all_play_df, current_week, df_cleaned):
    total_df = pd.merge(records_df, points_df, on="Team")
    total_df["Net Points"] = total_df["Points For"] - total_df["Points Against"]
    total_df["Total Games"] = total_df["Wins"] + total_df["Losses"] + total_df["Ties"]
    total_df["Win Pct"] = (total_df["Wins"] + 0.5 * total_df["Ties"]) / total_df["Total Games"]
    total_df["PPG"] = total_df["Points For"] / total_df["Total Games"]

    all_play_agg = all_play_df.groupby('Team').agg({
        "All-Play Wins": "sum",
        "All-Play Losses": "sum",
        "All-Play Win Pct": "mean"
    }).reset_index()

    total_df = pd.merge(total_df, all_play_agg, on="Team", how='left')
    total_df["Luck Differential"] = total_df["Win Pct"] - total_df["All-Play Win Pct"]
    total_df["Standings Score"] = total_df["Win Pct"] + (total_df["Points For"] / 10000)

    total_df["Rem. Games"] = TOTAL_WEEKS - current_week
    total_df["Max Wins"] = total_df["Wins"] + total_df["Rem. Games"]

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
                               (total_df["Points For"] * total_df["Win Pct"]) + \
                               (total_df["Points For"] * total_df["Win Pct vs Median"])

    # Calculate Luck Rating based on Difference
    conditions = [
        (total_df["Luck Differential"] >= 0.35),
        (total_df["Luck Differential"] >= 0.25),
        (total_df["Luck Differential"] >= 0.15),
        (total_df["Luck Differential"] == 0),
        (total_df["Luck Differential"] >= -0.1),
        (total_df["Luck Differential"] >= -0.2),
        (total_df["Luck Differential"] >= -0.3),
        (total_df["Luck Differential"] < -0.3)
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

    # Calculate Strength of Schedule
    sos_df = calculate_strength_of_schedule(df_cleaned, total_df)
    total_df = pd.merge(total_df, sos_df, on="Team", how="left")

    # **New Step: Calculate Schedule Difficulty Rank**
    total_df["Schedule Difficulty Rank"] = total_df["Strength of Schedule"].rank(ascending=False, method='dense').astype(int)

    columns_order = [
        "Team",
        "Wins",
        "Losses",
        "Ties",
        "Standings Score",
        "Win Pct",
        "Power Score",
        "Points For",
        "Points Against",
        "Net Points",
        "PPG",
        "All-Play Wins",
        "All-Play Losses",
        "All-Play Win Pct",
        "Luck Differential",
        "Luck Rating",
        "Schedule Difficulty Rank",  # Newly added metric
        "Rem. Games",
        "Max Wins",
        "Win Pct vs Median",
        "Magic Number"
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

def sanitize_sheet_name(name):
    """
    Sanitizes the sheet name to ensure it is <=30 characters and replaces invalid characters.
    """
    name = name.replace("Pct", "Pct")
    name = name.replace("/", "_").replace("\\", "_")  # Replace invalid characters
    if len(name) > 30:
        name = name[:30]
    return name

def create_download_link(df, filename, sheet_name="Sheet1"):
    """
    Creates a downloadable Excel file from a DataFrame with a sanitized sheet name.
    """
    sanitized_sheet_name = sanitize_sheet_name(sheet_name)
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=sanitized_sheet_name, index=False)
    except Exception as e:
        st.error(f"An error occurred while creating the Excel file: {e}")
        return
    processed_data = output.getvalue()
    return st.download_button(
        label="📥 Download Excel",
        data=processed_data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ---------------------------
# Main Script Execution within Streamlit
# ---------------------------

def main():
    # ---------------------------
    # Header with Logo
    # ---------------------------
    try:
        logo = Image.open("assets/logo.png")
        st.image(logo, width=300)  # Adjust the width as needed
    except FileNotFoundError:
        st.warning("Logo image not found. Please ensure 'assets/logo.png' exists.")
    except Exception as e:
        st.error(f"An error occurred while loading the logo: {e}")

    st.title("🏈 Fantasy Football Dashboard")
    st.subheader("Welcome to the nerd zone where you can see just how poorly your team stacks up against all the other mediocre squads in the league.")

    # ---------------------------
    # Navigation
    # ---------------------------
    selection = st.sidebar.radio("Navigate to", ["Overview", "Team Analysis", "Weekly Matchups", "Metrics Over Time", "Advanced Visualizations"])

    # ---------------------------
    # Fetch and Prepare Data
    # ---------------------------
    data = fetch_data()

    # Extract teams
    teams = {}
    for team in data["teams"]:
        team_id = team["id"]
        team_name = team.get("name") or team.get("nickname") or f"Team {team_id}"
        teams[team_id] = team_name

    # Parse Matchup Data
    matchups = []
    for matchup in data["schedule"]:
        week = matchup["matchupPeriodId"]

        home_team_id = matchup["home"]["teamId"]
        home_team_name = teams.get(home_team_id, f"Team {home_team_id}")
        home_score = matchup["home"]["totalPoints"]

        if "away" in matchup and matchup["away"] is not None:
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

    # Data Cleaning
    df_cleaned = master_df[
        ~((master_df["Home Score"] == 0.00) & (master_df["Away Score"] == 0.00))
    ]

    # Calculate Metrics
    records_df = calculate_team_records(df_cleaned)
    points_df = calculate_points(df_cleaned)
    all_play_df = calculate_all_play_records(df_cleaned)
    current_week = df_cleaned['Week'].max()
    total_table_df = generate_total_table(records_df, points_df, all_play_df, current_week, df_cleaned)
    weekly_total_tables = generate_weekly_total_tables(df_cleaned, all_play_df)

    # Define All Tracked Metrics
    all_metrics = [
        "Wins", "Losses", "Ties", "Points For", "Points Against",
        "Net Points", "Win Pct", "PPG",
        "All-Play Wins", "All-Play Losses", "All-Play Win Pct",
        "Luck Differential", "Magic Number",
        "Power Score", "Standings Score", "Rem. Games", "Max Wins",
        "Win Pct vs Median", "Luck Rating",
        "Strength of Schedule",  # Retained for detailed views
        "Schedule Difficulty Rank"  # Newly added metric
    ]

    # ---------------------------
    # Overview Page
    # ---------------------------
    if selection == "Overview":
        st.header("🏆 League Overview")

        # Display Total Table
        st.subheader("Total Table")
        st.dataframe(total_table_df)

        # Download Total Table
        create_download_link(
            total_table_df,
            "Total_Table.xlsx",
            sheet_name="Total Table"
        )

        # Display Points For by Team
        st.subheader("Points For by Team")
        fig_pf = px.bar(
            total_table_df,
            x='Team',
            y='Points For',
            title='Points For by Team',
            color='Points For',
            text='Points For'
        )
        fig_pf.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig_pf.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig_pf, use_container_width=True)

        # Display Points Against by Team
        st.subheader("Points Against by Team")
        fig_pa = px.bar(
            total_table_df,
            x='Team',
            y='Points Against',
            title='Points Against by Team',
            color='Points Against',
            text='Points Against'
        )
        fig_pa.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig_pa.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig_pa, use_container_width=True)

        # Display Net Points
        st.subheader("Net Points by Team")
        fig_np = px.bar(
            total_table_df,
            x='Team',
            y='Net Points',
            title='Net Points by Team',
            color='Net Points',
            text='Net Points'
        )
        fig_np.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig_np.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig_np, use_container_width=True)

        # Display Strength of Schedule Rank
        st.subheader("Schedule Difficulty Rank by Team")
        fig_rank = px.bar(
            total_table_df,
            x='Team',
            y='Schedule Difficulty Rank',
            title='Schedule Difficulty Rank by Team (1 = Hardest)',
            color='Schedule Difficulty Rank',
            text='Schedule Difficulty Rank'
        )
        fig_rank.update_traces(texttemplate='%{text}', textposition='outside')
        fig_rank.update_layout(yaxis=dict(autorange='reversed'), uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig_rank, use_container_width=True)

    # ---------------------------
    # Team Analysis Page
    # ---------------------------
    elif selection == "Team Analysis":
        st.header("🔍 Team Analysis")

        # Select Team
        teams_available = sorted(total_table_df['Team'].unique())
        selected_team = st.selectbox("Select Team", options=teams_available)

        # Apply Team Filter
        team_record = total_table_df[total_table_df['Team'] == selected_team]
        if team_record.empty:
            st.warning("No data available for the selected team.")
        else:
            # Display Team Record
            st.subheader(f"Record for {selected_team}")
            st.table(team_record[[
                'Wins', 'Losses', 'Ties', 'Win Pct',
                'Points For', 'Points Against', 'Net Points',
                'Strength of Schedule', 'Schedule Difficulty Rank'
            ]])

            # Display Weekly Performance
            st.subheader(f"Weekly Performance for {selected_team}")

            # Select Metric
            metric = st.selectbox(
                "Select Metric",
                options=all_metrics,
                index=all_metrics.index("Points For") if "Points For" in all_metrics else 0
            )

            # Prepare Data for Plotting
            weeks = sorted(weekly_total_tables.keys())
            metric_values = []
            for week in weeks:
                weekly_df = weekly_total_tables[week]
                if metric in weekly_df.columns:
                    value = weekly_df[weekly_df['Team'] == selected_team][metric].values
                    value = value[0] if len(value) > 0 else 0
                else:
                    value = 0
                metric_values.append(value)

            # Plot Metric Over Weeks
            fig_metric = px.line(
                x=weeks,
                y=metric_values,
                markers=True,
                title=f'{metric} Over Weeks for {selected_team}',
                labels={'x': 'Week', 'y': metric}
            )
            st.plotly_chart(fig_metric, use_container_width=True)

            # Download Team Data
            team_weekly_df = pd.DataFrame({
                "Week": weeks,
                metric: metric_values
            })
            create_download_link(
                team_weekly_df,
                f"{selected_team}_{metric}_Weekly_Data.xlsx",
                sheet_name=f"{selected_team} {metric}"
            )

            # Display Strength of Schedule
            st.subheader(f"Schedule Difficulty Rank for {selected_team}")
            sos_rank = team_record['Schedule Difficulty Rank'].values[0]
            if not np.isnan(sos_rank):
                st.metric(label="Schedule Difficulty Rank", value=f"{sos_rank}")
            else:
                st.write("No matchups played yet to calculate Schedule Difficulty Rank.")

    # ---------------------------
    # Weekly Matchups Page
    # ---------------------------
    elif selection == "Weekly Matchups":
        st.header("📅 Weekly Matchups")

        # Select Week
        weeks_available = sorted(weekly_total_tables.keys())
        default_week_index = min(len(weeks_available)-1, current_week-1) if current_week <= TOTAL_WEEKS else 0
        selected_week = st.selectbox(
            "Select Week",
            options=weeks_available,
            index=default_week_index
        )

        # Apply Week Filter
        if selected_week not in weekly_total_tables:
            st.warning("Selected week data is not available.")
        else:
            # Display Total Table for Selected Week
            st.subheader(f"Total Table for Week {selected_week}")
            weekly_table = weekly_total_tables.get(selected_week)
            if weekly_table is not None and not weekly_table.empty:
                st.dataframe(weekly_table)

                # Download Weekly Total Table
                create_download_link(
                    weekly_table,
                    f"Week_{selected_week}_Total_Table.xlsx",
                    sheet_name=f"Week {selected_week} Total Table"
                )
            else:
                st.warning("No data available for the selected week.")

            # Additional: Display Matchups Details
            st.subheader(f"Matchups Details for Week {selected_week}")
            matchups_week = master_df[
                (master_df['Week'] == selected_week) &
                (master_df['Home Team'].isin(total_table_df['Team'])) &
                (master_df['Away Team'].isin(total_table_df['Team'].tolist() + ["Bye"]))
            ]
            if not matchups_week.empty:
                st.table(matchups_week)
            else:
                st.warning("No matchup details available for the selected week.")

    # ---------------------------
    # Metrics Over Time Page
    # ---------------------------
    elif selection == "Metrics Over Time":
        st.header("📈 Metrics Over Time")

        # Select Metric
        metric = st.selectbox(
            "Select Metric to Visualize",
            options=all_metrics,
            index=all_metrics.index("Wins") if "Wins" in all_metrics else 0
        )

        if not metric:
            st.warning("Please select a metric to visualize.")
        else:
            # Initialize Plot
            fig_metric_over_time = px.line(
                title=f'{metric} Over Time',
                labels={'x': 'Week', 'y': metric}
            )

            # Plot for Each Team
            for team in sorted(total_table_df['Team'].unique()):
                metric_values = []
                for week in sorted(weekly_total_tables.keys()):
                    weekly_df = weekly_total_tables[week]
                    if metric in weekly_df.columns:
                        value = weekly_df[weekly_df['Team'] == team][metric].values
                        value = value[0] if len(value) > 0 else 0
                    else:
                        value = 0
                    metric_values.append(value)
                fig_metric_over_time.add_scatter(
                    x=sorted(weekly_total_tables.keys()),
                    y=metric_values,
                    mode='lines+markers',
                    name=team
                )

            fig_metric_over_time.update_layout(
                xaxis_title='Week',
                yaxis_title=metric,
                legend_title='Team',
                template='plotly_dark'
            )

            st.plotly_chart(fig_metric_over_time, use_container_width=True)

            # Download Metrics Over Time
            # Create a combined DataFrame for all teams
            metrics_over_time_df = pd.DataFrame({'Week': sorted(weekly_total_tables.keys())})
            for team in sorted(total_table_df['Team'].unique()):
                metrics_over_time_df[team] = [
                    weekly_total_tables[week].set_index('Team').loc[team, metric] if (team in weekly_total_tables[week].set_index('Team').index and metric in weekly_total_tables[week].columns) else 0
                    for week in sorted(weekly_total_tables.keys())
                ]
            create_download_link(
                metrics_over_time_df,
                f"{metric}_Over_Time.xlsx",
                sheet_name=f"{metric} Over Time"
            )

    # ---------------------------
    # Advanced Visualizations Page
    # ---------------------------
    elif selection == "Advanced Visualizations":
        st.header("🔧 Advanced Visualizations")

        # Select Visualization Type
        viz_type = st.selectbox("Select Visualization Type", options=["Scatter Plot", "Pie Chart", "Heatmap", "Line Graph"])

        if viz_type == "Scatter Plot":
            st.subheader("🔀 Scatter Plot")

            # Select X and Y Metrics
            x_metric = st.selectbox(
                "Select X-axis Metric",
                options=all_metrics,
                index=all_metrics.index("Points For") if "Points For" in all_metrics else 0
            )
            y_metric = st.selectbox(
                "Select Y-axis Metric",
                options=all_metrics,
                index=all_metrics.index("Points Against") if "Points Against" in all_metrics else 1
            )

            if x_metric and y_metric:
                fig_scatter = px.scatter(
                    total_table_df,
                    x=x_metric,
                    y=y_metric,
                    color="Team",
                    size="Win Pct",
                    hover_data=["Wins", "Losses", "Ties", "Strength of Schedule", "Schedule Difficulty Rank"],
                    title=f'{y_metric} vs {x_metric}',
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

                # Download Scatter Plot Data
                scatter_data = total_table_df[[
                    'Team', x_metric, y_metric, 'Wins', 'Losses', 'Ties', 'Win Pct', 'Strength of Schedule', 'Schedule Difficulty Rank'
                ]]
                create_download_link(
                    scatter_data,
                    f"Scatter_Plot_{y_metric}_vs_{x_metric}.xlsx",
                    sheet_name=f"{y_metric} vs {x_metric}"
                )

        elif viz_type == "Pie Chart":
            st.subheader("🥧 Pie Chart")

            # Select Metric for Pie Chart
            pie_metric = st.selectbox(
                "Select Metric for Pie Chart",
                options=all_metrics,
                index=all_metrics.index("Points For") if "Points For" in all_metrics else 0
            )

            if pie_metric:
                fig_pie = px.pie(
                    total_table_df,
                    names='Team',
                    values=pie_metric,
                    title=f'{pie_metric} Distribution by Team',
                    hole=0.3
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Download Pie Chart Data
                pie_data = total_table_df[['Team', pie_metric]]
                create_download_link(
                    pie_data,
                    f"Pie_Chart_{pie_metric}_Distribution.xlsx",
                    sheet_name=f"{pie_metric} Distribution"
                )

        elif viz_type == "Heatmap":
            st.subheader("🔥 Heatmap")

            # Select Metrics for Heatmap
            heatmap_metrics = st.multiselect(
                "Select Metrics for Heatmap",
                options=all_metrics,
                default=["Win Pct", "Points For", "Points Against"]
            )

            if heatmap_metrics and len(heatmap_metrics) >= 2:
                heatmap_data = total_table_df[["Team"] + heatmap_metrics].set_index("Team")
                fig_heatmap = px.imshow(
                    heatmap_data,
                    labels=dict(x="Metrics", y="Teams", color="Value"),
                    x=heatmap_metrics,
                    y=total_table_df['Team'],
                    aspect="auto",
                    title="Heatmap of Selected Metrics"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

                # Download Heatmap Data
                heatmap_download_df = heatmap_data.reset_index()
                create_download_link(
                    heatmap_download_df,
                    f"Heatmap_Selected_Metrics.xlsx",
                    sheet_name="Heatmap Data"
                )
            else:
                st.warning("Please select at least two metrics for the heatmap.")

        elif viz_type == "Line Graph":
            st.subheader("📈 Line Graph")

            # Select Metrics for Line Graph
            line_metrics = st.multiselect(
                "Select Metrics for Line Graph",
                options=all_metrics,
                default=["Win Pct", "Points For"]
            )

            if line_metrics:
                fig_line = px.line(
                    title="Selected Metrics Over Time",
                    labels={'x': 'Week', 'y': 'Value'}
                )

                for metric in line_metrics:
                    for team in sorted(total_table_df['Team'].unique()):
                        metric_values = []
                        for week in sorted(weekly_total_tables.keys()):
                            weekly_df = weekly_total_tables[week]
                            if metric in weekly_df.columns:
                                try:
                                    value = weekly_df[weekly_df['Team'] == team][metric].values
                                    value = value[0] if len(value) > 0 else 0
                                except KeyError:
                                    value = 0
                            else:
                                value = 0
                            metric_values.append(value)
                        fig_line.add_scatter(
                            x=sorted(weekly_total_tables.keys()),
                            y=metric_values,
                            mode='lines+markers',
                            name=f"{team} - {metric}"
                        )

                fig_line.update_layout(
                    xaxis_title='Week',
                    yaxis_title='Value',
                    legend_title='Team - Metric',
                    template='plotly_dark'
                )

                st.plotly_chart(fig_line, use_container_width=True)

                # Download Line Graph Data
                # Create a combined DataFrame for all teams and selected metrics
                line_graph_df = pd.DataFrame({'Week': sorted(weekly_total_tables.keys())})
                for metric in line_metrics:
                    for team in sorted(total_table_df['Team'].unique()):
                        column_name = f"{team} - {metric}"
                        line_graph_df[column_name] = [
                            weekly_total_tables[week].set_index('Team').loc[team, metric] if (team in weekly_total_tables[week].set_index('Team').index and metric in weekly_total_tables[week].columns) else 0
                            for week in sorted(weekly_total_tables.keys())
                        ]
                create_download_link(
                    line_graph_df,
                    f"Line_Graph_Selected_Metrics.xlsx",
                    sheet_name="Line Graph Data"
                )
            else:
                st.warning("Please select at least one metric for the line graph.")

# ---------------------------
# Run the App
# ---------------------------
if __name__ == "__main__":
    main()
