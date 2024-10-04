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

st.set_page_config(
    page_title="🏈 Fantasy Football Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="assets/favicon.ico"  # Path to your favicon
)

# ---------------------------
# Configuration and Constants
# ---------------------------

LEAGUE_ID = "487244852"
SEASON_YEAR = "2024"
TOTAL_WEEKS = 14  # Total number of weeks in the season
PLAYOFF_SPOTS = 6  # Number of playoff spots

# Accessing credentials securely from Streamlit secrets
SWID = st.secrets["SWID"]
ESPNS2 = st.secrets["ESPNS2"]

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

    response = requests.get(url, cookies=cookies)

    if response.status_code != 200:
        st.error(f"Failed to fetch data: {response.status_code}")
        st.stop()

    data = response.json()
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
    total_df["Difference (Actual vs All-Play Win Percentage)"] = total_df["Win Percentage"] - total_df["All-Play Win Percentage"]
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

    # Calculate Luck Rating based on Difference
    conditions = [
        (total_df["Difference (Actual vs All-Play Win Percentage)"] >= 0.35),
        (total_df["Difference (Actual vs All-Play Win Percentage)"] >= 0.25),
        (total_df["Difference (Actual vs All-Play Win Percentage)"] >= 0.15),
        (total_df["Difference (Actual vs All-Play Win Percentage)"] == 0),
        (total_df["Difference (Actual vs All-Play Win Percentage)"] >= -0.1),
        (total_df["Difference (Actual vs All-Play Win Percentage)"] >= -0.2),
        (total_df["Difference (Actual vs All-Play Win Percentage)"] >= -0.3),
        (total_df["Difference (Actual vs All-Play Win Percentage)"] < -0.3)
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
        "Difference (Actual vs All-Play Win Percentage)",
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

def sanitize_sheet_name(name):
    """
    Sanitizes the sheet name to ensure it is <=30 characters and replaces 'Percentage' with 'Pct'.
    """
    name = name.replace("Percentage", "Pct")
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
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sanitized_sheet_name, index=False)
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
    logo = Image.open("assets/logo.png")
    st.image(logo, width=300)  # Adjust the width as needed

    st.title("🏈 Fantasy Football Dashboard")
    st.markdown("""
    Welcome to your personalized Fantasy Football Dashboard! Monitor your league's performance, track team statistics, and visualize data with interactive charts and tables.
    """)

    # ---------------------------
    # Sidebar with Logo
    # ---------------------------
    st.sidebar.image("assets/logo.png", width=200)  # Adjust the width as needed

    # Navigation
    selection = st.sidebar.radio("Go to", ["Overview", "Team Analysis", "Weekly Matchups", "Metrics Over Time", "Advanced Visualizations", "About"])

    # Filters
    st.sidebar.subheader("Filters")
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
        "Net Points", "Win Percentage", "Points Per Game",
        "All-Play Wins", "All-Play Losses", "All-Play Win Percentage",
        "Difference (Actual vs All-Play Win Percentage)", "Magic Number",
        "Power Score", "Tiebreak Score", "Remaining Games", "Max Wins"
    ]

    # Dynamic Filters
    selected_teams = st.sidebar.multiselect(
        "Select Teams",
        options=sorted(total_table_df['Team'].unique()),
        default=sorted(total_table_df['Team'].unique())
    )
    selected_weeks = st.sidebar.slider(
        "Select Weeks",
        min_value=1,
        max_value=TOTAL_WEEKS,
        value=(1, TOTAL_WEEKS),
        step=1
    )
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics",
        options=all_metrics,
        default=["Wins", "Losses", "Points For", "Points Against"]
    )

    # Apply Filters
    filtered_total_df = total_table_df[
        total_table_df['Team'].isin(selected_teams)
    ]
    filtered_weekly_total_tables = {
        week: df[df['Team'].isin(selected_teams)]
        for week, df in weekly_total_tables.items()
        if selected_weeks[0] <= week <= selected_weeks[1]
    }

    # ---------------------------
    # Overview Page
    # ---------------------------
    if selection == "Overview":
        st.header("🏆 League Overview")

        # Display Total Table
        st.subheader("Total Table")
        st.dataframe(filtered_total_df)

        # Download Total Table
        create_download_link(
            filtered_total_df,
            "Total_Table.xlsx",
            sheet_name="Total Table"
        )

        # Display Points For by Team
        st.subheader("Points For by Team")
        fig_pf = px.bar(
            filtered_total_df,
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
            filtered_total_df,
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
            filtered_total_df,
            x='Team',
            y='Net Points',
            title='Net Points by Team',
            color='Net Points',
            text='Net Points'
        )
        fig_np.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig_np.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig_np, use_container_width=True)

    # ---------------------------
    # Team Analysis Page
    # ---------------------------
    elif selection == "Team Analysis":
        st.header("🔍 Team Analysis")

        # Select Team
        teams_available = sorted(total_table_df['Team'].unique())
        selected_team = st.selectbox("Select Team", options=teams_available)

        # Apply Team Filter
        team_record = filtered_total_df[filtered_total_df['Team'] == selected_team]
        if team_record.empty:
            st.warning("No data available for the selected team and filters.")
        else:
            # Display Team Record
            st.subheader(f"Record for {selected_team}")
            st.table(team_record[[
                'Wins', 'Losses', 'Ties', 'Win Percentage',
                'Points For', 'Points Against', 'Net Points'
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
            weeks = sorted(filtered_weekly_total_tables.keys())
            metric_values = []
            for week in weeks:
                weekly_df = filtered_weekly_total_tables[week]
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

    # ---------------------------
    # Weekly Matchups Page
    # ---------------------------
    elif selection == "Weekly Matchups":
        st.header("📅 Weekly Matchups")

        # Select Week
        weeks_available = sorted(weekly_total_tables.keys())
        selected_week = st.selectbox(
            "Select Week",
            options=weeks_available,
            index=min(len(weeks_available)-1, current_week-1) if current_week <= TOTAL_WEEKS else 0
        )

        # Apply Week Filter
        if selected_week not in weekly_total_tables:
            st.warning("Selected week data is not available.")
        else:
            # Display Total Table for Selected Week
            st.subheader(f"Total Table for Week {selected_week}")
            weekly_table = filtered_weekly_total_tables.get(selected_week)
            if weekly_table is not None and not weekly_table.empty:
                st.dataframe(weekly_table)

                # Download Weekly Total Table
                create_download_link(
                    weekly_table,
                    f"Week_{selected_week}_Total_Table.xlsx",
                    sheet_name=f"Week {selected_week} Total Table"
                )
            else:
                st.warning("No data available for the selected week and filters.")

            # Additional: Display Matchups Details
            st.subheader(f"Matchups Details for Week {selected_week}")
            matchups_week = master_df[
                (master_df['Week'] == selected_week) &
                (master_df['Home Team'].isin(selected_teams)) &
                (master_df['Away Team'].isin(selected_teams + ["Bye"]))
            ]
            if not matchups_week.empty:
                st.table(matchups_week)
            else:
                st.warning("No matchup details available for the selected week and filters.")

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
            for team in sorted(filtered_total_df['Team'].unique()):
                metric_values = []
                for week in sorted(filtered_weekly_total_tables.keys()):
                    weekly_df = weekly_total_tables[week]
                    if metric in weekly_df.columns:
                        value = weekly_df[weekly_df['Team'] == team][metric].values
                        value = value[0] if len(value) > 0 else 0
                    else:
                        value = 0
                    metric_values.append(value)
                fig_metric_over_time.add_scatter(
                    x=sorted(filtered_weekly_total_tables.keys()),
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
            metrics_over_time_df = pd.DataFrame({'Week': sorted(filtered_weekly_total_tables.keys())})
            for team in sorted(filtered_total_df['Team'].unique()):
                metrics_over_time_df[team] = [
                    weekly_total_tables[week].set_index('Team').loc[team, metric] if team in weekly_total_tables[week].set_index('Team').index else 0
                    for week in sorted(filtered_weekly_total_tables.keys())
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
        viz_type = st.selectbox("Select Visualization Type", options=["Scatter Plot", "Pie Chart", "Heatmap"])

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
                    filtered_total_df,
                    x=x_metric,
                    y=y_metric,
                    color="Team",
                    size="Win Percentage",
                    hover_data=["Wins", "Losses", "Ties"],
                    title=f'{y_metric} vs {x_metric}',
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

                # Download Scatter Plot Data
                scatter_data = filtered_total_df[[
                    'Team', x_metric, y_metric, 'Wins', 'Losses', 'Ties', 'Win Percentage'
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
                    filtered_total_df,
                    names='Team',
                    values=pie_metric,
                    title=f'{pie_metric} Distribution by Team',
                    hole=0.3
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Download Pie Chart Data
                pie_data = filtered_total_df[['Team', pie_metric]]
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
                default=["Win Percentage", "Points For", "Points Against"]
            )

            if heatmap_metrics and len(heatmap_metrics) >= 2:
                heatmap_data = filtered_total_df[["Team"] + heatmap_metrics].set_index("Team")
                fig_heatmap = px.imshow(
                    heatmap_data,
                    labels=dict(x="Metrics", y="Teams", color="Value"),
                    x=heatmap_metrics,
                    y=filtered_total_df['Team'],
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

    # ---------------------------
    # About Page
    # ---------------------------
    elif selection == "About":
        st.header("ℹ️ About This Dashboard")
        st.markdown("""
        This Fantasy Football Dashboard provides comprehensive insights into your league's performance. Track team records, points, and various metrics over the season with interactive charts and tables.

        **Features:**
        - **League Overview:** View the overall standings, points for and against, and net points for each team.
        - **Team Analysis:** Dive into individual team performance, including weekly metrics.
        - **Weekly Matchups:** Analyze matchups and results on a week-by-week basis.
        - **Metrics Over Time:** Visualize selected metrics across the season for all teams.
        - **Advanced Visualizations:** Create scatter plots, pie charts, and heatmaps for deeper analysis.
        - **Filters & Download Options:** Customize views with filters and download data as Excel files.

        **Built With:**
        - [Streamlit](https://streamlit.io/) for the interactive dashboard.
        - [Pandas](https://pandas.pydata.org/) for data manipulation.
        - [Plotly Express](https://plotly.com/python/plotly-express/) for interactive visualizations.

        **Data Source:** ESPN Fantasy Football API

        *Feel free to customize and enhance this dashboard to suit your needs!*
        """)

    # ---------------------------
    # Footer
    # ---------------------------
    st.markdown("---")
    st.markdown("© 2024 poorsportsmen fantasy fun league. All rights reserved.")

# ---------------------------
# Run the App
# ---------------------------
if __name__ == "__main__":
    main()