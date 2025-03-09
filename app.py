"""
Fitness Tracker Dashboard Application

A comprehensive analytics dashboard for Fitbit data that provides insights into
physical activity, sleep patterns, heart rate, and weight metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import logging
import time

# Import custom modules
from src.data_processing import load_and_process_all_data, get_all_users, save_processed_data
from src.analysis import (detect_activity_patterns, analyze_sleep_activity_correlation,
                         detect_heart_rate_anomalies, predict_calories_from_activity,
                         perform_time_series_decomposition, calculate_user_similarity,
                         analyze_weekly_patterns, identify_consistent_active_periods)
from src.visualization import (create_activity_summary_cards, plot_daily_steps_trend,
                              plot_activity_heatmap, plot_sleep_patterns,
                              plot_heart_rate_analysis, plot_weight_trend,
                              plot_activity_distribution, plot_health_score_components,
                              plot_activity_correlation, create_achievement_badges)
from src.utils import (date_to_str, str_to_date, get_date_range_options,
                       create_comparison_dataframe, format_duration,
                       export_data_to_csv, calculate_progress_percentage,
                       generate_daily_insights, get_default_goals,
                       save_user_goals, load_user_goals, format_number)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Fitness Tracker Dashboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define constants
RAW_DATA_DIR = os.path.join('data', 'raw', 'fitbit')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Define CSS styles
def load_css():
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 600;
            color: #1E88E5;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            font-weight: 500;
            color: #333;
            margin-bottom: 1rem;
        }
        .card {
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 600;
            color: #1E88E5;
            margin-bottom: 0.2rem;
        }
        .metric-label {
            font-size: 1rem;
            color: #6c757d;
            margin-bottom: 0.5rem;
        }
        .achievement-badge {
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
            border-left: 5px solid #4CAF50;
        }
        .badge-gold {
            border-left: 5px solid #FFD700;
        }
        .badge-silver {
            border-left: 5px solid #C0C0C0;
        }
        .badge-bronze {
            border-left: 5px solid #CD7F32;
        }
        .insight-card {
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
            border-left: 5px solid #1E88E5;
        }
        .suggestion-card {
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
            border-left: 5px solid #FFC107;
        }
        .progress-container {
            margin-bottom: 1rem;
        }
        .progress-bar {
            height: 0.5rem;
            border-radius: 0.25rem;
            background-color: #e9ecef;
        }
        .progress-fill {
            height: 100%;
            border-radius: 0.25rem;
            background-color: #1E88E5;
        }
        </style>
    """, unsafe_allow_html=True)


# Function to load data
@st.cache_data
def load_data():
    """
    Load and process all Fitbit data.
    
    Returns:
    --------
    dict
        Dictionary containing all processed dataframes
    """
    with st.spinner("Loading and processing data... This may take a few moments."):
        # Check if processed data exists
        if os.path.exists(os.path.join(PROCESSED_DATA_DIR, "merged_daily.csv")):
            logger.info("Loading processed data from files")
            
            data_dict = {}
            processed_files = [
                "daily_activity", "sleep", "heart_rate", "heart_rate_daily",
                "weight", "hourly_steps", "hourly_calories", "hourly_intensities",
                "merged_daily"
            ]
            
            for key in processed_files:
                file_path = os.path.join(PROCESSED_DATA_DIR, f"{key}.csv")
                if os.path.exists(file_path):
                    data_dict[key] = pd.read_csv(file_path, low_memory=False)
                    
                    # Convert date columns
                    date_cols = [col for col in data_dict[key].columns if 'Date' in col or 'Day' in col or 'Time' in col]
                    for col in date_cols:
                        try:
                            if col in data_dict[key].columns:
                                data_dict[key][col] = pd.to_datetime(data_dict[key][col], errors='coerce')
                        except Exception as e:
                            logger.warning(f"Error converting {col} to datetime: {str(e)}")
                else:
                    data_dict[key] = None
        else:
            logger.info("Processing raw data")
            # Load and process data from raw files
            data_dict = load_and_process_all_data(RAW_DATA_DIR)
            
            # Save processed data for future use
            save_processed_data(data_dict, PROCESSED_DATA_DIR)
        
        # Get all users
        all_users = get_all_users(data_dict)
        data_dict['all_users'] = all_users
        
        # Debug: Display available columns in key dataframes
        for key in ["daily_activity", "merged_daily"]:
            if key in data_dict and data_dict[key] is not None:
                logger.info(f"Columns in {key}: {data_dict[key].columns.tolist()}")
        
        return data_dict


# User selection component
def user_selector(data_dict):
    """
    Create a user selection dropdown.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed dataframes
        
    Returns:
    --------
    str
        Selected user ID
    """
    if 'all_users' not in data_dict or not data_dict['all_users']:
        st.error("No user data found. Please check your data files.")
        return None
    
    # Get user list
    users = data_dict['all_users']
    
    # If there's a previous selection, use it as default
    default_index = 0
    if 'selected_user_index' in st.session_state:
        default_index = st.session_state.selected_user_index
    
    # Create the dropdown
    selected_index = st.selectbox(
        "Select User:",
        range(len(users)),
        format_func=lambda i: f"User {users[i]}",
        index=default_index
    )
    
    # Store the selected index in session state
    st.session_state.selected_user_index = selected_index
    
    return users[selected_index]


# Date range selector component
def date_range_selector(data_dict, user_id):
    """
    Create a date range selection component.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed dataframes
    user_id : str
        Selected user ID
        
    Returns:
    --------
    tuple
        (start_date, end_date)
    """
    # Get date range for the selected user
    user_dates = []
    
    for key in ['daily_activity', 'sleep', 'weight']:
        if key in data_dict and data_dict[key] is not None and 'Id' in data_dict[key].columns:
            user_data = data_dict[key][data_dict[key]['Id'] == user_id]
            if 'Date' in user_data.columns and not user_data.empty:
                user_dates.extend(user_data['Date'].tolist())
    
    if not user_dates:
        st.warning("No date data found for the selected user.")
        return None, None
    
    # Get min and max dates
    min_date = min(user_dates).date() if isinstance(min(user_dates), pd.Timestamp) else min(user_dates)
    max_date = max(user_dates).date() if isinstance(max(user_dates), pd.Timestamp) else max(user_dates)
    
    # Create date range options
    date_ranges = get_date_range_options()
    
    # Adjust the date ranges based on available data
    for key, range_data in date_ranges.items():
        if range_data['start_date'] < min_date:
            date_ranges[key]['start_date'] = min_date
        if range_data['end_date'] > max_date:
            date_ranges[key]['end_date'] = max_date
    
    # Add a custom option
    date_ranges['custom'] = {
        'label': 'Custom Range',
        'start_date': min_date,
        'end_date': max_date
    }
    
    # If there's a previous selection, use it as default
    default_range = 'last_30_days'
    if 'selected_date_range' in st.session_state:
        default_range = st.session_state.selected_date_range
    
    # Create selection widget
    selected_range = st.selectbox(
        "Select Date Range:",
        list(date_ranges.keys()),
        format_func=lambda x: date_ranges[x]['label'],
        index=list(date_ranges.keys()).index(default_range)
    )
    
    # Store selection in session state
    st.session_state.selected_date_range = selected_range
    
    # If custom range is selected, show date input widgets
    if selected_range == 'custom':
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        # Ensure start_date <= end_date
        if start_date > end_date:
            st.warning("Start date should be before or equal to end date.")
            start_date = end_date
    else:
        # Use predefined range
        start_date = date_ranges[selected_range]['start_date']
        end_date = date_ranges[selected_range]['end_date']
    
    # Convert to pandas datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    return start_date, end_date


# Progress bar component
def progress_bar(value, max_value, label):
    """
    Create a custom progress bar.
    
    Parameters:
    -----------
    value : float
        Current value
    max_value : float
        Maximum value (target)
    label : str
        Label for the progress bar
    """
    progress_pct = calculate_progress_percentage(value, max_value)
    
    st.markdown(f"""
        <div class="progress-container">
            <div class="metric-label">{label}</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_pct}%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #6c757d;">
                <span>0</span>
                <span>{value:.0f}/{max_value:.0f} ({progress_pct:.0f}%)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


# Achievement badge component
def display_badge(badge):
    """
    Display an achievement badge.
    
    Parameters:
    -----------
    badge : dict
        Badge information with name, description, and level
    """
    level_class = f"badge-{badge['level'].lower()}"
    
    st.markdown(f"""
        <div class="achievement-badge {level_class}">
            <div style="display: flex; align-items: center;">
                <div style="margin-right: 1rem;">🏆</div>
                <div>
                    <div style="font-weight: 600; font-size: 1.1rem;">{badge['name']}</div>
                    <div style="color: #6c757d;">{badge['description']}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# Insights card component
def display_insights(insights):
    """
    Display insights for a specific day.
    
    Parameters:
    -----------
    insights : dict
        Dictionary with insights
    """
    if not insights:
        st.info("No insights available for the selected date.")
        return
    
    st.markdown(f"### Insights for {insights['date']} ({insights['day_of_week']})")
    
    # Display summary
    for insight in insights['summary']:
        st.markdown(f"""
            <div class="insight-card">
                <div>{insight}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Display suggestions if any
    if insights['suggestions']:
        st.markdown("### Suggestions")
        for suggestion in insights['suggestions']:
            st.markdown(f"""
                <div class="suggestion-card">
                    <div>💡 {suggestion}</div>
                </div>
            """, unsafe_allow_html=True)


# Metric card component
def metric_card(title, value, delta=None, delta_suffix="vs. previous period"):
    """
    Display a metric card with title, value, and optional delta.
    
    Parameters:
    -----------
    title : str
        Metric title
    value : str or float
        Metric value
    delta : float or None
        Change from previous period
    delta_suffix : str
        Suffix for the delta text
    """
    if delta:
        st.metric(label=title, value=value, delta=delta, delta_color="normal", help=delta_suffix)
    else:
        st.metric(label=title, value=value)


# Overview page
def overview_page(data_dict, user_id, date_range):
    """
    Create the overview dashboard page.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed dataframes
    user_id : str
        Selected user ID
    date_range : tuple
        (start_date, end_date)
    """
    st.markdown('<div class="main-header">Overview Dashboard</div>', unsafe_allow_html=True)
    
    start_date, end_date = date_range
    
    # Get daily activity data for the selected user and date range
    daily_df = data_dict.get('merged_daily')
    
    if daily_df is None or user_id not in daily_df['Id'].unique():
        st.error("No data available for the selected user.")
        return
    
    # Filter data for the selected user and date range
    user_data = daily_df[daily_df['Id'] == user_id]
    filtered_data = user_data[(user_data['Date'] >= pd.Timestamp(start_date)) & 
                              (user_data['Date'] <= pd.Timestamp(end_date))]
    
    if filtered_data.empty:
        st.warning(f"No data available for the selected date range: {start_date} to {end_date}")
        return
    
    # Calculate summary statistics
    summary = create_activity_summary_cards(filtered_data)
    
    # Create columns for summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'avg_steps' in summary:
            metric_card("Average Steps", f"{summary['avg_steps']:,}")
        
        if 'avg_distance' in summary:
            metric_card("Average Distance", f"{summary['avg_distance']:.2f} km")
    
    with col2:
        if 'avg_calories' in summary:
            metric_card("Average Calories", f"{summary['avg_calories']:,}")
        
        if 'avg_active_minutes' in summary:
            metric_card("Average Active Minutes", f"{summary['avg_active_minutes']}")
    
    with col3:
        if 'active_days_percentage' in summary:
            metric_card("Active Days", f"{summary['active_days_percentage']}%", 
                      delta_suffix="days with 30+ min of moderate/vigorous activity")
        
        if 'steps_goal_achievement' in summary:
            metric_card("Step Goal Achievement", f"{summary['steps_goal_achievement']}%",
                      delta_suffix="days with 10,000+ steps")
    
    with col4:
        if 'avg_health_score' in summary:
            metric_card("Average Health Score", f"{summary['avg_health_score']}/100")
        
        # Display date range info
        if 'num_days' in summary:
            metric_card("Days in Period", f"{summary['num_days']}")
    
    st.markdown('<div class="sub-header">Activity Trends</div>', unsafe_allow_html=True)
    
    # Create columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot daily steps trend
        steps_trend = plot_daily_steps_trend(filtered_data, rolling_window=7)
        if steps_trend:
            st.plotly_chart(steps_trend, use_container_width=True)
    
    with col2:
        # Plot activity distribution
        activity_dist = plot_activity_distribution(filtered_data)
        if activity_dist:
            st.plotly_chart(activity_dist, use_container_width=True)
    
    # Display activity heatmap
    st.markdown('<div class="sub-header">Activity Patterns</div>', unsafe_allow_html=True)
    
    hourly_steps_df = data_dict.get('hourly_steps')
    if hourly_steps_df is not None:
        # Filter data for the selected user
        user_hourly_steps = hourly_steps_df[hourly_steps_df['Id'] == user_id]
        
        # Plot activity heatmap
        heatmap = plot_activity_heatmap(user_hourly_steps, date_range=(start_date, end_date))
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)
    
    # Display achievements
    st.markdown('<div class="sub-header">Achievements</div>', unsafe_allow_html=True)
    
    badges = create_achievement_badges(filtered_data, user_id)
    if badges:
        badge_cols = st.columns(2)
        for i, (badge_id, badge) in enumerate(badges.items()):
            with badge_cols[i % 2]:
                display_badge(badge)
    else:
        st.info("No achievements earned in the selected period.")
    
    # Display daily insights for the most recent day
    st.markdown('<div class="sub-header">Recent Insights</div>', unsafe_allow_html=True)
    
    most_recent_date = filtered_data['Date'].max()
    daily_insights = generate_daily_insights(daily_df, user_id, most_recent_date)
    
    if daily_insights:
        display_insights(daily_insights)
    else:
        st.info("No insights available for the most recent day.")


# Activity Analysis page
def activity_analysis_page(data_dict, user_id, date_range):
    """
    Create the activity analysis page.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed dataframes
    user_id : str
        Selected user ID
    date_range : tuple
        (start_date, end_date)
    """
    st.markdown('<div class="main-header">Activity Analysis</div>', unsafe_allow_html=True)
    
    start_date, end_date = date_range
    
    # Get daily activity data for the selected user and date range
    daily_df = data_dict.get('daily_activity')
    merged_daily_df = data_dict.get('merged_daily')
    hourly_steps_df = data_dict.get('hourly_steps')
    hourly_intensities_df = data_dict.get('hourly_intensities')
    
    if daily_df is None or user_id not in daily_df['Id'].unique():
        st.error("No activity data available for the selected user.")
        return
    
    # Filter data for the selected user and date range
    filtered_daily = daily_df[(daily_df['Id'] == user_id) & 
                             (daily_df['Date'] >= pd.Timestamp(start_date)) & 
                             (daily_df['Date'] <= pd.Timestamp(end_date))]
    
    if filtered_daily.empty:
        st.warning(f"No activity data available for the selected date range: {start_date} to {end_date}")
        return
    
    # Create tabs for different analyses
    tabs = st.tabs(["Daily Patterns", "Weekly Patterns", "Step Analysis", "Activity Intensity", "Correlations"])
    
    # Tab 1: Daily Patterns
    with tabs[0]:
        st.markdown('<div class="sub-header">Daily Activity Patterns</div>', unsafe_allow_html=True)
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Check if Steps column exists
            if 'TotalSteps' in filtered_daily.columns:
                avg_steps = filtered_daily['TotalSteps'].mean()
                max_steps = filtered_daily['TotalSteps'].max()
            else:
                # Try to find the right column (may be case sensitive)
                steps_col = [col for col in filtered_daily.columns if col.lower() == 'steps']
                if steps_col:
                    avg_steps = filtered_daily[steps_col[0]].mean()
                    max_steps = filtered_daily[steps_col[0]].max()
                else:
                    avg_steps = 0
                    max_steps = 0
                    st.warning('Steps data not found')
            metric_card("Average Steps", f"{avg_steps:.0f}", 
                       delta=f"{max_steps:.0f}", delta_suffix="max in period")
        
        with col2:
            if 'VeryActiveMinutes' in filtered_daily.columns and 'FairlyActiveMinutes' in filtered_daily.columns:
                avg_active = (filtered_daily['VeryActiveMinutes'] + filtered_daily['FairlyActiveMinutes']).mean()
                max_active = (filtered_daily['VeryActiveMinutes'] + filtered_daily['FairlyActiveMinutes']).max()
                metric_card("Average Active Minutes", f"{avg_active:.0f}", 
                           delta=f"{max_active:.0f}", delta_suffix="max in period")
        
        with col3:
            if 'Calories' in filtered_daily.columns:
                avg_calories = filtered_daily['Calories'].mean()
                max_calories = filtered_daily['Calories'].max()
                metric_card("Average Calories", f"{avg_calories:.0f}", 
                           delta=f"{max_calories:.0f}", delta_suffix="max in period")
        
        # Plot steps trend
        steps_trend = plot_daily_steps_trend(filtered_daily, user_id, date_range, rolling_window=7)
        if steps_trend:
            st.plotly_chart(steps_trend, use_container_width=True)
        
        # Activity distribution
        activity_dist = plot_activity_distribution(filtered_daily, user_id, date_range)
        if activity_dist:
            st.plotly_chart(activity_dist, use_container_width=True)
        
        # Hourly patterns
        if hourly_steps_df is not None:
            st.markdown('<div class="sub-header">Hourly Activity Patterns</div>', unsafe_allow_html=True)
            
            # Filter hourly data
            try:
                # Ensure ActivityHour is a datetime column
                hourly_steps_df_copy = hourly_steps_df.copy()
                hourly_steps_df_copy['ActivityHour'] = pd.to_datetime(hourly_steps_df_copy['ActivityHour'], errors='coerce')
                
                # Filter only rows where Id matches and ActivityHour is valid
                filtered_hourly = hourly_steps_df_copy[
                    (hourly_steps_df_copy['Id'] == user_id) & 
                    (hourly_steps_df_copy['ActivityHour'].notna())
                ]
                
                # Extract date from ActivityHour and filter by date range
                filtered_hourly['ActivityDate'] = filtered_hourly['ActivityHour'].dt.date
                filtered_hourly = filtered_hourly[
                    (filtered_hourly['ActivityDate'] >= pd.to_datetime(start_date).date()) & 
                    (filtered_hourly['ActivityDate'] <= pd.to_datetime(end_date).date())
                ]
            except Exception as e:
                st.error(f"Error processing hourly data: {str(e)}")
                filtered_hourly = pd.DataFrame()
            
            if not filtered_hourly.empty:
                # Plot activity heatmap
                heatmap = plot_activity_heatmap(filtered_hourly, date_range=(start_date, end_date))
                if heatmap:
                    st.plotly_chart(heatmap, use_container_width=True)
                
                # Identify active periods
                active_periods = identify_consistent_active_periods(filtered_hourly)
                if active_periods and user_id in active_periods:
                    st.markdown('<div class="sub-header">Consistent Active Periods</div>', unsafe_allow_html=True)
                    
                    periods = active_periods[user_id]
                    if periods:
                        for period in periods:
                            st.markdown(f"🕒 **{period}**")
                    else:
                        st.info("No consistent active periods detected.")
            else:
                st.info("No hourly data available for the selected period.")
    
    # Tab 2: Weekly Patterns
    with tabs[1]:
        st.markdown('<div class="sub-header">Weekly Activity Patterns</div>', unsafe_allow_html=True)
        
        # Analyze weekly patterns
        weekly_patterns = analyze_weekly_patterns(filtered_daily)
        
        if weekly_patterns is not None:
            # Plot weekly patterns
            weekly_steps = weekly_patterns.pivot(index='Day', columns='', values=('TotalSteps', 'mean')).reset_index()
            weekly_steps = weekly_steps.sort_values('DayOfWeek')
            
            # Create weekly steps bar chart
            fig = px.bar(
                weekly_steps,
                x='Day',
                y=('TotalSteps', 'mean'),
                labels={'value': 'Average Steps', 'Day': 'Day of Week'},
                title='Average Steps by Day of Week',
                color_discrete_sequence=['#1E88E5']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed weekly stats
            st.markdown('<div class="sub-header">Detailed Weekly Statistics</div>', unsafe_allow_html=True)
            
            # Convert multi-level column names to single level
            weekly_stats = weekly_patterns.copy()
            weekly_stats.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in weekly_stats.columns]
            
            # Reorder columns for better display
            display_cols = ['Day', 'DayOfWeek']
            for metric in ['TotalSteps', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'Calories']:
                for stat in ['mean', 'median', 'max']:
                    col_name = f"{metric}_{stat}"
                    if col_name in weekly_stats.columns:
                        display_cols.append(col_name)
            
            weekly_display = weekly_stats[display_cols].sort_values('DayOfWeek')
            
            # Format columns
            for col in weekly_display.columns:
                if col not in ['Day', 'DayOfWeek'] and weekly_display[col].dtype.kind in 'if':
                    weekly_display[col] = weekly_display[col].round(1)
            
            # Rename columns for display
            rename_dict = {
                'Steps_mean': 'Avg Steps',
                'Steps_median': 'Median Steps',
                'Steps_max': 'Max Steps',
                'VeryActiveMinutes_mean': 'Avg Very Active',
                'FairlyActiveMinutes_mean': 'Avg Fairly Active',
                'Calories_mean': 'Avg Calories'
            }
            weekly_display = weekly_display.rename(columns=rename_dict)
            
            # Drop DayOfWeek column for display
            weekly_display = weekly_display.drop(columns=['DayOfWeek'])
            
            st.dataframe(weekly_display, use_container_width=True)
            
            # Identify most and least active days
            most_active_day = weekly_steps.loc[weekly_steps[('TotalSteps', 'mean')].idxmax(), 'Day']
            least_active_day = weekly_steps.loc[weekly_steps[('TotalSteps', 'mean')].idxmin(), 'Day']
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📈 Most active day: **{most_active_day}**")
            with col2:
                st.info(f"📉 Least active day: **{least_active_day}**")
        else:
            st.info("Not enough data to analyze weekly patterns.")
    
    # Tab 3: Step Analysis
    with tabs[2]:
        st.markdown('<div class="sub-header">Step Analysis</div>', unsafe_allow_html=True)
        
        # Step goal achievement
        steps_goal = 10000  # Default goal
        
        # Calculate achievement stats
        days_over_goal = (filtered_daily['TotalSteps'] >= steps_goal).sum()
        total_days = len(filtered_daily)
        achievement_pct = (days_over_goal / total_days * 100) if total_days > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display goal achievement stats
            st.markdown(f"""
                <div class="card">
                    <div class="metric-label">Step Goal Achievement</div>
                    <div class="metric-value">{achievement_pct:.1f}%</div>
                    <div>{days_over_goal} out of {total_days} days</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Display streak information
            streak_data = filtered_daily.sort_values('Date')
            streak_data['goal_achieved'] = streak_data['TotalSteps'] >= steps_goal
            streak_data['streak_group'] = (streak_data['goal_achieved'] != streak_data['goal_achieved'].shift()).cumsum()
            
            # Calculate current and max streaks
            streaks = streak_data.groupby(['goal_achieved', 'streak_group']).size().reset_index(name='streak_length')
            current_streak = streaks[streaks['goal_achieved']].iloc[-1]['streak_length'] if not streaks[streaks['goal_achieved']].empty else 0
            max_streak = streaks[streaks['goal_achieved']]['streak_length'].max() if not streaks[streaks['goal_achieved']].empty else 0
            
            st.markdown(f"""
                <div class="card">
                    <div class="metric-label">Step Goal Streaks</div>
                    <div class="metric-value">{current_streak}</div>
                    <div>Current streak (days)</div>
                    <div style="margin-top: 0.5rem;">Longest streak: {max_streak} days</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Step distribution histogram
            fig = px.histogram(
                filtered_daily,
                x='TotalSteps',
                nbins=20,
                title='Step Count Distribution',
                color_discrete_sequence=['#1E88E5']
            )
            
            # Add goal line
            fig.add_shape(
                type='line',
                x0=steps_goal,
                y0=0,
                x1=steps_goal,
                y1=1,
                yref='paper',
                line=dict(color='red', width=2, dash='dash'),
            )
            
            fig.add_annotation(
                x=steps_goal,
                y=0.95,
                yref='paper',
                text='Goal',
                showarrow=True,
                arrowhead=2,
                arrowcolor='red',
                arrowsize=1,
                arrowwidth=2,
                ax=40,
                ay=-40
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Step count trends by day of week
        fig = px.box(
            filtered_daily,
            x='Day',
            y='TotalSteps',
            title='Step Count by Day of Week',
            color_discrete_sequence=['#1E88E5'],
            category_orders={
                'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            }
        )
        
        # Add goal line
        fig.add_shape(
            type='line',
            x0=-0.5,
            y0=steps_goal,
            x1=6.5,
            y1=steps_goal,
            line=dict(color='red', width=2, dash='dash'),
        )
        
        fig.add_annotation(
            x=6,
            y=steps_goal,
            text='Goal',
            showarrow=False,
            font=dict(color='red')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Activity Intensity
    with tabs[3]:
        st.markdown('<div class="sub-header">Activity Intensity Analysis</div>', unsafe_allow_html=True)
        
        intensity_cols = ['VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes']
        if all(col in filtered_daily.columns for col in intensity_cols):
            # Calculate daily averages
            avg_very = filtered_daily['VeryActiveMinutes'].mean()
            avg_fairly = filtered_daily['FairlyActiveMinutes'].mean()
            avg_lightly = filtered_daily['LightlyActiveMinutes'].mean()
            avg_sedentary = filtered_daily['SedentaryMinutes'].mean()
            
            # Create columns for metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                metric_card("Very Active", f"{avg_very:.0f} min")
            
            with col2:
                metric_card("Fairly Active", f"{avg_fairly:.0f} min")
            
            with col3:
                metric_card("Lightly Active", f"{avg_lightly:.0f} min")
            
            with col4:
                metric_card("Sedentary", f"{avg_sedentary:.0f} min")
            
            # Create activity breakdown chart
            activity_data = filtered_daily[['Date'] + intensity_cols].melt(
                id_vars=['Date'],
                value_vars=intensity_cols,
                var_name='Activity Type',
                value_name='Minutes'
            )
            
            # Rename activity types for better display
            activity_map = {
                'VeryActiveMinutes': 'Very Active',
                'FairlyActiveMinutes': 'Fairly Active',
                'LightlyActiveMinutes': 'Lightly Active',
                'SedentaryMinutes': 'Sedentary'
            }
            activity_data['Activity Type'] = activity_data['Activity Type'].map(activity_map)
            
            # Create stacked bar chart
            fig = px.bar(
                activity_data,
                x='Date',
                y='Minutes',
                color='Activity Type',
                title='Daily Activity Breakdown',
                barmode='stack',
                color_discrete_map={
                    'Very Active': '#e31a1c',
                    'Fairly Active': '#fb9a99',
                    'Lightly Active': '#a6cee3',
                    'Sedentary': '#c7c7c7'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create activity intensity trend chart
            filtered_daily['TotalActiveMinutes'] = (
                filtered_daily['VeryActiveMinutes'] + 
                filtered_daily['FairlyActiveMinutes'] + 
                filtered_daily['LightlyActiveMinutes']
            )
            
            filtered_daily['ActivePercentage'] = (
                filtered_daily['TotalActiveMinutes'] / 
                (filtered_daily['TotalActiveMinutes'] + filtered_daily['SedentaryMinutes']) * 100
            )
            
            # Create line chart for active percentage
            fig = px.line(
                filtered_daily,
                x='Date',
                y='ActivePercentage',
                title='Daily Active vs. Sedentary Time (%)',
                labels={'ActivePercentage': 'Active Time (%)'},
                color_discrete_sequence=['#1E88E5']
            )
            
            # Add rolling average
            filtered_daily['RollingActivePercentage'] = filtered_daily['ActivePercentage'].rolling(window=7, min_periods=1).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=filtered_daily['Date'],
                    y=filtered_daily['RollingActivePercentage'],
                    mode='lines',
                    name='7-day Average',
                    line=dict(color='rgba(219, 68, 55, 0.9)', width=2)
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show active minutes recommendations
            st.markdown("""
                <div class="card">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">Activity Recommendations</div>
                    <div>The World Health Organization recommends:</div>
                    <ul>
                        <li>At least 150 minutes of moderate activity or 75 minutes of vigorous activity per week</li>
                        <li>Spread activity throughout the week (e.g., 30 minutes of moderate activity on 5 days)</li>
                        <li>Minimize sedentary time</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Activity intensity data not available for the selected user.")
    
    # Tab 5: Correlations
    with tabs[4]:
        st.markdown('<div class="sub-header">Activity Correlations</div>', unsafe_allow_html=True)
        
        if merged_daily_df is not None:
            # Filter merged data
            filtered_merged = merged_daily_df[
                (merged_daily_df['Id'] == user_id) & 
                (merged_daily_df['Date'] >= pd.Timestamp(start_date)) & 
                (merged_daily_df['Date'] <= pd.Timestamp(end_date))
            ]
            
            if not filtered_merged.empty:
                # Create columns for metric selection
                col1, col2 = st.columns(2)
                
                with col1:
                    # Available metrics
                    x_metrics = ['TotalSteps', 'TotalDistance', 'VeryActiveMinutes', 'FairlyActiveMinutes', 
                               'LightlyActiveMinutes', 'SedentaryMinutes', 'Calories']
                    
                    # Filter to available columns
                    available_x = [m for m in x_metrics if m in filtered_merged.columns]
                    
                    x_metric = st.selectbox("X-axis Metric:", available_x, index=0)
                
                with col2:
                    # Available metrics
                    y_metrics = ['Calories', 'TotalSteps', 'TotalDistance', 'VeryActiveMinutes', 
                               'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes', 
                               'TotalMinutesAsleep', 'SleepEfficiency', 'AvgHeartRate']
                    
                    # Filter to available columns
                    available_y = [m for m in y_metrics if m in filtered_merged.columns and m != x_metric]
                    
                    y_metric = st.selectbox("Y-axis Metric:", available_y, index=0)
                
                # Plot correlation
                corr_plot = plot_activity_correlation(filtered_merged, x_metric, y_metric)
                if corr_plot:
                    st.plotly_chart(corr_plot, use_container_width=True)
                
                # If we have sleep data, show sleep-activity correlation
                if 'TotalMinutesAsleep' in filtered_merged.columns:
                    st.markdown('<div class="sub-header">Sleep and Activity Correlation</div>', unsafe_allow_html=True)
                    
                    # Analyze sleep-activity correlation
                    sleep_corr = analyze_sleep_activity_correlation(filtered_merged)
                    
                    if sleep_corr:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### Same Day Correlations")
                            
                            # Create a dataframe for same day correlations
                            same_day_data = []
                            for metric, values in sleep_corr['same_day_correlations'].items():
                                same_day_data.append({
                                    'Metric': metric,
                                    'Sleep Duration': values['sleep_duration'],
                                    'Sleep Efficiency': values['sleep_efficiency']
                                })
                            
                            same_day_df = pd.DataFrame(same_day_data)
                            st.dataframe(same_day_df, use_container_width=True)
                        
                        with col2:
                            st.markdown("##### Previous Day's Activity Effect")
                            
                            # Create a dataframe for previous day correlations
                            prev_day_data = []
                            for metric, values in sleep_corr['previous_day_correlations'].items():
                                prev_day_data.append({
                                    'Metric': metric,
                                    'Sleep Duration': values['sleep_duration'],
                                    'Sleep Efficiency': values['sleep_efficiency']
                                })
                            
                            prev_day_df = pd.DataFrame(prev_day_data)
                            st.dataframe(prev_day_df, use_container_width=True)
                        
                        # Interpretation of correlations
                        st.markdown("""
                            <div class="card">
                                <div style="font-weight: 600; margin-bottom: 0.5rem;">Correlation Interpretation</div>
                                <div>
                                    <ul>
                                        <li>Positive values (0 to 1) indicate that as one metric increases, the other tends to increase as well</li>
                                        <li>Negative values (-1 to 0) indicate that as one metric increases, the other tends to decrease</li>
                                        <li>Values closer to -1 or 1 indicate stronger correlations</li>
                                        <li>Values close to 0 indicate weak or no correlation</li>
                                    </ul>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Calories prediction
                if 'Calories' in filtered_merged.columns:
                    st.markdown('<div class="sub-header">Calories Prediction Analysis</div>', unsafe_allow_html=True)
                    
                    # Calculate calorie prediction model
                    model, importance, metrics = predict_calories_from_activity(filtered_merged)
                    
                    if model and importance and metrics:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### Feature Importance")
                            
                            # Create a bar chart of feature importance
                            fig = px.bar(
                                x=list(importance.values()),
                                y=list(importance.keys()),
                                orientation='h',
                                labels={'x': 'Importance', 'y': 'Feature'},
                                title='Factors Influencing Calorie Burn',
                                color_discrete_sequence=['#1E88E5']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("##### Model Performance")
                            
                            # Display metrics
                            st.markdown(f"""
                                <div class="card">
                                    <div style="margin-bottom: 0.5rem;"><b>RMSE:</b> {metrics['rmse']} calories</div>
                                    <div style="margin-bottom: 0.5rem;"><b>R²:</b> {metrics['r2']}</div>
                                    <div><b>Mean Absolute Error:</b> {metrics['mean_absolute_error']} calories</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("""
                                <div style="margin-top: 1rem;">
                                    <b>Interpretation:</b><br>
                                    The model accuracy shows how well we can predict calories burned based on your activity metrics.
                                    A higher R² value (closer to 1.0) indicates better prediction quality.
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("No data available for correlation analysis in the selected period.")
        else:
            st.warning("Merged data not available for correlation analysis.")


# Sleep Insights page
def sleep_insights_page(data_dict, user_id, date_range):
    """
    Create the sleep insights page.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed dataframes
    user_id : str
        Selected user ID
    date_range : tuple
        (start_date, end_date)
    """
    st.markdown('<div class="main-header">Sleep Insights</div>', unsafe_allow_html=True)
    
    start_date, end_date = date_range
    
    # Get sleep data
    sleep_df = data_dict.get('sleep')
    merged_daily_df = data_dict.get('merged_daily')
    
    if sleep_df is None or merged_daily_df is None or user_id not in sleep_df['Id'].unique():
        st.error("No sleep data available for the selected user.")
        return
    
    # Filter data for the selected user and date range
    filtered_sleep = sleep_df[
        (sleep_df['Id'] == user_id) & 
        (sleep_df['Date'] >= pd.Timestamp(start_date)) & 
        (sleep_df['Date'] <= pd.Timestamp(end_date))
    ]
    
    filtered_merged = merged_daily_df[
        (merged_daily_df['Id'] == user_id) & 
        (merged_daily_df['Date'] >= pd.Timestamp(start_date)) & 
        (merged_daily_df['Date'] <= pd.Timestamp(end_date))
    ]
    
    if filtered_sleep.empty:
        st.warning(f"No sleep data available for the selected date range: {start_date} to {end_date}")
        return
    
    # Create tabs for different analyses
    tabs = st.tabs(["Sleep Summary", "Sleep Patterns", "Sleep Quality", "Sleep-Activity Relationship"])
    
    # Tab 1: Sleep Summary
    with tabs[0]:
        st.markdown('<div class="sub-header">Sleep Summary</div>', unsafe_allow_html=True)
        
        # Calculate summary statistics
        avg_sleep_min = filtered_sleep['TotalMinutesAsleep'].mean()
        avg_sleep_hours = avg_sleep_min / 60
        avg_time_in_bed_min = filtered_sleep['TotalTimeInBed'].mean()
        avg_time_in_bed_hours = avg_time_in_bed_min / 60
        
        if 'SleepEfficiency' in filtered_sleep.columns:
            avg_efficiency = filtered_sleep['SleepEfficiency'].mean() * 100
        else:
            avg_efficiency = (avg_sleep_min / avg_time_in_bed_min * 100) if avg_time_in_bed_min > 0 else 0
        
        # Calculate days within recommended range (7-9 hours)
        sleep_hours = filtered_sleep['TotalMinutesAsleep'] / 60
        within_range = ((sleep_hours >= 7) & (sleep_hours <= 9)).mean() * 100
        
        # Create metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            metric_card("Average Sleep", f"{avg_sleep_hours:.1f} hours")
        
        with col2:
            metric_card("Average Time in Bed", f"{avg_time_in_bed_hours:.1f} hours")
        
        with col3:
            metric_card("Sleep Efficiency", f"{avg_efficiency:.1f}%")
        
        with col4:
            metric_card("Within 7-9h Range", f"{within_range:.1f}%")
        
        # Plot sleep patterns
        sleep_chart = plot_sleep_patterns(filtered_sleep, user_id, date_range)
        if sleep_chart:
            st.plotly_chart(sleep_chart, use_container_width=True)
        
        # Sleep duration distribution
        fig = px.histogram(
            filtered_sleep,
            x=filtered_sleep['TotalMinutesAsleep'] / 60,
            nbins=20,
            labels={'x': 'Sleep Duration (hours)'},
            title='Sleep Duration Distribution',
            color_discrete_sequence=['#1E88E5']
        )
        
        # Add recommended range
        fig.add_shape(
            type='rect',
            x0=7,
            y0=0,
            x1=9,
            y1=1,
            yref='paper',
            fillcolor='rgba(15, 157, 88, 0.1)',
            line=dict(width=0)
        )
        
        fig.add_annotation(
            x=8,
            y=0.95,
            yref='paper',
            text='Recommended Range',
            showarrow=False,
            font=dict(color='rgba(15, 157, 88, 0.9)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Sleep Patterns
    with tabs[1]:
        st.markdown('<div class="sub-header">Sleep Patterns</div>', unsafe_allow_html=True)
        
        # Calculate sleep timing
        if 'SleepDay' in filtered_sleep.columns:
            # Estimate bedtime and wake time based on difference between time in bed and asleep
            filtered_sleep['BedtimeEstimate'] = pd.to_datetime(filtered_sleep['SleepDay']) - pd.to_timedelta(filtered_sleep['TotalTimeInBed'] - filtered_sleep['TotalMinutesAsleep'], unit='m')
            
            # Create a box plot of bedtime
            bedtime_hour = filtered_sleep['BedtimeEstimate'].dt.hour + filtered_sleep['BedtimeEstimate'].dt.minute / 60
            
            # Adjust for bedtimes after midnight
            bedtime_hour = bedtime_hour.apply(lambda x: x if x >= 20 else x + 24)
            
            fig = px.box(
                x=bedtime_hour,
                labels={'x': 'Estimated Bedtime (hour)'},
                title='Bedtime Distribution',
                color_discrete_sequence=['#1E88E5']
            )
            
            # Format x-axis
            tick_vals = list(range(20, 29))
            tick_text = [f"{h if h < 24 else h - 24}:00" for h in tick_vals]
            
            fig.update_layout(
                xaxis=dict(
                    tickvals=tick_vals,
                    ticktext=tick_text
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Sleep by day of week
        if 'Day' in filtered_sleep.columns:
            # Create a box plot of sleep duration by day of week
            fig = px.box(
                filtered_sleep,
                x='Day',
                y=filtered_sleep['TotalMinutesAsleep'] / 60,
                labels={'y': 'Sleep Duration (hours)', 'Day': 'Day of Week'},
                title='Sleep Duration by Day of Week',
                color_discrete_sequence=['#1E88E5'],
                category_orders={
                    'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                }
            )
            
            # Add recommended range
            fig.add_shape(
                type='rect',
                x0=-0.5,
                y0=7,
                x1=6.5,
                y1=9,
                fillcolor='rgba(15, 157, 88, 0.1)',
                line=dict(width=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate average sleep by day of week
            sleep_by_day = filtered_sleep.groupby('Day')['TotalMinutesAsleep'].mean() / 60
            
            # Sort by day of week
            day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            sleep_by_day = sleep_by_day.reset_index()
            sleep_by_day['DayOrder'] = sleep_by_day['Day'].map(day_order)
            sleep_by_day = sleep_by_day.sort_values('DayOrder')
            
            # Identify days with most and least sleep
            most_sleep_day = sleep_by_day.loc[sleep_by_day['TotalMinutesAsleep'].idxmax(), 'Day']
            least_sleep_day = sleep_by_day.loc[sleep_by_day['TotalMinutesAsleep'].idxmin(), 'Day']
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📈 Day with most sleep: **{most_sleep_day}**")
            with col2:
                st.info(f"📉 Day with least sleep: **{least_sleep_day}**")
    
    # Tab 3: Sleep Quality
    with tabs[2]:
        st.markdown('<div class="sub-header">Sleep Quality</div>', unsafe_allow_html=True)
        
        if 'SleepEfficiency' in filtered_sleep.columns:
            # Calculate sleep efficiency stats
            avg_efficiency = filtered_sleep['SleepEfficiency'].mean() * 100
            min_efficiency = filtered_sleep['SleepEfficiency'].min() * 100
            max_efficiency = filtered_sleep['SleepEfficiency'].max() * 100
            
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                metric_card("Average Efficiency", f"{avg_efficiency:.1f}%")
            
            with col2:
                metric_card("Minimum Efficiency", f"{min_efficiency:.1f}%")
            
            with col3:
                metric_card("Maximum Efficiency", f"{max_efficiency:.1f}%")
            
            # Plot sleep efficiency over time
            fig = px.line(
                filtered_sleep,
                x='Date',
                y=filtered_sleep['SleepEfficiency'] * 100,
                labels={'y': 'Sleep Efficiency (%)', 'Date': 'Date'},
                title='Sleep Efficiency Over Time',
                color_discrete_sequence=['#1E88E5']
            )
            
            # Add rolling average
            filtered_sleep['RollingEfficiency'] = filtered_sleep['SleepEfficiency'].rolling(window=7, min_periods=1).mean() * 100
            
            fig.add_trace(
                go.Scatter(
                    x=filtered_sleep['Date'],
                    y=filtered_sleep['RollingEfficiency'],
                    mode='lines',
                    name='7-day Average',
                    line=dict(color='rgba(219, 68, 55, 0.9)', width=2)
                )
            )
            
            # Add recommended range
            fig.add_shape(
                type='rect',
                x0=filtered_sleep['Date'].min(),
                y0=85,
                x1=filtered_sleep['Date'].max(),
                y1=100,
                fillcolor='rgba(15, 157, 88, 0.1)',
                line=dict(width=0)
            )
            
            fig.add_annotation(
                x=filtered_sleep['Date'].max(),
                y=92.5,
                text='Good: 85%+',
                showarrow=False,
                xshift=-70,
                font=dict(color='rgba(15, 157, 88, 0.9)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Efficiency vs. duration scatter plot
            fig = px.scatter(
                filtered_sleep,
                x=filtered_sleep['TotalMinutesAsleep'] / 60,
                y=filtered_sleep['SleepEfficiency'] * 100,
                labels={'x': 'Sleep Duration (hours)', 'y': 'Sleep Efficiency (%)'},
                title='Sleep Efficiency vs. Duration',
                color='Date',
                trendline='ols',
                color_discrete_sequence=['#1E88E5']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sleep quality classification
            filtered_sleep['SleepQuality'] = pd.cut(
                filtered_sleep['SleepEfficiency'] * 100,
                bins=[0, 65, 75, 85, 100],
                labels=['Poor', 'Fair', 'Good', 'Excellent']
            )
            
            quality_counts = filtered_sleep['SleepQuality'].value_counts().reset_index()
            quality_counts.columns = ['Quality', 'Count']
            quality_counts['Percentage'] = quality_counts['Count'] / quality_counts['Count'].sum() * 100
            
            # Create pie chart
            fig = px.pie(
                quality_counts,
                values='Percentage',
                names='Quality',
                title='Sleep Quality Distribution',
                color='Quality',
                color_discrete_map={
                    'Excellent': '#4CAF50',
                    'Good': '#8BC34A',
                    'Fair': '#FFC107',
                    'Poor': '#F44336'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sleep efficiency data not available for the selected user.")
    
    # Tab 4: Sleep-Activity Relationship
    with tabs[3]:
        st.markdown('<div class="sub-header">Sleep-Activity Relationship</div>', unsafe_allow_html=True)
        
        if not filtered_merged.empty and 'TotalMinutesAsleep' in filtered_merged.columns:
            # Analyze correlation between activity and sleep
            corr = analyze_sleep_activity_correlation(filtered_merged)
            
            if corr:
                # Display correlations
                st.markdown("#### How Does Your Activity Affect Sleep?")
                
                # Create a dataframe for same day correlations
                same_day_data = []
                for metric, values in corr['same_day_correlations'].items():
                    same_day_data.append({
                        'Metric': metric,
                        'Sleep Duration Correlation': values['sleep_duration'],
                        'Sleep Efficiency Correlation': values['sleep_efficiency']
                    })
                
                same_day_df = pd.DataFrame(same_day_data)
                
                # Sort by absolute correlation with sleep duration
                same_day_df['AbsCorr'] = abs(same_day_df['Sleep Duration Correlation'])
                same_day_df = same_day_df.sort_values('AbsCorr', ascending=False).drop(columns=['AbsCorr'])
                
                # Create a horizontal bar chart for correlations
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    y=same_day_df['Metric'],
                    x=same_day_df['Sleep Duration Correlation'],
                    name='Sleep Duration',
                    orientation='h',
                    marker=dict(color='rgba(66, 133, 244, 0.7)')
                ))
                
                fig.add_trace(go.Bar(
                    y=same_day_df['Metric'],
                    x=same_day_df['Sleep Efficiency Correlation'],
                    name='Sleep Efficiency',
                    orientation='h',
                    marker=dict(color='rgba(15, 157, 88, 0.7)')
                ))
                
                fig.update_layout(
                    title='Impact of Daily Activity on Sleep',
                    xaxis_title='Correlation',
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display previous day effect
                st.markdown("#### Does Yesterday's Activity Affect Tonight's Sleep?")
                
                if 'previous_day_correlations' in corr and corr['previous_day_correlations']:
                    # Create a dataframe for previous day correlations
                    prev_day_data = []
                    for metric, values in corr['previous_day_correlations'].items():
                        prev_day_data.append({
                            'Metric': metric,
                            'Sleep Duration Correlation': values['sleep_duration'],
                            'Sleep Efficiency Correlation': values['sleep_efficiency']
                        })
                    
                    prev_day_df = pd.DataFrame(prev_day_data)
                    
                    # Sort by absolute correlation with sleep duration
                    prev_day_df['AbsCorr'] = abs(prev_day_df['Sleep Duration Correlation'])
                    prev_day_df = prev_day_df.sort_values('AbsCorr', ascending=False).drop(columns=['AbsCorr'])
                    
                    # Create a horizontal bar chart for correlations
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        y=prev_day_df['Metric'],
                        x=prev_day_df['Sleep Duration Correlation'],
                        name='Sleep Duration',
                        orientation='h',
                        marker=dict(color='rgba(66, 133, 244, 0.7)')
                    ))
                    
                    fig.add_trace(go.Bar(
                        y=prev_day_df['Metric'],
                        x=prev_day_df['Sleep Efficiency Correlation'],
                        name='Sleep Efficiency',
                        orientation='h',
                        marker=dict(color='rgba(15, 157, 88, 0.7)')
                    ))
                    
                    fig.update_layout(
                        title='Impact of Previous Day\'s Activity on Sleep',
                        xaxis_title='Correlation',
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data to analyze previous day effects on sleep.")
                
                # Plot step count vs. sleep duration
                if 'TotalSteps' in filtered_merged.columns:
                    fig = px.scatter(
                        filtered_merged,
                        x='TotalSteps',
                        y=filtered_merged['TotalMinutesAsleep'] / 60,
                        labels={'x': 'TotalSteps', 'y': 'Sleep Duration (hours)'},
                        title='Steps vs. Sleep Duration',
                        color='Date',
                        trendline='ols',
                        color_discrete_sequence=['#1E88E5']
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sedentary time vs. sleep quality
                if 'SedentaryMinutes' in filtered_merged.columns and 'SleepEfficiency' in filtered_merged.columns:
                    fig = px.scatter(
                        filtered_merged,
                        x='SedentaryMinutes',
                        y=filtered_merged['SleepEfficiency'] * 100,
                        labels={'x': 'Sedentary Minutes', 'y': 'Sleep Efficiency (%)'},
                        title='Sedentary Time vs. Sleep Efficiency',
                        color='Date',
                        trendline='ols',
                        color_discrete_sequence=['#1E88E5']
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to analyze sleep-activity relationship.")
            
            # Display sleep recommendations
            st.markdown("#### Sleep Recommendations")
            
            st.markdown("""
                <div class="card">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">For Better Sleep Quality:</div>
                    <ul>
                        <li>Aim for 7-9 hours of sleep per night</li>
                        <li>Maintain a consistent sleep schedule</li>
                        <li>Engage in regular physical activity, but avoid intense exercise close to bedtime</li>
                        <li>Minimize screen time before bed</li>
                        <li>Create a comfortable sleep environment (cool, dark, quiet)</li>
                        <li>Avoid caffeine and alcohol close to bedtime</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Not enough data to analyze sleep-activity relationship.")


# Heart Rate Analysis page
def heart_rate_analysis_page(data_dict, user_id, date_range):
    """
    Create the heart rate analysis page.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed dataframes
    user_id : str
        Selected user ID
    date_range : tuple
        (start_date, end_date)
    """
    st.markdown('<div class="main-header">Heart Rate Analysis</div>', unsafe_allow_html=True)
    
    start_date, end_date = date_range
    
    # Get heart rate data
    heart_rate_df = data_dict.get('heart_rate')
    heart_rate_daily_df = data_dict.get('heart_rate_daily')
    
    if heart_rate_df is None or user_id not in heart_rate_df['Id'].unique():
        st.error("No heart rate data available for the selected user.")
        return
    
    # Filter data for the selected user and date range
    filtered_hr = heart_rate_df[
        (heart_rate_df['Id'] == user_id) & 
        (heart_rate_df['Date'] >= pd.Timestamp(start_date)) & 
        (heart_rate_df['Date'] <= pd.Timestamp(end_date))
    ]
    
    filtered_hr_daily = None
    if heart_rate_daily_df is not None:
        filtered_hr_daily = heart_rate_daily_df[
            (heart_rate_daily_df['Date'] >= pd.Timestamp(start_date)) & 
            (heart_rate_daily_df['Date'] <= pd.Timestamp(end_date))
        ]
    
    if filtered_hr.empty:
        st.warning(f"No heart rate data available for the selected date range: {start_date} to {end_date}")
        return
    
    # Create tabs for different analyses
    tabs = st.tabs(["Heart Rate Overview", "Daily Analysis", "Zone Analysis", "Anomaly Detection"])
    
    # Tab 1: Heart Rate Overview
    with tabs[0]:
        st.markdown('<div class="sub-header">Heart Rate Overview</div>', unsafe_allow_html=True)
        
        # Daily statistics
        if filtered_hr_daily is not None:
            # Calculate summary statistics
            avg_hr = filtered_hr_daily['AvgHeartRate'].mean()
            min_hr = filtered_hr_daily['MinHeartRate'].mean()
            max_hr = filtered_hr_daily['MaxHeartRate'].mean()
            resting_hr = filtered_hr_daily['RestingHeartRate'].mean() if 'RestingHeartRate' in filtered_hr_daily.columns else min_hr
            
            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                metric_card("Average HR", f"{avg_hr:.1f} bpm")
            
            with col2:
                metric_card("Resting HR", f"{resting_hr:.1f} bpm")
            
            with col3:
                metric_card("Min HR", f"{min_hr:.1f} bpm")
            
            with col4:
                metric_card("Max HR", f"{max_hr:.1f} bpm")
            
            # Create heart rate trends chart
            fig = go.Figure()
            
            # Add max heart rate
            fig.add_trace(go.Scatter(
                x=filtered_hr_daily['Date'],
                y=filtered_hr_daily['MaxHeartRate'],
                mode='lines+markers',
                name='Max HR',
                line=dict(color='rgba(219, 68, 55, 0.8)', width=1),
                marker=dict(size=5)
            ))
            
            # Add average heart rate
            fig.add_trace(go.Scatter(
                x=filtered_hr_daily['Date'],
                y=filtered_hr_daily['AvgHeartRate'],
                mode='lines+markers',
                name='Avg HR',
                line=dict(color='rgba(66, 133, 244, 0.8)', width=2),
                marker=dict(size=5)
            ))
            
            # Add resting heart rate
            hr_col = 'RestingHeartRate' if 'RestingHeartRate' in filtered_hr_daily.columns else 'MinHeartRate'
            hr_name = 'Resting HR' if 'RestingHeartRate' in filtered_hr_daily.columns else 'Min HR'
            
            fig.add_trace(go.Scatter(
                x=filtered_hr_daily['Date'],
                y=filtered_hr_daily[hr_col],
                mode='lines+markers',
                name=hr_name,
                line=dict(color='rgba(15, 157, 88, 0.8)', width=1),
                marker=dict(size=5)
            ))
            
            fig.update_layout(
                title='Heart Rate Trends',
                xaxis_title='Date',
                yaxis_title='Heart Rate (bpm)',
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                margin=dict(l=40, r=40, t=60, b=40),
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Heart rate distribution
            fig = px.histogram(
                filtered_hr,
                x='Value',
                nbins=30,
                labels={'Value': 'Heart Rate (bpm)'},
                title='Heart Rate Distribution',
                color_discrete_sequence=['#1E88E5']
            )
            
            # Add zones if available
            if 'HeartRateZone' in filtered_hr.columns:
                # Get zone boundaries
                zones = filtered_hr['HeartRateZone'].unique()
                zone_colors = {
                    'Rest': 'rgba(66, 133, 244, 0.2)',
                    'Fat Burn': 'rgba(15, 157, 88, 0.2)',
                    'Cardio': 'rgba(251, 188, 5, 0.2)',
                    'Peak': 'rgba(219, 68, 55, 0.2)'
                }
                
                # Add zone shapes
                for zone in zones:
                    if zone in zone_colors and zone != 'Abnormal':
                        # Get min and max values for this zone
                        zone_mask = filtered_hr['HeartRateZone'] == zone
                        zone_data = filtered_hr[zone_mask]
                        
                        if not zone_data.empty:
                            zone_min = zone_data['Value'].min()
                            zone_max = zone_data['Value'].max()
                            
                            fig.add_shape(
                                type='rect',
                                x0=zone_min,
                                y0=0,
                                x1=zone_max,
                                y1=1,
                                yref='paper',
                                fillcolor=zone_colors[zone],
                                line=dict(width=0)
                            )
                            
                            fig.add_annotation(
                                x=(zone_min + zone_max) / 2,
                                y=0.95,
                                yref='paper',
                                text=zone,
                                showarrow=False,
                                font=dict(size=10, color='rgba(0, 0, 0, 0.6)')
                            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Daily heart rate summary data not available.")
    
    # Tab 2: Daily Analysis
    with tabs[1]:
        st.markdown('<div class="sub-header">Daily Heart Rate Analysis</div>', unsafe_allow_html=True)
        
        # Allow user to select a specific date
        available_dates = sorted(filtered_hr['Date'].unique())
        
        if not available_dates:
            st.warning("No heart rate data available for the selected date range.")
            return
        
        selected_date = st.selectbox(
            "Select Date to Analyze:",
            available_dates,
            index=len(available_dates) - 1  # Default to most recent date
        )
        
        # Plot heart rate for the selected date
        hr_plot = plot_heart_rate_analysis(filtered_hr, user_id, selected_date)
        if hr_plot:
            st.plotly_chart(hr_plot, use_container_width=True)
        
        # Calculate daily statistics for the selected date
        selected_hr = filtered_hr[filtered_hr['Date'] == selected_date]
        
        if not selected_hr.empty:
            avg_hr = selected_hr['Value'].mean()
            min_hr = selected_hr['Value'].min()
            max_hr = selected_hr['Value'].max()
            resting_hr = np.percentile(selected_hr['Value'], 5)  # Approximate resting HR
            
            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                metric_card("Average HR", f"{avg_hr:.1f} bpm")
            
            with col2:
                metric_card("Resting HR", f"{resting_hr:.1f} bpm")
            
            with col3:
                metric_card("Min HR", f"{min_hr:.0f} bpm")
            
            with col4:
                metric_card("Max HR", f"{max_hr:.0f} bpm")
            
            # Calculate heart rate variability for the selected date
            hr_stddev = selected_hr['Value'].std()
            hr_range = max_hr - min_hr
            
            col1, col2 = st.columns(2)
            
            with col1:
                metric_card("HR Variability", f"{hr_stddev:.1f} bpm")
            
            with col2:
                metric_card("HR Range", f"{hr_range:.0f} bpm")
            
            # Heart rate by hour of day
            selected_hr['Hour'] = selected_hr['Time'].dt.hour
            hourly_hr = selected_hr.groupby('Hour')['Value'].agg(['mean', 'min', 'max']).reset_index()
            
            # Create hourly heart rate chart
            fig = go.Figure()
            
            # Add max heart rate
            fig.add_trace(go.Scatter(
                x=hourly_hr['Hour'],
                y=hourly_hr['max'],
                mode='lines',
                name='Max HR',
                line=dict(color='rgba(219, 68, 55, 0.5)', width=1),
                fill=None
            ))
            
            # Add min heart rate
            fig.add_trace(go.Scatter(
                x=hourly_hr['Hour'],
                y=hourly_hr['min'],
                mode='lines',
                name='Min HR',
                line=dict(color='rgba(15, 157, 88, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(66, 133, 244, 0.1)'
            ))
            
            # Add average heart rate
            fig.add_trace(go.Scatter(
                x=hourly_hr['Hour'],
                y=hourly_hr['mean'],
                mode='lines+markers',
                name='Avg HR',
                line=dict(color='rgba(66, 133, 244, 0.8)', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=f'Heart Rate by Hour of Day ({selected_date})',
                xaxis_title='Hour of Day',
                yaxis_title='Heart Rate (bpm)',
                xaxis=dict(
                    tickvals=list(range(0, 24)),
                    ticktext=[f"{h:02d}:00" for h in range(0, 24)]
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                margin=dict(l=40, r=40, t=60, b=40),
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No heart rate data available for {selected_date}.")
    
    # Tab 3: Zone Analysis
    with tabs[2]:
        st.markdown('<div class="sub-header">Heart Rate Zone Analysis</div>', unsafe_allow_html=True)
        
        if 'HeartRateZone' in filtered_hr.columns:
            # Calculate time in each zone
            zone_counts = filtered_hr['HeartRateZone'].value_counts()
            
            # Convert to minutes (assuming data is in seconds)
            zone_minutes = zone_counts / 60
            
            # Create a bar chart of time in each zone
            zone_df = pd.DataFrame({
                'Zone': zone_counts.index,
                'Minutes': zone_minutes.values
            })
            
            # Define zone order
            zone_order = ['Rest', 'Fat Burn', 'Cardio', 'Peak', 'Abnormal']
            zone_df['ZoneOrder'] = zone_df['Zone'].map({zone: i for i, zone in enumerate(zone_order)})
            zone_df = zone_df.sort_values('ZoneOrder')
            
            # Define zone colors
            zone_colors = {
                'Rest': 'rgba(66, 133, 244, 0.7)',
                'Fat Burn': 'rgba(15, 157, 88, 0.7)',
                'Cardio': 'rgba(251, 188, 5, 0.7)',
                'Peak': 'rgba(219, 68, 55, 0.7)',
                'Abnormal': 'rgba(0, 0, 0, 0.7)'
            }
            
            # Create bar chart
            fig = px.bar(
                zone_df,
                x='Zone',
                y='Minutes',
                title='Time Spent in Each Heart Rate Zone',
                color='Zone',
                color_discrete_map=zone_colors,
                category_orders={'Zone': [z for z in zone_order if z in zone_df['Zone'].values]}
            )
            
            fig.update_layout(
                xaxis_title='Heart Rate Zone',
                yaxis_title='Time (minutes)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display zone information
            st.markdown("""
                <div class="card">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">Heart Rate Zones</div>
                    <ul>
                        <li><b>Rest:</b> Resting heart rate, typically while sleeping or very relaxed</li>
                        <li><b>Fat Burn:</b> Low to moderate intensity, good for fat metabolism and endurance</li>
                        <li><b>Cardio:</b> Moderate to high intensity, improves cardiovascular fitness</li>
                        <li><b>Peak:</b> High intensity, improves performance and speed</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            # Zone distribution by day
            if filtered_hr_daily is not None and all(col in filtered_hr_daily.columns for col in ['TimeInRestZone', 'TimeInFatBurnZone', 'TimeInCardioZone', 'TimeInPeakZone']):
                # Create a stacked bar chart of zones by day
                zone_columns = ['TimeInRestZone', 'TimeInFatBurnZone', 'TimeInCardioZone', 'TimeInPeakZone']
                
                # Convert to minutes
                for col in zone_columns:
                    # Ensure the column contains numeric values
                    if col in filtered_hr_daily.columns:
                        filtered_hr_daily = filtered_hr_daily.copy()
                        # Convert any non-numeric values to NaN and then to 0
                        filtered_hr_daily[col] = pd.to_numeric(filtered_hr_daily[col], errors='coerce').fillna(0)
                        filtered_hr_daily[f"{col}Min"] = filtered_hr_daily[col] / 60
                
                # Create melted dataframe for plotting
                zone_melted = filtered_hr_daily.melt(
                    id_vars=['Date'],
                    value_vars=[f"{col}Min" for col in zone_columns],
                    var_name='Zone',
                    value_name='Minutes'
                )
                
                # Map zone names
                zone_map = {
                    'TimeInRestZoneMin': 'Rest',
                    'TimeInFatBurnZoneMin': 'Fat Burn',
                    'TimeInCardioZoneMin': 'Cardio',
                    'TimeInPeakZoneMin': 'Peak'
                }
                zone_melted['Zone'] = zone_melted['Zone'].map(zone_map)
                
                # Create stacked bar chart
                fig = px.bar(
                    zone_melted,
                    x='Date',
                    y='Minutes',
                    color='Zone',
                    title='Heart Rate Zones by Day',
                    barmode='stack',
                    color_discrete_map=zone_colors
                )
                
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Time (minutes)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate average time in each zone
                st.markdown('<div class="sub-header">Average Daily Time in Each Zone</div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_rest = filtered_hr_daily['TimeInRestZoneMin'].mean()
                    metric_card("Rest Zone", f"{avg_rest:.1f} min")
                
                with col2:
                    avg_fat_burn = filtered_hr_daily['TimeInFatBurnZoneMin'].mean()
                    metric_card("Fat Burn Zone", f"{avg_fat_burn:.1f} min")
                
                with col3:
                    avg_cardio = filtered_hr_daily['TimeInCardioZoneMin'].mean()
                    metric_card("Cardio Zone", f"{avg_cardio:.1f} min")
                
                with col4:
                    avg_peak = filtered_hr_daily['TimeInPeakZoneMin'].mean()
                    metric_card("Peak Zone", f"{avg_peak:.1f} min")
        else:
            st.warning("Heart rate zone data not available for the selected user.")
    
    # Tab 4: Anomaly Detection
    with tabs[3]:
        st.markdown('<div class="sub-header">Heart Rate Anomaly Detection</div>', unsafe_allow_html=True)
        
        # Run anomaly detection
        anomalies = detect_heart_rate_anomalies(filtered_hr, contamination=0.01)
        
        if anomalies is not None and not anomalies.empty:
            # Count anomalies by date
            anomaly_counts = anomalies.groupby(['Date'])['anomaly'].sum().reset_index()
            anomaly_counts.columns = ['Date', 'AnomalyCount']
            
            # Calculate percentage of readings that are anomalies
            reading_counts = anomalies.groupby(['Date']).size().reset_index()
            reading_counts.columns = ['Date', 'TotalReadings']
            
            anomaly_summary = pd.merge(anomaly_counts, reading_counts, on='Date')
            anomaly_summary['AnomalyPercentage'] = (anomaly_summary['AnomalyCount'] / anomaly_summary['TotalReadings'] * 100).round(2)
            
            # Create bar chart of anomalies by date
            fig = px.bar(
                anomaly_summary,
                x='Date',
                y='AnomalyCount',
                title='Heart Rate Anomalies by Date',
                color='AnomalyPercentage',
                color_continuous_scale='Reds',
                labels={'AnomalyCount': 'Number of Anomalies', 'AnomalyPercentage': 'Anomaly %'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display days with highest anomaly percentages
            st.markdown('<div class="sub-header">Dates with Highest Anomaly Percentages</div>', unsafe_allow_html=True)
            
            top_anomaly_days = anomaly_summary.sort_values('AnomalyPercentage', ascending=False).head(5)
            
            for _, row in top_anomaly_days.iterrows():
                st.markdown(f"""
                    <div class="card">
                        <div style="font-weight: 600;">{row['Date'].strftime('%Y-%m-%d')}</div>
                        <div>{row['AnomalyCount']} anomalies detected ({row['AnomalyPercentage']}% of readings)</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Allow user to select a date with anomalies to view
            anomaly_dates = anomaly_summary[anomaly_summary['AnomalyCount'] > 0]['Date'].tolist()
            
            if anomaly_dates:
                selected_anomaly_date = st.selectbox(
                    "Select date to view anomalies:",
                    anomaly_dates,
                    index=0
                )
                
                # Filter to the selected date
                date_anomalies = anomalies[anomalies['Date'] == selected_anomaly_date]
                date_anomalies = date_anomalies.sort_values('Time')
                
                # Create a plot showing heart rate with anomalies highlighted
                fig = go.Figure()
                
                # Add all heart rate points
                fig.add_trace(go.Scatter(
                    x=date_anomalies['Time'],
                    y=date_anomalies['Value'],
                    mode='lines',
                    name='Heart Rate',
                    line=dict(color='rgba(66, 133, 244, 0.7)', width=1)
                ))
                
                # Add anomalies
                anomaly_points = date_anomalies[date_anomalies['anomaly'] == 1]
                
                if not anomaly_points.empty:
                    fig.add_trace(go.Scatter(
                        x=anomaly_points['Time'],
                        y=anomaly_points['Value'],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(
                            size=8,
                            color='rgba(219, 68, 55, 1.0)',
                            symbol='circle',
                            line=dict(width=1, color='rgba(219, 68, 55, 1.0)')
                        )
                    ))
                
                fig.update_layout(
                    title=f'Heart Rate Anomalies on {selected_anomaly_date.strftime("%Y-%m-%d")}',
                    xaxis_title='Time',
                    yaxis_title='Heart Rate (bpm)',
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    ),
                    margin=dict(l=40, r=40, t=60, b=40),
                    hovermode='x'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add an explanation of what anomalies might indicate
                st.markdown("""
                    <div class="card">
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">About Heart Rate Anomalies</div>
                        <div>
                            <p>Anomalies represent unusual heart rate patterns that differ significantly from typical values. These could be caused by:</p>
                            <ul>
                                <li>Intense physical activity</li>
                                <li>Stress or anxiety</li>
                                <li>Device measurement errors</li>
                                <li>Certain medications</li>
                                <li>Health conditions</li>
                            </ul>
                            <p><em>Note: This is not medical advice. Consult a healthcare professional for concerns about heart rate patterns.</em></p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No anomalies detected in the selected date range.")


# Weight Management page
def weight_management_page(data_dict, user_id, date_range):
    """
    Create the weight management page.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed dataframes
    user_id : str
        Selected user ID
    date_range : tuple
        (start_date, end_date)
    """
    st.markdown('<div class="main-header">Weight Management</div>', unsafe_allow_html=True)
    
    start_date, end_date = date_range
    
    # Get weight data
    weight_df = data_dict.get('weight')
    daily_activity_df = data_dict.get('daily_activity')
    
    if weight_df is None or user_id not in weight_df['Id'].unique():
        st.error("No weight data available for the selected user.")
        return
    
    # Filter data for the selected user and date range
    filtered_weight = weight_df[
        (weight_df['Id'] == user_id) & 
        (weight_df['Date'] >= pd.Timestamp(start_date)) & 
        (weight_df['Date'] <= pd.Timestamp(end_date))
    ]
    
    if filtered_weight.empty:
        st.warning(f"No weight data available for the selected date range: {start_date} to {end_date}")
        return
    
    # Create tabs for different analyses
    tabs = st.tabs(["Weight Trends", "BMI Analysis", "Weight and Activity", "Metabolic Calculations"])
    
    # Tab 1: Weight Trends
    with tabs[0]:
        st.markdown('<div class="sub-header">Weight Trends</div>', unsafe_allow_html=True)
        
        # Calculate summary statistics
        current_weight = filtered_weight.iloc[-1]['WeightKg']
        initial_weight = filtered_weight.iloc[0]['WeightKg']
        weight_change = current_weight - initial_weight
        avg_weight = filtered_weight['WeightKg'].mean()
        min_weight = filtered_weight['WeightKg'].min()
        max_weight = filtered_weight['WeightKg'].max()
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            metric_card("Current Weight", f"{current_weight:.1f} kg", delta=f"{weight_change:.1f} kg", delta_suffix="since first record")
        with col2:
            metric_card("Average Weight", f"{avg_weight:.1f} kg")
        
        with col3:
            metric_card("Weight Range", f"{min_weight:.1f} - {max_weight:.1f} kg")
        
        # Plot weight trend
        weight_trend = plot_weight_trend(filtered_weight, user_id, date_range)
        if weight_trend:
            st.plotly_chart(weight_trend, use_container_width=True)
        
        # Calculate weight change rate
        if len(filtered_weight) > 1:
            days_diff = (filtered_weight.iloc[-1]['Date'] - filtered_weight.iloc[0]['Date']).days
            if days_diff > 0:
                weight_change_rate = weight_change / (days_diff / 7)  # Change per week
                
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-label">Weight Change Rate</div>
                        <div class="metric-value">{weight_change_rate:.2f} kg/week</div>
                        <div>Based on data from {days_diff} days</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Add an interpretation of the weight change rate
                if abs(weight_change_rate) > 1:
                    st.warning("This rate of weight change is relatively fast. A sustainable rate is typically 0.5-1 kg per week.")
                elif abs(weight_change_rate) > 0:
                    st.info("This is a moderate and generally sustainable rate of weight change.")
    
    # Tab 2: BMI Analysis
    with tabs[1]:
        st.markdown('<div class="sub-header">BMI Analysis</div>', unsafe_allow_html=True)
        
        if 'BMI' in filtered_weight.columns:
            # Calculate BMI statistics
            current_bmi = filtered_weight.iloc[-1]['BMI']
            initial_bmi = filtered_weight.iloc[0]['BMI']
            bmi_change = current_bmi - initial_bmi
            avg_bmi = filtered_weight['BMI'].mean()
            
            # BMI categories
            bmi_categories = {
                'Underweight': (0, 18.5),
                'Normal': (18.5, 25),
                'Overweight': (25, 30),
                'Obese': (30, float('inf'))
            }
            
            # Determine current BMI category
            current_category = None
            for category, (lower, upper) in bmi_categories.items():
                if lower <= current_bmi < upper:
                    current_category = category
                    break
            
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                metric_card("Current BMI", f"{current_bmi:.1f}", delta=f"{bmi_change:.1f}", delta_suffix="since first record")
            
            with col2:
                metric_card("BMI Category", current_category)
            
            with col3:
                metric_card("Average BMI", f"{avg_bmi:.1f}")
            
            # Create BMI over time chart
            fig = px.line(
                filtered_weight,
                x='Date',
                y='BMI',
                title='BMI Trend Over Time',
                color_discrete_sequence=['#1E88E5']
            )
            
            # Add BMI category zones
            for category, (lower, upper) in bmi_categories.items():
                if category != 'Obese':  # Skip the upper limit for obese
                    fig.add_shape(
                        type='line',
                        x0=filtered_weight['Date'].min(),
                        y0=upper,
                        x1=filtered_weight['Date'].max(),
                        y1=upper,
                        line=dict(color='rgba(0, 0, 0, 0.3)', width=1, dash='dash')
                    )
                    
                    # Add annotation for the category boundary
                    fig.add_annotation(
                        x=filtered_weight['Date'].max(),
                        y=upper,
                        text=f"{category} / {list(bmi_categories.keys())[list(bmi_categories.keys()).index(category) + 1]}",
                        showarrow=False,
                        xshift=10,
                        yshift=0,
                        font=dict(size=10)
                    )
            
            # Highlight the normal BMI range
            fig.add_shape(
                type='rect',
                x0=filtered_weight['Date'].min(),
                y0=18.5,
                x1=filtered_weight['Date'].max(),
                y1=25,
                fillcolor='rgba(15, 157, 88, 0.1)',
                line=dict(width=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show BMI distribution
            st.markdown('<div class="sub-header">BMI Category Distribution</div>', unsafe_allow_html=True)
            
            # Categorize all BMI readings
            filtered_weight['BMI_Category'] = pd.cut(
                filtered_weight['BMI'],
                bins=[0, 18.5, 25, 30, float('inf')],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese']
            )
            
            # Count readings in each category
            bmi_counts = filtered_weight['BMI_Category'].value_counts().reset_index()
            bmi_counts.columns = ['Category', 'Count']
            bmi_counts['Percentage'] = (bmi_counts['Count'] / bmi_counts['Count'].sum() * 100).round(1)
            
            # Create pie chart
            fig = px.pie(
                bmi_counts,
                values='Percentage',
                names='Category',
                title='BMI Category Distribution',
                color='Category',
                color_discrete_map={
                    'Underweight': '#90CAF9',
                    'Normal': '#4CAF50',
                    'Overweight': '#FFC107',
                    'Obese': '#F44336'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display BMI information
            st.markdown("""
                <div class="card">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">About BMI (Body Mass Index)</div>
                    <div>
                        <p>BMI is a measure of body fat based on height and weight that applies to adult men and women. The categories are:</p>
                        <ul>
                            <li><b>Underweight:</b> BMI less than 18.5</li>
                            <li><b>Normal weight:</b> BMI 18.5 to 24.9</li>
                            <li><b>Overweight:</b> BMI 25 to 29.9</li>
                            <li><b>Obesity:</b> BMI 30 or greater</li>
                        </ul>
                        <p><em>Note: BMI has limitations and doesn't account for factors like muscle mass or body composition.</em></p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("BMI data not available for the selected user.")
    
    # Tab 3: Weight and Activity
    with tabs[2]:
        st.markdown('<div class="sub-header">Weight and Activity Relationship</div>', unsafe_allow_html=True)
        
        if daily_activity_df is not None and 'Calories' in daily_activity_df.columns:
            # Filter activity data
            filtered_activity = daily_activity_df[
                (daily_activity_df['Id'] == user_id) & 
                (daily_activity_df['Date'] >= pd.Timestamp(start_date)) & 
                (daily_activity_df['Date'] <= pd.Timestamp(end_date))
            ]
            
            # Ensure expected columns exist (using TotalSteps instead of Steps)
            expected_cols = ['Date', 'TotalSteps', 'Calories', 'VeryActiveMinutes', 'FairlyActiveMinutes']
            available_cols = ['Date']
            col_mapping = {
                'TotalSteps': ['TotalSteps', 'Steps', 'totalsteps', 'steps'],
                'Calories': ['Calories', 'calories'],
                'VeryActiveMinutes': ['VeryActiveMinutes', 'veryactiveminutes'],
                'FairlyActiveMinutes': ['FairlyActiveMinutes', 'fairlyactiveminutes'],
            }
            
            # Find available columns with case insensitivity
            for target_col, possible_cols in col_mapping.items():
                for col in possible_cols:
                    if col in filtered_activity.columns:
                        available_cols.append(col)
                        break
            
            if not filtered_activity.empty and not filtered_weight.empty:
                # Merge weight and activity data on the nearest date
                # First, sort both dataframes by date
                filtered_weight = filtered_weight.sort_values('Date')
                filtered_activity = filtered_activity.sort_values('Date')
                
                # Create a merged dataframe for days with both weight and activity data
                merged_dates = pd.merge_asof(
                    filtered_weight[['Date', 'WeightKg', 'BMI']].sort_values('Date'),
                    filtered_activity[['Date', 'TotalSteps', 'Calories', 'VeryActiveMinutes', 'FairlyActiveMinutes']].sort_values('Date'),
                    on='Date',
                    direction='nearest'
                )
                
                if not merged_dates.empty:
                    # Create scatter plot of steps vs weight
                    if 'TotalSteps' in merged_dates.columns:
                        fig = px.scatter(
                            merged_dates,
                            x='TotalSteps',
                            y='WeightKg',
                            labels={'TotalSteps': 'Daily Steps', 'WeightKg': 'Weight (kg)'},
                            title='Steps vs. Weight',
                            trendline='ols',
                            color_discrete_sequence=['#1E88E5']
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Create scatter plot of calories vs weight
                    if 'Calories' in merged_dates.columns:
                        fig = px.scatter(
                            merged_dates,
                            x='Calories',
                            y='WeightKg',
                            labels={'Calories': 'Daily Calories Burned', 'WeightKg': 'Weight (kg)'},
                            title='Calories Burned vs. Weight',
                            trendline='ols',
                            color_discrete_sequence=['#1E88E5']
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Create scatter plot of active minutes vs weight
                    if 'VeryActiveMinutes' in merged_dates.columns and 'FairlyActiveMinutes' in merged_dates.columns:
                        merged_dates['ActiveMinutes'] = merged_dates['VeryActiveMinutes'] + merged_dates['FairlyActiveMinutes']
                        
                        fig = px.scatter(
                            merged_dates,
                            x='ActiveMinutes',
                            y='WeightKg',
                            labels={'ActiveMinutes': 'Active Minutes (Moderate + Vigorous)', 'WeightKg': 'Weight (kg)'},
                            title='Active Minutes vs. Weight',
                            trendline='ols',
                            color_discrete_sequence=['#1E88E5']
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display correlation information
                    st.markdown('<div class="sub-header">Activity-Weight Correlations</div>', unsafe_allow_html=True)
                    
                    correlations = {}
                    if 'TotalSteps' in merged_dates.columns:
                        correlations['TotalSteps'] = merged_dates['TotalSteps'].corr(merged_dates['WeightKg'])
                    if 'Calories' in merged_dates.columns:
                        correlations['Calories'] = merged_dates['Calories'].corr(merged_dates['WeightKg'])
                    if 'ActiveMinutes' in merged_dates.columns:
                        correlations['Active Minutes'] = merged_dates['ActiveMinutes'].corr(merged_dates['WeightKg'])
                    
                    if correlations:
                        # Create a bar chart of correlations
                        corr_df = pd.DataFrame({
                            'Metric': list(correlations.keys()),
                            'Correlation with Weight': list(correlations.values())
                        })
                        
                        fig = px.bar(
                            corr_df,
                            x='Metric',
                            y='Correlation with Weight',
                            title='Correlation between Activity Metrics and Weight',
                            color_discrete_sequence=['#1E88E5']
                        )
                        
                        # Add a line at zero
                        fig.add_shape(
                            type='line',
                            x0=-0.5,
                            y0=0,
                            x1=len(correlations) - 0.5,
                            y1=0,
                            line=dict(color='black', width=1, dash='dash')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Provide an interpretation of the correlations
                        st.markdown("""
                            <div class="card">
                                <div style="font-weight: 600; margin-bottom: 0.5rem;">Interpretation of Correlations</div>
                                <div>
                                    <ul>
                                        <li>Negative correlation means that as activity increases, weight tends to decrease</li>
                                        <li>Positive correlation means that as activity increases, weight tends to increase</li>
                                        <li>Values closer to -1 or 1 indicate stronger relationships</li>
                                        <li>Values close to 0 indicate weak or no relationship</li>
                                    </ul>
                                    <p><em>Note: Correlation does not necessarily imply causation. Many factors influence weight.</em></p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Not enough overlapping data between weight and activity records to perform analysis.")
            else:
                st.warning("Not enough activity or weight data available for the selected period.")
        else:
            st.warning("Activity data not available for correlation analysis.")
    
    # Tab 4: Metabolic Calculations
    with tabs[3]:
        st.markdown('<div class="sub-header">Metabolic Calculations</div>', unsafe_allow_html=True)
        
        # Create form for user inputs
        with st.form("metabolic_form"):
            st.markdown("##### Enter Additional Information for Metabolic Calculations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            
            with col2:
                age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
            
            with col3:
                gender = st.selectbox("Gender", ["Male", "Female"])
            
            activity_level = st.select_slider(
                "Activity Level",
                options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"],
                value="Moderately Active"
            )
            
            activity_map = {
                "Sedentary": "sedentary",
                "Lightly Active": "lightly_active",
                "Moderately Active": "moderately_active",
                "Very Active": "very_active",
                "Extra Active": "extra_active"
            }
            
            submit_button = st.form_submit_button(label="Calculate")
        
        # Process form submission
        if submit_button or 'weight_calc_done' in st.session_state:
            st.session_state.weight_calc_done = True
            
            # Use the most recent weight
            weight_kg = current_weight
            
            # Calculate BMI
            bmi = weight_kg / ((height_cm / 100) ** 2)
            
            # Calculate BMR (Basal Metabolic Rate) using Mifflin-St Jeor Equation
            if gender == "Male":
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
            else:  # Female
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
            
            # Calculate TDEE (Total Daily Energy Expenditure)
            activity_multipliers = {
                "sedentary": 1.2,
                "lightly_active": 1.375,
                "moderately_active": 1.55,
                "very_active": 1.725,
                "extra_active": 1.9
            }
            
            tdee = bmr * activity_multipliers[activity_map[activity_level]]
            
            # Display results
            st.markdown('<div class="sub-header">Metabolic Results</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-label">BMI (Body Mass Index)</div>
                        <div class="metric-value">{bmi:.1f}</div>
                        <div>Height: {height_cm} cm, Weight: {weight_kg:.1f} kg</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-label">BMR (Basal Metabolic Rate)</div>
                        <div class="metric-value">{bmr:.0f}</div>
                        <div>Calories burned at complete rest</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-label">TDEE (Total Daily Energy Expenditure)</div>
                        <div class="metric-value">{tdee:.0f}</div>
                        <div>Estimated daily calorie expenditure</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Weight goals section
            st.markdown('<div class="sub-header">Weight Management Goals</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-label">Weight Loss Plan</div>
                        <div class="metric-value">{tdee - 500:.0f}</div>
                        <div>Daily calories for 0.5 kg/week loss</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-label">Maintenance Plan</div>
                        <div class="metric-value">{tdee:.0f}</div>
                        <div>Daily calories to maintain weight</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="card">
                        <div class="metric-label">Weight Gain Plan</div>
                        <div class="metric-value">{tdee + 500:.0f}</div>
                        <div>Daily calories for 0.5 kg/week gain</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Add explanatory information
            st.markdown("""
                <div class="card">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">Metabolic Calculations Information</div>
                    <div>
                        <p><b>BMR (Basal Metabolic Rate):</b> The number of calories your body needs to maintain basic functions while at rest. This includes breathing, circulation, cell production, and basic neurological functions.</p>
                        <p><b>TDEE (Total Daily Energy Expenditure):</b> Your total daily calorie expenditure, including your BMR plus all additional activities.</p>
                        <p><b>Weight Management:</b></p>
                        <ul>
                            <li>For weight loss, create a calorie deficit by consuming less than your TDEE</li>
                            <li>For weight maintenance, consume calories equal to your TDEE</li>
                            <li>For weight gain, create a calorie surplus by consuming more than your TDEE</li>
                        </ul>
                        <p><em>A safe rate of weight change is typically 0.5-1 kg per week. This requires a daily calorie deficit or surplus of approximately 500-1000 calories.</em></p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

def main():
    """
    Main application function.
    """
    # Load CSS
    load_css()
    
    # App title and description
    st.title("Personal Fitness Tracker")
    st.markdown("Comprehensive analytics dashboard for Fitbit data")
    
    # Load data
    data_dict = load_data()
    
    if not data_dict or 'all_users' not in data_dict or not data_dict['all_users']:
        st.error("No data available. Please check the data directory.")
        return
    
    # Create sidebar
    st.sidebar.markdown("## Settings")
    
    # User selection
    user_id = user_selector(data_dict)
    
    if user_id is None:
        return
    
    # Date range selection
    date_range = date_range_selector(data_dict, user_id)
    
    if date_range[0] is None or date_range[1] is None:
        return
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select Page:",
        ["Overview", "Activity Analysis", "Sleep Insights", "Heart Rate Analysis", "Weight Management"],
        index=0
    )
    
    # Display the selected page
    if page == "Overview":
        overview_page(data_dict, user_id, date_range)
    elif page == "Activity Analysis":
        activity_analysis_page(data_dict, user_id, date_range)
    elif page == "Sleep Insights":
        sleep_insights_page(data_dict, user_id, date_range)
    elif page == "Heart Rate Analysis":
        heart_rate_analysis_page(data_dict, user_id, date_range)
    elif page == "Weight Management":
        weight_management_page(data_dict, user_id, date_range)

if __name__ == "__main__":
    main()