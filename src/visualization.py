"""
Visualization Module for Fitness Tracker Application.

This module provides visualization functions for Fitbit data
including charts, plots, and interactive components.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set default styles for matplotlib
plt.style.use('ggplot')
sns.set_style('whitegrid')

def create_activity_summary_cards(daily_activity_df, user_id=None, date_range=None):
    """
    Create summary cards for activity data.
    
    Parameters:
    -----------
    daily_activity_df : pandas.DataFrame
        DataFrame containing daily activity data
    user_id : str or None
        User ID to filter data for
    date_range : tuple or None
        (start_date, end_date) to filter data
        
    Returns:
    --------
    dict
        Dictionary with summary statistics
    """
    if daily_activity_df is None:
        return None
    
    # Filter data if user_id is provided
    if user_id:
        filtered_df = daily_activity_df[daily_activity_df['Id'] == user_id]
    else:
        filtered_df = daily_activity_df
    
    # Filter data if date_range is provided
    if date_range and 'Date' in filtered_df.columns:
        start_date, end_date = date_range
        filtered_df = filtered_df[(pd.to_datetime(filtered_df['Date']) >= start_date) & 
                                  (pd.to_datetime(filtered_df['Date']) <= end_date)]
    
    # Check if we have data after filtering
    if filtered_df.empty:
        return None
    
    # Calculate summary statistics
    summary = {}
    
    # Steps
    if 'Steps' in filtered_df.columns:
        summary['avg_steps'] = int(filtered_df['TotalSteps'] if 'TotalSteps' in df.columns else df['Steps'].mean())
        summary['max_steps'] = int(filtered_df['TotalSteps'] if 'TotalSteps' in df.columns else df['Steps'].max())
        summary['total_steps'] = int(filtered_df['TotalSteps'] if 'TotalSteps' in df.columns else df['Steps'].sum())
        
        # Goal achievement (assuming 10,000 steps goal)
        goal_achievement = (filtered_df['TotalSteps'] if 'TotalSteps' in df.columns else df['Steps'] >= 10000).mean() * 100
        summary['steps_goal_achievement'] = round(goal_achievement, 1)
    
    # Distance
    if 'TotalDistance' in filtered_df.columns:
        summary['avg_distance'] = round(filtered_df['TotalDistance'].mean(), 2)
        summary['max_distance'] = round(filtered_df['TotalDistance'].max(), 2)
        summary['total_distance'] = round(filtered_df['TotalDistance'].sum(), 2)
    
    # Calories
    if 'Calories' in filtered_df.columns:
        summary['avg_calories'] = int(filtered_df['Calories'].mean())
        summary['max_calories'] = int(filtered_df['Calories'].max())
        summary['total_calories'] = int(filtered_df['Calories'].sum())
    
    # Active minutes
    active_columns = ['VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes']
    if all(col in filtered_df.columns for col in active_columns):
        # Total active minutes (sum of all active categories)
        filtered_df['TotalActiveMinutes'] = (
            filtered_df['VeryActiveMinutes'] + 
            filtered_df['FairlyActiveMinutes'] + 
            filtered_df['LightlyActiveMinutes']
        )
        
        summary['avg_active_minutes'] = int(filtered_df['TotalActiveMinutes'].mean())
        summary['avg_very_active'] = int(filtered_df['VeryActiveMinutes'].mean())
        summary['avg_fairly_active'] = int(filtered_df['FairlyActiveMinutes'].mean())
        summary['avg_lightly_active'] = int(filtered_df['LightlyActiveMinutes'].mean())
        
        # Active days (at least 30 min of moderate/vigorous activity)
        active_days = (filtered_df['VeryActiveMinutes'] + filtered_df['FairlyActiveMinutes'] >= 30).mean() * 100
        summary['active_days_percentage'] = round(active_days, 1)
    
    # Health score
    if 'HealthScore' in filtered_df.columns:
        summary['avg_health_score'] = round(filtered_df['HealthScore'].mean(), 1)
        summary['max_health_score'] = round(filtered_df['HealthScore'].max(), 1)
        summary['min_health_score'] = round(filtered_df['HealthScore'].min(), 1)
    
    # Date range
    if 'Date' in filtered_df.columns:
        summary['start_date'] = filtered_df['Date'].min()
        summary['end_date'] = filtered_df['Date'].max()
        summary['num_days'] = len(filtered_df['Date'].unique())
    
    return summary

def plot_daily_steps_trend(daily_activity_df, user_id=None, date_range=None, rolling_window=7):
    """
    Plot trend of daily steps.
    
    Parameters:
    -----------
    daily_activity_df : pandas.DataFrame
        DataFrame containing daily activity data
    user_id : str or None
        User ID to filter data for
    date_range : tuple or None
        (start_date, end_date) to filter data
    rolling_window : int
        Window size for rolling average
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if daily_activity_df is None or 'Steps' not in daily_activity_df.columns:
        return None
    
    # Filter data if user_id is provided
    if user_id:
        filtered_df = daily_activity_df[daily_activity_df['Id'] == user_id]
    else:
        filtered_df = daily_activity_df
    
    # Filter data if date_range is provided
    if date_range and 'Date' in filtered_df.columns:
        start_date, end_date = date_range
        filtered_df = filtered_df[(pd.to_datetime(filtered_df['Date']) >= start_date) & 
                                  (pd.to_datetime(filtered_df['Date']) <= end_date)]
    
    # Check if we have data after filtering
    if filtered_df.empty:
        return None
    
    # Ensure data is sorted by date
    filtered_df = filtered_df.sort_values('Date')
    
    # Calculate rolling average
    filtered_df['RollingSteps'] = filtered_df['TotalSteps'] if 'TotalSteps' in df.columns else df['Steps'].rolling(window=rolling_window, min_periods=1).mean()
    
    # Create the figure
    fig = go.Figure()
    
    # Add daily steps
    fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['TotalSteps'] if 'TotalSteps' in df.columns else df['Steps'],
        mode='markers',
        name='Daily Steps',
        marker=dict(
            size=8,
            color='rgba(66, 133, 244, 0.7)',
            line=dict(width=1, color='rgba(66, 133, 244, 1.0)')
        )
    ))
    
    # Add rolling average
    fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['RollingSteps'],
        mode='lines',
        name=f'{rolling_window}-day Moving Average',
        line=dict(color='rgba(219, 68, 55, 0.9)', width=2)
    ))
    
    # Add goal line (10,000 steps)
    fig.add_shape(
        type='line',
        x0=filtered_df['Date'].min(),
        y0=10000,
        x1=filtered_df['Date'].max(),
        y1=10000,
        line=dict(
            color='rgba(15, 157, 88, 0.7)',
            width=2,
            dash='dash'
        )
    )
    
    # Add annotation for goal line
    fig.add_annotation(
        x=filtered_df['Date'].max(),
        y=10000,
        text='Goal: 10,000 steps',
        showarrow=False,
        yshift=10,
        font=dict(color='rgba(15, 157, 88, 0.9)')
    )
    
    # Update layout
    fig.update_layout(
        title='Daily Steps Trend',
        xaxis_title='Date',
        yaxis_title='Steps',
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
    
    return fig

def plot_activity_heatmap(hourly_steps_df, user_id=None, date_range=None):
    """
    Create a heatmap of activity by hour and day of week.
    
    Parameters:
    -----------
    hourly_steps_df : pandas.DataFrame
        DataFrame containing hourly steps data
    user_id : str or None
        User ID to filter data for
    date_range : tuple or None
        (start_date, end_date) to filter data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if hourly_steps_df is None or 'StepTotal' not in hourly_steps_df.columns:
        return None
    
    # Ensure ActivityHour is datetime
    if 'ActivityHour' in hourly_steps_df.columns and not pd.api.types.is_datetime64_any_dtype(hourly_steps_df['ActivityHour']):
        hourly_steps_df['ActivityHour'] = pd.to_datetime(hourly_steps_df['ActivityHour'])
    
    # Filter data if user_id is provided
    if user_id:
        filtered_df = hourly_steps_df[hourly_steps_df['Id'] == user_id]
    else:
        filtered_df = hourly_steps_df
    
    # Filter data if date_range is provided
    if date_range:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['ActivityHour'].dt.date >= start_date) & 
                                  (filtered_df['ActivityHour'].dt.date <= end_date)]
    
    # Check if we have data after filtering
    if filtered_df.empty:
        return None
    
    # Create a fresh copy to avoid SettingWithCopyWarning
    filtered_df = filtered_df.copy()
    
    # Ensure ActivityHour is datetime
    filtered_df['ActivityHour'] = pd.to_datetime(filtered_df['ActivityHour'], errors='coerce')
    
    # Add columns for Hour, DayOfWeek, and DayName
    filtered_df['Hour'] = 0
    filtered_df['DayOfWeek'] = 0
    filtered_df['DayName'] = "Unknown"
    
    # Extract hour and day of week (for non-NaT values)
    valid_mask = ~filtered_df['ActivityHour'].isna()
    if valid_mask.any():
        # Apply updates to valid rows only
        filtered_df.loc[valid_mask, 'Hour'] = filtered_df.loc[valid_mask, 'ActivityHour'].dt.hour
        filtered_df.loc[valid_mask, 'DayOfWeek'] = filtered_df.loc[valid_mask, 'ActivityHour'].dt.dayofweek
        filtered_df.loc[valid_mask, 'DayName'] = filtered_df.loc[valid_mask, 'ActivityHour'].dt.day_name()
    
    # Aggregate steps by day of week and hour
    heatmap_data = filtered_df.groupby(['DayOfWeek', 'Hour'])['StepTotal'].mean().reset_index()
    
    # Create a pivot table
    heatmap_pivot = heatmap_data.pivot(index='DayOfWeek', columns='Hour', values='StepTotal')
    
    # Get day names in correct order
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_map = {i: day for i, day in enumerate(days)}
    
    # Create the heatmap
    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x='Hour of Day', y='Day of Week', color='Steps'),
        x=[f"{hour}:00" for hour in range(24)],
        y=[day_map[i] for i in range(7)],
        color_continuous_scale='Blues',
        aspect='auto'
    )
    
    # Update layout
    fig.update_layout(
        title='Activity Heatmap by Hour and Day of Week',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        coloraxis_colorbar=dict(title='Average Steps'),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    return fig

def plot_sleep_patterns(sleep_df, user_id=None, date_range=None, rolling_window=7):
    """
    Plot sleep duration and efficiency.
    
    Parameters:
    -----------
    sleep_df : pandas.DataFrame
        DataFrame containing sleep data
    user_id : str or None
        User ID to filter data for
    date_range : tuple or None
        (start_date, end_date) to filter data
    rolling_window : int
        Window size for rolling average
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if sleep_df is None or 'TotalMinutesAsleep' not in sleep_df.columns:
        return None
    
    # Filter data if user_id is provided
    if user_id:
        filtered_df = sleep_df[sleep_df['Id'] == user_id]
    else:
        filtered_df = sleep_df
    
    # Filter data if date_range is provided
    if date_range and 'Date' in filtered_df.columns:
        start_date, end_date = date_range
        filtered_df = filtered_df[(pd.to_datetime(filtered_df['Date']) >= start_date) & 
                                  (pd.to_datetime(filtered_df['Date']) <= end_date)]
    
    # Check if we have data after filtering
    if filtered_df.empty:
        return None
    
    # Ensure data is sorted by date
    filtered_df = filtered_df.sort_values('Date')
    
    # Convert minutes to hours for better readability
    filtered_df['SleepHours'] = filtered_df['TotalMinutesAsleep'] / 60
    filtered_df['TimeInBedHours'] = filtered_df['TotalTimeInBed'] / 60
    
    # Calculate rolling averages
    filtered_df['RollingSleepHours'] = filtered_df['SleepHours'].rolling(window=rolling_window, min_periods=1).mean()
    
    if 'SleepEfficiency' in filtered_df.columns:
        filtered_df['RollingSleepEfficiency'] = filtered_df['SleepEfficiency'].rolling(window=rolling_window, min_periods=1).mean()
    
    # Create subplots: sleep duration and efficiency
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Sleep Duration', 'Sleep Efficiency')
    )
    
    # Add sleep duration
    fig.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['SleepHours'],
            mode='markers',
            name='Sleep Duration',
            marker=dict(
                size=8,
                color='rgba(66, 133, 244, 0.7)',
                line=dict(width=1, color='rgba(66, 133, 244, 1.0)')
            )
        ),
        row=1, col=1
    )
    
    # Add time in bed
    fig.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['TimeInBedHours'],
            mode='markers',
            name='Time in Bed',
            marker=dict(
                size=8,
                color='rgba(168, 168, 168, 0.7)',
                line=dict(width=1, color='rgba(168, 168, 168, 1.0)')
            )
        ),
        row=1, col=1
    )
    
    # Add rolling average for sleep duration
    fig.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['RollingSleepHours'],
            mode='lines',
            name=f'{rolling_window}-day Moving Average',
            line=dict(color='rgba(219, 68, 55, 0.9)', width=2)
        ),
        row=1, col=1
    )
    
    # Add recommended sleep range (7-9 hours)
    fig.add_shape(
        type='rect',
        x0=filtered_df['Date'].min(),
        y0=7,
        x1=filtered_df['Date'].max(),
        y1=9,
        fillcolor='rgba(15, 157, 88, 0.1)',
        line=dict(width=0),
        row=1, col=1
    )
    
    # Add annotation for recommended range
    fig.add_annotation(
        x=filtered_df['Date'].max(),
        y=8,
        text='Recommended: 7-9 hours',
        showarrow=False,
        xshift=-100,
        font=dict(color='rgba(15, 157, 88, 0.9)'),
        row=1, col=1
    )
    
    # Add sleep efficiency if available
    if 'SleepEfficiency' in filtered_df.columns:
        # Convert to percentage for better readability
        filtered_df['SleepEfficiencyPercent'] = filtered_df['SleepEfficiency'] * 100
        filtered_df['RollingSleepEfficiencyPercent'] = filtered_df['RollingSleepEfficiency'] * 100
        
        fig.add_trace(
            go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df['SleepEfficiencyPercent'],
                mode='markers',
                name='Sleep Efficiency',
                marker=dict(
                    size=8,
                    color='rgba(251, 188, 5, 0.7)',
                    line=dict(width=1, color='rgba(251, 188, 5, 1.0)')
                )
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df['RollingSleepEfficiencyPercent'],
                mode='lines',
                name=f'{rolling_window}-day Moving Average',
                line=dict(color='rgba(219, 68, 55, 0.9)', width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add good efficiency range (85%+)
        fig.add_shape(
            type='rect',
            x0=filtered_df['Date'].min(),
            y0=85,
            x1=filtered_df['Date'].max(),
            y1=100,
            fillcolor='rgba(15, 157, 88, 0.1)',
            line=dict(width=0),
            row=2, col=1
        )
        
        fig.add_annotation(
            x=filtered_df['Date'].max(),
            y=92.5,
            text='Good: 85%+',
            showarrow=False,
            xshift=-70,
            font=dict(color='rgba(15, 157, 88, 0.9)'),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='Sleep Patterns Over Time',
        xaxis2_title='Date',
        yaxis_title='Hours',
        yaxis2_title='Efficiency (%)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=700,
        hovermode='x'
    )
    
    return fig

def plot_heart_rate_analysis(heart_rate_df, user_id=None, date=None):
    """
    Plot heart rate data for a specific day.
    
    Parameters:
    -----------
    heart_rate_df : pandas.DataFrame
        DataFrame containing heart rate data
    user_id : str or None
        User ID to filter data for
    date : datetime.date or None
        Specific date to analyze
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if heart_rate_df is None or 'Value' not in heart_rate_df.columns:
        return None
    
    # Filter data if user_id is provided
    if user_id:
        filtered_df = heart_rate_df[heart_rate_df['Id'] == user_id]
    else:
        filtered_df = heart_rate_df
    
    # Filter data for the specific date or use the most recent date if not provided
    if date:
        filtered_df = filtered_df[filtered_df['Date'] == date]
    else:
        # Use the most recent date
        most_recent_date = filtered_df['Date'].max()
        filtered_df = filtered_df[filtered_df['Date'] == most_recent_date]
        date = most_recent_date
    
    # Check if we have data after filtering
    if filtered_df.empty:
        return None
    
    # Sort by time
    filtered_df = filtered_df.sort_values('Time')
    
    # Create the figure
    fig = go.Figure()
    
    # Add heart rate trace
    fig.add_trace(go.Scatter(
        x=filtered_df['Time'],
        y=filtered_df['Value'],
        mode='lines',
        name='Heart Rate',
        line=dict(
            color='rgba(219, 68, 55, 0.9)',
            width=1.5
        )
    ))
    
    # Add zones if they exist
    if 'HeartRateZone' in filtered_df.columns:
        # Define zone colors
        zone_colors = {
            'Rest': 'rgba(66, 133, 244, 0.7)',
            'Fat Burn': 'rgba(15, 157, 88, 0.7)',
            'Cardio': 'rgba(251, 188, 5, 0.7)',
            'Peak': 'rgba(219, 68, 55, 0.7)',
            'Abnormal': 'rgba(0, 0, 0, 0.7)'
        }
        
        # Add shapes for zones
        zones = filtered_df['HeartRateZone'].unique()
        
        for zone in zones:
            if zone not in zone_colors:
                continue
                
            # Get min and max values for this zone
            zone_mask = filtered_df['HeartRateZone'] == zone
            zone_data = filtered_df[zone_mask]
            
            if zone_data.empty:
                continue
                
            zone_min = zone_data['Value'].min()
            zone_max = zone_data['Value'].max()
            
            # Add a transparent rectangle for this zone
            fig.add_trace(go.Scatter(
                x=[filtered_df['Time'].min(), filtered_df['Time'].max()],
                y=[zone_min, zone_min],
                fill='tonexty',
                fillcolor=zone_colors[zone],
                line=dict(width=0),
                showlegend=False,
                opacity=0.1
            ))
    
    # Update layout
    fig.update_layout(
        title=f'Heart Rate Analysis for {date}',
        xaxis_title='Time',
        yaxis_title='Heart Rate (bpm)',
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
        hovermode='x',
        yaxis=dict(
            range=[
                max(40, filtered_df['Value'].min() - 10),  # Lower bound
                min(200, filtered_df['Value'].max() + 10)  # Upper bound
            ]
        )
    )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig

def plot_weight_trend(weight_df, user_id=None, date_range=None, rolling_window=7):
    """
    Plot weight trend over time.
    
    Parameters:
    -----------
    weight_df : pandas.DataFrame
        DataFrame containing weight data
    user_id : str or None
        User ID to filter data for
    date_range : tuple or None
        (start_date, end_date) to filter data
    rolling_window : int
        Window size for rolling average
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if weight_df is None or 'WeightKg' not in weight_df.columns:
        return None
    
    # Filter data if user_id is provided
    if user_id:
        filtered_df = weight_df[weight_df['Id'] == user_id]
    else:
        filtered_df = weight_df
    
    # Filter data if date_range is provided
    if date_range and 'Date' in filtered_df.columns:
        start_date, end_date = date_range
        filtered_df = filtered_df[(pd.to_datetime(filtered_df['Date']) >= start_date) & 
                                  (pd.to_datetime(filtered_df['Date']) <= end_date)]
    
    # Check if we have data after filtering
    if filtered_df.empty:
        return None
    
    # Ensure data is sorted by date
    filtered_df = filtered_df.sort_values('Date')
    
    # Calculate rolling average
    if len(filtered_df) >= rolling_window:
        filtered_df['RollingWeight'] = filtered_df['WeightKg'].rolling(window=rolling_window, min_periods=1).mean()
    else:
        filtered_df['RollingWeight'] = filtered_df['WeightKg']
    
    # Create the figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Weight', 'BMI'),
        row_heights=[0.7, 0.3]
    )
    
    # Add weight data
    fig.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['WeightKg'],
            mode='markers',
            name='Weight (kg)',
            marker=dict(
                size=8,
                color='rgba(66, 133, 244, 0.7)',
                line=dict(width=1, color='rgba(66, 133, 244, 1.0)')
            )
        ),
        row=1, col=1
    )
    
    # Add rolling average
    fig.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['RollingWeight'],
            mode='lines',
            name=f'{rolling_window}-day Moving Average',
            line=dict(color='rgba(219, 68, 55, 0.9)', width=2)
        ),
        row=1, col=1
    )
    
    # Add BMI if available
    if 'BMI' in filtered_df.columns:
        fig.add_trace(
            go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df['BMI'],
                mode='markers+lines',
                name='BMI',
                marker=dict(
                    size=8,
                    color='rgba(15, 157, 88, 0.7)',
                    line=dict(width=1, color='rgba(15, 157, 88, 1.0)')
                ),
                line=dict(
                    color='rgba(15, 157, 88, 0.5)',
                    width=1
                )
            ),
            row=2, col=1
        )
        
        # Add normal BMI range (18.5-24.9)
        fig.add_shape(
            type='rect',
            x0=filtered_df['Date'].min(),
            y0=18.5,
            x1=filtered_df['Date'].max(),
            y1=24.9,
            fillcolor='rgba(15, 157, 88, 0.1)',
            line=dict(width=0),
            row=2, col=1
        )
        
        fig.add_annotation(
            x=filtered_df['Date'].max(),
            y=21.7,
            text='Normal BMI: 18.5-24.9',
            showarrow=False,
            xshift=-100,
            font=dict(color='rgba(15, 157, 88, 0.9)'),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='Weight Trend Over Time',
        xaxis2_title='Date',
        yaxis_title='Weight (kg)',
        yaxis2_title='BMI',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=600,
        hovermode='x'
    )
    
    return fig

def plot_activity_distribution(daily_activity_df, user_id=None, date_range=None):
    """
    Plot distribution of activity minutes by category.
    
    Parameters:
    -----------
    daily_activity_df : pandas.DataFrame
        DataFrame containing daily activity data
    user_id : str or None
        User ID to filter data for
    date_range : tuple or None
        (start_date, end_date) to filter data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if daily_activity_df is None:
        return None
    
    active_columns = ['VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes']
    if not all(col in daily_activity_df.columns for col in active_columns):
        return None
    
    # Filter data if user_id is provided
    if user_id:
        filtered_df = daily_activity_df[daily_activity_df['Id'] == user_id]
    else:
        filtered_df = daily_activity_df
    
    # Filter data if date_range is provided
    if date_range and 'Date' in filtered_df.columns:
        start_date, end_date = date_range
        filtered_df = filtered_df[(pd.to_datetime(filtered_df['Date']) >= start_date) & 
                                  (pd.to_datetime(filtered_df['Date']) <= end_date)]
    
    # Check if we have data after filtering
    if filtered_df.empty:
        return None
    
    # Calculate daily average minutes for each category
    avg_minutes = {
        'Very Active': filtered_df['VeryActiveMinutes'].mean(),
        'Fairly Active': filtered_df['FairlyActiveMinutes'].mean(),
        'Lightly Active': filtered_df['LightlyActiveMinutes'].mean(),
        'Sedentary': filtered_df['SedentaryMinutes'].mean()
    }
    
    # Convert to percentage of day
    total_minutes = sum(avg_minutes.values())
    percentages = {k: (v / total_minutes * 100) for k, v in avg_minutes.items()}
    
    # Create a pie chart for the distribution
    labels = list(avg_minutes.keys())
    values = list(avg_minutes.values())
    percentages_list = list(percentages.values())
    
    # Custom text showing both minutes and percentages
    text = [f"{value:.0f} min ({percentage:.1f}%)" for value, percentage in zip(values, percentages_list)]
    
    # Colors for each category
    colors = ['rgba(219, 68, 55, 0.7)', 'rgba(244, 180, 0, 0.7)', 
              'rgba(15, 157, 88, 0.7)', 'rgba(66, 133, 244, 0.7)']
    
    # Create the pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        text=text,
        hoverinfo='label+text',
        textinfo='label',
        hole=.4,
        marker=dict(colors=colors)
    )])
    
    # Update layout
    fig.update_layout(
        title='Activity Distribution',
        annotations=[dict(
            text=f"Total<br>{total_minutes:.0f} min",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )],
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_health_score_components(daily_df, user_id=None, date=None):
    """
    Plot the components of the health score for a specific day.
    
    Parameters:
    -----------
    daily_df : pandas.DataFrame
        DataFrame containing daily merged data with health scores
    user_id : str or None
        User ID to filter data for
    date : datetime.date or None
        Specific date to analyze
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if daily_df is None:
        return None
    
    # Check if health score components are available
    required_cols = ['HealthScore', 'StepsScore', 'ActivityScore', 'SleepScore', 'HeartRateScore']
    if not all(col in daily_df.columns for col in required_cols):
        return None
    
    # Filter data if user_id is provided
    if user_id:
        filtered_df = daily_df[daily_df['Id'] == user_id]
    else:
        filtered_df = daily_df
    
    # Filter data for the specific date or use the most recent date if not provided
    if date:
        filtered_df = filtered_df[filtered_df['Date'] == date]
    else:
        # Use the most recent date
        most_recent_date = filtered_df['Date'].max()
        filtered_df = filtered_df[filtered_df['Date'] == most_recent_date]
        date = most_recent_date
    
    # Check if we have data after filtering
    if filtered_df.empty:
        return None
    
    # Get the first row (should be only one after filtering by user and date)
    row = filtered_df.iloc[0]
    
    # Create radar chart data
    categories = ['Steps', 'Activity', 'Sleep', 'Heart Rate']
    values = [row['StepsScore'], row['ActivityScore'], row['SleepScore'], row['HeartRateScore']]
    
    # Create the radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Health Components',
        line=dict(color='rgba(66, 133, 244, 0.8)'),
        fillcolor='rgba(66, 133, 244, 0.3)'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Health Score Components for {date}',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        annotations=[dict(
            text=f"Health Score<br>{row['HealthScore']:.1f}/100",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )]
    )
    
    return fig

def plot_activity_correlation(daily_activity_df, x_metric='Steps', y_metric='Calories', user_id=None, date_range=None):
    """
    Plot correlation between two activity metrics.
    
    Parameters:
    -----------
    daily_activity_df : pandas.DataFrame
        DataFrame containing daily activity data
    x_metric : str
        Metric to plot on x-axis
    y_metric : str
        Metric to plot on y-axis
    user_id : str or None
        User ID to filter data for
    date_range : tuple or None
        (start_date, end_date) to filter data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if daily_activity_df is None or x_metric not in daily_activity_df.columns or y_metric not in daily_activity_df.columns:
        return None
    
    # Filter data if user_id is provided
    if user_id:
        filtered_df = daily_activity_df[daily_activity_df['Id'] == user_id]
    else:
        filtered_df = daily_activity_df
    
    # Filter data if date_range is provided
    if date_range and 'Date' in filtered_df.columns:
        start_date, end_date = date_range
        filtered_df = filtered_df[(pd.to_datetime(filtered_df['Date']) >= start_date) & 
                                  (pd.to_datetime(filtered_df['Date']) <= end_date)]
    
    # Check if we have data after filtering
    if filtered_df.empty:
        return None
    
    # Calculate correlation
    correlation = filtered_df[x_metric].corr(filtered_df[y_metric])
    
    # Create the scatter plot
    fig = px.scatter(
        filtered_df,
        x=x_metric,
        y=y_metric,
        color='Date' if 'Date' in filtered_df.columns else None,
        hover_data=['Id', 'Date'] if 'Date' in filtered_df.columns else ['Id'],
        trendline='ols'
    )
    
    # Update layout
    fig.update_layout(
        title=f'Correlation between {x_metric} and {y_metric} (r = {correlation:.3f})',
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )
    
    return fig

def create_achievement_badges(daily_activity_df, user_id=None):
    """
    Create achievement badges based on activity data.
    
    Parameters:
    -----------
    daily_activity_df : pandas.DataFrame
        DataFrame containing daily activity data
    user_id : str or None
        User ID to filter data for
        
    Returns:
    --------
    dict
        Dictionary of badges earned with descriptions
    """
    if daily_activity_df is None:
        return None
    
    # Filter data if user_id is provided
    if user_id:
        filtered_df = daily_activity_df[daily_activity_df['Id'] == user_id]
    else:
        filtered_df = daily_activity_df
    
    # Check if we have data after filtering
    if filtered_df.empty:
        return None
    
    # Initialize badges dictionary
    badges = {}
    
    # Steps badges
    if 'Steps' in filtered_df.columns:
        max_steps = filtered_df['TotalSteps'] if 'TotalSteps' in df.columns else df['Steps'].max()
        if max_steps >= 20000:
            badges['marathon_walker'] = {
                'name': 'Marathon Walker',
                'description': 'Achieved over 20,000 steps in a single day',
                'level': 'Gold'
            }
        elif max_steps >= 15000:
            badges['super_stepper'] = {
                'name': 'Super Stepper',
                'description': 'Achieved over 15,000 steps in a single day',
                'level': 'Silver'
            }
        elif max_steps >= 10000:
            badges['step_achiever'] = {
                'name': 'Step Achiever',
                'description': 'Achieved over 10,000 steps in a single day',
                'level': 'Bronze'
            }
        
        # Consistency badge
        steps_consistency = (filtered_df['TotalSteps'] if 'TotalSteps' in df.columns else df['Steps'] >= 10000).mean() * 100
        if steps_consistency >= 80:
            badges['step_consistency'] = {
                'name': 'Step Consistency Master',
                'description': f'Achieved 10,000+ steps on {steps_consistency:.1f}% of days',
                'level': 'Gold'
            }
        elif steps_consistency >= 50:
            badges['step_consistency'] = {
                'name': 'Step Consistency Pro',
                'description': f'Achieved 10,000+ steps on {steps_consistency:.1f}% of days',
                'level': 'Silver'
            }
        elif steps_consistency >= 30:
            badges['step_consistency'] = {
                'name': 'Step Consistency Beginner',
                'description': f'Achieved 10,000+ steps on {steps_consistency:.1f}% of days',
                'level': 'Bronze'
            }
    
    # Active minutes badges
    active_columns = ['VeryActiveMinutes', 'FairlyActiveMinutes']
    if all(col in filtered_df.columns for col in active_columns):
        # Calculate total active minutes (moderate to vigorous)
        filtered_df['ModerateVigorousMinutes'] = filtered_df['VeryActiveMinutes'] + filtered_df['FairlyActiveMinutes']
        
        max_active = filtered_df['ModerateVigorousMinutes'].max()
        if max_active >= 120:
            badges['active_champion'] = {
                'name': 'Active Champion',
                'description': 'Achieved over 2 hours of moderate to vigorous activity in a single day',
                'level': 'Gold'
            }
        elif max_active >= 60:
            badges['active_warrior'] = {
                'name': 'Active Warrior',
                'description': 'Achieved over 1 hour of moderate to vigorous activity in a single day',
                'level': 'Silver'
            }
        elif max_active >= 30:
            badges['active_starter'] = {
                'name': 'Active Starter',
                'description': 'Achieved over 30 minutes of moderate to vigorous activity in a single day',
                'level': 'Bronze'
            }
        
        # Weekly activity badge (WHO recommends 150 min moderate or 75 min vigorous per week)
        # Approximate by checking if average daily moderate+vigorous minutes meets weekly goal
        avg_active = filtered_df['ModerateVigorousMinutes'].mean()
        if avg_active >= 30:  # ~210 minutes per week
            badges['weekly_active'] = {
                'name': 'Weekly Activity Champion',
                'description': 'Exceeds WHO recommendation for weekly physical activity',
                'level': 'Gold'
            }
        elif avg_active >= 21.4:  # ~150 minutes per week
            badges['weekly_active'] = {
                'name': 'Weekly Activity Achiever',
                'description': 'Meets WHO recommendation for weekly physical activity',
                'level': 'Silver'
            }
        elif avg_active >= 15:  # ~105 minutes per week
            badges['weekly_active'] = {
                'name': 'Weekly Activity Starter',
                'description': 'Making progress toward WHO recommendation for weekly physical activity',
                'level': 'Bronze'
            }
    
    # Calories badges
    if 'Calories' in filtered_df.columns:
        max_calories = filtered_df['Calories'].max()
        if max_calories >= 3500:
            badges['calorie_burner'] = {
                'name': 'Master Calorie Burner',
                'description': 'Burned over 3,500 calories in a single day',
                'level': 'Gold'
            }
        elif max_calories >= 3000:
            badges['calorie_burner'] = {
                'name': 'Advanced Calorie Burner',
                'description': 'Burned over 3,000 calories in a single day',
                'level': 'Silver'
            }
        elif max_calories >= 2500:
            badges['calorie_burner'] = {
                'name': 'Calorie Burner',
                'description': 'Burned over 2,500 calories in a single day',
                'level': 'Bronze'
            }
    
    # Health score badges
    if 'HealthScore' in filtered_df.columns:
        max_health_score = filtered_df['HealthScore'].max()
        if max_health_score >= 90:
            badges['health_master'] = {
                'name': 'Health Master',
                'description': 'Achieved a health score of 90 or above',
                'level': 'Gold'
            }
        elif max_health_score >= 80:
            badges['health_pro'] = {
                'name': 'Health Pro',
                'description': 'Achieved a health score of 80 or above',
                'level': 'Silver'
            }
        elif max_health_score >= 70:
            badges['health_achiever'] = {
                'name': 'Health Achiever',
                'description': 'Achieved a health score of 70 or above',
                'level': 'Bronze'
            }
    
    return badges