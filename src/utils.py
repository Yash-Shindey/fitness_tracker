"""
Utility functions for Fitness Tracker Application.

This module provides helper functions and utilities used across the application.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def date_to_str(date):
    """
    Convert a date object to string format.
    
    Parameters:
    -----------
    date : datetime.date
        Date to convert
        
    Returns:
    --------
    str
        Formatted date string (YYYY-MM-DD)
    """
    if isinstance(date, pd.Timestamp):
        return date.strftime('%Y-%m-%d')
    elif isinstance(date, datetime) or hasattr(date, 'strftime'):
        return date.strftime('%Y-%m-%d')
    else:
        return str(date)

def str_to_date(date_str):
    """
    Convert a string to date object.
    
    Parameters:
    -----------
    date_str : str
        Date string to convert
        
    Returns:
    --------
    datetime.date
        Date object
    """
    try:
        return pd.to_datetime(date_str).date()
    except:
        return None

def get_date_range_options():
    """
    Get predefined date range options for the application.
    
    Returns:
    --------
    dict
        Dictionary of date range options with labels and date tuples
    """
    today = datetime.now().date()
    
    return {
        'last_7_days': {
            'label': 'Last 7 Days',
            'start_date': today - timedelta(days=7),
            'end_date': today
        },
        'last_30_days': {
            'label': 'Last 30 Days',
            'start_date': today - timedelta(days=30),
            'end_date': today
        },
        'last_90_days': {
            'label': 'Last 90 Days',
            'start_date': today - timedelta(days=90),
            'end_date': today
        },
        'this_week': {
            'label': 'This Week',
            'start_date': today - timedelta(days=today.weekday()),
            'end_date': today
        },
        'this_month': {
            'label': 'This Month',
            'start_date': today.replace(day=1),
            'end_date': today
        },
        'all_time': {
            'label': 'All Time',
            'start_date': today - timedelta(days=365*10),  # 10 years ago
            'end_date': today
        }
    }

def create_comparison_dataframe(daily_df, metric, date_ranges):
    """
    Create a dataframe for comparing a metric across different date ranges.
    
    Parameters:
    -----------
    daily_df : pandas.DataFrame
        DataFrame containing daily data
    metric : str
        Column name of the metric to compare
    date_ranges : list of tuple
        List of (label, start_date, end_date) tuples
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with comparison data
    """
    if daily_df is None or metric not in daily_df.columns:
        return None
    
    results = []
    
    for label, start_date, end_date in date_ranges:
        # Filter data for this date range
        mask = (daily_df['Date'] >= start_date) & (daily_df['Date'] <= end_date)
        period_df = daily_df[mask]
        
        # Skip if no data for this period
        if period_df.empty:
            continue
        
        # Calculate statistics
        stats = {
            'period': label,
            'mean': period_df[metric].mean(),
            'median': period_df[metric].median(),
            'min': period_df[metric].min(),
            'max': period_df[metric].max(),
            'std': period_df[metric].std(),
            'total': period_df[metric].sum(),
            'count': len(period_df)
        }
        
        results.append(stats)
    
    # Create dataframe from results
    if results:
        comparison_df = pd.DataFrame(results)
        return comparison_df
    else:
        return None

def format_duration(minutes):
    """
    Format a duration in minutes to a readable string (HH:MM).
    
    Parameters:
    -----------
    minutes : float
        Duration in minutes
        
    Returns:
    --------
    str
        Formatted duration string
    """
    if pd.isna(minutes) or minutes < 0:
        return "00:00"
    
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    
    return f"{hours:02d}:{mins:02d}"

def export_data_to_csv(df, filename, output_dir="exports"):
    """
    Export a dataframe to a CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to export
    filename : str
        Output filename
    output_dir : str
        Output directory
        
    Returns:
    --------
    str
        Path to the exported file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Full output path
    output_path = os.path.join(output_dir, filename)
    
    # Export the dataframe
    df.to_csv(output_path, index=False)
    logger.info(f"Exported data to {output_path}")
    
    return output_path

def calculate_progress_percentage(current_value, target_value):
    """
    Calculate progress percentage towards a target.
    
    Parameters:
    -----------
    current_value : float
        Current value
    target_value : float
        Target value
        
    Returns:
    --------
    float
        Progress percentage (0-100)
    """
    if target_value <= 0 or pd.isna(current_value) or pd.isna(target_value):
        return 0
    
    progress = (current_value / target_value) * 100
    return min(100, max(0, progress))  # Clamp between 0 and 100

def generate_daily_insights(daily_df, user_id, date):
    """
    Generate insights for a specific day.
    
    Parameters:
    -----------
    daily_df : pandas.DataFrame
        DataFrame containing daily data
    user_id : str
        User ID to filter data for
    date : datetime.date
        Specific date to analyze
        
    Returns:
    --------
    dict
        Dictionary with insights
    """
    if daily_df is None:
        return None
    
    # Filter data for the specific user and date
    mask = (daily_df['Id'] == user_id) & (daily_df['Date'] == date)
    day_data = daily_df[mask]
    
    # Check if we have data for this day
    if day_data.empty:
        return None
    
    # Get the row for this day
    row = day_data.iloc[0]
    
    # Initialize insights
    insights = {
        'user_id': user_id,
        'date': date,
        'day_of_week': row['Day'] if 'Day' in row else None,
        'summary': [],
        'achievements': [],
        'suggestions': []
    }
    
    # Steps insights
    if 'Steps' in row:
        steps = row['Steps']
        if steps >= 10000:
            insights['summary'].append(f"Great job! You took {steps:,} steps, exceeding the 10,000 steps goal.")
            insights['achievements'].append("Exceeded daily step goal")
        elif steps >= 7500:
            insights['summary'].append(f"Good progress! You took {steps:,} steps, approaching the 10,000 steps goal.")
            insights['suggestions'].append("Take a short walk to reach 10,000 steps")
        else:
            insights['summary'].append(f"You took {steps:,} steps. The recommended goal is 10,000 steps per day.")
            insights['suggestions'].append("Try to incorporate more walking into your daily routine")
    
    # Active minutes insights
    active_cols = ['VeryActiveMinutes', 'FairlyActiveMinutes']
    if all(col in row for col in active_cols):
        moderate_vigorous = row['VeryActiveMinutes'] + row['FairlyActiveMinutes']
        if moderate_vigorous >= 30:
            insights['summary'].append(f"Excellent! You had {moderate_vigorous} minutes of moderate to vigorous activity.")
            insights['achievements'].append("Met daily active minutes recommendation")
        else:
            insights['summary'].append(f"You had {moderate_vigorous} minutes of moderate to vigorous activity. Aim for at least 30 minutes daily.")
            insights['suggestions'].append("Increase moderate to vigorous activity to at least 30 minutes")
    
    # Sleep insights
    if 'TotalMinutesAsleep' in row and not pd.isna(row['TotalMinutesAsleep']):
        sleep_hours = row['TotalMinutesAsleep'] / 60
        if sleep_hours >= 7 and sleep_hours <= 9:
            insights['summary'].append(f"You slept for {sleep_hours:.1f} hours, which is within the recommended 7-9 hour range.")
            insights['achievements'].append("Optimal sleep duration")
        elif sleep_hours < 7:
            insights['summary'].append(f"You slept for {sleep_hours:.1f} hours, which is below the recommended 7-9 hour range.")
            insights['suggestions'].append("Aim for at least 7 hours of sleep for optimal health")
        else:
            insights['summary'].append(f"You slept for {sleep_hours:.1f} hours, which is above the recommended 7-9 hour range.")
    
    # Sleep efficiency insights
    if 'SleepEfficiency' in row and not pd.isna(row['SleepEfficiency']):
        efficiency = row['SleepEfficiency'] * 100
        if efficiency >= 85:
            insights['summary'].append(f"Your sleep efficiency was {efficiency:.1f}%, which is very good.")
            insights['achievements'].append("High sleep efficiency")
        elif efficiency >= 75:
            insights['summary'].append(f"Your sleep efficiency was {efficiency:.1f}%, which is acceptable.")
        else:
            insights['summary'].append(f"Your sleep efficiency was {efficiency:.1f}%, which could be improved.")
            insights['suggestions'].append("Improve sleep environment and bedtime routine for better sleep efficiency")
    
    # Heart rate insights
    hr_cols = ['AvgHeartRate', 'MinHeartRate', 'MaxHeartRate']
    if 'AvgHeartRate' in row and not pd.isna(row['AvgHeartRate']):
        avg_hr = row['AvgHeartRate']
        insights['summary'].append(f"Your average heart rate was {avg_hr:.0f} bpm.")
        
        if 'MinHeartRate' in row and not pd.isna(row['MinHeartRate']):
            min_hr = row['MinHeartRate']
            if min_hr < 60:
                insights['summary'].append(f"Your resting heart rate was {min_hr:.0f} bpm, indicating good cardiovascular fitness.")
                insights['achievements'].append("Low resting heart rate")
    
    # Calories insights
    if 'Calories' in row:
        calories = row['Calories']
        insights['summary'].append(f"You burned approximately {calories:,} calories.")
        
        # Compare to average
        avg_calories = daily_df[daily_df['Id'] == user_id]['Calories'].mean()
        if calories > avg_calories * 1.2:
            insights['summary'].append(f"You burned 20% more calories than your average ({avg_calories:.0f}).")
            insights['achievements'].append("Above average calorie burn")
    
    # Health score insights
    if 'HealthScore' in row:
        health_score = row['HealthScore']
        if health_score >= 80:
            insights['summary'].append(f"Your health score was {health_score:.1f}, which is excellent!")
            insights['achievements'].append("Excellent health score")
        elif health_score >= 70:
            insights['summary'].append(f"Your health score was {health_score:.1f}, which is good.")
        elif health_score >= 60:
            insights['summary'].append(f"Your health score was {health_score:.1f}, which is fair.")
            insights['suggestions'].append("Focus on increasing activity and improving sleep for a higher health score")
        else:
            insights['summary'].append(f"Your health score was {health_score:.1f}, which has room for improvement.")
            insights['suggestions'].append("Work on improving all health metrics: steps, activity, sleep, and heart rate")
    
    return insights

def get_default_goals():
    """
    Get default health and fitness goals.
    
    Returns:
    --------
    dict
        Dictionary with default goals
    """
    return {
        'steps': 10000,
        'active_minutes': 30,
        'sleep_hours': 8,
        'sleep_efficiency': 0.85,
        'calories': 2500,
        'health_score': 75
    }

def save_user_goals(user_id, goals, output_dir="user_data"):
    """
    Save user goals to a JSON file.
    
    Parameters:
    -----------
    user_id : str
        User ID
    goals : dict
        Dictionary with user's goals
    output_dir : str
        Output directory
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output filename
    filename = f"user_{user_id}_goals.json"
    output_path = os.path.join(output_dir, filename)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(goals, f)
        logger.info(f"Saved user goals to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving user goals: {str(e)}")
        return False

def load_user_goals(user_id, input_dir="user_data"):
    """
    Load user goals from a JSON file.
    
    Parameters:
    -----------
    user_id : str
        User ID
    input_dir : str
        Input directory
        
    Returns:
    --------
    dict
        Dictionary with user's goals, or default goals if file doesn't exist
    """
    # Input filename
    filename = f"user_{user_id}_goals.json"
    input_path = os.path.join(input_dir, filename)
    
    # Check if file exists
    if not os.path.exists(input_path):
        logger.info(f"User goals file not found. Using default goals.")
        return get_default_goals()
    
    try:
        with open(input_path, 'r') as f:
            goals = json.load(f)
        logger.info(f"Loaded user goals from {input_path}")
        return goals
    except Exception as e:
        logger.error(f"Error loading user goals: {str(e)}")
        return get_default_goals()

def format_number(value, precision=0):
    """
    Format a number with thousands separator and specified precision.
    
    Parameters:
    -----------
    value : float or int
        Number to format
    precision : int
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted number
    """
    if pd.isna(value):
        return "N/A"
    
    try:
        if precision == 0:
            return f"{int(value):,}"
        else:
            return f"{value:,.{precision}f}"
    except:
        return str(value)

def calculate_bmi(weight_kg, height_cm):
    """
    Calculate BMI from weight and height.
    
    Parameters:
    -----------
    weight_kg : float
        Weight in kilograms
    height_cm : float
        Height in centimeters
        
    Returns:
    --------
    float
        BMI value
    """
    if pd.isna(weight_kg) or pd.isna(height_cm) or height_cm <= 0:
        return None
    
    # Convert height to meters
    height_m = height_cm / 100
    
    # Calculate BMI
    bmi = weight_kg / (height_m * height_m)
    
    return round(bmi, 1)

def get_bmi_category(bmi):
    """
    Get BMI category based on BMI value.
    
    Parameters:
    -----------
    bmi : float
        BMI value
        
    Returns:
    --------
    str
        BMI category
    """
    if pd.isna(bmi):
        return "Unknown"
    
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def calculate_tdee(weight_kg, height_cm, age, gender, activity_level):
    """
    Calculate Total Daily Energy Expenditure (TDEE).
    
    Parameters:
    -----------
    weight_kg : float
        Weight in kilograms
    height_cm : float
        Height in centimeters
    age : int
        Age in years
    gender : str
        'male' or 'female'
    activity_level : str
        Activity level: 'sedentary', 'lightly_active', 'moderately_active', 'very_active', or 'extra_active'
        
    Returns:
    --------
    float
        TDEE in calories
    """
    if (pd.isna(weight_kg) or pd.isna(height_cm) or 
        pd.isna(age) or gender not in ['male', 'female']):
        return None
    
    # Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor Equation
    if gender == 'male':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:  # female
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    
    # Activity multipliers
    activity_multipliers = {
        'sedentary': 1.2,         # Little or no exercise
        'lightly_active': 1.375,   # Light exercise/sports 1-3 days/week
        'moderately_active': 1.55, # Moderate exercise/sports 3-5 days/week
        'very_active': 1.725,      # Hard exercise/sports 6-7 days/week
        'extra_active': 1.9        # Very hard exercise, physical job or training twice a day
    }
    
    # Calculate TDEE
    multiplier = activity_multipliers.get(activity_level, 1.2)
    tdee = bmr * multiplier
    
    return round(tdee)