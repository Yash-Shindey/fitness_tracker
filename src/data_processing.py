"""
Data Processing Module for Fitness Tracker Application.

This module handles loading, cleaning, preprocessing, and merging of Fitbit data
from various CSV files.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define data directories
RAW_DATA_DIR = os.path.join('data', 'raw')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_csv_file(file_path):
    """
    Load a CSV file into a pandas DataFrame.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing the data or None if file doesn't exist
    """
    try:
        if os.path.exists(file_path):
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            return df
        else:
            logger.warning(f"File not found: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None

def process_date_columns(df, date_columns):
    """
    Convert date columns to datetime format.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    date_columns : list
        List of column names containing date information
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with processed date columns
    """
    for col in date_columns:
        if col in df.columns:
            try:
                # Try multiple formats since Fitbit data can have inconsistent formats
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    # Try with a specific format if the automatic conversion fails
                    df[col] = pd.to_datetime(df[col], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
                logger.info(f"Converted {col} to datetime")
            except Exception as e:
                logger.warning(f"Failed to convert {col} to datetime: {str(e)}")
    return df

def process_daily_activity(file_path):
    """
    Process daily activity data.
    
    Parameters:
    -----------
    file_path : str
        Path to the daily activity CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Processed daily activity data
    """
    df = load_csv_file(file_path)
    if df is None:
        return None
    
    # Convert date columns
    df = process_date_columns(df, ['ActivityDate'])
    
    # Create additional columns for easier analysis
    if 'ActivityDate' in df.columns:
        df['Date'] = df['ActivityDate'].dt.date
        df['Day'] = df['ActivityDate'].dt.day_name()
        df['Month'] = df['ActivityDate'].dt.month_name()
        df['Year'] = df['ActivityDate'].dt.year
        df['DayOfWeek'] = df['ActivityDate'].dt.dayofweek
        df['Week'] = df['ActivityDate'].dt.isocalendar().week
    
    # Calculate activity ratios
    if all(col in df.columns for col in ['LightlyActiveMinutes', 'FairlyActiveMinutes', 'VeryActiveMinutes', 'SedentaryMinutes']):
        total_active_minutes = df['LightlyActiveMinutes'] + df['FairlyActiveMinutes'] + df['VeryActiveMinutes']
        total_minutes = total_active_minutes + df['SedentaryMinutes']
        df['ActiveRatio'] = (total_active_minutes / total_minutes).round(4)
        df['IntensityRatio'] = ((df['FairlyActiveMinutes'] + 2 * df['VeryActiveMinutes']) / total_active_minutes).round(4)
        
    return df

def process_sleep_data(file_path):
    """
    Process sleep data.
    
    Parameters:
    -----------
    file_path : str
        Path to the sleep CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Processed sleep data
    """
    df = load_csv_file(file_path)
    if df is None:
        return None
    
    # Convert date columns
    df = process_date_columns(df, ['SleepDay'])
    
    # Create additional columns for easier analysis
    if 'SleepDay' in df.columns:
        df['Date'] = df['SleepDay'].dt.date
        df['Day'] = df['SleepDay'].dt.day_name()
        df['Month'] = df['SleepDay'].dt.month_name()
        df['Year'] = df['SleepDay'].dt.year
        df['DayOfWeek'] = df['SleepDay'].dt.dayofweek
        df['Week'] = df['SleepDay'].dt.isocalendar().week
    
    # Calculate sleep efficiency
    if all(col in df.columns for col in ['TotalMinutesAsleep', 'TotalTimeInBed']):
        df['SleepEfficiency'] = (df['TotalMinutesAsleep'] / df['TotalTimeInBed']).round(4)
        df['SleepEfficiencyPercentage'] = (df['SleepEfficiency'] * 100).round(2)
    
    return df

def process_heart_rate_data(file_path):
    """
    Process heart rate data.
    
    Parameters:
    -----------
    file_path : str
        Path to the heart rate CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Processed heart rate data
    """
    df = load_csv_file(file_path)
    if df is None:
        return None
    
    # Convert date columns
    df = process_date_columns(df, ['Time'])
    
    # Create additional columns for easier analysis
    if 'Time' in df.columns:
        df['Date'] = df['Time'].dt.date
        df['Hour'] = df['Time'].dt.hour
        df['Minute'] = df['Time'].dt.minute
        df['Day'] = df['Time'].dt.day_name()
        df['Month'] = df['Time'].dt.month_name()
        df['Year'] = df['Time'].dt.year
        df['DayOfWeek'] = df['Time'].dt.dayofweek
        df['Week'] = df['Time'].dt.isocalendar().week
    
    # Classify heart rate zones
    # These are general zones and can be customized based on user's max heart rate
    if 'Value' in df.columns:
        df['HeartRateZone'] = pd.cut(
            df['Value'], 
            bins=[0, 60, 70, 85, 100, 300],
            labels=['Rest', 'Fat Burn', 'Cardio', 'Peak', 'Abnormal']
        )
    
    return df

def process_weight_data(file_path):
    """
    Process weight data.
    
    Parameters:
    -----------
    file_path : str
        Path to the weight CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Processed weight data
    """
    df = load_csv_file(file_path)
    if df is None:
        return None
    
    # Convert date columns
    df = process_date_columns(df, ['Date'])
    
    # Create additional columns for easier analysis
    if 'Date' in df.columns:
        df['Day'] = df['Date'].dt.day_name()
        df['Month'] = df['Date'].dt.month_name()
        df['Year'] = df['Date'].dt.year
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Week'] = df['Date'].dt.isocalendar().week
    
    # Convert weight to standard units if necessary
    if 'WeightKg' in df.columns and 'WeightPounds' not in df.columns:
        df['WeightPounds'] = df['WeightKg'] * 2.20462
    elif 'WeightPounds' in df.columns and 'WeightKg' not in df.columns:
        df['WeightKg'] = df['WeightPounds'] / 2.20462
    
    # Calculate BMI if height data is available
    # Note: This is a placeholder. Actual implementation would need user height data.
    
    return df

def aggregate_heart_rate_daily(heart_rate_df):
    """
    Aggregate heart rate data to daily level.
    
    Parameters:
    -----------
    heart_rate_df : pandas.DataFrame
        Processed heart rate data
        
    Returns:
    --------
    pandas.DataFrame
        Daily aggregated heart rate data
    """
    if heart_rate_df is None or 'Value' not in heart_rate_df.columns:
        return None
    
    # Group by date and calculate metrics
    daily_hr = heart_rate_df.groupby('Date').agg(
        AvgHeartRate=('Value', 'mean'),
        MinHeartRate=('Value', 'min'),
        MaxHeartRate=('Value', 'max'),
        RestingHeartRate=('Value', lambda x: np.percentile(x, 5)),  # Approximate resting heart rate
        TimeInRestZone=('HeartRateZone', lambda x: (x == 'Rest').sum()),
        TimeInFatBurnZone=('HeartRateZone', lambda x: (x == 'Fat Burn').sum()),
        TimeInCardioZone=('HeartRateZone', lambda x: (x == 'Cardio').sum()),
        TimeInPeakZone=('HeartRateZone', lambda x: (x == 'Peak').sum()),
        Measurements=('Value', 'count')
    )
    
    # Round numeric columns
    for col in daily_hr.columns:
        if daily_hr[col].dtype.kind in 'if':  # if column is float or integer
            daily_hr[col] = daily_hr[col].round(1)
    
    # Reset index for easier merging
    daily_hr = daily_hr.reset_index()
    
    return daily_hr

def merge_daily_data(activity_df, sleep_df, weight_df, heart_rate_daily_df):
    """
    Merge daily activity, sleep, weight, and heart rate data.
    
    Parameters:
    -----------
    activity_df : pandas.DataFrame
        Processed daily activity data
    sleep_df : pandas.DataFrame
        Processed sleep data
    weight_df : pandas.DataFrame
        Processed weight data
    heart_rate_daily_df : pandas.DataFrame
        Daily aggregated heart rate data
        
    Returns:
    --------
    pandas.DataFrame
        Merged daily data
    """
    # Start with activity data as the base
    if activity_df is None:
        return None
    
    merged_df = activity_df.copy()
    
    # Ensure Date column is datetime type
    if 'Date' in merged_df.columns and not pd.api.types.is_datetime64_dtype(merged_df['Date']):
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    
    # Merge with sleep data
    if sleep_df is not None:
        # Ensure Date column is datetime type in sleep_df
        if 'Date' in sleep_df.columns and not pd.api.types.is_datetime64_dtype(sleep_df['Date']):
            sleep_df = sleep_df.copy()
            sleep_df['Date'] = pd.to_datetime(sleep_df['Date'])
            
        merged_df = pd.merge(
            merged_df, 
            sleep_df[['Id', 'Date', 'TotalMinutesAsleep', 'TotalTimeInBed', 'SleepEfficiency']],
            on=['Id', 'Date'],
            how='left'
        )
    
    # Merge with weight data
    if weight_df is not None:
        # Ensure Date column is datetime type in weight_df
        if 'Date' in weight_df.columns and not pd.api.types.is_datetime64_dtype(weight_df['Date']):
            weight_df = weight_df.copy()
            weight_df['Date'] = pd.to_datetime(weight_df['Date'])
            
        merged_df = pd.merge(
            merged_df, 
            weight_df[['Id', 'Date', 'WeightKg', 'BMI']],
            on=['Id', 'Date'],
            how='left'
        )
    
    # Merge with heart rate data
    if heart_rate_daily_df is not None:
        # Ensure Date column is datetime type in heart_rate_daily_df
        if 'Date' in heart_rate_daily_df.columns and not pd.api.types.is_datetime64_dtype(heart_rate_daily_df['Date']):
            heart_rate_daily_df = heart_rate_daily_df.copy()
            heart_rate_daily_df['Date'] = pd.to_datetime(heart_rate_daily_df['Date'])
            
        merged_df = pd.merge(
            merged_df, 
            heart_rate_daily_df,
            on=['Date'],
            how='left'
        )
    
    # Fill NAs with appropriate values
    # For numeric columns, fill with 0
    numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
    
    return merged_df
        
def calculate_health_score(merged_df):
    """
    Calculate an overall health score based on multiple metrics.
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame
        Merged daily data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added health score column
    """
    if merged_df is None:
        return None
    
    # Define weights for different factors
    weights = {
        'steps': 0.3,
        'activity': 0.25,
        'sleep': 0.25,
        'heart_rate': 0.2
    }
    
    # Create individual component scores (0-100 scale)
    # Steps score: 10000 steps = 100 points
    if 'Steps' in merged_df.columns:
        merged_df['StepsScore'] = (merged_df['Steps'] / 10000 * 100).clip(0, 100)
    else:
        merged_df['StepsScore'] = 0
    
    # Activity score: Based on active minutes
    if all(col in merged_df.columns for col in ['VeryActiveMinutes', 'FairlyActiveMinutes']):
        # WHO recommends 150 minutes of moderate activity or 75 minutes of vigorous activity per week
        # Using a daily target of ~30 min moderate or ~15 min vigorous
        merged_df['ActivityScore'] = (
            (merged_df['FairlyActiveMinutes'] + 2 * merged_df['VeryActiveMinutes']) / 30 * 100
        ).clip(0, 100)
    else:
        merged_df['ActivityScore'] = 0
    
    # Sleep score: Based on sleep duration and efficiency
    if all(col in merged_df.columns for col in ['TotalMinutesAsleep', 'SleepEfficiency']):
        # Ideal sleep is 7-9 hours (420-540 minutes) with high efficiency
        sleep_duration_score = (
            (merged_df['TotalMinutesAsleep'] - 360).clip(0, 180) / 180 * 100
        ).clip(0, 100)
        
        sleep_efficiency_score = (merged_df['SleepEfficiency'] * 100).clip(0, 100)
        
        merged_df['SleepScore'] = (0.7 * sleep_duration_score + 0.3 * sleep_efficiency_score)
    else:
        merged_df['SleepScore'] = 0
    
    # Heart rate score: Based on resting heart rate and time in zones
    if 'AvgHeartRate' in merged_df.columns:
        # Lower resting heart rate is generally better (within normal range)
        # A resting HR between 40-60 is excellent, 60-70 is good
        # Using RestingHeartRate if available, otherwise using MinHeartRate as a proxy
        hr_col = 'RestingHeartRate' if 'RestingHeartRate' in merged_df.columns else 'MinHeartRate'
        
        if hr_col in merged_df.columns:
            # Score decreases as resting heart rate increases (within normal range)
            normal_hr_mask = merged_df[hr_col].between(40, 100)
            merged_df.loc[normal_hr_mask, 'HeartRateScore'] = (
                100 - ((merged_df.loc[normal_hr_mask, hr_col] - 40) / 60 * 100)
            ).clip(0, 100)
            
            # Abnormal heart rates get a lower score
            merged_df.loc[~normal_hr_mask, 'HeartRateScore'] = 40
        else:
            merged_df['HeartRateScore'] = 50  # Neutral score if no data
    else:
        merged_df['HeartRateScore'] = 0
    
    # Calculate weighted health score
    merged_df['HealthScore'] = (
        weights['steps'] * merged_df['StepsScore'] +
        weights['activity'] * merged_df['ActivityScore'] +
        weights['sleep'] * merged_df['SleepScore'] +
        weights['heart_rate'] * merged_df['HeartRateScore']
    ).round(1)
    
    return merged_df

def load_and_process_all_data(base_folder_path=None):
    """
    Load and process all Fitbit data files.
    
    Parameters:
    -----------
    base_folder_path : str or None
        Base folder containing Fitbit data folders
        If None, will use the default RAW_DATA_DIR
        
    Returns:
    --------
    dict
        Dictionary containing all processed dataframes
    """
    if base_folder_path is None:
        base_folder_path = RAW_DATA_DIR
    
    # Find all data folders
    data_folders = []
    for root, dirs, files in os.walk(base_folder_path):
        # Check if the folder contains Fitbit data files
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            data_folders.append(root)
    
    logger.info(f"Found {len(data_folders)} data folders")
    
    # Initialize result dictionary
    result = {
        'daily_activity': [],
        'sleep': [],
        'heart_rate': [],
        'weight': [],
        'hourly_steps': [],
        'hourly_calories': [],
        'hourly_intensities': []
    }
    
    # Process each folder
    for folder in data_folders:
        logger.info(f"Processing folder: {folder}")
        
        # Process daily activity data
        daily_activity_file = os.path.join(folder, 'dailyActivity_merged.csv')
        daily_activity_df = process_daily_activity(daily_activity_file)
        if daily_activity_df is not None:
            result['daily_activity'].append(daily_activity_df)
        
        # Process sleep data
        sleep_file = os.path.join(folder, 'sleepDay_merged.csv')
        if not os.path.exists(sleep_file):
            sleep_file = os.path.join(folder, 'minuteSleep_merged.csv')
        sleep_df = process_sleep_data(sleep_file)
        if sleep_df is not None:
            result['sleep'].append(sleep_df)
        
        # Process heart rate data
        heart_rate_file = os.path.join(folder, 'heartrate_seconds_merged.csv')
        heart_rate_df = process_heart_rate_data(heart_rate_file)
        if heart_rate_df is not None:
            result['heart_rate'].append(heart_rate_df)
        
        # Process weight data
        weight_file = os.path.join(folder, 'weightLogInfo_merged.csv')
        weight_df = process_weight_data(weight_file)
        if weight_df is not None:
            result['weight'].append(weight_df)
        
        # Process hourly data
        hourly_steps_file = os.path.join(folder, 'hourlySteps_merged.csv')
        hourly_steps_df = load_csv_file(hourly_steps_file)
        if hourly_steps_df is not None:
            result['hourly_steps'].append(hourly_steps_df)
        
        hourly_calories_file = os.path.join(folder, 'hourlyCalories_merged.csv')
        hourly_calories_df = load_csv_file(hourly_calories_file)
        if hourly_calories_df is not None:
            result['hourly_calories'].append(hourly_calories_df)
        
        hourly_intensities_file = os.path.join(folder, 'hourlyIntensities_merged.csv')
        hourly_intensities_df = load_csv_file(hourly_intensities_file)
        if hourly_intensities_df is not None:
            result['hourly_intensities'].append(hourly_intensities_df)
    
    # Combine data from different folders
    for key in result:
        if result[key]:
            result[key] = pd.concat(result[key], ignore_index=True)
        else:
            result[key] = None
    
    # Perform additional processing and merging
    if result['heart_rate'] is not None:
        result['heart_rate_daily'] = aggregate_heart_rate_daily(result['heart_rate'])
    else:
        result['heart_rate_daily'] = None
    
    # Merge daily data
    result['merged_daily'] = merge_daily_data(
        result['daily_activity'],
        result['sleep'],
        result['weight'],
        result['heart_rate_daily']
    )
    
    # Calculate health score
    if result['merged_daily'] is not None:
        result['merged_daily'] = calculate_health_score(result['merged_daily'])
    
    return result

def get_all_users(data_dict):
    """
    Get a list of all unique user IDs from the data.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed dataframes
        
    Returns:
    --------
    list
        List of unique user IDs
    """
    user_ids = set()
    
    # Collect user IDs from all dataframes that contain 'Id' column
    for key, df in data_dict.items():
        if df is not None and 'Id' in df.columns:
            user_ids.update(df['Id'].unique())
    
    return sorted(list(user_ids))

def save_processed_data(data_dict, output_dir=None):
    """
    Save processed data to CSV files.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing all processed dataframes
    output_dir : str or None
        Directory to save processed files
        If None, will use the default PROCESSED_DATA_DIR
        
    Returns:
    --------
    None
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each dataframe
    for key, df in data_dict.items():
        if df is not None:
            output_file = os.path.join(output_dir, f"{key}.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {key} to {output_file}")

if __name__ == "__main__":
    # Example usage
    print("Processing Fitbit data...")
    processed_data = load_and_process_all_data()
    save_processed_data(processed_data)
    print("Data processing complete.")