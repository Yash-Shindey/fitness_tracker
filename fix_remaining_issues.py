
import pandas as pd
import os

# 1. Fix the plot_activity_heatmap function
def fix_activity_heatmap():
    with open('src/visualization.py', 'r') as file:
        content = file.read()
    
    # Replace the problematic code
    old_code = '''
    # Filter data if date_range is provided
    if date_range:
        start_date, end_date = date_range
        # Ensure start_date and end_date are pandas datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Ensure ActivityHour is datetime
        if 'ActivityHour' in filtered_df.columns:
            filtered_df = filtered_df.copy()
            filtered_df['ActivityHour'] = pd.to_datetime(filtered_df['ActivityHour'], errors='coerce')
            
            # Now filter by date
            filtered_df = filtered_df[filtered_df['ActivityHour'].notna()]  # Remove NaT values
            filtered_df['ActivityDate'] = filtered_df['ActivityHour'].dt.date  # Extract date
            filtered_df = filtered_df[(filtered_df['ActivityDate'] >= start_date.date()) & 
                                  (filtered_df['ActivityDate'] <= end_date.date())]'''
    
    new_code = '''
    # Filter data if date_range is provided
    if date_range:
        start_date, end_date = date_range
        
        # Convert start_date and end_date to pandas timestamp
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Ensure ActivityHour is datetime
        if 'ActivityHour' in filtered_df.columns:
            filtered_df = filtered_df.copy()
            filtered_df['ActivityHour'] = pd.to_datetime(filtered_df['ActivityHour'], errors='coerce')
            
            # Now filter by date - FIXED COMPARISON
            filtered_df = filtered_df[filtered_df['ActivityHour'].notna()]  # Remove NaT values
            # Use string date format for comparison to avoid type issues
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            filtered_df = filtered_df[(filtered_df['ActivityHour'].dt.strftime('%Y-%m-%d') >= start_date_str) & 
                                  (filtered_df['ActivityHour'].dt.strftime('%Y-%m-%d') <= end_date_str)]'''
    
    content = content.replace(old_code, new_code)
    
    # Fix the hour and day extraction
    old_code = '''
    # Ensure ActivityHour is datetime
    filtered_df = filtered_df.copy()
    filtered_df['ActivityHour'] = pd.to_datetime(filtered_df['ActivityHour'], errors='coerce')
    
    # Extract hour and day of week (for non-NaT values)
    valid_mask = ~filtered_df['ActivityHour'].isna()
    if valid_mask.any():
        filtered_df.loc[valid_mask, 'Hour'] = filtered_df.loc[valid_mask, 'ActivityHour'].dt.hour
        filtered_df.loc[valid_mask, 'DayOfWeek'] = filtered_df.loc[valid_mask, 'ActivityHour'].dt.dayofweek
        filtered_df.loc[valid_mask, 'DayName'] = filtered_df.loc[valid_mask, 'ActivityHour'].dt.day_name()
    else:
        filtered_df['Hour'] = 0
        filtered_df['DayOfWeek'] = 0
        filtered_df['DayName'] = "Unknown" '''
    
    new_code = '''
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
        filtered_df.loc[valid_mask, 'DayName'] = filtered_df.loc[valid_mask, 'ActivityHour'].dt.day_name()'''
    
    content = content.replace(old_code, new_code)
    
    with open('src/visualization.py', 'w') as file:
        file.write(content)
    
    print("Fixed plot_activity_heatmap function")

# 2. Fix Sleep Insights page
def fix_sleep_insights():
    with open('app.py', 'r') as file:
        content = file.read()
    
    # Replace the problematic timestamp division code
    old_code = '''
    # Calculate summary statistics
    if isinstance(avg_sleep_min, pd.Timestamp):
        avg_sleep_min = 0
    avg_sleep_hours = avg_sleep_min / 60 if avg_sleep_min else 0
    
    avg_time_in_bed_min = filtered_sleep['TotalTimeInBed'].mean()
    # Ensure avg_time_in_bed_min is a numeric value
    if isinstance(avg_time_in_bed_min, pd.Timestamp):
        avg_time_in_bed_min = 0
    avg_time_in_bed_hours = avg_time_in_bed_min / 60 if avg_time_in_bed_min else 0'''
    
    new_code = '''
    # Calculate summary statistics - ensure all values are numeric
    avg_sleep_min = filtered_sleep['TotalMinutesAsleep'].mean()
    if avg_sleep_min is None or pd.isna(avg_sleep_min) or isinstance(avg_sleep_min, pd.Timestamp):
        avg_sleep_min = 0
    avg_sleep_hours = float(avg_sleep_min) / 60
    
    avg_time_in_bed_min = filtered_sleep['TotalTimeInBed'].mean()
    if avg_time_in_bed_min is None or pd.isna(avg_time_in_bed_min) or isinstance(avg_time_in_bed_min, pd.Timestamp):
        avg_time_in_bed_min = 0
    avg_time_in_bed_hours = float(avg_time_in_bed_min) / 60'''
    
    content = content.replace(old_code, new_code)
    
    # Make sure we have a fallback even if the old code is missing
    if old_code not in content:
        old_code = '''
    # Calculate summary statistics
    avg_sleep_min = filtered_sleep['TotalMinutesAsleep'].mean()
    avg_sleep_hours = avg_sleep_min / 60
    avg_time_in_bed_min = filtered_sleep['TotalTimeInBed'].mean()
    avg_time_in_bed_hours = avg_time_in_bed_min / 60'''
        
        content = content.replace(old_code, new_code)
    
    with open('app.py', 'w') as file:
        file.write(content)
    
    print("Fixed sleep insights page")

# 3. Fix Heart Rate Anomaly Detection
def fix_heart_rate_anomalies():
    with open('src/analysis.py', 'r') as file:
        content = file.read()
    
    # Fix the anomaly detection function
    old_code = '''
        # Apply Isolation Forest
        model = IsolationForest(contamination=contamination, random_state=42)
        X['anomaly'] = model.fit_predict(X)
        X['anomaly_score'] = model.decision_function(X)'''
    
    new_code = '''
        # Apply Isolation Forest
        model = IsolationForest(contamination=contamination, random_state=42)
        # Save the feature columns for fitting and prediction
        feature_cols = X.columns.tolist()
        # Fit the model
        model.fit(X[feature_cols])
        # Predict and add results as new columns
        X_copy = X.copy()
        X_copy['anomaly'] = model.predict(X[feature_cols])
        X_copy['anomaly_score'] = model.decision_function(X[feature_cols])
        X = X_copy'''
    
    content = content.replace(old_code, new_code)
    
    with open('src/analysis.py', 'w') as file:
        file.write(content)
    
    print("Fixed heart rate anomaly detection")

# Run all fixes
if __name__ == "__main__":
    print("Applying targeted fixes for remaining issues...")
    fix_activity_heatmap()
    fix_sleep_insights()
    fix_heart_rate_anomalies()
    print("All targeted fixes applied! Try running the app again.")
