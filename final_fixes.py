
import os
import shutil
import pandas as pd

# Function to backup the current files
def backup_files():
    if not os.path.exists("backup"):
        os.makedirs("backup")
    
    # Backup app.py
    if os.path.exists("app.py"):
        shutil.copy("app.py", "backup/app.py.bak")
    
    # Backup visualization.py
    if os.path.exists("src/visualization.py"):
        shutil.copy("src/visualization.py", "backup/visualization.py.bak")
    
    print("Files backed up to backup directory")

# Fix column name mappings
def fix_column_mappings_in_app():
    with open('app.py', 'r') as file:
        content = file.read()
    
    # Fix 'Steps' to 'TotalSteps' references
    content = content.replace("filtered_daily['Steps']", "filtered_daily['TotalSteps']")
    content = content.replace("['Steps']", "['TotalSteps']")
    content = content.replace("'Steps'", "'TotalSteps'")
    
    # Fix timestamp division errors in sleep_insights_page
    old_code = """
    # Calculate summary statistics
    avg_sleep_min = filtered_sleep['TotalMinutesAsleep'].mean()
    avg_sleep_hours = avg_sleep_min / 60
    avg_time_in_bed_min = filtered_sleep['TotalTimeInBed'].mean()
    avg_time_in_bed_hours = avg_time_in_bed_min / 60"""
    
    new_code = """
    # Calculate summary statistics
    avg_sleep_min = filtered_sleep['TotalMinutesAsleep'].mean()
    # Ensure avg_sleep_min is a numeric value
    if isinstance(avg_sleep_min, pd.Timestamp):
        avg_sleep_min = 0
    avg_sleep_hours = avg_sleep_min / 60 if avg_sleep_min else 0
    
    avg_time_in_bed_min = filtered_sleep['TotalTimeInBed'].mean()
    # Ensure avg_time_in_bed_min is a numeric value
    if isinstance(avg_time_in_bed_min, pd.Timestamp):
        avg_time_in_bed_min = 0
    avg_time_in_bed_hours = avg_time_in_bed_min / 60 if avg_time_in_bed_min else 0"""
    
    content = content.replace(old_code, new_code)
    
    # Fix heart_rate_analysis_page timestamp division errors
    old_code = """
                # Convert to minutes
                for col in zone_columns:
                    filtered_hr_daily[f"{col}Min"] = filtered_hr_daily[col] / 60"""
    
    new_code = """
                # Convert to minutes
                for col in zone_columns:
                    # Ensure the column contains numeric values
                    if col in filtered_hr_daily.columns:
                        filtered_hr_daily = filtered_hr_daily.copy()
                        # Convert any non-numeric values to NaN and then to 0
                        filtered_hr_daily[col] = pd.to_numeric(filtered_hr_daily[col], errors='coerce').fillna(0)
                        filtered_hr_daily[f"{col}Min"] = filtered_hr_daily[col] / 60"""
    
    content = content.replace(old_code, new_code)
    
    # Fix weight_management_page for Steps column
    old_code = """
            # Filter activity data
            filtered_activity = daily_activity_df[
                (daily_activity_df['Id'] == user_id) & 
                (daily_activity_df['Date'] >= pd.Timestamp(start_date)) & 
                (daily_activity_df['Date'] <= pd.Timestamp(end_date))
            ]"""
    
    new_code = """
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
                        break"""
    
    content = content.replace(old_code, new_code)
    
    # Fix accessing filtered_activity DataFrame
    old_code = """
                # Merge weight and activity data on the nearest date
                # First, sort both dataframes by date
                filtered_weight = filtered_weight.sort_values('Date')
                filtered_activity = filtered_activity.sort_values('Date')
                
                # Create a merged dataframe for days with both weight and activity data
                merged_dates = pd.merge_asof(
                    filtered_weight[['Date', 'WeightKg', 'BMI']].sort_values('Date'),
                    filtered_activity[['Date', 'Steps', 'Calories', 'VeryActiveMinutes', 'FairlyActiveMinutes']].sort_values('Date'),
                    on='Date',
                    direction='nearest'
                )"""
    
    new_code = """
                # Merge weight and activity data on the nearest date
                # First, sort both dataframes by date
                filtered_weight = filtered_weight.sort_values('Date')
                filtered_activity = filtered_activity.sort_values('Date')
                
                # Create a merged dataframe for days with both weight and activity data
                try:
                    # Use only available columns for merging
                    activity_cols = [col for col in available_cols if col in filtered_activity.columns]
                    if len(activity_cols) > 1:  # Make sure we have at least Date and one more column
                        merged_dates = pd.merge_asof(
                            filtered_weight[['Date', 'WeightKg', 'BMI']].sort_values('Date'),
                            filtered_activity[activity_cols].sort_values('Date'),
                            on='Date',
                            direction='nearest'
                        )
                    else:
                        merged_dates = pd.DataFrame()
                except Exception as e:
                    st.error(f"Error merging data: {str(e)}")
                    merged_dates = pd.DataFrame()"""
    
    content = content.replace(old_code, new_code)
    
    with open('app.py', 'w') as file:
        file.write(content)
    
    print("Fixed column mappings and data type issues in app.py")

# Fix visualization.py to handle datetime and column issues
def fix_visualization_file():
    with open('src/visualization.py', 'r') as file:
        content = file.read()
    
    # Fix ActivityHour dt.date issue
    old_code = """
    # Filter data if date_range is provided
    if date_range:
        start_date, end_date = date_range
        # Ensure start_date and end_date are pandas datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_df = filtered_df[(filtered_df['ActivityHour'].dt.date >= pd.to_datetime(start_date)) & 
                              (filtered_df['ActivityHour'].dt.date <= pd.to_datetime(end_date))]"""
    
    new_code = """
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
                                  (filtered_df['ActivityDate'] <= end_date.date())]"""
    
    content = content.replace(old_code, new_code)
    
    # Fix plot_daily_steps_trend to use TotalSteps
    content = content.replace("if df is None or 'Steps' not in df.columns:", 
                             "if df is None or ('Steps' not in df.columns and 'TotalSteps' not in df.columns):")
    
    content = content.replace("df['Steps']", "df['TotalSteps'] if 'TotalSteps' in df.columns else df['Steps']")
    
    # Fix all date comparisons
    content = content.replace("filtered_df['Date'] >= ", 
                             "pd.to_datetime(filtered_df['Date']).dt.date >= pd.to_datetime(")
    
    content = content.replace("filtered_df['Date'] <= ", 
                             "pd.to_datetime(filtered_df['Date']).dt.date <= pd.to_datetime(")
    
    # Add date conversions for plot_activity_heatmap
    old_code = """
    # Extract hour and day of week
    filtered_df['Hour'] = filtered_df['ActivityHour'].dt.hour
    filtered_df['DayOfWeek'] = filtered_df['ActivityHour'].dt.dayofweek
    filtered_df['DayName'] = filtered_df['ActivityHour'].dt.day_name()"""
    
    new_code = """
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
        filtered_df['DayName'] = "Unknown" """
    
    content = content.replace(old_code, new_code)
    
    with open('src/visualization.py', 'w') as file:
        file.write(content)
    
    print("Fixed visualization.py for datetime and column issues")

# Fix activity_analysis_page function
def fix_activity_analysis_page():
    with open('app.py', 'r') as file:
        content = file.read()
    
    # Fix hourly data filtering
    old_code = """
            # Filter hourly data
            filtered_hourly = hourly_steps_df[
                (hourly_steps_df['Id'] == user_id) & 
                (hourly_steps_df['ActivityHour'].dt.date >= start_date) & 
                (hourly_steps_df['ActivityHour'].dt.date <= end_date)
            ]"""
    
    new_code = """
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
                filtered_hourly = pd.DataFrame()"""
    
    content = content.replace(old_code, new_code)
    
    with open('app.py', 'w') as file:
        file.write(content)
    
    print("Fixed activity_analysis_page function")

# Execute all fixes
if __name__ == "__main__":
    print("Applying final fixes to the Fitness Tracker app...")
    backup_files()
    fix_column_mappings_in_app()
    fix_visualization_file()
    fix_activity_analysis_page()
    print("All fixes applied! Try running the app again.")
        