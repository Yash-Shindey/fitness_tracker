import os
import pandas as pd

# Define constants
RAW_DATA_DIR = os.path.join('data', 'raw', 'fitbit')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# First, modify the date_range_selector function in app.py
def modify_date_range_selector():
    with open('app.py', 'r') as file:
        content = file.read()
    
    # Find the return statement in date_range_selector
    old_code = "    return start_date, end_date"
    new_code = """    # Convert to pandas datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    return start_date, end_date"""
    
    # Replace the code
    updated_content = content.replace(old_code, new_code)
    
    with open('app.py', 'w') as file:
        file.write(updated_content)
    
    print("Modified date_range_selector in app.py")

# Fix visualization.py to handle date comparisons properly
def fix_visualization_file():
    with open('src/visualization.py', 'r') as file:
        content = file.read()
    
    # Fix plot_daily_steps_trend function
    old_code = "    # Filter data if date_range is provided\n    if date_range and 'Date' in filtered_df.columns:\n        start_date, end_date = date_range\n        filtered_df = filtered_df[(filtered_df['Date'] >= pd.Timestamp(start_date)) & \n                              (filtered_df['Date'] <= pd.Timestamp(end_date))]"
    new_code = "    # Filter data if date_range is provided\n    if date_range and 'Date' in filtered_df.columns:\n        start_date, end_date = date_range\n        # Ensure start_date and end_date are pandas datetime\n        start_date = pd.to_datetime(start_date)\n        end_date = pd.to_datetime(end_date)\n        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & \n                              (filtered_df['Date'] <= end_date)]"
    
    content = content.replace(old_code, new_code)
    
    # Fix similar patterns in other functions (plot_sleep_patterns, plot_heart_rate_analysis, plot_weight_trend)
    # This is a general pattern replacement
    old_pattern = "[(filtered_df['Date'] >= "
    new_pattern = "[(pd.to_datetime(filtered_df['Date']) >= "
    
    content = content.replace(old_pattern, new_pattern)
    
    old_pattern = "(filtered_df['Date'] <= "
    new_pattern = "(pd.to_datetime(filtered_df['Date']) <= "
    
    content = content.replace(old_pattern, new_pattern)
    
    # Fix heart rate analysis selected_hr['Time'].dt.hour issue
    old_code = "    # Heart rate by hour of day\n    selected_hr['Hour'] = selected_hr['Time'].dt.hour"
    new_code = "    # Heart rate by hour of day\n    if 'Time' in selected_hr.columns:\n        # Ensure Time column is datetime\n        selected_hr = selected_hr.copy()\n        selected_hr['Time'] = pd.to_datetime(selected_hr['Time'], errors='coerce')\n        selected_hr['Hour'] = selected_hr['Time'].dt.hour"
    
    content = content.replace(old_code, new_code)
    
    with open('src/visualization.py', 'w') as file:
        file.write(content)
    
    print("Fixed visualization.py")

# Fix the activity_analysis_page function to handle missing columns
def fix_activity_analysis_page():
    with open('app.py', 'r') as file:
        content = file.read()
    
    # Add column check before accessing 'Steps'
    old_code = "        with col1:\n            avg_steps = filtered_daily['Steps'].mean()\n            max_steps = filtered_daily['Steps'].max()"
    new_code = "        with col1:\n            # Check if Steps column exists\n            if 'Steps' in filtered_daily.columns:\n                avg_steps = filtered_daily['Steps'].mean()\n                max_steps = filtered_daily['Steps'].max()\n            else:\n                # Try to find the right column (may be case sensitive)\n                steps_col = [col for col in filtered_daily.columns if col.lower() == 'steps']\n                if steps_col:\n                    avg_steps = filtered_daily[steps_col[0]].mean()\n                    max_steps = filtered_daily[steps_col[0]].max()\n                else:\n                    avg_steps = 0\n                    max_steps = 0\n                    st.warning('Steps data not found')"
    
    content = content.replace(old_code, new_code)
    
    with open('app.py', 'w') as file:
        file.write(content)
    
    print("Fixed activity_analysis_page in app.py")

# Improve data loading to ensure proper datetime conversion
def improve_data_loading():
    with open('app.py', 'r') as file:
        content = file.read()
    
    # Modify the load_data function
    old_code = "                    # Convert date columns\n                    date_cols = [col for col in data_dict[key].columns if 'Date' in col or 'Day' in col]\n                    for col in date_cols:\n                        try:\n                            data_dict[key][col] = pd.to_datetime(data_dict[key][col])\n                        except:\n                            pass"
    new_code = "                    # Convert date columns\n                    date_cols = [col for col in data_dict[key].columns if 'Date' in col or 'Day' in col or 'Time' in col]\n                    for col in date_cols:\n                        try:\n                            if col in data_dict[key].columns:\n                                data_dict[key][col] = pd.to_datetime(data_dict[key][col], errors='coerce')\n                        except Exception as e:\n                            logger.warning(f\"Error converting {col} to datetime: {str(e)}\")"
    
    content = content.replace(old_code, new_code)
    
    # Add debug information
    old_code = "        # Get all users\n        all_users = get_all_users(data_dict)\n        data_dict['all_users'] = all_users"
    new_code = "        # Get all users\n        all_users = get_all_users(data_dict)\n        data_dict['all_users'] = all_users\n        \n        # Debug: Display available columns in key dataframes\n        for key in [\"daily_activity\", \"merged_daily\"]:\n            if key in data_dict and data_dict[key] is not None:\n                logger.info(f\"Columns in {key}: {data_dict[key].columns.tolist()}\")"
    
    content = content.replace(old_code, new_code)
    
    # Modify CSV loading to avoid DtypeWarning
    old_code = "                    data_dict[key] = pd.read_csv(file_path)"
    new_code = "                    data_dict[key] = pd.read_csv(file_path, low_memory=False)"
    
    content = content.replace(old_code, new_code)
    
    with open('app.py', 'w') as file:
        file.write(content)
    
    print("Improved data loading in app.py")

# Fix the constants in app.py to include fitbit folder
def fix_constants():
    with open('app.py', 'r') as file:
        content = file.read()
    
    # Update RAW_DATA_DIR
    old_code = "RAW_DATA_DIR = os.path.join('data', 'raw')"
    new_code = "RAW_DATA_DIR = os.path.join('data', 'raw', 'fitbit')"
    
    content = content.replace(old_code, new_code)
    
    with open('app.py', 'w') as file:
        file.write(content)
    
    print("Fixed constants in app.py")

# Execute all fixes
if __name__ == "__main__":
    print("Applying fixes to the Fitness Tracker app...")
    fix_constants()
    modify_date_range_selector()
    fix_visualization_file()
    fix_activity_analysis_page()
    improve_data_loading()
    print("All fixes applied! Try running the app again.")