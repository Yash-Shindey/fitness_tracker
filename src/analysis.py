"""
Analysis Module for Fitness Tracker Application.

This module provides advanced analysis functions for Fitbit data including:
- Time series analysis
- Pattern detection
- Statistical analysis
- Predictive modeling
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_activity_patterns(hourly_steps_df, n_clusters=4):
    """
    Detect patterns in hourly activity data using clustering.
    
    Parameters:
    -----------
    hourly_steps_df : pandas.DataFrame
        Dataframe containing hourly step data
    n_clusters : int
        Number of clusters to identify
        
    Returns:
    --------
    tuple
        (DataFrame with cluster assignments, cluster centers)
    """
    if hourly_steps_df is None or 'StepTotal' not in hourly_steps_df.columns:
        return None, None
    
    # Prepare the data
    # Convert ActivityHour to datetime if it's not already
    if 'ActivityHour' in hourly_steps_df.columns and not pd.api.types.is_datetime64_any_dtype(hourly_steps_df['ActivityHour']):
        hourly_steps_df['ActivityHour'] = pd.to_datetime(hourly_steps_df['ActivityHour'])
    
    # Extract hour and create user-day identifier
    hourly_steps_df['Hour'] = hourly_steps_df['ActivityHour'].dt.hour
    hourly_steps_df['Date'] = hourly_steps_df['ActivityHour'].dt.date
    hourly_steps_df['UserDay'] = hourly_steps_df['Id'].astype(str) + '_' + hourly_steps_df['Date'].astype(str)
    
    # Create a pivot table with hours as columns
    activity_patterns = hourly_steps_df.pivot_table(
        index='UserDay',
        columns='Hour',
        values='StepTotal',
        fill_value=0
    )
    
    # Normalize the data
    scaler = StandardScaler()
    scaled_patterns = scaler.fit_transform(activity_patterns)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_patterns)
    
    # Add cluster labels to the original data
    activity_patterns['Cluster'] = cluster_labels
    
    # Get cluster centers (in original scale)
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=activity_patterns.columns[:-1]  # Exclude the Cluster column
    )
    
    return activity_patterns, cluster_centers

def analyze_sleep_activity_correlation(merged_daily_df):
    """
    Analyze correlation between daily activity and sleep quality.
    
    Parameters:
    -----------
    merged_daily_df : pandas.DataFrame
        Dataframe containing merged daily activity and sleep data
        
    Returns:
    --------
    dict
        Dictionary containing correlation analysis results
    """
    if merged_daily_df is None:
        return None
    
    required_cols = ['Steps', 'VeryActiveMinutes', 'FairlyActiveMinutes', 
                     'LightlyActiveMinutes', 'SedentaryMinutes', 
                     'TotalMinutesAsleep', 'TotalTimeInBed', 'SleepEfficiency']
    
    if not all(col in merged_daily_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in merged_daily_df.columns]
        logger.warning(f"Missing columns for sleep-activity correlation analysis: {missing_cols}")
        return None
    
    # Filter out days with no sleep data
    sleep_activity_df = merged_daily_df[merged_daily_df['TotalMinutesAsleep'] > 0].copy()
    
    if len(sleep_activity_df) == 0:
        logger.warning("No days with sleep data available for correlation analysis")
        return None
    
    # Calculate correlations
    correlation_cols = [
        'Steps', 'VeryActiveMinutes', 'FairlyActiveMinutes', 
        'LightlyActiveMinutes', 'SedentaryMinutes', 'Calories'
    ]
    
    correlations = {}
    for col in correlation_cols:
        if col in sleep_activity_df.columns:
            sleep_duration_corr = sleep_activity_df[col].corr(sleep_activity_df['TotalMinutesAsleep'])
            sleep_efficiency_corr = sleep_activity_df[col].corr(sleep_activity_df['SleepEfficiency'])
            
            correlations[col] = {
                'sleep_duration': round(sleep_duration_corr, 3),
                'sleep_efficiency': round(sleep_efficiency_corr, 3)
            }
    
    # Calculate lag effects (previous day's activity vs. current day's sleep)
    sleep_activity_df = sleep_activity_df.sort_values(['Id', 'Date'])
    
    # Create lagged features (previous day's activity)
    for col in correlation_cols:
        if col in sleep_activity_df.columns:
            sleep_activity_df[f'Prev_{col}'] = sleep_activity_df.groupby('Id')[col].shift(1)
    
    # Calculate lag correlations
    lag_correlations = {}
    for col in correlation_cols:
        lagged_col = f'Prev_{col}'
        if lagged_col in sleep_activity_df.columns:
            # Remove rows with NaN values (first day for each user)
            lag_df = sleep_activity_df.dropna(subset=[lagged_col])
            
            if len(lag_df) > 0:
                sleep_duration_lag_corr = lag_df[lagged_col].corr(lag_df['TotalMinutesAsleep'])
                sleep_efficiency_lag_corr = lag_df[lagged_col].corr(lag_df['SleepEfficiency'])
                
                lag_correlations[col] = {
                    'sleep_duration': round(sleep_duration_lag_corr, 3),
                    'sleep_efficiency': round(sleep_efficiency_lag_corr, 3)
                }
    
    return {
        'same_day_correlations': correlations,
        'previous_day_correlations': lag_correlations
    }

def detect_heart_rate_anomalies(heart_rate_df, contamination=0.01):
    """
    Detect anomalies in heart rate data using Isolation Forest.
    
    Parameters:
    -----------
    heart_rate_df : pandas.DataFrame
        Dataframe containing heart rate data
    contamination : float
        Expected proportion of anomalies in the dataset
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with anomaly flags
    """
    if heart_rate_df is None or 'Value' not in heart_rate_df.columns:
        return None
    
    # Group by user and date
    user_dates = heart_rate_df.groupby(['Id', 'Date'])
    
    # Initialize result dataframe
    results = []
    
    # Process each user-date combination
    for (user_id, date), group in user_dates:
        if len(group) < 10:  # Skip days with too few measurements
            continue
            
        # Prepare features (heart rate value and its rolling statistics)
        X = group[['Value']].copy()
        
        # Add rolling statistics if there are enough measurements
        if len(X) >= 20:
            X['rolling_mean'] = group['Value'].rolling(window=10, min_periods=5).mean()
            X['rolling_std'] = group['Value'].rolling(window=10, min_periods=5).std()
            X = X.dropna()
        
        if len(X) < 10:  # Skip if not enough data after preprocessing
            continue
            
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
        X = X_copy
        
        # Convert predictions: -1 (anomaly) to 1, 1 (normal) to 0
        X['anomaly'] = (X['anomaly'] == -1).astype(int)
        
        # Add user and date info
        X['Id'] = user_id
        X['Date'] = date
        
        # Add to results
        results.append(X.reset_index())
    
    if results:
        # Combine all results
        anomalies_df = pd.concat(results, ignore_index=True)
        return anomalies_df
    else:
        return None

def predict_calories_from_activity(daily_activity_df, test_size=0.2):
    """
    Train a model to predict calorie burn based on activity metrics.
    
    Parameters:
    -----------
    daily_activity_df : pandas.DataFrame
        Dataframe containing daily activity data
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    tuple
        (trained model, feature importance dict, evaluation metrics dict)
    """
    if daily_activity_df is None:
        return None, None, None
    
    required_cols = [
        'Steps', 'TotalDistance', 'VeryActiveMinutes', 
        'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes',
        'Calories'
    ]
    
    if not all(col in daily_activity_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in daily_activity_df.columns]
        logger.warning(f"Missing columns for calorie prediction: {missing_cols}")
        return None, None, None
    
    # Prepare the data
    X = daily_activity_df[required_cols[:-1]]  # All except Calories
    y = daily_activity_df['Calories']
    
    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    
    # Sort feature importance
    feature_importance = {k: round(v, 4) for k, v in sorted(feature_importance.items(), 
                                                            key=lambda item: item[1], 
                                                            reverse=True)}
    
    # Create evaluation metrics
    eval_metrics = {
        'mse': round(mse, 2),
        'rmse': round(rmse, 2),
        'r2': round(r2, 4),
        'mean_absolute_error': round(np.mean(np.abs(y_test - y_pred)), 2)
    }
    
    return model, feature_importance, eval_metrics

def perform_time_series_decomposition(daily_df, metric='Steps', period=7):
    """
    Perform time series decomposition to identify trend, seasonality, and residual components.
    
    Parameters:
    -----------
    daily_df : pandas.DataFrame
        Dataframe containing daily data
    metric : str
        Column name of the metric to analyze
    period : int
        Period for seasonality detection (e.g., 7 for weekly patterns)
        
    Returns:
    --------
    dict
        Dictionary containing decomposition results
    """
    if daily_df is None or metric not in daily_df.columns:
        return None
    
    # Ensure data is sorted by date
    daily_df = daily_df.sort_values('Date')
    
    # Prepare time series data by user
    user_ids = daily_df['Id'].unique()
    results = {}
    
    for user_id in user_ids:
        user_data = daily_df[daily_df['Id'] == user_id].copy()
        
        # Check if we have enough data for this user
        if len(user_data) < 2 * period:
            continue
        
        # Set date as index
        user_data = user_data.set_index('Date')
        
        try:
            # Perform decomposition
            decomposition = seasonal_decompose(
                user_data[metric], 
                model='additive', 
                period=period,
                extrapolate_trend='freq'
            )
            
            user_results = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed
            }
            
            results[user_id] = user_results
            
        except Exception as e:
            logger.warning(f"Time series decomposition failed for user {user_id}: {str(e)}")
    
    return results

def calculate_user_similarity(daily_activity_df):
    """
    Calculate similarity between users based on activity patterns.
    
    Parameters:
    -----------
    daily_activity_df : pandas.DataFrame
        Dataframe containing daily activity data
        
    Returns:
    --------
    pandas.DataFrame
        Similarity matrix
    """
    if daily_activity_df is None:
        return None
    
    # Features to use for similarity calculation
    features = [
        'Steps', 'TotalDistance', 'VeryActiveMinutes', 
        'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes',
        'Calories'
    ]
    
    missing_features = [f for f in features if f not in daily_activity_df.columns]
    if missing_features:
        # Use available features
        features = [f for f in features if f in daily_activity_df.columns]
        logger.warning(f"Missing features for similarity calculation: {missing_features}")
        
    if not features:
        return None
    
    # Calculate average metrics for each user
    user_profiles = daily_activity_df.groupby('Id')[features].mean()
    
    # Standardize the data
    scaler = StandardScaler()
    user_profiles_scaled = scaler.fit_transform(user_profiles)
    user_profiles_scaled = pd.DataFrame(user_profiles_scaled, index=user_profiles.index, columns=user_profiles.columns)
    
    # Calculate similarity (correlation-based)
    similarity_matrix = user_profiles_scaled.T.corr()
    
    return similarity_matrix

def analyze_weekly_patterns(daily_activity_df):
    """
    Analyze weekly patterns in activity data.
    
    Parameters:
    -----------
    daily_activity_df : pandas.DataFrame
        Dataframe containing daily activity data
        
    Returns:
    --------
    pandas.DataFrame
        Weekly pattern analysis results
    """
    if daily_activity_df is None or 'DayOfWeek' not in daily_activity_df.columns:
        return None
    
    # Metrics to analyze
    metrics = [
        'Steps', 'TotalDistance', 'VeryActiveMinutes', 
        'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes',
        'Calories'
    ]
    
    available_metrics = [m for m in metrics if m in daily_activity_df.columns]
    if not available_metrics:
        return None
    
    # Group by day of week and calculate statistics
    weekly_patterns = daily_activity_df.groupby('DayOfWeek')[available_metrics].agg(
        ['mean', 'median', 'std', 'min', 'max']
    )
    
    # Reset index for easier access
    weekly_patterns = weekly_patterns.reset_index()
    
    # Add day names
    day_names = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    weekly_patterns['Day'] = weekly_patterns['DayOfWeek'].map(day_names)
    
    # Reorder columns to have Day first
    cols = weekly_patterns.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    weekly_patterns = weekly_patterns[cols]
    
    return weekly_patterns

def identify_consistent_active_periods(hourly_steps_df, threshold=500):
    """
    Identify consistently active periods during the day.
    
    Parameters:
    -----------
    hourly_steps_df : pandas.DataFrame
        Dataframe containing hourly step data
    threshold : int
        Minimum number of steps to consider an hour as active
        
    Returns:
    --------
    dict
        Dictionary containing active periods by user
    """
    if hourly_steps_df is None or 'StepTotal' not in hourly_steps_df.columns:
        return None
    
    # Convert ActivityHour to datetime if needed
    if 'ActivityHour' in hourly_steps_df.columns and not pd.api.types.is_datetime64_any_dtype(hourly_steps_df['ActivityHour']):
        hourly_steps_df['ActivityHour'] = pd.to_datetime(hourly_steps_df['ActivityHour'])
    
    # Extract hour
    hourly_steps_df['Hour'] = hourly_steps_df['ActivityHour'].dt.hour
    
    # Group by user and hour, calculate average steps
    hourly_avg = hourly_steps_df.groupby(['Id', 'Hour'])['StepTotal'].agg(['mean', 'std', 'count']).reset_index()
    
    # Identify active hours (above threshold)
    hourly_avg['is_active'] = hourly_avg['mean'] > threshold
    
    # Find consecutive active hours
    active_periods = {}
    
    for user_id in hourly_avg['Id'].unique():
        user_hours = hourly_avg[hourly_avg['Id'] == user_id].sort_values('Hour')
        active_hours = user_hours[user_hours['is_active']]['Hour'].tolist()
        
        # Find consecutive periods
        periods = []
        current_period = []
        
        for hour in range(24):  # Check all hours
            if hour in active_hours:
                current_period.append(hour)
            else:
                if current_period:
                    periods.append((min(current_period), max(current_period)))
                    current_period = []
        
        # Add the last period if exists
        if current_period:
            periods.append((min(current_period), max(current_period)))
        
        # Format periods for readability
        formatted_periods = []
        for start, end in periods:
            start_str = f"{start:02d}:00"
            end_str = f"{end:02d}:59"
            formatted_periods.append(f"{start_str} - {end_str}")
        
        active_periods[user_id] = formatted_periods
    
    return active_periods