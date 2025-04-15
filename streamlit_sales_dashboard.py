# Importing packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
import joblib

# Loading variables
@st.cache_data
def load_data():
    sales_complete = joblib.load("data/sales_complete.pkl")
    sales_monthly = joblib.load("data/sales_monthly.pkl")
    X_monthly = joblib.load("data/X_monthly.pkl")

    return sales_complete, sales_monthly, X_monthly

sales_complete, sales_monthly, X_monthly = load_data()

# App title
st.title("Superstore Sales Data Interactive Dashboard")

# --- SECTION 1: Filtered Segment Sales ---    
# Making sidebar filters
st.sidebar.header("Filters")
selected_region = st.sidebar.multiselect("Region", sales_complete["Region"].unique(), default=sales_complete["Region"].unique())
selected_category = st.sidebar.multiselect("Category", sales_complete["Category"].unique(), default=sales_complete["Category"].unique())
selected_year = st.sidebar.multiselect("Year", sales_complete["Year"].unique(), default=sales_complete["Year"].unique())
selected_month = st.sidebar.multiselect("Month", sales_complete["Month"].unique(), default=sales_complete["Month"].unique())

# Filtering sales dataset
filtered_sales = sales_complete[
    (sales_complete["Region"].isin(selected_region)) &
    (sales_complete["Category"].isin(selected_category)) &
    (sales_complete["Year"].isin(selected_year)) &
    (sales_complete["Month"].isin(selected_month))
]

# Showing filtered dataset
st.subheader("Filtered Sales")
st.write(filtered_sales.head())

# Defining segment order
segment_order = ["Consumer", "Corporate", "Home Office"]
segment_sales = (
    filtered_sales
    .groupby("Segment")["Sales"]
    .sum()
    .reindex(segment_order)
    .reset_index()
)

# Plotting filtered segment sales
fig = plt.figure(figsize=(10, 6))
sns.barplot(data=segment_sales, x="Segment", y="Sales")

plt.title("Total sales by segment")
plt.xlabel("Segment")
plt.ylabel("Sales")
st.pyplot(fig)

# --- SECTION 2: SARIMAX Forecast vs Actual Sales  --- 
st.subheader("Sales Forecasting with the SARIMAX Model")

min_date = sales_monthly.index[0].date()
max_date = sales_monthly.index[-1].date()

# Making a slider for actual historical sales
start_month, end_month = st.slider(
    "Select period for historical sales data:", 
    min_value = min_date, 
    max_value=max_date, 
    value=(min_date, max_date), 
    format="Y-M"
)
# Converting to time period to ensure accurate indexing
start_month = pd.Timestamp(start_month).to_period('M')
end_month = pd.Timestamp(end_month).to_period('M')

months_selected = (end_month.year - start_month.year) * 12 + (end_month.month - start_month.month)

# Making a slider for forecast horizon
forecast_horizon = st.slider(
    "Select forecasting period (1-6 months):", 
    min_value = 1, 
    max_value = 6, 
    value = 3, 
    step = 1
)

if months_selected < 24:
    st.warning("Please select a range of at least 24 months.")
else:
    if st.button("Run Forecast"):    
        # Creating evaluation function
        def evaluate_model(y_true, y_pred, start_month, end_month, forecast_horizon):
            mae = mean_absolute_error(y_true, y_pred) # Absolute forecasting errors
            rmse = root_mean_squared_error(y_true, y_pred) # Penalizes larger forecasting errors
            mape = mean_absolute_percentage_error(y_true, y_pred) # Relative forecasting errors
            forecast_bias = np.mean(y_pred - y_true) # Indicates tendency to overpredict or underpredict
        
            evaluations = (
                f"SARIMAX model results:\n"
                f"Period: {start_month} to {end_month}\n"
                f"Forecast horizon: {forecast_horizon} months\n"
                f"mae: {mae}\n"
                f"mape: {mape}\n"
                f"rmse:{rmse}\n"
                f"forecast_bias: {forecast_bias}"
            )
        
            return evaluations
    
            
        # Creating forecast function
        def forecast_sarimax(sales_monthly, X_monthly, start_month, end_month, forecast_horizon=6, m=12):    
            # Getting the integer location in sales_monthly for start_month and end_month
            window_start = sales_monthly.index.get_loc(start_month.strftime("%Y-%m-%d"))
            window_end = sales_monthly.index.get_loc(end_month.strftime("%Y-%m-%d"))
            
            # Getting the historical sales for the specified window
            historical_sales = sales_monthly.iloc[window_start:window_end + 1]
            
            # Getting the logged sales for the training period
            y_log_train = sales_monthly["Log Sales"].iloc[:window_end-forecast_horizon + 1]
        
            # Getting the true sales values for the forecasting horizon
            y_true = historical_sales["Sales"].iloc[-forecast_horizon:]
            
            # Splitting exogenous variables into training and testing
            X_train = X_monthly.iloc[:window_end-forecast_horizon + 1]
            X_test = X_monthly.iloc[window_end-forecast_horizon + 1:window_end+1]
    
            # Handling seasonal differencing issues
            try:
                # Try fitting the model 
                sarima_model = pm.auto_arima(
                    y=y_log_train, 
                    X=X_train, 
                    seasonal=True, 
                    m=m, 
                    stepwise=True, 
                    trace=False, 
                    error_action="ignore" # Ignore minor errors during fitting
                )
            except ValueError as e: 
                # If there is a ValueError remove seasonal differencing
                sarima_model = pm.auto_arima(
                    y=y_log_train, 
                    X=X_train, 
                    seasonal=True, 
                    m=m, 
                    stepwise=True, 
                    trace=False, 
                    error_action="ignore",
                    D=0 # Do not apply seasonal differencing
                )
        
            # Forecasting the logged sales values for the forecasting horizon
            sarimax_log_forecast = sarima_model.predict(n_periods=forecast_horizon, X=X_test)
            
            # Inverse transformation to get sales values in the original scale
            sarimax_forecast = np.expm1(sarimax_log_forecast)
        
            # Evaluating the model
            sarima_evaluation = evaluate_model(y_true, sarimax_forecast, start_month, end_month, forecast_horizon)
            st.text(sarima_evaluation)
        
            # Plotting actual sales data for the last 24 months
            figure = plt.figure(figsize=(12,6))
            plt.plot(historical_sales.index, historical_sales["Sales"], marker="o", 
                     label=f"Actual sales for {start_month} to {end_month}", color="blue")
            
            # Plotting SARIMA forecast for the next 6 months
            plt.plot(y_true.index, sarimax_forecast, marker="o", 
                     label=f"SARIMAX forecast with {forecast_horizon} months horizon ", color="red", 
                     linestyle="dashed")
            
            # Formatting plot
            plt.title(f"SARIMAX Forecast vs Actual Sales")
            plt.xlabel("Date")
            plt.xticks(historical_sales.index, rotation=45)
            plt.ylabel("Sales")
            plt.axvline(x=y_true.index[0], color='black', linestyle='--', label="Forecast Start")
            plt.legend()
            plt.grid(True)
            st.pyplot(figure)
        
            return sarimax_forecast, y_true
        
        # Making a SARIMAX forecast
        sarimax_forecast, y_true = forecast_sarimax(
            sales_monthly, X_monthly, start_month=start_month, end_month=end_month, forecast_horizon = forecast_horizon, m=12
        )
        
        # --- SECTION 3: Residuals of Forecasted Sales  ---        
        st.subheader("Residuals of Forecasted Sales")
        
        # Calculating residuals between true sales values and forecasted sales values
        residuals = y_true - sarimax_forecast.values
        
        # Plotting residuals
        fig = plt.figure(figsize=(12, 6))
        plt.plot(y_true.index, residuals, label="Residuals", color="green")
        plt.axhline(0, color="black", linestyle="--")
        plt.title("Residuals of SARIMAX Forecast")
        plt.xlabel("Month")
        plt.xticks(y_true.index)
        plt.ylabel("Residuals")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)

        # --- SECTION 4: Percentage Errors for Forecasted Sales  ---    
        st.subheader("Percentage Errors of Forecasted Sales")
        
        # Calculating percentage errors and MAPE for reference
        error_percentages = np.abs(residuals / y_true) * 100
        mape = np.mean(np.abs(error_percentages))
        
        # Plotting percentage errors for forecasted values
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x=sarimax_forecast.index, y=error_percentages, color="purple")
        plt.axhline(y=mape, color="red", linestyle="--", label=f"Mean Absolute Percentage Error: {mape:.2f}%")
        plt.title("Percentage Error Of Forecasted Sales")
        plt.xlabel("Month")
        plt.ylabel("Percentage Error")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)