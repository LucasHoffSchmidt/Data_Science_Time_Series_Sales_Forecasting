# Importing packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Loading variables
@st.cache_data
def load_data():
    return pd.read_parquet("data/sales_complete.parquet"), pd.read_parquet("data/sales_monthly.parquet")

sales_complete, sales_monthly = load_data()

# App title
st.title("Superstore Sales Data Interactive Dashboard")

# Filtered segment sales
## Making sidebar filters
st.sidebar.header("Filters")
selected_region = st.sidebar.multiselect("Region", sales_complete["Region"].unique(), default=sales_complete["Region"].unique())
selected_category = st.sidebar.multiselect("Category", sales_complete["Category"].unique(), default=sales_complete["Category"].unique())
selected_year = st.sidebar.multiselect("Year", sales_complete["Year"].unique(), default=sales_complete["Year"].unique())
selected_month = st.sidebar.multiselect("Month", sales_complete["Month"].unique(), default=sales_complete["Month"].unique())

## Filtering sales dataset
filtered_sales = sales_complete[
    (sales_complete["Region"].isin(selected_region)) &
    (sales_complete["Category"].isin(selected_category)) &
    (sales_complete["Year"].isin(selected_year)) &
    (sales_complete["Month"].isin(selected_month))
]

## Showing filtered dataset
st.subheader("Filtered Sales")
st.write(filtered_sales.head())

## Defining segment order
segment_order = ["Consumer", "Corporate", "Home Office"]
filtered_sales["Segment"] = pd.Categorical(filtered_sales["Segment"], categories=segment_order)

## Plotting filtered segment sales
fig = plt.figure(figsize=(10, 6))
sns.barplot(data=filtered_sales, x="Segment", y="Sales", estimator="sum")

plt.title("Total sales by segment")
plt.xlabel("Segment")
plt.ylabel("Sales")
st.pyplot(fig)

# Forecast vs actual sales 
st.subheader("Sales forecasting with Holt-Winters model")

min_date = sales_monthly.index[24].date()
max_date = sales_monthly.index[-1].date()

## Making a slider for actual historical data
start_month, end_month = st.slider(
    "Select period for historical sales data:", 
    min_value = min_date, 
    max_value=max_date, 
    value=(min_date, max_date), 
    format="Y-M"
)

months_selected = (end_month.year - start_month.year) * 12 + (end_month.month - start_month.month)

if months_selected < 12:
    st.warning("Please select a range of at least 12 months.")

## Making a slider for forecasted data
forecast_horizon = st.slider(
    "Select forecast period (1-6 months):", 
    min_value = 1, 
    max_value = 6, 
    value = 3, 
    step = 1
)

## Training and forecasting with the Holt-Winters model
train_data = sales_monthly.loc[:end_month - pd.DateOffset(months=forecast_horizon)]
forecast_start = end_month - pd.DateOffset(months=forecast_horizon -1)

hw_model = ExponentialSmoothing(train_data["Log_sales"], trend="additive", seasonal="additive", initialization_method="estimated").fit()
hw_log_forecast = hw_model.forecast(forecast_horizon)
hw_forecast = np.expm1(hw_log_forecast)

hw_forecast.index = pd.date_range(start=forecast_start, periods=forecast_horizon, freq="M")

## Creating actual sales values
actual_sales = sales_monthly.loc[start_month:forecast_start + pd.DateOffset(months=forecast_horizon)]

## Plotting the actual sales and forecasted sales
fig = plt.figure(figsize=(12, 6))
plt.plot(actual_sales.index, actual_sales["Sales"], label="Actual Sales", marker="o", color="blue")
plt.plot(hw_forecast.index, hw_forecast, label=f"{forecast_horizon}-month forecast", marker="o", color="red", linestyle="dashed")
plt.axvline(x=hw_forecast.index[0], color='black', linestyle='--', label="Forecast Start")
plt.title("Holt-Winters Forecast vs Actual Sales")
plt.xlabel("Date")
plt.xticks(actual_sales.index, rotation=45)
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
st.pyplot(fig)

# Residuals of forecasted values
st.subheader("Residuals of forecasted values")

## Calculating residuals between test data and forecasted values
test_data = actual_sales.iloc[-forecast_horizon:]
residuals = test_data["Sales"].values - hw_forecast.values

## Plotting residuals
fig = plt.figure(figsize=(12, 6))
plt.plot(test_data.index, residuals, label="Residuals", color="green")
plt.axhline(0, color="black", linestyle="--")
plt.title("Residuals of Holt-Winters Forecast")
plt.xlabel("Month")
plt.xticks(test_data.index)
plt.ylabel("Residuals")
plt.legend()
plt.grid(True)
st.pyplot(fig)

# Percentage errors for forecasted values
st.subheader("Evaluation of forecast over time")

## Calculating percentage errors and MAPE for reference
error_percentages = np.abs(residuals / test_data["Sales"].values) * 100
mape = np.mean(np.abs(error_percentages))

## Plotting percentage errors for forecasted values
fig = plt.figure(figsize=(10, 6))
sns.barplot(x=hw_forecast.index, y=error_percentages, color="purple")
plt.axhline(y=mape, color="red", linestyle="--", label=f"Mean Absolute Percentage Error: {mape:.2f}%")
plt.title("Percentage Error Of Forecasted Values")
plt.xlabel("Month")
plt.ylabel("Percentage Error")
plt.legend()
plt.grid(True)
st.pyplot(fig)