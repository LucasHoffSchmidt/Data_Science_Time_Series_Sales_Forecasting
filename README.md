# Data Science: Time Series Sales Forecasting
This project seeks to predict future sales of a global superstore to assist in inventory planning and optimizing pricing strategies. 

## 📊 Business Problem
We are a global superstore selling all kinds of products in the categories of furniture, technology and office supplies. We are interested in predicting our future sales to assist in inventory planning and optimizing pricing strategies.

## 📂 Data Collection
### Objective
Find sales data for a variety of products over a significant period of time. 

### Process
- Searched for sales data.
- Found a kaggle dataset containing an array of superstore product sales data for the years 2015-2018. 
- Checked the size of the dataset, which is 9800 rows and 18 columns. 
- Got an overview of the different attributes included in the dataset:
  - **Row ID:** Unique identifier for each product sale.  
  - **Order ID:** Order ID related to each product sale.  
  - **Order Date:** The date the order was placed.  
  - **Ship Date:** The date the order was shipped.  
  - **Ship Mode:** The type of shipping used to ship the order.  
  - **Customer ID:** The unique identifier for the customer.  
  - **Customer Name:** The name of the customer.  
  - **Segment:** The type of customer (Consumer, Home Office, or Corporate).  
  - **Country:** The country of the customer.  
  - **City:** The city of the customer.  
  - **State:** The state of the customer.  
  - **Postal Code:** The postal code of the customer.  
  - **Region:** The region of the customer.  
  - **Product ID:** The unique identifier for the particular product.  
  - **Category:** The product category.  
  - **Sub-Category:** The product sub-category.  
  - **Product Name:** The name of the product.  
  - **Sales:** The price of the product.  

### Findings
- We notice that the sales target variable has a very high variation, which could make it difficult for predictive models to distinguish the pattern.

---

## 🔍 Data Understanding: Exploratory Data Analysis (EDA)
### Objective
Identify and fix obvious data errors such as incorrect data types and missing values and visualize relationships between features and sales. 

### Process
- Checked for null values and wrong data types.
- Changed the data type of Order Date and Ship Date to datetime.
- Checked the **statistical distribution** of attributes and noticed the high variation between sales with minimum value 0.4, maximum value 22638 and 75% quartile at 210.
- Checked cardinality of attributes.
- Made a histogram of sales to determine its skewness.
- Made a boxplot of sales to determine outliers.
- Created date features from Order Date and made line plots with these to visualize trends and seasonality.
- Made barplots to visualize:
  - Mean difference between order and ship dates by month
  - Total sales by ship mode
  - Total sales by segment and region
  - Total sales by category and subcategory
  - Total sales by state
- Made a folium map over total sales by state  

### Findings
- The sales data is heavily skewed to the left, with most values closer to 0 and quite a few outliers. 
- We see a yearly linear upwards trend, and a repeating yearly seasonal cycle in the sales values. 
- The mean difference between order date and shipping date is fairly consistent around 4 days. 
- We see that by far the most popular shipping mode is the standard class.
- The region and segment with the highest total sales is the west region and the consumer segment.
- The subcategories with highest total sales are phones, chairs and storage with phones and chairs being far higher than the rest.
- The categories of technology, furniture and office supplies are quite equal in total sales, with technology being just a bit higher.
- The states with highest total sales are California, New York and Texas with California and New York far higher than the rest of the states. 

---

## 🔧 Data Preprocessing
### Objective
Perform final data cleaning, check for invalid values as well as duplicates and drop unnecessary features.  

### Process
- Checking for duplicates.
- Checked that invalid names containing numbers.
- Removed any leading and trailing spaces from categorical features.
- Dropped high cardinality, redundant and unique identifier features from the final dataframe. 

### Findings
- There are no duplicates or invalid values in the dataset.
- There are 6 high cardinality features with more than 10% unique values, 2 unique identifiers of Customer ID and Customer Name and 1 redundant feature of Postal Code, since we use the lower cardinality state feature.  

---

## 🧠 Model Training and Evaluation
### Objective
Train statistical models to make sales forecasts and evaluate the models to determine the best. 

### Process
- Imported packages
- Prepared monthly sales data with winsorization to cap extreme outliers and log sales to reduce the skewness of the sales data.
- Splitted the data into training and test data with a test period of 6 months.
- Made a historical sales period of the last 24 months of sales values.
- Made a function to produce evaluation metrics for the given model.
- Made an ARIMA forecast.
- Made seasonal decomposition plots for monthly log sales of the training data.
- Made a SARIMA forecast with yearly seasonality, based on the seasonal decomposition plot.
- Made a Holt-Winters' seasonal forecast with linear trend and seasonality based on the seasonal decomposition plot.
- Evaluated statistical models.  

### Findings
- The seasonality of the sales values are yearly and linear.
- The trend of the sales values is growing linearly.
- The best statistical model with the lowest errors is Holt-Winters' seasonal model. 

---

## 🧐 Model Interpretation
### Objective
Explain how the Holt-Winters' seasonal model makes sales forecasts. 

### Process
- Plotted the level, trend and seasonal components of the Holt-Winters' seasonal model.
- Plotted the residuals for the model's predicted values of the training data and the training data.
- Plotted the forecasted values vs actual values of the last 6 months of the actual sales.  

### Findings
- The growing sales data is captured entirely by the level component, leaving a horizontal trend component.
- The seasonal cycle follows a drop in the beginning of the year, followed by a gradual increase during the year, and spiking in the end of the year.
- The residuals of the forecasted values oscillate around 0, indicating that the model is unbiased with no consistent over- or underprediction of values.
- The residuals spike in the beginning of 2015 with significantly smaller variations after this point, suggesting that the model initially struggled to fit the data accurately, but then improved in later periods.
- The model is quite decent at capturing movements of the sales, but consistently over- or underpredicts the actual sales value. 

---

## 🚀 Model Deployment
### Objective
Deploy the model to an interactive streamlit dashboard, showing a variety of visualizations. 

### Process
- We create an interactive streamlit dashboard with the following features:
   - A **title** for the dashboard.
   - **Sidebar filters** to change the feature variables and update visuals dynamically.
   - A **filtered dataset** that reflects the changes based on the selected filters.
   - A barplot showing **total segment sales** for the filtered dataset.
   - **Forecasted values** vs actual sales values of the selected historical period and forecast horizon.
   - **Residuals** of forecasted values on test data
   - **Percentage errors** for forecasted values
- We save the monthly sales dataset and the complete dataset in the performant **.parquet** format for loading into the Streamlit app.

The streamlit app can be accessed here: [Sales Forecasting Streamlit App](https://data-science-customer-churn-prediction.streamlit.app/)

### Findings
 - The launched model performs well initially with caching of data only marginally improving the speed of execution. 

---

## Conclusion
- The sales values show clear yearly seasonality with a linearly growing trend over time.
- The Holt-Winters' Seasonal model has been found to quite effectively capture the overall movements of future sales values, but it tends to under- or overevaluate the actual sales values.
- The model provides a reliable forecast of the directional movement of sales values, but individual predictions of sales values should be used with caution.
