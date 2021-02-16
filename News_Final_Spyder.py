# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:35:10 2020

@author: Ralph Tranquille
"""

import plotly
import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
#from fbprophet.diagnostics import performance_modeletrics
#from fbprophet.plot import plot_cross_validation_modeletric
from fbprophet.plot import plot_plotly, plot_components_plotly


# Read the data AND View the first 2 rows of the data
df = pd.read_csv('News_Final.csv') 
df.head(2)


# Read only the PublishDate and the SentimentTitle ----------------------------
df2 = pd.read_csv("News_Final.csv", usecols = ['PublishDate','SentimentHeadlines'])
df2.head(2)
df2.tail(2)
df2


# Select a certain date from the PublishDate Column ---------------------------
df3 = df2.loc[df2['PublishDate'].between('2015-03-01  12:06:00 AM','2018-03-01  12:06:00 AM', inclusive=False)]
df3.tail()



# Rename the column to sd and y -----------------------------------------------
Sentiment=df3.rename(columns={"PublishDate": "ds", "SentimentHeadlines": "y"})
Sentiment.head(2)


# Instantiate the new prophet model / W/ CHANGEPOINT  or seasonality_mode------
m = Prophet (n_changepoints=7)
# or
# m = Prophet(changepoint_prior_scale=0.5)
# # or 
# m = Prophet(seasonality_mode='multiplicative')
# # or 
# m = Prophet(mcmc_samples=300)


# ADD US holidays -------------------------------------------------------------
m.add_country_holidays(country_name='US')


### OPTIONAL ### ADD Seasonality AND Regressor Parameters ---------------------
# m.add_seasonality('quarterly', period=91.25, fourier_order=8, mode='additive')


# Fit the model ---------------------------------------------------------------
m.fit(Sentiment)


# Create the future dataframe we want to forecast into ------------------------
future = m.make_future_dataframe(periods=12, freq='MS', include_history=False)
future.head()
future.tail()


# Build Forcast/Prediction ----------------------------------------------------
forecast = m.predict(future)


# The upper/lower bounds of the Forcast/Prediction ----------------------------
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# plot the Forcast/Prediction -------------------------------------------------
fig1 = m.plot(forecast)


# Plot the decomposition of the Forcast/Prediction ----------------------------
fig2 = m.plot_components(forecast)


# ADD CHANGEPOINTS TO THE MODEL  and Plot -------------------------------------
plot_plotly(m, forecast, xlabel='DATE', ylabel='SENTIMENT')
plot_components_plotly(m,forecast)
fig3 = m.plot(forecast)
a= add_changepoints_to_plot(fig3.gca(), m,forecast)



# # Cross-validation to determine the accuracy of the model -------------------
# df2_cv = cross_validation(m, horizon = '10 days')


# # Performance Metrics of the Cross Validation -------------------------------
# df2_p = performance_metrics(df2_cv)
# df2_p.head()

# # Plot the Corss Validation d2 wit metric mape ------------------------------ 
# fig = plot_cross_validation_metric(df2_cv, metric='mape')





























