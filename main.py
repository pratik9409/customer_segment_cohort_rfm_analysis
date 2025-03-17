from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import datetime as dt
import numpy as np
import psycopg2
from sqlalchemy import create_engine


app = Flask(__name__)

# def load_data():
#     df = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')
#     df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
#     df.dropna(subset=['CustomerID'], inplace=True)
#     return df

# Database connection setup
DATABASE_URL = "postgresql://postgres:quacking90@localhost:5432/postgres"
engine = create_engine(DATABASE_URL)



def load_data():
    query = "SELECT * FROM online_retail"
    df = pd.read_sql(query, con=engine)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df.dropna(subset=['customer_id'], inplace=True)
    return df

def rfm_analysis(df):
    snapshot_date = df['invoice_date'].max() + dt.timedelta(days=1)
    rfm = df.groupby('customer_id').agg({
        'invoice_date': lambda x: (snapshot_date - x.max()).days,
        'invoice_no': 'count',
        'unit_price': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

def kmeans_clustering(rfm, clusters=4):
    model = KMeans(n_clusters=clusters, random_state=42)
    rfm['Cluster'] = model.fit_predict(rfm[['Recency', 'Frequency', 'Monetary']])
    return rfm

def cohort_analysis(df):
    df['OrderMonth'] = df['invoice_date'].dt.to_period('M').astype(str)
    df['Cohort'] = df.groupby('customer_id')['invoice_date'].transform('min').dt.to_period('M').astype(str)
    cohort_counts = df.groupby(['Cohort', 'OrderMonth']).customer_id.nunique().unstack(0)
    return cohort_counts

def country_sales_analysis(df):
    country_sales = df.groupby('country')['unit_price'].sum().reset_index()
    return country_sales

def calculate_ltv(df):
    df['Revenue'] = df['unit_price'] * df['quantity']
    ltv = df.groupby('customer_id')['Revenue'].sum().reset_index()
    return ltv

def churn_prediction(df):
    df['DaysSinceLastPurchase'] = (df['invoice_date'].max() - df['invoice_date']).dt.days
    churn_threshold = df['DaysSinceLastPurchase'].quantile(0.75)
    df['ChurnRisk'] = np.where(df['DaysSinceLastPurchase'] > churn_threshold, 'High', 'Low')
    churn = df.groupby('ChurnRisk').customer_id.nunique().reset_index()
    return churn

def time_series_forecasting(df):
    df['InvoiceMonth'] = df['invoice_date'].dt.to_period('M')
    monthly_sales = df.groupby('InvoiceMonth')['unit_price'].sum().reset_index()
    monthly_sales['InvoiceMonth'] = monthly_sales['InvoiceMonth'].astype(str)
    # Ensure enough data points before applying seasonality
    seasonal_periods = min(12, len(monthly_sales)//2)  # Adjust if data is too small

    if seasonal_periods >= 2:
        model = ExponentialSmoothing(monthly_sales['unit_price'], trend='add', seasonal='add', seasonal_periods=seasonal_periods)
    else:
        model = ExponentialSmoothing(monthly_sales['unit_price'], trend='add', seasonal=None)  # No seasonality if insufficient data
    # model = ExponentialSmoothing(monthly_sales['unit_price'], trend='add', seasonal='add', seasonal_periods=12)
    fit = model.fit()
    forecast = fit.forecast(6)
    return monthly_sales, forecast

def detect_anomalies(df):
    df['Revenue'] = df['quantity'] * df['unit_price']
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = model.fit_predict(df[['Revenue']])
    anomalies = df[df['Anomaly'] == -1]
    return anomalies

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/segmentation', methods=['GET', 'POST'])
def index():
    df = load_data()
    rfm = rfm_analysis(df)
    clusters = int(request.form.get('clusters', 4))
    rfm = kmeans_clustering(rfm, clusters)
    
    fig = px.scatter(rfm, x='Recency', y='Monetary', color='Cluster', title='Customer Segmentation')
    plot_html = fig.to_html(full_html=False)
    
    return render_template('index.html', plot_html=plot_html, clusters=clusters)

@app.route('/cohort')
def cohort():
    df = load_data()
    cohort_data = cohort_analysis(df)
    cohort_data.index = cohort_data.index.astype(str)
    cohort_data.columns = cohort_data.columns.astype(str)
    fig = px.imshow(cohort_data, labels=dict(x='Cohort', y='Order Month', color='Customers'))
    # fig = px.imshow(cohort_data, labels=dict(x='Cohort Month', y='Order Month', color='Customers'), title='Cohort Analysis')
    plot_html = fig.to_html(full_html=False)
    return render_template('cohort.html', plot_html=plot_html)

@app.route('/country-sales')
def country_sales():
    df = load_data()
    country_sales = country_sales_analysis(df)
    fig = px.bar(country_sales, x='country', y='unit_price')
    plot_html = fig.to_html(full_html=False)
    return render_template('country_sales.html', plot_html=plot_html)

@app.route('/ltv')
def ltv():
    df = load_data()
    ltv_data = calculate_ltv(df)
    fig = px.histogram(ltv_data, x='Revenue', nbins=50, title='Customer Lifetime Value Distribution')
    plot_html = fig.to_html(full_html=False)
    return render_template('ltv.html', plot_html=plot_html)

@app.route('/churn')
def churn():
    df = load_data()
    churn_data = churn_prediction(df)
    fig = px.pie(churn_data, names='ChurnRisk', values='customer_id', title='Churn Prediction')
    plot_html = fig.to_html(full_html=False)
    return render_template('churn.html', plot_html=plot_html)

@app.route('/forecast')
def forecast():
    df = load_data()
    sales_data, forecast_values = time_series_forecasting(df)
    fig = px.line(sales_data, x='InvoiceMonth', y='unit_price', title='Monthly Sales with Forecast')
    forecast_df = pd.DataFrame({'InvoiceMonth': list(range(len(sales_data), len(sales_data) + 6)), 'unit_price': forecast_values})
    fig.add_scatter(x=forecast_df['InvoiceMonth'], y=forecast_df['unit_price'], mode='lines', name='Forecast')
    plot_html = fig.to_html(full_html=False)
    return render_template('forecast.html', plot_html=plot_html)


@app.route('/anomalies')
def anomalies():
    df = load_data()
    anomaly_data = detect_anomalies(df)
    fig = px.scatter(anomaly_data, x='invoice_date', y='Revenue', title='Customer Anomalies', color_discrete_sequence=['red'])
    plot_html = fig.to_html(full_html=False)
    return render_template('anomalies.html', plot_html=plot_html)

if __name__ == '__main__':
    app.run(debug=True)
