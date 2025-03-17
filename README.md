# Customer Analytics Dashboard
## 📌 Project Overview
This project focuses on analyzing customer purchasing behavior using Cohort Analysis, RFM (Recency, Frequency, Monetary) Analysis, and K-Means Clustering. By leveraging these techniques, businesses can gain insights into customer retention patterns, segment customers based on their transaction history, and tailor marketing strategies accordingly.
Customer Analytics Dashboard designed to provide businesses with deep insights into customer purchasing behavior using machine learning techniques. The dashboard is built using Flask, PostgreSQL, and advanced data science methodologies to segment customers, predict churn, forecast sales, and detect anomalies.

The dashboard provides an interactive and visually appealing user interface that enables users to explore various customer analytics metrics efficiently.
##🔥 Features
### 🏷 Customer Segmentation
**RFM Analysis**: Evaluates customers based on:
  **Recency**: How recently a customer made a purchase.
  **Frequency**: How often a customer makes a purchase.
  **Monetary**: How much money a customer spends on purchases.
  **K-Means Clustering**: Groups customers into distinct segments based on their RFM scores, allowing businesses to identify high-value customers and those requiring                           attention.
### 📊 Cohort Analysis
**Monthly Cohorts**: Analyzes customer retention by grouping them based on their first purchase month and tracking their behavior over subsequent months. 
                     Tracks customer retention trends and repeat purchase behavior. 
                     Helps businesses understand customer loyalty over time.

### 🌍 Sales by Country
Provides a regional breakdown of sales performance.
Identifies high-revenue geographical locations.

### 💰 Customer Lifetime Value (LTV)
Predicts future revenue generation from existing customers.
Uses historical purchase data to estimate long-term profitability.

### ⚠️ Churn Prediction
Implements machine learning classification models to identify customers likely to stop purchasing.
Helps businesses create retention strategies.

### 📈 Sales Forecasting
Utilizes time-series forecasting techniques such as ARIMA, Prophet, or LSTM to predict future revenue.
Helps in inventory and revenue planning.

### 🚨 Anomaly Detection
Uses unsupervised learning techniques (e.g., Isolation Forest, DBSCAN) to flag unusual transactions.
Helps detect fraudulent activities or uncommon purchasing behavior.

### 🛢 Database Integration
Stores customer transaction data in PostgreSQL for scalability and efficient querying.
Allows dynamic updates and supports real-time data analysis.


## 🏗 Tech Stack
**Data Analysis & Modeling**: Python, Jupyter Notebook
**Machine Learning**: Scikit-learn, Statsmodels, XGBoost
**Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
**Database**: PostgreSQL
**Web Framework(Backend)**: Flask(Python)
**Frontend**: HTML, CSS, Bootstrap, JavaScript
**Visualization**:Matplotlib, Seaborn, Plotly, Dash

## 🚀 Installation & Setup
### 1️⃣ Clone the Repository

```bash
git clone https://github.com/pratik9409/customer_segment_cohort_rfm_analysis.git
cd customer_segment_cohort_rfm_analysis
```
### 2️⃣ Set Up PostgreSQL Database
**Install PostgreSQL**: Ensure PostgreSQL is installed on your system.
**Create Database**: Create a new database named customer_analysis.
**Update Credentials**: Modify the db.py file with your PostgreSQL username and password.
### 3️⃣ Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Import Data into PostgreSQL
**CSV File**: Ensure the OnlineRetail.csv file is in the project directory.
**Run Import Script**: Execute the db.py script to import data into PostgreSQL.
```bash
python db.py
```
### 5️⃣ Run the Flask Application
```bash
python main.py
```
Access the application at http://127.0.0.1:5000/


