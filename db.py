import pandas as pd
from sqlalchemy import create_engine

# Database connection
DATABASE_URL = "postgresql://username:password@localhost:5432/postgres"
engine = create_engine(DATABASE_URL)

# Load CSV file
csv_file = "OnlineRetail.csv"
df = pd.read_csv(csv_file, encoding='ISO-8859-1')

# Rename columns for consistency
column_mapping = {
    'InvoiceNo': 'invoice_no',
    'StockCode': 'stock_code',
    'Description': 'description',
    'Quantity': 'quantity',
    'InvoiceDate': 'invoice_date',
    'UnitPrice': 'unit_price',
    'CustomerID': 'customer_id',
    'Country': 'country'
}
df.rename(columns=column_mapping, inplace=True)

# Convert date column to datetime format
df['invoice_date'] = pd.to_datetime(df['invoice_date'])

# Drop duplicates and NaN values
df.dropna(subset=['customer_id'], inplace=True)
df.drop_duplicates(inplace=True)

# Save to PostgreSQL
df.to_sql('online_retail', con=engine, if_exists='replace', index=False)

print("Data successfully imported into PostgreSQL!")