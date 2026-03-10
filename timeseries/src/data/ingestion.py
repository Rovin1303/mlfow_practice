import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def download_btcusd_data(symbol="BTC-USD", period="max", interval="1d"):
    """
    Downloads historical BTC-USD data from Yahoo Finance.
    """
    print(f"Downloading data for {symbol}...")
    try:
        # Fetch data and reset index to handle MultiIndex columns
        data = yf.download(symbol, period=period, interval=interval)
        if data.empty:
            raise ValueError("Downloaded data is empty.")
        
        # Yahoo Finance returns MultiIndex for columns if single ticker is used sometimes, 
        # or just standard columns. Let's flatten if needed.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Ensure the directory exists
        os.makedirs("data/raw", exist_ok=True)
        
        # Save to CSV
        file_path = f"data/raw/{symbol.lower()}_historical.csv"
        data.to_csv(file_path)
        print(f"Data saved to {file_path}")
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def preprocess_data(df):
    """
    Basic preprocessing: handling missing values and ensuring correct types.
    """
    if df is None:
        return None
    
    # Fill missing values if any
    df = df.ffill()
    
    # Add some basic technical indicators as features
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Drop rows with NaN from rolling calculations
    df = df.dropna()
    
    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    processed_path = "data/processed/btcusd_processed.csv"
    df.to_csv(processed_path)
    print(f"Processed data saved to {processed_path}")
    return df

if __name__ == "__main__":
    raw_data = download_btcusd_data()
    if raw_data is not None:
        processed_data = preprocess_data(raw_data)
        print("Data Ingestion and Preprocessing Complete.")
