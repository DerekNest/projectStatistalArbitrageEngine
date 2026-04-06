import pandas as pd
import os

def build_russell_universe(input_path="data/IWB_holdings.csv", output_path="data/russell_1000.csv"):
    """
    parses the ishares russell 1000 (iwb) holdings csv file.
    dynamically finds the header row to prevent breakages when blackrock updates formatting.
    """
    if not os.path.exists(input_path):
        print(f"error: could not find {input_path}")
        return

    # dynamically find the header row by reading line-by-line
    skip_lines = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # the header row begins with 'Ticker'
            if line.strip().startswith("Ticker"):
                skip_lines = i
                break

    # read the csv starting from the correct header row
    df = pd.read_csv(input_path, skiprows=skip_lines)

    # clean column names to strip leading/trailing whitespace
    df.columns = df.columns.str.strip()

    # filter out cash, derivatives, and strictly map columns
    if 'Asset Class' in df.columns:
        df = df[df['Asset Class'] == 'Equity']

    # standardize column names for the engine
    df = df.rename(columns={'Ticker': 'ticker', 'Sector': 'sector'})

    # subset to only the columns we need and drop missing values
    df = df[['ticker', 'sector']].dropna()

    # save to the expected format for the engine
    df.to_csv(output_path, index=False)
    
    print(f"successfully extracted {len(df)} tickers to {output_path}")
    return df

if __name__ == "__main__":
    # make sure your downloaded file is named exactly this in your data folder
    build_russell_universe(input_path="data/IWB_holdings.csv")