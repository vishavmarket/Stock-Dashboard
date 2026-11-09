import datetime
import time

import numpy as np
import pandas as pd
import yfinance as yf

# ---------- 1. Read your stock list ----------
stocks = pd.read_csv("stocks.csv")
stocks.columns = [c.strip() for c in stocks.columns]
stocks["Symbol"] = stocks["Symbol"].astype(str).str.strip().str.upper()
stocks["Yahoo Code"] = stocks["Symbol"] + ".NS"

# Number of trading days to look back for each performance period
PERIODS = {"1W %": 5, "1M %": 21, "3M %": 63, "6M %": 126, "12M %": 252}


# ---------- 2. Work out the numbers for one stock ----------
def calculate(d):
    d = d.dropna(subset=["Close"])
    if len(d) < 30:
        raise ValueError("not enough price history")
    last_date = d.index[-1].date()
    if (datetime.date.today() - last_date).days > 5:
        raise ValueError("price data is old")
    close = d["Close"]
    price = float(close.iloc[-1])
    high_52 = float(d["High"].tail(252).max())
    row = {
        "Data as of": str(last_date),
        "Price": round(price, 2),
        "52W High": round(high_52, 2),
        "% From 52W High": round((price / high_52 - 1) * 100, 2),
    }
    for label, days in PERIODS.items():
        if len(close) > days:
            past = float(close.iloc[-1 - days])
            row[label] = round((price / past - 1) * 100, 2)
        else:
            row[label] = np.nan
    return row


# ---------- 3. Download prices in batches of 100 ----------
yahoo_codes = stocks["Yahoo Code"].tolist()
rows = []
failed = []
CHUNK = 100

for start in range(0, len(yahoo_codes), CHUNK):
    batch = yahoo_codes[start:start + CHUNK]
    print(f"Downloading stocks {start + 1} to {start + len(batch)}")
    try:
        prices = yf.download(batch, period="14mo", interval="1d", auto_adjust=True,
                             group_by="ticker", threads=True, progress=False)
    except Exception as error:
        print("Batch failed:", error)
        failed.extend(batch)
        continue
    for code in batch:
        try:
            row = calculate(prices[code])
            row["Yahoo Code"] = code
            rows.append(row)
        except Exception:
            failed.append(code)
    time.sleep(2)

# ---------- 4. Retry the failures one at a time ----------
still_failed = []
for code in failed:
    time.sleep(1)
    try:
        prices = yf.Ticker(code).history(period="14mo", auto_adjust=True)
        row = calculate(prices)
        row["Yahoo Code"] = code
        rows.append(row)
    except Exception:
        still_failed.append(code)

print(f"Done. Succeeded: {len(rows)}. Failed: {len(still_failed)}")

# ---------- 5. Combine everything into result tables ----------
results = stocks.merge(pd.DataFrame(rows), on="Yahoo Code", how="left")
results = results.drop(columns=["Yahoo Code"])

# Stock is flagged if it is within 20% of its 52-week high
results["Within 20% of High"] = np.where(
    results["Price"].notna(),
    (results["% From 52W High"] >= -20).astype(int),
    np.nan,
)


def make_summary(group_column):
    ok = results[results["Price"].notna()]
    grouped = ok.groupby(group_column)
    table = pd.DataFrame({
        "Stocks": grouped.size(),
        "Within 20% of High": grouped["Within 20% of High"].sum(),
        "Avg 1M %": grouped["1M %"].mean(),
        "Avg 3M %": grouped["3M %"].mean(),
        "Avg 6M %": grouped["6M %"].mean(),
        "Avg 12M %": grouped["12M %"].mean(),
    })
    table = table[table["Stocks"] >= 5].copy()  # ignore groups with fewer than 5 stocks
    table["% Within 20% of High"] = (table["Within 20% of High"] / table["Stocks"] * 100).round(1)
    table = table.round(2).sort_values("% Within 20% of High", ascending=False)
    return table.reset_index()


# ---------- 6. Save the files ----------
results.to_csv("results.csv", index=False)
make_summary("Sector").to_csv("sector_summary.csv", index=False)
make_summary("Industry").to_csv("industry_summary.csv", index=False)
make_summary("Macro-Economic Sector").to_csv("macro_summary.csv", index=False)
pd.DataFrame({"Symbols that failed": still_failed}).to_csv("failed.csv", index=False)
print("Files saved.")
