import time

import numpy as np
import pandas as pd
import yfinance as yf

# =====================================================================
# SETTINGS - change only the numbers in this block
# =====================================================================
MIN_PRICE = 10            # rupees: ignore stocks priced below this
MIN_AVG_VALUE_CR = 7      # rupees crore: minimum average daily traded value
VALUE_DAYS = 20           # trading days used for average traded value
RS_CUTOFF = 80            # top 20% = RS rating of 80 or above
NEAR_HIGH_PCT = 20        # "near 52-week high" = within 20% of it
MIN_GROUP_SIZE = 4        # sectors/industries with fewer eligible stocks are ignored
HISTORY_DAYS = 5          # trading days of history saved
WEIGHTS = {"3M": 0.40, "6M": 0.20, "9M": 0.20, "12M": 0.20}   # RS 1Y formula
PERIODS = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "9M": 189, "12M": 252}
CHUNK = 100               # stocks per download batch

PERIOD_LABELS = list(PERIODS.keys())
META_COLS = ["S. No.", "Macro-Economic Sector", "Sector", "Industry",
             "Basic Industry", "Symbol", "Company Name"]

STOCK_COLS = (META_COLS
              + ["Data as of", "Price", "52W High", "% From 52W High",
                 "Avg Value (Cr, 20D)", "Passes Filters", "Within 20% of High"]
              + [f"{p} %" for p in PERIOD_LABELS]
              + [f"RS {p}" for p in PERIOD_LABELS]
              + ["RS 1Y", "RS 1Y Rank", "Top 20% RS"])

GROUP_MID = (["Stocks"]
             + [f"Median {p} %" for p in PERIOD_LABELS]
             + ["Median % From 52W High", "Within 20% of High (count)",
                "% Within 20% of High", "Top 20% RS (count)"]
             + [f"RS {p}" for p in PERIOD_LABELS]
             + ["RS 1Y", "Top 20% Group", "Rank"])
SECTOR_COLS = ["Sector"] + GROUP_MID
INDUSTRY_COLS = ["Industry Key", "Sector", "Industry"] + GROUP_MID

# ---------------------------------------------------------------------
# 1. Read the stock list
# ---------------------------------------------------------------------
meta = pd.read_csv("stocks.csv")
meta.columns = [c.strip() for c in meta.columns]
for col in META_COLS:
    meta[col] = meta[col].astype(str).str.strip()
meta["Symbol"] = meta["Symbol"].str.upper()
meta = meta.drop_duplicates(subset="Symbol").reset_index(drop=True)
meta["Yahoo Code"] = meta["Symbol"] + ".NS"
codes = meta["Yahoo Code"].tolist()


# ---------------------------------------------------------------------
# 2. Download two years of daily prices
# ---------------------------------------------------------------------
def clean(df):
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df.dropna(subset=["Close"])


price_data = {}
for start in range(0, len(codes), CHUNK):
    batch = codes[start:start + CHUNK]
    print(f"Downloading stocks {start + 1} to {start + len(batch)} of {len(codes)}")
    try:
        raw = yf.download(batch, period="2y", interval="1d", auto_adjust=True,
                          group_by="ticker", threads=True, progress=False)
    except Exception as error:
        print("Batch failed:", error)
        continue
    for code in batch:
        try:
            df = clean(raw[code])
            if len(df) > 0:
                price_data[code] = df
        except Exception:
            pass
    time.sleep(2)

# Retry anything that was missing, one stock at a time
for code in codes:
    if code in price_data:
        continue
    time.sleep(1)
    try:
        df = clean(yf.Ticker(code).history(period="2y", auto_adjust=True))
        if len(df) > 0:
            price_data[code] = df
    except Exception:
        pass

print(f"Price data found for {len(price_data)} of {len(codes)} stocks")

# Find the last few trading days (days where at least half the stocks traded)
date_counts = {}
for df in price_data.values():
    for ts in df.index:
        date_counts[ts] = date_counts.get(ts, 0) + 1
enough = 0.5 * len(price_data)
good_dates = sorted(ts for ts, n in date_counts.items() if n >= enough)
if not good_dates:
    raise SystemExit("No price data was downloaded. Check the Actions log.")
as_of_dates = good_dates[-HISTORY_DAYS:]


# ---------------------------------------------------------------------
# 3. Calculate each stock as on a given date
# ---------------------------------------------------------------------
def stock_row(code, df, as_of):
    d = df[df.index <= as_of]
    if len(d) < 30:
        return None
    last_date = d.index[-1]
    close = d["Close"].astype(float)
    price = float(close.iloc[-1])
    high_52 = float(d["High"].astype(float).tail(252).max())
    value = (close * d["Volume"].astype(float)).tail(VALUE_DAYS).mean()
    row = {
        "Yahoo Code": code,
        "Data as of": last_date.date().isoformat(),
        "Price": round(price, 2),
        "52W High": round(high_52, 2),
        "% From 52W High": round((price / high_52 - 1) * 100, 2),
        "Avg Value (Cr, 20D)": round(float(value) / 1e7, 2),
        "Fresh": (as_of - last_date).days <= 5,
    }
    for label, days in PERIODS.items():
        if len(close) > days:
            past = float(close.iloc[-1 - days])
            row[f"{label} %"] = round((price / past - 1) * 100, 2)
        else:
            row[f"{label} %"] = np.nan
    return row


def percentile(values, mask):
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result[mask] = values[mask].rank(pct=True) * 100
    return result


def build_stocks(as_of):
    rows = []
    for code, df in price_data.items():
        row = stock_row(code, df, as_of)
        if row:
            rows.append(row)
    t = meta.merge(pd.DataFrame(rows), on="Yahoo Code", how="left")

    numeric = (["Price", "52W High", "% From 52W High", "Avg Value (Cr, 20D)"]
               + [f"{p} %" for p in PERIOD_LABELS])
    for col in numeric:
        t[col] = pd.to_numeric(t[col], errors="coerce")

    fresh = t["Fresh"].fillna(False).astype(bool)
    ok = (t["Price"] >= MIN_PRICE) & (t["Avg Value (Cr, 20D)"] >= MIN_AVG_VALUE_CR) & fresh

    t["Passes Filters"] = np.where(ok, "Yes", "No")
    t["Within 20% of High"] = np.where(ok & (t["% From 52W High"] >= -NEAR_HIGH_PCT), "Yes", "No")
    for p in PERIOD_LABELS:
        t[f"RS {p}"] = percentile(t[f"{p} %"], ok).round(1)
    t["RS 1Y"] = sum(w * t[f"RS {p}"] for p, w in WEIGHTS.items()).round(1)
    t["RS 1Y Rank"] = t["RS 1Y"].where(ok).rank(ascending=False, method="min").astype("Int64")
    t["Top 20% RS"] = np.where(ok & (t["RS 1Y"] >= RS_CUTOFF), "Yes", "No")
    t["Industry Key"] = t["Sector"] + " | " + t["Industry"]
    return t, ok


# ---------------------------------------------------------------------
# 4. Group sectors and industries
# ---------------------------------------------------------------------
def build_groups(t, ok, keys):
    src = t[ok].copy()
    src["Within"] = (src["Within 20% of High"] == "Yes").astype(int)
    src["Top"] = (src["Top 20% RS"] == "Yes").astype(int)

    agg = {"Stocks": ("Symbol", "count")}
    for p in PERIOD_LABELS:
        agg[f"Median {p} %"] = (f"{p} %", "median")
    agg["Median % From 52W High"] = ("% From 52W High", "median")
    agg["Within 20% of High (count)"] = ("Within", "sum")
    agg["Top 20% RS (count)"] = ("Top", "sum")

    g = src.groupby(keys).agg(**agg).reset_index()
    g = g[g["Stocks"] >= MIN_GROUP_SIZE].copy()
    g["% Within 20% of High"] = (g["Within 20% of High (count)"] / g["Stocks"] * 100).round(1)
    for p in PERIOD_LABELS:
        g[f"RS {p}"] = (g[f"Median {p} %"].rank(pct=True) * 100).round(1)
    g["RS 1Y"] = sum(w * g[f"RS {p}"] for p, w in WEIGHTS.items()).round(1)
    g["Top 20% Group"] = np.where(g["RS 1Y"] >= RS_CUTOFF, "Yes", "No")
    g["Rank"] = g["RS 1Y"].rank(ascending=False, method="min").astype("Int64")
    return g.sort_values("RS 1Y", ascending=False).round(2)


# ---------------------------------------------------------------------
# 5. Run for each of the last 5 days and build history
# ---------------------------------------------------------------------
stock_hist, sector_hist, industry_hist = [], [], []
for as_of in as_of_dates:
    t, ok = build_stocks(as_of)
    sec = build_groups(t, ok, ["Sector"])
    ind = build_groups(t, ok, ["Industry Key", "Sector", "Industry"])
    day = pd.Timestamp(as_of).date().isoformat()

    s = sec[["Sector", "Stocks", "% Within 20% of High", "RS 1Y", "Rank"]].copy()
    s.insert(0, "Date", day)
    sector_hist.append(s)

    i = ind[["Industry Key", "Sector", "Industry", "Stocks",
             "% Within 20% of High", "RS 1Y", "Rank"]].copy()
    i.insert(0, "Date", day)
    industry_hist.append(i)

    st = t[ok][["Symbol", "Company Name", "Sector", "Industry",
                "Within 20% of High", "RS 1Y", "RS 1Y Rank", "Top 20% RS"]].copy()
    st.insert(0, "Date", day)
    stock_hist.append(st)

latest_stocks, latest_sectors, latest_industries = t, sec, ind

# ---------------------------------------------------------------------
# 6. Save the files
# ---------------------------------------------------------------------
latest_stocks[STOCK_COLS].sort_values("RS 1Y", ascending=False,
                                      na_position="last").to_csv("stocks_rs.csv", index=False)
latest_sectors[SECTOR_COLS].to_csv("sectors_rs.csv", index=False)
latest_industries[INDUSTRY_COLS].to_csv("industries_rs.csv", index=False)
pd.concat(stock_hist).to_csv("history_stocks.csv", index=False)
pd.concat(sector_hist).to_csv("history_sectors.csv", index=False)
pd.concat(industry_hist).to_csv("history_industries.csv", index=False)

no_data = meta.loc[~meta["Yahoo Code"].isin(list(price_data.keys())), "Symbol"]
pd.DataFrame({"Symbols with no price data": no_data.values}).to_csv("failed.csv", index=False)

print("Eligible stocks:", int((latest_stocks["Passes Filters"] == "Yes").sum()))
print("Top 20% RS stocks:", int((latest_stocks["Top 20% RS"] == "Yes").sum()))
print("Files saved.")
