import time

import numpy as np
import pandas as pd
import yfinance as yf

# =====================================================================
# SETTINGS - change only the numbers in this block
# =====================================================================
MIN_PRICE = 10               # rupees: ignore stocks priced below this for rankings
MIN_AVG_VALUE_CR = 7         # rupees crore: minimum 20-day average daily value for rankings
VALUE_DAYS = 20              # trading days used for average traded value
RS_CUTOFF = 80               # top 20% = RS rating of 80 or above
NEAR_HIGH_PCT = 20           # "near 52-week high" = within 20% of it
MIN_GROUP_SIZE = 4           # sectors/industries with fewer eligible stocks are ignored
HISTORY_DAYS = 5             # trading days of history (RS and group performance)
SCREEN_HISTORY_DAYS = 21     # trading days of screen history (about one month)
SWING_NEAR_HIGH_PCT = 15     # swing: within this % of the 52-week high
SWING_RS3M_MIN = 80          # swing: RS 3M at or above this

# Stage 2 screen
S2_MIN_PRICE = 20            # rupees
S2_MIN_MCAP_CR = 200         # rupees crore (Rs 2 billion = Rs 200 crore)
S2_NEAR_HIGH_PCT = 20        # within this % of the 52-week high
S2_MIN_52W_LOW_PCT = 30      # at least this % above the 52-week low
S2_MIN_RS1Y = 80             # RS 1Y must be above this
S2_MIN_VALUE_CR = 7          # rupees crore average daily value

# 20% in one month screen
M1_MIN_PRICE = 5             # rupees
M1_MIN_RETURN_PCT = 20       # 1 month return above this %
M1_MIN_VALUE_CR = 5          # rupees crore average daily value
M1_MIN_VOLUME_RATIO = 1.5    # 5-day average volume / 50-day average, at least this

MARKET_INFO_LIMIT_SEC = 2400 # stop fetching share counts after 40 minutes
WEIGHTS = {"3M": 0.40, "6M": 0.20, "9M": 0.20, "12M": 0.20}   # RS 1Y formula
PERIODS = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "9M": 189, "12M": 252}
GROUP_RETURNS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63}          # money-flow periods
CHUNK = 100                  # stocks per download batch

PERIOD_LABELS = list(PERIODS.keys())
META_COLS = ["S. No.", "Macro-Economic Sector", "Sector", "Industry",
             "Basic Industry", "Symbol", "Company Name"]

BASE_STOCK_COLS = (META_COLS
                   + ["Data as of", "Price", "52W High", "% From 52W High",
                      "Avg Value (Cr, 20D)", "Passes Filters", "Within 20% of High"]
                   + [f"{p} %" for p in PERIOD_LABELS]
                   + [f"RS {p}" for p in PERIOD_LABELS]
                   + ["RS 1Y", "RS 1Y Rank", "Top 20% RS"])
TREND_COLS = ["50 DMA", "200 DMA", "Above 50 DMA", "Above 200 DMA", "50 Above 200",
              "200 DMA Rising", "52W Low", "% Above 52W Low", "Volume Ratio 5D/50D",
              "Swing Setup", "RS 3M Rank"]
SCREEN_TREND_COLS = ["150 DMA", "Market Cap (Cr)", "Stage 2 Screen", "20% Month Screen"]
STOCK_COLS = BASE_STOCK_COLS + TREND_COLS + SCREEN_TREND_COLS + ["1D %"]

SCREEN_COLS = ["Symbol", "Company Name", "Sector", "Industry", "Price",
               "% From 52W High", "Within 20% of High", "RS 3M", "RS 1Y",
               "Above 50 DMA", "Above 200 DMA", "50 Above 200", "200 DMA Rising",
               "Volume Ratio 5D/50D", "Swing Setup", "1M %", "Avg Value (Cr, 20D)",
               "Market Cap (Cr)", "150 DMA", "Stage 2 Screen", "20% Month Screen"]

GROUP_MID = (["Stocks"]
             + [f"Median {p} %" for p in PERIOD_LABELS]
             + ["Median % From 52W High", "Within 20% of High (count)",
                "% Within 20% of High", "Top 20% RS (count)"]
             + [f"RS {p}" for p in PERIOD_LABELS]
             + ["RS 1Y", "Top 20% Group", "Rank", "RS 3M Rank"])
SECTOR_COLS = ["Sector"] + GROUP_MID
INDUSTRY_COLS = ["Industry Key", "Sector", "Industry"] + GROUP_MID

GROUP_PERF_COLS = (["Date", "Level", "Key", "Group", "Sector", "Industry", "Stocks"]
                   + [f"Median {p} %" for p in GROUP_RETURNS]
                   + ["% Within 20% of High"])


def positive_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(number) and number > 0:
        return number
    return None


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


# ---------------------------------------------------------------------
# 2b. Share counts and market caps (used for the market cap filter)
# ---------------------------------------------------------------------
def fetch_market_info(code):
    try:
        info = yf.Ticker(code).fast_info
    except Exception:
        return None, None
    shares, mcap = None, None
    try:
        shares = positive_number(info["shares"])
    except Exception:
        pass
    try:
        mcap = positive_number(info["market_cap"])
    except Exception:
        pass
    return shares, mcap


SHARES, MCAP_NOW = {}, {}
started = time.time()
print("Fetching share counts and market caps")
for code in price_data:
    if time.time() - started > MARKET_INFO_LIMIT_SEC:
        print("Time limit reached for market data; the rest will be left blank.")
        break
    shares, mcap = fetch_market_info(code)
    if shares:
        SHARES[code] = shares
    if mcap:
        MCAP_NOW[code] = mcap
    time.sleep(0.2)
print(f"Share counts for {len(SHARES)} stocks; market caps for {len(MCAP_NOW)} stocks")

# Find the last trading days (days where at least half the stocks traded)
date_counts = {}
for df in price_data.values():
    for ts in df.index:
        date_counts[ts] = date_counts.get(ts, 0) + 1
enough = 0.5 * len(price_data)
good_dates = sorted(ts for ts, n in date_counts.items() if n >= enough)
if not good_dates:
    raise SystemExit("No price data was downloaded. Check the Actions log.")
as_of_dates = good_dates[-SCREEN_HISTORY_DAYS:]
rs_dates = as_of_dates[-HISTORY_DAYS:]


# ---------------------------------------------------------------------
# 3. Calculate each stock as on a given date
# ---------------------------------------------------------------------
def stock_row(code, df, as_of):
    d = df[df.index <= as_of]
    if len(d) < 30:
        return None
    last_date = d.index[-1]
    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    volume = d["Volume"].astype(float)
    price = float(close.iloc[-1])
    high_52 = float(high.tail(252).max())
    low_52 = float(low.tail(252).min())
    value = float((close * volume).tail(VALUE_DAYS).mean())

    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else np.nan
    sma150 = float(close.tail(150).mean()) if len(close) >= 150 else np.nan
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else np.nan
    sma200_past = float(close.iloc[-220:-20].mean()) if len(close) >= 220 else np.nan
    vol50 = float(volume.tail(50).mean()) if len(volume) >= 50 else np.nan
    vol5 = float(volume.tail(5).mean()) if len(volume) >= 5 else np.nan
    vol_ratio = vol5 / vol50 if (vol50 and vol50 > 0) else np.nan

    shares = SHARES.get(code)
    if shares:
        mcap_cr = shares * price / 1e7
    elif code in MCAP_NOW:
        mcap_cr = MCAP_NOW[code] / 1e7
    else:
        mcap_cr = np.nan

    row = {
        "Yahoo Code": code,
        "Data as of": last_date.date().isoformat(),
        "Price": round(price, 2),
        "52W High": round(high_52, 2),
        "% From 52W High": round((price / high_52 - 1) * 100, 2),
        "Avg Value (Cr, 20D)": round(value / 1e7, 2),
        "Fresh": (as_of - last_date).days <= 5,
        "52W Low": round(low_52, 2),
        "% Above 52W Low": round((price / low_52 - 1) * 100, 2),
        "50 DMA": round(sma50, 2),
        "150 DMA": round(sma150, 2),
        "200 DMA": round(sma200, 2),
        "Above 50 DMA": "Yes" if price > sma50 else "No",
        "Above 200 DMA": "Yes" if price > sma200 else "No",
        "50 Above 200": "Yes" if sma50 > sma200 else "No",
        "200 DMA Rising": "Yes" if sma200 > sma200_past else "No",
        "Volume Ratio 5D/50D": round(vol_ratio, 2),
        "Market Cap (Cr)": round(mcap_cr, 0),
    }
    for label, days in {**GROUP_RETURNS, **PERIODS}.items():
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

    numeric = (["Price", "52W High", "% From 52W High", "Avg Value (Cr, 20D)",
                "52W Low", "% Above 52W Low", "50 DMA", "150 DMA", "200 DMA",
                "Volume Ratio 5D/50D", "Market Cap (Cr)"]
               + [f"{p} %" for p in GROUP_RETURNS]
               + [f"{p} %" for p in PERIOD_LABELS])
    for col in numeric:
        t[col] = pd.to_numeric(t[col], errors="coerce")

    fresh = t["Fresh"].fillna(False).astype(bool)
    base = fresh & t["Price"].notna()
    ok = (t["Price"] >= MIN_PRICE) & (t["Avg Value (Cr, 20D)"] >= MIN_AVG_VALUE_CR) & fresh

    t["Passes Filters"] = np.where(ok, "Yes", "No")
    t["Within 20% of High"] = np.where(ok & (t["% From 52W High"] >= -NEAR_HIGH_PCT), "Yes", "No")
    for p in PERIOD_LABELS:
        t[f"RS {p}"] = percentile(t[f"{p} %"], ok).round(1)
    t["RS 1Y"] = sum(w * t[f"RS {p}"] for p, w in WEIGHTS.items()).round(1)
    t["RS 1Y Rank"] = t["RS 1Y"].where(ok).rank(ascending=False, method="min").astype("Int64")
    t["RS 3M Rank"] = t["RS 3M"].where(ok).rank(ascending=False, method="min").astype("Int64")
    t["Top 20% RS"] = np.where(ok & (t["RS 1Y"] >= RS_CUTOFF), "Yes", "No")
    t["Swing Setup"] = np.where(
        ok
        & (t["Above 50 DMA"] == "Yes")
        & (t["Above 200 DMA"] == "Yes")
        & (t["50 Above 200"] == "Yes")
        & (t["200 DMA Rising"] == "Yes")
        & (t["% From 52W High"] >= -SWING_NEAR_HIGH_PCT)
        & (t["RS 3M"] >= SWING_RS3M_MIN),
        "Yes", "No")

    # Stage 2 screen
    t["Stage 2 Screen"] = np.where(
        base
        & (t["Price"] > S2_MIN_PRICE)
        & (t["Market Cap (Cr)"] > S2_MIN_MCAP_CR)
        & (t["Price"] > t["50 DMA"])
        & (t["50 DMA"] > t["150 DMA"])
        & (t["150 DMA"] > t["200 DMA"])
        & (t["200 DMA Rising"] == "Yes")
        & (t["% From 52W High"] >= -S2_NEAR_HIGH_PCT)
        & (t["% Above 52W Low"] >= S2_MIN_52W_LOW_PCT)
        & (t["RS 1Y"] > S2_MIN_RS1Y)
        & (t["Avg Value (Cr, 20D)"] >= S2_MIN_VALUE_CR),
        "Yes", "No")

    # 20% in one month screen
    t["20% Month Screen"] = np.where(
        base
        & (t["Price"] > M1_MIN_PRICE)
        & (t["1M %"] > M1_MIN_RETURN_PCT)
        & (t["Price"] > t["200 DMA"])
        & (t["Avg Value (Cr, 20D)"] >= M1_MIN_VALUE_CR)
        & (t["Volume Ratio 5D/50D"] >= M1_MIN_VOLUME_RATIO),
        "Yes", "No")

    t["Industry Key"] = t["Sector"] + " | " + t["Industry"]
    return t, ok


# ---------------------------------------------------------------------
# 4. Group sectors and industries (RS ranking)
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
    g["RS 3M Rank"] = g["RS 3M"].rank(ascending=False, method="min").astype("Int64")
    return g.sort_values("RS 1Y", ascending=False).round(2)


# ---------------------------------------------------------------------
# 4b. Money-flow table: median returns and breadth per group
# ---------------------------------------------------------------------
def build_group_performance(t, ok, day):
    src = t[ok].copy()
    src["Near"] = (src["Within 20% of High"] == "Yes").astype(int)
    frames = []
    for level in ["Sector", "Industry"]:
        keys = ["Sector"] if level == "Sector" else ["Sector", "Industry"]
        agg = {"Stocks": ("Symbol", "count"), "Near": ("Near", "sum")}
        for p in GROUP_RETURNS:
            agg[f"Median {p} %"] = (f"{p} %", "median")
        g = src.groupby(keys).agg(**agg).reset_index()
        g = g[g["Stocks"] >= MIN_GROUP_SIZE].copy()
        if level == "Sector":
            g["Industry"] = ""
            g["Key"] = g["Sector"]
            g["Group"] = g["Sector"]
        else:
            g["Key"] = g["Sector"] + " | " + g["Industry"]
            g["Group"] = g["Industry"]
        g["Level"] = level
        g["Date"] = day
        g["% Within 20% of High"] = g["Near"] / g["Stocks"] * 100
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    return out[GROUP_PERF_COLS]


# ---------------------------------------------------------------------
# 5. Run for each saved day and build history
# ---------------------------------------------------------------------
stock_hist, sector_hist, industry_hist, screen_hist, group_hist = [], [], [], [], []
for as_of in as_of_dates:
    t, ok = build_stocks(as_of)
    sec = build_groups(t, ok, ["Sector"])
    ind = build_groups(t, ok, ["Industry Key", "Sector", "Industry"])
    day = pd.Timestamp(as_of).date().isoformat()

    any_screen = ((t["Swing Setup"] == "Yes")
                  | (t["Stage 2 Screen"] == "Yes")
                  | (t["20% Month Screen"] == "Yes"))
    screen = t[any_screen][SCREEN_COLS].copy()
    screen.insert(0, "Date", day)
    screen_hist.append(screen)

    if as_of in rs_dates:
        s = sec[["Sector", "Stocks", "% Within 20% of High", "RS 1Y", "Rank",
                 "RS 3M", "RS 3M Rank"]].copy()
        s = s.rename(columns={"Rank": "Rank 1Y", "RS 3M Rank": "Rank 3M"})
        s.insert(0, "Date", day)
        sector_hist.append(s)

        i = ind[["Industry Key", "Sector", "Industry", "Stocks", "% Within 20% of High",
                 "RS 1Y", "Rank", "RS 3M", "RS 3M Rank"]].copy()
        i = i.rename(columns={"Rank": "Rank 1Y", "RS 3M Rank": "Rank 3M"})
        i.insert(0, "Date", day)
        industry_hist.append(i)

        st = t[ok][["Symbol", "Company Name", "Sector", "Industry", "Within 20% of High",
                    "RS 1Y", "RS 1Y Rank", "RS 3M", "RS 3M Rank"]].copy()
        st.insert(0, "Date", day)
        stock_hist.append(st)

        group_hist.append(build_group_performance(t, ok, day))

# The last loop pass is the latest day
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
pd.concat(screen_hist).to_csv("screen_history.csv", index=False)

group_out = pd.concat(group_hist).round(2)
group_out = group_out.sort_values(["Date", "Level", "Group"])
group_out.to_csv("group_performance.csv", index=False)

no_data = meta.loc[~meta["Yahoo Code"].isin(list(price_data.keys())), "Symbol"]
pd.DataFrame({"Symbols with no price data": no_data.values}).to_csv("failed.csv", index=False)

print("Eligible stocks:", int((latest_stocks["Passes Filters"] == "Yes").sum()))
print("Top 20% RS stocks:", int((latest_stocks["Top 20% RS"] == "Yes").sum()))
print("Swing setups today:", int((latest_stocks["Swing Setup"] == "Yes").sum()))
print("Stage 2 screen today:", int((latest_stocks["Stage 2 Screen"] == "Yes").sum()))
print("20% in one month screen today:", int((latest_stocks["20% Month Screen"] == "Yes").sum()))
print("Group performance rows saved:", len(group_out))
print("Files saved.")
