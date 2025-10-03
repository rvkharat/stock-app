# app.py
# Trading Price Action Analysis & Prediction Web App (for Render + GitHub)
# Author: adapted for deployment
# Note: Educational/demo only. Not financial advice.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime, timedelta
import os

# ---------------------------
# Helpers / Data fetching
# ---------------------------
@st.cache_data(ttl=600)
def fetch_data(ticker: str, period: str = "1y", interval: str = "1d"):
    t = ticker.strip().upper()
    if not t:
        raise ValueError("Ticker cannot be empty")
    # If user doesn't specify exchange suffix, assume NSE (.NS)
    if not (t.endswith(".NS") or t.endswith(".BO") or "." in t):
        t = t + ".NS"
    try:
        df = yf.download(t, period=period, interval=interval, progress=False, threads=False)
    except Exception as e:
        raise RuntimeError("yfinance download error: " + str(e))
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {t}. Try different ticker/period.")
    df.index = pd.to_datetime(df.index)
    df.dropna(how="all", inplace=True)
    return df

# ---------------------------
# Technical indicators
# ---------------------------
@st.cache_data(ttl=600)
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["SMA20"] = df2["Close"].rolling(20).mean()
    df2["SMA50"] = df2["Close"].rolling(50).mean()
    df2["EMA12"] = df2["Close"].ewm(span=12, adjust=False).mean()
    df2["EMA26"] = df2["Close"].ewm(span=26, adjust=False).mean()
    df2["RSI14"] = ta.rsi(df2["Close"], length=14)
    macd = ta.macd(df2["Close"])
    for c in macd.columns:
        df2[c] = macd[c]
    df2["ATR14"] = ta.atr(df2["High"], df2["Low"], df2["Close"], length=14)
    bb = ta.bbands(df2["Close"], length=20, std=2)
    for c in bb.columns:
        df2[c] = bb[c]
    df2["Return"] = df2["Close"].pct_change()
    df2["LogRet"] = np.log(df2["Close"]).diff()
    df2 = df2.dropna()
    return df2

# ---------------------------
# Fundamental analytics
# ---------------------------
@st.cache_data(ttl=3600)
def get_fundamentals(ticker: str):
    t = ticker.strip().upper()
    if not (t.endswith(".NS") or "." in t):
        t = t + ".NS"
    tk = yf.Ticker(t)
    info = {}
    try:
        info = tk.info
    except Exception:
        info = {}
    fundamentals = {
        "shortName": info.get("shortName", ""),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "marketCap": info.get("marketCap", None),
        "trailingPE": info.get("trailingPE", None),
        "forwardPE": info.get("forwardPE", None),
        "priceToBook": info.get("priceToBook", None),
        "beta": info.get("beta", None),
        "dividendYield": info.get("dividendYield", None),
        "longBusinessSummary": info.get("longBusinessSummary", "")[:800]
    }
    try:
        fundamentals["financials"] = tk.financials.fillna("").to_dict()
        fundamentals["balance_sheet"] = tk.balance_sheet.fillna("").to_dict()
        fundamentals["cashflow"] = tk.cashflow.fillna("").to_dict()
    except Exception:
        fundamentals["financials"] = {}
        fundamentals["balance_sheet"] = {}
        fundamentals["cashflow"] = {}
    return fundamentals

# ---------------------------
# Quantitative analytics
# ---------------------------
def compute_quant_metrics(df: pd.DataFrame):
    metrics = {}
    returns = df["Return"].dropna()
    if returns.empty:
        return {}
    mean_daily = returns.mean()
    vol_daily = returns.std()
    metrics["annualized_return"] = ((1 + mean_daily) ** 252 - 1)
    metrics["annualized_volatility"] = vol_daily * np.sqrt(252)
    metrics["sharpe"] = metrics["annualized_return"] / metrics["annualized_volatility"] if metrics["annualized_volatility"] != 0 else np.nan
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    metrics["max_drawdown"] = drawdown.min()
    total_days = (df.index[-1] - df.index[0]).days
    total_return = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
    metrics["CAGR"] = (1 + total_return) ** (365 / total_days) - 1 if total_days > 0 else np.nan
    return metrics

# ---------------------------
# Price action patterns
# ---------------------------
def detect_price_action_patterns(df: pd.DataFrame, lookback: int = 50):
    patterns = []
    d = df.copy().tail(lookback)
    d["body"] = d["Close"] - d["Open"]
    d["range"] = d["High"] - d["Low"]
    for i in range(1, len(d)):
        cur = d.iloc[i]
        prev = d.iloc[i - 1]
        # Bullish engulfing
        if prev["Close"] < prev["Open"] and cur["Close"] > cur["Open"]:
            if (cur["Close"] > prev["Open"]) and (cur["Open"] < prev["Close"]):
                patterns.append({"date": cur.name, "pattern": "Bullish Engulfing"})
        # Bearish engulfing
        if prev["Close"] > prev["Open"] and cur["Close"] < cur["Open"]:
            if (cur["Open"] > prev["Close"]) and (cur["Close"] < prev["Open"]):
                patterns.append({"date": cur.name, "pattern": "Bearish Engulfing"})
        # Hammer / Shooting star
        if cur["range"] > 0:
            lower_wick = cur["Open"] - cur["Low"] if cur["body"] >= 0 else cur["Close"] - cur["Low"]
            upper_wick = cur["High"] - cur["Close"] if cur["body"] >= 0 else cur["High"] - cur["Open"]
            if lower_wick > 2 * abs(cur["body"]) and upper_wick < abs(cur["body"]):
                patterns.append({"date": cur.name, "pattern": "Hammer"})
            if upper_wick > 2 * abs(cur["body"]) and lower_wick < abs(cur["body"]):
                patterns.append({"date": cur.name, "pattern": "Shooting Star"})
    return patterns

# ---------------------------
# Signals (rules + model)
# ---------------------------
def make_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lag1"] = df["Close"].shift(1)
    df["lag2"] = df["Close"].shift(2)
    df["lag3"] = df["Close"].shift(3)
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["rsi14"] = ta.rsi(df["Close"], length=14)
    macd = ta.macd(df["Close"])
    for c in macd.columns:
        df[c] = macd[c]
    df["atr14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    feature_cols = ["lag1", "lag2", "lag3", "ma5", "ma10", "rsi14", "MACD_12_26_9", "MACDs_12_26_9", "atr14"]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    X = X.dropna()
    return X

def train_model(df: pd.DataFrame, test_size=0.2, random_state=42, n_estimators=100):
    X = make_features_for_model(df)
    y = df["Close"].shift(-1).loc[X.index]
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    if len(X) < 20:
        raise RuntimeError("Not enough data to train model. Increase period or interval.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"model": model, "scaler": scaler, "mae": mae, "r2": r2, "X_test_index": X_test.index, "y_test": y_test, "preds": preds}

def generate_signals(df: pd.DataFrame, model=None, prob_threshold=0.005):
    df = df.copy()
    df["signal"] = ""
    df["ma_signal"] = 0
    df.loc[df["SMA20"] > df["SMA50"], "ma_signal"] = 1
    df.loc[df["SMA20"] < df["SMA50"], "ma_signal"] = -1
    df["rsi_signal"] = 0
    df.loc[df["RSI14"] < 30, "rsi_signal"] = 1
    df.loc[df["RSI14"] > 70, "rsi_signal"] = -1
    df["macd_signal"] = 0
    if "MACD_12_26_9" in df.columns and "MACDs_12_26_9" in df.columns:
        df.loc[df["MACD_12_26_9"] > df["MACDs_12_26_9"], "macd_signal"] = 1
        df.loc[df["MACD_12_26_9"] < df["MACDs_12_26_9"], "macd_signal"] = -1
    df["rule_score"] = df["ma_signal"] + df["rsi_signal"] + df["macd_signal"]
    df["rule_signal"] = df["rule_score"].apply(lambda x: "Buy" if x >= 2 else ("Sell" if x <= -2 else "Neutral"))
    df["model_pred"] = np.nan
    if model is not None:
        try:
            X = make_features_for_model(df)
            if not X.empty:
                preds = model.predict(X.fillna(0))
                df.loc[X.index, "model_pred"] = preds
                df["ml_signal"] = df["model_pred"].diff().apply(lambda x: "Buy" if x > prob_threshold else ("Sell" if x < -prob_threshold else "Neutral"))
            else:
                df["ml_signal"] = "Neutral"
        except Exception:
            df["ml_signal"] = "Neutral"
    else:
        df["ml_signal"] = "Neutral"
    df["final_signal"] = df.apply(lambda row: row["ml_signal"] if row["ml_signal"] != "Neutral" else row["rule_signal"], axis=1)
    return df

# ---------------------------
# Risk management helpers
# ---------------------------
def position_sizing(account_size, risk_percent, entry_price, stop_loss_price):
    risk_amount = account_size * (risk_percent / 100.0)
    if entry_price == stop_loss_price:
        return {"size": 0, "risk_amount": risk_amount, "reason": "Entry equals stop loss"}
    per_share_risk = abs(entry_price - stop_loss_price)
    if per_share_risk == 0:
        return {"size": 0, "risk_amount": risk_amount, "reason": "zero per-share risk"}
    size = int(risk_amount / per_share_risk)
    return {"size": size, "risk_amount": risk_amount, "per_share_risk": per_share_risk}

# ---------------------------
# Plot helpers
# ---------------------------
def plot_candles_with_indicators(df: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    if "SMA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], line=dict(color="orange", width=1), name="SMA20"))
    if "SMA50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], line=dict(color="blue", width=1), name="SMA50"))
    if "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color="lightgray", name="Volume", yaxis="y2", opacity=0.4))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, position=0.15, title="Volume"))
    fig.update_layout(title=f"{ticker} Price Chart", xaxis_rangeslider_visible=False, height=700)
    return fig

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Indian Stocks PA & Predictor", layout="wide", initial_sidebar_state="expanded")
st.title("Indian Stocks — Price Action, Analytics & Prediction (GitHub + Render)")

with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker (NSE - e.g., RELIANCE or RELIANCE.NS)", value="RELIANCE")
    period = st.selectbox("History period", options=["6mo", "1y", "2y", "5y", "10y"], index=1)
    interval = st.selectbox("Interval", options=["1d", "1wk", "1mo"], index=0)
    run_model = st.checkbox("Enable ML predictor", value=True)
    retrain = st.checkbox("Retrain model now (may take time)", value=False)
    test_size = st.slider("ML test size (%)", 10, 40, 20)
    n_estimators = st.slider("RF estimators", 10, 300, 100)
    account_size = st.number_input("Account size (for position sizing, INR)", min_value=1000, value=100000)
    risk_percent = st.slider("Risk % per trade", 0.1, 10.0, 1.0)
    st.markdown("Disclaimer: This is for educational purposes only. Not financial advice.")
    st.markdown("Data source: Yahoo Finance (yfinance). Some fundamentals may be incomplete.")

try:
    df_raw = fetch_data(ticker, period=period, interval=interval)
    df = add_technical_indicators(df_raw)
except Exception as e:
    st.error(f"Error fetching or preparing data: {e}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
latest = df.iloc[-1]
col1.metric("Latest Close", f"{latest['Close']:.2f}", delta=f"{(latest['Close']/df['Close'].iloc[-2]-1)*100:.2f}%")
col2.metric("SMA20", f"{latest.get('SMA20', np.nan):.2f}")
col3.metric("RSI14", f"{latest.get('RSI14', np.nan):.2f}")
col4.metric("ATR14", f"{latest.get('ATR14', np.nan):.2f}")

tabs = st.tabs(["Home", "Technical Analysis", "Fundamentals", "Quantitative", "Price Action", "Prediction & Signals", "Risk Management", "Download"])
with tabs[0]:
    st.header("Overview & Quick Insights")
    st.markdown("""
    This app is deployed on the web (GitHub + Render) and provides:
    - Interactive candlestick charts with technical indicators
    - Simple price-action pattern detection
    - Fundamental metrics (via yfinance)
    - Quantitative analytics (Sharpe, volatility, drawdown)
    - Rule-based buy/sell signals + optional ML-based next-day close predictor (RandomForest)
    - Position sizing and stop-loss suggestions
    """)
    st.plotly_chart(plot_candles_with_indicators(df, ticker), use_container_width=True)
    signals_df_quick = generate_signals(df.tail(60))
    last_signal = signals_df_quick["final_signal"].iloc[-1]
    st.subheader(f"Latest signal: {last_signal}")
    st.dataframe(signals_df_quick[["Close", "SMA20", "SMA50", "RSI14", "final_signal"]].tail(10))

with tabs[1]:
    st.header("Technical Indicators & Charts")
    st.plotly_chart(plot_candles_with_indicators(df, ticker), use_container_width=True)
    st.subheader("Indicator Table (latest rows)")
    st.dataframe(df[["Close", "SMA20", "SMA50", "EMA12", "EMA26", "RSI14", "ATR14"]].tail(20))

with tabs[2]:
    st.header("Fundamental Analytics")
    fund = get_fundamentals(ticker)
    st.subheader(fund.get("shortName", ticker))
    st.write("Sector:", fund.get("sector", "N/A"), " — Industry:", fund.get("industry", "N/A"))
    st.write("Market Cap:", fund.get("marketCap", "N/A"))
    st.write("P/E (trailing):", fund.get("trailingPE", "N/A"), "Forward P/E:", fund.get("forwardPE", "N/A"))
    st.write("P/B:", fund.get("priceToBook", "N/A"), "Beta:", fund.get("beta", "N/A"))
    st.markdown("Company summary:")
    st.write(fund.get("longBusinessSummary", "N/A"))

with tabs[3]:
    st.header("Quantitative Analytics")
    metrics = compute_quant_metrics(df)
    if metrics:
        st.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
        st.metric("Annualized Volatility", f"{metrics['annualized_volatility']:.2%}")
        st.metric("Sharpe (rf=0)", f"{metrics['sharpe']:.2f}")
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        st.metric("CAGR", f"{metrics['CAGR']:.2%}")
    else:
        st.info("Not enough return data for quantitative metrics.")
    st.subheader("Price distribution & returns histogram")
    st.bar_chart(df["Return"].dropna().tail(252))

with tabs[4]:
    st.header("Price Action Patterns")
    patterns = detect_price_action_patterns(df, lookback=120)
    if patterns:
        st.write("Detected patterns (recent)")
        pat_df = pd.DataFrame(patterns).sort_values("date", ascending=False).head(20)
        st.dataframe(pat_df)
    else:
        st.info("No common patterns detected in the lookback period.")
    st.subheader("Recent candles")
    st.dataframe(df[["Open", "High", "Low", "Close", "Volume"]].tail(20))

with tabs[5]:
    st.header("Signals & ML Prediction")
    model_obj = None
    model_loaded = False
    safe_name = ticker.replace(".", "_")
    model_filename = f"rf_model_{safe_name}.joblib"
    scaler_filename = f"scaler_{safe_name}.joblib"

    if run_model:
        try:
            if retrain:
                st.info("Training model... This may take a minute depending on data and n_estimators.")
                with st.spinner("Training RandomForest..."):
                    result = train_model(df, test_size=int(test_size)/100.0, n_estimators=n_estimators)
                # Save to local disk (ephemeral on Render)
                joblib.dump(result["model"], model_filename)
                joblib.dump(result["scaler"], scaler_filename)
                st.success(f"Trained RF model. MAE: {result['mae']:.4f}, R2: {result['r2']:.4f}")
                model_obj = result["model"]
                model_loaded = True
            else:
                try:
                    model_obj = joblib.load(model_filename)
                    # scaler may be used later if needed
                    model_loaded = True
                    st.info("Loaded saved model from disk.")
                except Exception:
                    st.warning("No saved model found. Train model to enable ML predictions (tick retrain).")
                    model_loaded = False
        except Exception as e:
            st.error("Model training error: " + str(e))
            model_loaded = False

    signals_df = generate_signals(df, model=model_obj if model_loaded else None)
    st.subheader("Latest signals (last 30 rows)")
    st.dataframe(signals_df[["Close", "SMA20", "SMA50", "RSI14", "final_signal"]].tail(30))

    if model_loaded:
        X_all = make_features_for_model(df)
        if not X_all.empty:
            last_index = X_all.index[-1]
            X_last = X_all.loc[[last_index]]
            try:
                scaler = joblib.load(scaler_filename)
                X_last_scaled = scaler.transform(X_last.fillna(0))
                pred = model_obj.predict(X_last_scaled)[0]
                st.metric("Model next-day predicted Close", f"{pred:.2f}", delta=f"{(pred/df['Close'].iloc[-1]-1)*100:.2f}%")
            except Exception:
                st.info("Scaler/prediction unavailable for last row.")
        else:
            st.info("Insufficient features to predict.")

with tabs[6]:
    st.header("Risk Management & Position Sizing")
    st.write("Estimate position size given an entry and a stop loss.")
    entry_price = st.number_input("Entry price (INR)", value=float(df["Close"].iloc[-1]))
    stop_loss = st.number_input("Stop loss price (INR)", value=float(df["Close"].iloc[-1] * 0.97))
    sizing = position_sizing(account_size, float(risk_percent), float(entry_price), float(stop_loss))
    st.write("Position sizing result:")
    st.json(sizing)
    st.markdown("Suggested stop loss must be set based on your strategy, volatility and timeframe.")

with tabs[7]:
    st.header("Download / Export")
    st.markdown("Download the recent processed data with indicators")
    csv = df.to_csv().encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=f"{ticker}_data.csv", mime="text/csv")
    st.markdown("Download trained model (if exists)")
    try:
        with open(model_filename, "rb") as f:
            model_bytes = f.read()
        st.download_button("Download RF model", data=model_bytes, file_name=model_filename)
    except Exception:
        st.info("No model saved yet. Retrain to enable model download.")

st.markdown("---")
st.caption("Built for demo / educational use. Validate before real trading. Data via Yahoo Finance (yfinance).")
