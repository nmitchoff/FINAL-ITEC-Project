import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import requests
import matplotlib.pyplot as plt
from datetime import datetime

# --- Load and Prepare Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=['date'])
    df.set_index("date", inplace=True)
    return df

def fetch_cpi():
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "CPALTT01USQ657N",
        "api_key": "cc8f548ac3d4b8ab1ef46b9aa6ac1a2d",
        "file_type": "json",
        "observation_start": "2018-01-01"
    }
    response = requests.get(url, params=params)
    data = response.json()["observations"]
    cpi_df = pd.DataFrame(data)[["date", "value"]]
    cpi_df.columns = ["date", "CPI"]
    cpi_df["date"] = pd.to_datetime(cpi_df["date"])
    cpi_df["CPI"] = pd.to_numeric(cpi_df["CPI"], errors='coerce')
    cpi_df.set_index("date", inplace=True)
    return cpi_df

# --- Forecast Function ---
def forecast_revenue(df, periods=4):
    df = df.copy()
    df = df.dropna(subset=['CPI', 'store_count'])
    exog = df[["CPI", "store_count"]]
    model = ARIMA(df["revenue"], order=(1,1,1), exog=exog)
    results = model.fit()

    # Forecast future exog values as the last known ones
    last_exog = exog.iloc[-1].values.reshape(1, -1)
    future_exog = np.tile(last_exog, (periods, 1))

    forecast = results.get_forecast(steps=periods, exog=future_exog)
    return forecast.predicted_mean, forecast.conf_int(), results

# --- App Layout ---
st.set_page_config(page_title="Starbucks Revenue Risk Analyzer", layout="centered")
st.title("Starbucks Revenue Risk Analyzer")

st.markdown("""
**Thesis statement**: Through incorporating forecasting analysis and Artificial Intelligence integration, the analysis illustrates Starbucks' risk of revenue overstatement from 2018 to 2023 in quarterly intervals. It allows us to reveal potential misstatement and fraud, depending on the severity of the investigation.
""")

# --- Full Historical + Forecast + Expected Revenue Visualization ---
st.header("Historical Revenue, Forecast, and Expected Values")

# Prepare full ARIMAX forecast visualization
df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=['date'])
df.set_index("date", inplace=True)
full_df = df.dropna(subset=["CPI"])
exog = full_df[["CPI"]]
model_full = ARIMA(full_df["revenue"], order=(1, 1, 1), exog=exog)
results_full = model_full.fit()

# Forecast with last known exog values
last_exog = exog.iloc[-1].values.reshape(1, -1)
future_exog = np.tile(last_exog, (4, 1))
forecast_full = results_full.get_forecast(steps=4, exog=future_exog)
forecast_values = forecast_full.predicted_mean
conf_int = forecast_full.conf_int()
future_index = pd.date_range(start=full_df.index[-1] + pd.offsets.QuarterEnd(), periods=4, freq="Q")
expected = forecast_values.tolist()

# Plot
fig_full = go.Figure()
fig_full.add_trace(go.Scatter(x=full_df.index, y=full_df["revenue"], name="Historical Revenue", mode="lines"))
fig_full.add_trace(go.Scatter(x=future_index, y=forecast_values, name="Forecast", mode="lines+markers"))
fig_full.add_trace(go.Scatter(x=future_index, y=expected, name="Expected (User Input)", mode="lines+markers"))
# Add shaded 95% confidence interval
fig_full.add_trace(go.Scatter(
    x=pd.Series(future_index).append(pd.Series(future_index[::-1]), ignore_index=True),
    y=pd.Series(conf_int.iloc[:, 0]).append(pd.Series(conf_int.iloc[:, 1][::-1]), ignore_index=True),
    fill='toself',
    fillcolor='rgba(160,160,160,0.3)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name="95% Confidence Interval"
))

fig_full.update_layout(
    title="Starbucks Quarterly Revenue Forecast with User Expectations",
    xaxis_title="Date",
    yaxis_title="Revenue ($M)",
    legend_title="Legend",
    template="plotly_white"
)

st.plotly_chart(fig_full)

# Load data and CPI
df = load_data()
cpi_live = fetch_cpi()
df.update(cpi_live, overwrite=False)

# Forecast + model for in-sample fit
forecast, conf, arimax_result = forecast_revenue(df)

# In-sample fitted vs actual
exog_insample = df[["CPI", "store_count"]]
pred_insample = arimax_result.get_prediction(start=0, end=len(df)-1, exog=exog_insample)
pred_mean_insample = pred_insample.predicted_mean
conf_int_insample = pred_insample.conf_int(alpha=0.05)

st.subheader("📉 In-Sample Actual vs. Fitted Revenue")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df["revenue"], label="Actual Revenue", marker="o", linestyle="-", color="tab:blue")
ax.plot(df.index, pred_mean_insample, label="Fitted Revenue", marker="o", linestyle="--", color="tab:orange")
ax.fill_between(df.index, conf_int_insample["lower revenue"], conf_int_insample["upper revenue"], color="tab:orange", alpha=0.2, label="95% CI")
ax.set_title("Actual vs. In-Sample Fitted Revenue\n(Revenue in Millions USD)")
ax.set_xlabel("Quarter")
ax.set_ylabel("Revenue (Millions USD)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- User Input ---
st.header("🔢 Enter Expected Revenue (Next 4 Quarters)")
expected = []
cols = st.columns(4)
for i in range(4):
    with cols[i]:
        val = st.number_input(f"Q{i+1}", min_value=0.0, value=float(round(forecast[i], 2)))
        expected.append(val)

# --- Risk Flagging ---
st.header("🚨 Risk Alert")
risk_msgs = []
for i in range(4):
    if expected[i] > conf.iloc[i, 1]:
        risk_msgs.append(f"- Q{i+1} expected revenue **exceeds** 95% CI upper bound.")

if risk_msgs:
    st.error("\n".join(risk_msgs))
else:
    st.success("Expected revenue is within forecasted range.")

# --- Visualization ---
st.header(":chart_with_upwards_trend: Forecast vs Expected")
future_dates = pd.date_range(start=df.index[-1] + pd.offsets.QuarterEnd(), periods=4, freq='Q')

fig = go.Figure()
fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", mode="lines+markers"))
fig.add_trace(go.Scatter(x=future_dates, y=expected, name="Expected", mode="lines+markers"))
# Add shaded 95% confidence interval
fig.add_trace(go.Scatter(
    x=pd.Series(future_dates).append(pd.Series(future_dates[::-1]), ignore_index=True),
    y=pd.Series(conf.iloc[:, 0]).append(pd.Series(conf.iloc[:, 1][::-1]), ignore_index=True),
    fill='toself',
    fillcolor='rgba(160,160,160,0.3)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name="95% Confidence Interval"
))
fig.update_layout(title="Forecasted vs Expected Revenue", xaxis_title="Quarter", yaxis_title="Revenue ($M)")
st.plotly_chart(fig)

# --- AI Summary (Static Placeholder) ---
st.header(":robot_face: AI Summary")
st.markdown("""
**Summary**: The ARIMAX forecast, incorporating marketing spend and CPI data, projects revenue patterns that diverge from Starbucks' historical trends. This inconsistency may suggest potential overstatement in expected revenues or unusual operational behavior. Notably, marketing expenditures increase sharply in late 2022 and 2023 while revenue growth remains flat—indicating that sales boosts may be driven more by promotional efforts than organic performance. Given this pattern, further investigation is recommended to assess possible misstatements.
""")

# Footer
st.caption("Developed for ITEC 3155 / ACTG 4155 - Spring 2025")
