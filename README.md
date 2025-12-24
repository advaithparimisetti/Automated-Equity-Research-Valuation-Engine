# 📈 Automated Equity Research & Valuation Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B)
![Finance](https://img.shields.io/badge/Domain-Fintech-green)
![Status](https://img.shields.io/badge/Status-Active-success)

A professional-grade equity research platform that acts as an automated financial analyst. 

This tool converts complex financial data into actionable insights, performing institutional-grade valuation (DCF), risk assessment (VaR), and fundamental scoring. It generates **comprehensive PDF reports** and visual infographics suitable for investment memos or quick decision-making.

---

## 🚀 Key Features

### 📊 1. Core Fundamental Analysis
* **Dual-Score Algorithms:** * **Undervalued Score (0-40):** Detects value plays using P/E, P/B, ROE, and Debt/Equity metrics relative to sector benchmarks.
    * **Multibagger Score (0-50):** Identifies high-growth potential based on Revenue Growth, Capital Efficiency (ROIC), and Earnings Acceleration.
* **Visual Infographic:** Auto-generates a one-page "Cheatsheet" with a Strength Radar and Gauge charts.

### 💰 2. Institutional Valuation (DCF Models)
* **Automated DCF:** Automatically calculates WACC (Weighted Average Cost of Capital) and projects Free Cash Flows.
* **3-Way Modeling:**
    * **Perpetual Growth Method:** Standard academic valuation.
    * **Exit Multiple Method:** Private Equity style valuation (based on EBITDA multiples).
    * **Sensitivity Matrix:** A "Banker’s View" heatmap showing value across different WACC and Growth rate assumptions.

### ⚖️ 3. Risk Management Engine
* **Quantitative Metrics:** Calculates **Beta** (Volatility relative to market), **VaR (Value at Risk)**, and Maximum Drawdown.
* **Distribution Analysis:** Visualizes the "Fat Tail" risk of the stock using historical return histograms.

### 📝 4. Professional Reporting
* **PDF Report Generator:** Generates a multi-page, watermarked PDF equity research report including all charts, metrics, and analyst verdicts.
* **Excel Export:** Downloads raw financial statements and calculated models for further analysis.

### 🌍 5. Global Coverage
* **Multi-Region Support:** Native support for **US** (NYSE/NASDAQ), **India** (NSE/BSE), **UK**, **Germany**, **France**, **Canada**, and **Japan**.
* **Currency Normalization:** Option to convert all metrics to USD for standardized comparison.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (Custom CSS for "Dark/Fintech" aesthetic)
* **Data Pipeline:** `yfinance` (Yahoo Finance API) & `BeautifulSoup` (Google News scraping)
* **Financial Modeling:** `NumPy` & `Pandas` (Vectorized calculations for WACC & DCF)
* **Visualization:** * `Plotly` (Interactive candlestick charts, heatmaps)
    * `Matplotlib` & `Seaborn` (Static report charts, distribution curves)
* **Reporting:** `ReportLab` (Pixel-perfect PDF generation)

---

## ⚙️ Installation & Setup

Follow these steps to run the engine locally on your machine.

### Prerequisites
* Python 3.8 or higher

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/equity-research-engine.git](https://github.com/yourusername/equity-research-engine.git)
cd equity-research-engine
