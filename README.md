# 📈 Automated Equity Research & Valuation Engine
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![Finance](https://img.shields.io/badge/Domain-Fintech-green)

🚀 **Live Demo:** [Click here to use the Engine](https://automated-equity-research-valuation-engine.streamlit.app/)
A professional-grade equity research dashboard that converts complex financial data into a clean, **1-page visual report**. 

Built for investors and analysts, this engine automates the "grunt work" of fundamental analysis. It fetches real-time data, applies heuristic valuation models, and renders high-resolution infographics suitable for investment memos or quick decision-making.

---

## 🚀 Key Features

* **Global Coverage:** Supports stocks from **US** (NYSE/NASDAQ), **India** (NSE/BSE), **UK**, **Germany**, and **Japan**.
* **Dual-Score Algorithms:**
    * **🟢 Undervalued Score:** Identifies value plays based on P/E, P/B, and ROE.
    * **🔵 Multibagger Score:** Identifies high-growth potential based on Revenue Growth, Capital Efficiency, and Market Cap size.
* **Visual Intelligence:**
    * **Strength Radar:** A polar chart visualizing the balance between Growth, Valuation, Profitability, and Balance Sheet health.
    * **Trend Analysis:** Interactive 1-year price history overlaid with Moving Averages.
* **Export Ready:** One-click download of the analysis as a high-resolution PNG infographic.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (Custom CSS for "Dark/Light" fintech theming)
* **Data Pipeline:** `yfinance` (Yahoo Finance API)
* **Visualization:** `Matplotlib` & `Seaborn` (Custom GridSpec layouts)
* **Data Processing:** `Pandas` & `NumPy`

---

## ⚙️ Installation & Setup

Follow these steps to run the engine locally on your machine.

### Prerequisites
* Python 3.8 or higher
* VS Code (Recommended) or Terminal

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/equity-research-engine.git](https://github.com/yourusername/equity-research-engine.git)
cd equity-research-engine
