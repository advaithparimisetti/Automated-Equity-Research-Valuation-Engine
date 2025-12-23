import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
import traceback
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import re
import html

# --- DEPENDENCY CHECK ---
try:
    import openpyxl
except ImportError:
    st.error("⚠️ The 'openpyxl' library is missing. Please run `pip install openpyxl` to enable Excel exports.")

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

sns.set_style("whitegrid")

TICKER_CORRECTIONS = {
    "SHELL": "SHEL",
    "GOOG": "GOOGL",
    "BRK.B": "BRK-B",
    "FB": "META",
    "TWTR": "X"
}

COUNTRY_SUFFIX_MAP = {
    "IN": [".NS", ".BO"],
    "US": [""],
    "GB": [".L"],
    "DE": [".DE"],
    "FR": [".PA"],
    "CA": [".TO"],
    "AU": [".AX"],
    "HK": [".HK"],
    "JP": [".T"],
    "SG": [".SI"],
    "CH": [".SW"],
    "NL": [".AS"],
    "SE": [".ST"],
    "IT": [".MI"],
}

_CURRENCY_SYMBOLS = {
    "INR": "₹", "USD": "$", "EUR": "€", "GBP": "£",
    "JPY": "¥", "HKD": "HK$", "AUD": "A$", "CAD": "C$",
    "SGD": "S$", "CHF": "CHF ",
}

def _get_currency_symbol(currency_code):
    if not currency_code: return ""
    return _CURRENCY_SYMBOLS.get(currency_code.upper(), currency_code.upper() + " ")

@st.cache_data(ttl=3600, show_spinner=False)
def get_currency_rate(from_currency, to_currency="USD"):
    if not from_currency or from_currency.upper() == to_currency.upper():
        return 1.0
    pair = f"{from_currency.upper()}{to_currency.upper()}=X"
    try:
        ticker = yf.Ticker(pair)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
    except:
        pass
    return 1.0

def format_large_number(n):
    if n is None: return "N/A"
    try:
        n = float(n)
        if abs(n) >= 1e12: return f"{n/1e12:.2f}T"
        if abs(n) >= 1e9: return f"{n/1e9:.2f}B"
        if abs(n) >= 1e6: return f"{n/1e6:.2f}M"
        if abs(n) >= 1e3: return f"{n/1e3:.2f}K"
        return f"{n:,.2f}"
    except: return str(n)

def clean_financial_df(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df_T = df.transpose()
    df_T = df_T[sorted(df_T.columns, reverse=True)]
    df_T.columns = [d.strftime('%Y-%m-%d') if isinstance(d, datetime) else str(d) for d in df_T.columns]
    for col in df_T.columns:
        df_T[col] = df_T[col].apply(lambda x: format_large_number(x) if isinstance(x, (int, float)) else x)
    return df_T

def human_readable_inr(value):
    if value is None: return "n/a"
    try:
        cr = value / 1e7
        return f"₹{cr:,.2f} Cr"
    except Exception: return str(value)

def human_readable_generic(value, currency_code=None):
    if value is None: return "n/a"
    try:
        sym = _get_currency_symbol(currency_code)
        n = float(value)
        if abs(n) >= 1e12: return f"{sym}{n/1e12:.2f}T"
        if abs(n) >= 1e9: return f"{sym}{n/1e9:.2f}B"
        if abs(n) >= 1e6: return f"{sym}{n/1e6:.2f}M"
        if abs(n) >= 1e3: return f"{sym}{n/1e3:.2f}K"
        return f"{sym}{n:.2f}"
    except Exception: return str(value)

def human_readable_marketcap(value, currency_code=None):
    if value is None: return "n/a"
    currency = (currency_code or "").upper()
    if currency in ("INR", "₹"):
        local_str = human_readable_inr(value)
    else:
        local_str = human_readable_generic(value, currency)
    return local_str

def human_readable_price(price, currency_code=None):
    if price is None: return "n/a"
    sym = _get_currency_symbol(currency_code)
    if currency_code and currency_code.upper() == "INR": return f"{sym}{price:,.2f}"
    if currency_code and currency_code.upper() in ("JPY",): return f"{sym}{price:,.0f}"
    return f"{sym}{price:,.2f}"

def safe_numeric(x):
    if x is None: return None
    try:
        if isinstance(x, str):
            s = x.strip().replace(",", "")
            if s.endswith("%"): return float(s.rstrip("%"))/100.0
            return float(s)
        if isinstance(x, (int, float, np.floating, np.integer)): return float(x)
    except: return None
    return None

def format_pct_for_display(x):
    if x is None: return "n/a"
    try: return f"{x*100:.2f}%"
    except: return str(x)

def get_yf_ticker_variants(base_ticker, country_code):
    variants = []
    if "." in base_ticker: variants.append(base_ticker)
    suffixes = COUNTRY_SUFFIX_MAP.get(country_code, [])
    for s in suffixes: variants.append(f"{base_ticker}{s}")
    variants.append(base_ticker)
    if base_ticker.upper() not in variants: variants.append(base_ticker.upper())
    seen = set(); out = []
    for v in variants:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info_with_variants(base_ticker, country_code):
    tried = []
    variants = get_yf_ticker_variants(base_ticker, country_code)
    for variant in variants:
        try:
            t = yf.Ticker(variant)
            info = t.info
            if info and (info.get("regularMarketPrice") is not None or info.get("symbol") or info.get("shortName")):
                return info, variant
        except Exception:
            tried.append(variant)
    raise RuntimeError(f"No usable yfinance data found. Tried: {variants}")

def fetch_screener_in(ticker_base):
    try:
        url = f"https://www.screener.in/company/{ticker_base}/"
        headers = {"User-Agent":"Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200: return {}
        soup = BeautifulSoup(r.content, "html.parser")
        facts = {}
        for div in soup.select(".snapshot .data"):
            spans = div.select("span")
            if len(spans) >= 2:
                label = spans[0].get_text(strip=True).rstrip(':')
                value = spans[1].get_text(strip=True)
                facts[label] = value
        return facts
    except: return {}

def fetch_news_fallback(ticker):
    news_items = []
    try:
        query = f"{ticker} stock news"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, headers=headers, timeout=6)
        soup = BeautifulSoup(r.content, features="xml")
        items = soup.findAll('item')
        for item in items[:5]: 
            try:
                title = item.title.text
                link = item.link.text
                pub_date = item.pubDate.text
                try:
                    dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                    date_str = dt.strftime("%Y-%m-%d")
                except:
                    date_str = "Recent"
                news_items.append({'title': title, 'link': link, 'publisher': 'Google News', 'providerPublishTime': date_str})
            except: continue
    except Exception: pass
    return news_items

# --- SCORING LOGIC ---
def score_undervalued_from_info(info):
    score = 0
    details = {}
    pe = safe_numeric(info.get("trailingPE"))
    pb = safe_numeric(info.get("priceToBook"))
    roe = safe_numeric(info.get("returnOnEquity"))
    roic = safe_numeric(info.get("returnOnInvestment")) or safe_numeric(info.get("returnOnAssets")) or roe
    dte = safe_numeric(info.get("debtToEquity"))
    rev_growth = safe_numeric(info.get("revenueGrowth"))
    div_yield = safe_numeric(info.get("dividendYield"))
    details.update({"pe": pe, "pb": pb, "roe": roe, "roic": roic, "debtToEquity": dte, "revenueGrowth": rev_growth, "dividendYield": div_yield})

    if pe and pe > 0:
        if pe < 15: score += 8
        elif pe < 25: score += 4
    if pb and pb > 0:
        if pb < 1.5: score += 6
        elif pb < 3: score += 3
    if roe is not None:
        if roe > 0.15: score += 8
        elif roe > 0.08: score += 4
    if roic is not None and roic > 0.12: score += 6
    if dte is not None:
        if dte < 0.5: score += 5
        elif dte < 1.0: score += 2
    if rev_growth is not None and rev_growth > 0.05: score += 5
    if div_yield is not None and div_yield > 0.02: score += 2
    return score, details

def score_multibagger_from_info(info):
    score = 0
    details = {}
    market_cap = safe_numeric(info.get("marketCap"))
    eps_growth = safe_numeric(info.get("earningsQuarterlyGrowth"))
    roe = safe_numeric(info.get("returnOnEquity"))
    roic = safe_numeric(info.get("returnOnInvestment")) or roe
    payout = safe_numeric(info.get("payoutRatio"))
    rev_growth = safe_numeric(info.get("revenueGrowth"))
    gross = safe_numeric(info.get("grossMargins"))
    opm = safe_numeric(info.get("operatingMargins"))
    details.update({"marketCap": market_cap, "eps_growth": eps_growth, "roe": roe, "roic": roic, "payoutRatio": payout, "revenueGrowth": rev_growth, "grossMargins": gross, "operatingMargins": opm})

    if market_cap is not None and market_cap < 2e9: score += 7
    if eps_growth is not None:
        if eps_growth > 0.20: score += 10
        elif eps_growth > 0.10: score += 5
    if roe is not None:
        if roe > 0.20: score += 7
        elif roe > 0.12: score += 3
    if roic is not None and roic > 0.15: score += 3
    if payout is not None:
        if payout < 0.2: score += 6
        elif payout < 0.5: score += 2
    if rev_growth is not None:
        if rev_growth > 0.15: score += 7
        elif rev_growth > 0.07: score += 3
    if gross is not None and gross > 0.30: score += 3
    if opm is not None and opm > 0.15: score += 3
    return score, details

@st.cache_data(ttl=3600, show_spinner=False)
def run_playbook_for_ticker(ticker_input, country_code=""):
    try:
        info, used_variant = fetch_info_with_variants(ticker_input, country_code)
    except Exception as e:
        return None

    screener_facts = {}
    if country_code and country_code.upper() == "IN":
        screener_facts = fetch_screener_in(ticker_input)

    u_score, u_details = score_undervalued_from_info(info)
    m_score, m_details = score_multibagger_from_info(info)

    summary = {
        "ticker_requested": ticker_input,
        "ticker_used": used_variant,
        "company": info.get("shortName") or info.get("longName") or used_variant,
        "price": safe_numeric(info.get("regularMarketPrice")),
        "price_currency": info.get("currency"),
        "marketCap": safe_numeric(info.get("marketCap")),
        "marketCap_currency": info.get("currency"),
        "sector": info.get("sector") or info.get("industry") or screener_facts.get("Industry"),
        "undervalued_score": u_score,
        "undervalued_details": u_details,
        "multibagger_score": m_score,
        "multibagger_details": m_details,
        "decisions": {"undervalued_pass": u_score >= 25, "multibagger_pass": m_score >= 30},
        "fetched_info": info,
        "as_of": datetime.now(timezone.utc).isoformat() + "Z"
    }
    return summary

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

# --- PLOTTING UTILS ---
def draw_kpi(ax, title, value, subtitle=None, fontsize=12, small=False):
    ax.axis('off')
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, color='white', ec='#e6e6e6', lw=1, zorder=0))
    ax.text(0.5, 0.65, title, ha='center', va='center', fontsize=12 if not small else 10, color='#333')
    ax.text(0.5, 0.35, value, ha='center', va='center', fontsize=18 if not small else 14, fontweight='bold', color='#111')
    if subtitle:
        ax.text(0.5, 0.12, subtitle, ha='center', va='center', fontsize=9 if not small else 8, color='#666')

def gauge_bar(ax, pct, label, bgcolor='#f3f3f3'):
    ax.axis('off')
    ax.add_patch(patches.Rectangle((0, 0.25), 1, 0.5, color=bgcolor, zorder=0, ec='none', alpha=1.0))
    pct_clamped = max(0.0, min(1.0, pct))
    cmap_map = plt.get_cmap('RdYlGn')
    color = cmap_map(pct_clamped)
    ax.add_patch(patches.Rectangle((0, 0.25), pct_clamped, 0.5, color=color, zorder=1))
    ax.text(0.01, 0.75, label, ha='left', va='bottom', fontsize=10, color='#333')
    ax.text(0.99, 0.75, f"{int(pct_clamped*100)}%", ha='right', va='bottom', fontsize=12, fontweight='bold', color='#111')

def radar_plot(ax, labels, values, title=None):
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals = values + values[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11, color='#333', fontweight='bold')
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.plot(angles, vals, linewidth=2, linestyle='solid', color='#2b7fc1')
    ax.fill(angles, vals, color='#2b7fc1', alpha=0.25)
    if title:
        ax.set_title(title, y=1.1, fontsize=12, fontweight='bold', color='#111')

def fmt_pct_val(x):
    try: return f"{x*100:.1f}%"
    except: return "n/a"

def render_infographic(playbook_result, convert_usd=False, rate=1.0):
    if playbook_result is None: return None
    s = playbook_result
    
    price = s.get('price')
    mcap = s.get('marketCap')
    price_ccy = s.get('price_currency')
    mcap_ccy = s.get('marketCap_currency')
    
    if convert_usd:
        if price: price = price * rate
        if mcap: mcap = mcap * rate
        price_ccy = "USD"
        mcap_ccy = "USD"

    company = s.get('company') or s.get('ticker_used')
    ticker_used = s.get('ticker_used')
    sector = s.get('sector') or "N/A"
    u_score = s.get('undervalued_score') or 0
    m_score = s.get('multibagger_score') or 0
    u_pct = max(0, min(1, u_score / 40.0))
    m_pct = max(0, min(1, m_score / 50.0))
    u_details = s.get('undervalued_details', {})
    m_details = s.get('multibagger_details', {})
    
    fig = plt.figure(constrained_layout=True, figsize=(14,10))
    gs = GridSpec(6, 6, figure=fig)
    ax_header = fig.add_subplot(gs[0, :4])
    ax_right = fig.add_subplot(gs[0, 4:])
    ax_kpi_1 = fig.add_subplot(gs[1, 0])
    ax_kpi_2 = fig.add_subplot(gs[1, 1])
    ax_kpi_3 = fig.add_subplot(gs[1, 2])
    ax_kpi_4 = fig.add_subplot(gs[1, 3])
    ax_gauge_u = fig.add_subplot(gs[2, 0:3])
    ax_gauge_m = fig.add_subplot(gs[2, 3:6])
    ax_radar = fig.add_subplot(gs[3:5, 0:3], polar=True)
    ax_checklist_panels = [fig.add_subplot(gs[3+i, 3:6]) for i in range(2)]
    ax_price = fig.add_subplot(gs[5, 0:6])

    ax_header.axis('off')
    ax_header.text(0.01, 0.7, company, fontsize=20, fontweight='bold', color='#111')
    ax_header.text(0.01, 0.48, f"{ticker_used}   |   Sector: {sector}", fontsize=11, color='#444')
    ax_header.text(0.01, 0.28, f"Price: {human_readable_price(price, price_ccy)}    MarketCap: {human_readable_marketcap(mcap, mcap_ccy)}", fontsize=11, color='#333')
    ax_header.text(0.01, 0.08, f"Report generated: {datetime.now().strftime('%Y-%m-%d')} | Analysis by Sai Advaith Parimisetti", fontsize=9, color='#666')
    ax_header.text(0.01, -0.15, "Disclaimer: For informational purposes only. Not financial advice.", fontsize=7, color='#999', ha='left')

    ax_right.axis('off')
    ax_right.add_patch(patches.Rectangle((0.02, 0.12), 0.96, 0.76, color='#fafafa', ec='#e6e6e6'))
    ax_right.text(0.5, 0.78, "Core Snapshot", ha='center', fontsize=11, fontweight='bold')
    ax_right.text(0.07, 0.62, f"P/E: {u_details.get('pe') if u_details.get('pe') else 'n/a'}", fontsize=10)
    ax_right.text(0.07, 0.50, f"P/B: {u_details.get('pb') if u_details.get('pb') else 'n/a'}", fontsize=10)
    ax_right.text(0.07, 0.38, f"ROE: {fmt_pct_val(u_details.get('roe'))}", fontsize=10)
    ax_right.text(0.07, 0.26, f"Rev Growth: {fmt_pct_val(u_details.get('revenueGrowth'))}", fontsize=10)

    draw_kpi(ax_kpi_1, "Price", human_readable_price(price, price_ccy))
    draw_kpi(ax_kpi_2, "Market Cap", human_readable_marketcap(mcap, mcap_ccy))
    draw_kpi(ax_kpi_3, "Undervalued", f"{u_score}/40", subtitle=f"{int(u_pct*100)}%")
    draw_kpi(ax_kpi_4, "Multibagger", f"{m_score}/50", subtitle=f"{int(m_pct*100)}%")
    gauge_bar(ax_gauge_u, u_pct, "Undervalued Score")
    gauge_bar(ax_gauge_m, m_pct, "Multibagger Score")

    valuation_comp = 0.0
    pe_val = u_details.get('pe')
    if pe_val and pe_val > 0: valuation_comp = max(0, min(1, (40 - pe_val) / 40))
    profitability_comp = 0.0
    roe_val = u_details.get('roe')
    if roe_val: profitability_comp = max(0, min(1, roe_val / 0.25))
    growth_comp = 0.0
    revg_val = u_details.get('revenueGrowth')
    if revg_val: growth_comp = max(0, min(1, revg_val / 0.25))
    balance_comp = 0.0
    dte_val = u_details.get('debtToEquity')
    if dte_val: balance_comp = max(0, min(1, 1 - (dte_val / 2.0)))
    moat_comp = 0.0
    gm = m_details.get('grossMargins')
    if gm: moat_comp = max(0, min(1, gm / 0.60))

    labels = ["Valuation", "Profitability", "Growth", "Balance", "Moat"]
    values = [valuation_comp, profitability_comp, growth_comp, balance_comp, moat_comp]
    radar_plot(ax_radar, labels, values, title="Strength Radar")

    checklist = [
        ("Price < Intrinsic", s['decisions']['undervalued_pass']),
        ("Debt/Equity < 0.5", (u_details.get('debtToEquity') is not None and u_details.get('debtToEquity') < 0.5)),
        ("ROE > 15%", (u_details.get('roe') is not None and u_details.get('roe') > 0.15)),
        ("Rev growth > 5%", (u_details.get('revenueGrowth') is not None and u_details.get('revenueGrowth') > 0.05)),
        ("Small-cap < $2B", (m_details.get('marketCap') is not None and m_details.get('marketCap') < 2e9)),
        ("EPS Accel > 20%", (m_details.get('eps_growth') is not None and m_details.get('eps_growth') > 0.2)),
    ]

    for axc in ax_checklist_panels: axc.axis('off')
    half = (len(checklist)+1)//2
    for idx, (label, passed) in enumerate(checklist):
        col = 0 if idx < half else 1
        row_idx = idx if idx < half else idx - half
        ax_pos = ax_checklist_panels[col]
        y_top = 0.9 - row_idx * 0.18
        color = '#1aab2a' if passed else '#d93f3f'
        ax_pos.add_patch(patches.FancyBboxPatch((0.05, y_top-0.12), 0.9, 0.14, boxstyle="round,pad=0.02", color=color, ec=color))
        ax_pos.text(0.07, y_top-0.05, label, va='center', ha='left', color='white', fontsize=10)

    try:
        hist = yf.Ticker(ticker_used).history(period="12mo")
        if hist is not None and not hist.empty:
            if convert_usd: hist['Close'] = hist['Close'] * rate
            ax_price.plot(hist.index, hist['Close'], label='Close', lw=1.2)
            ax_price.set_title(f"Price History (12mo) — {ticker_used}")
            ax_price.legend()
        else:
            ax_price.text(0.5, 0.5, "Price history not available", ha='center', va='center')
    except:
        ax_price.text(0.5, 0.5, "Price history fetch failed", ha='center', va='center')

    plt.suptitle(f"{company} — Playbook Infographic", fontsize=16, fontweight='bold', y=0.995)
    return fig

# ==========================================
# 2. STREAMLIT APP UI
# ==========================================

st.set_page_config(page_title="Automated Equity Research & Valuation Engine", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
.stApp{font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;}
.stApp { background: linear-gradient(180deg,#071014 0%,#0f1724 100%); }
h1 { color: #9EE6A8; font-weight:800; font-size:44px; margin-bottom:4px; }
h2, h3 { color:#81C784; font-weight:600; }
body, p, span { color:#E6F0EA; font-size:15px; line-height:1.6; }
div.stButton > button:first-child{ background: linear-gradient(90deg,#2E8B45 0%,#173C20 100%); color:#fff; font-weight:700; border-radius:10px; padding:0.7rem 1.4rem; box-shadow:0 6px 20px rgba(0,0,0,0.5); }
div.stButton > button:first-child:hover{ transform: translateY(-3px); box-shadow:0 10px 30px rgba(0,0,0,0.6); }
[data-testid="stMetricContainer"]{ background: rgba(255,255,255,0.02); border-radius:10px; padding:1rem; border:1px solid rgba(255,255,255,0.04); box-shadow:0 6px 18px rgba(0,0,0,0.45); }
[data-testid="stMetricValue"]{ color:#AEE7B1; font-weight:800; font-size:20px; }
.streamlit-expanderHeader{ background: rgba(255,255,255,0.02); border-radius:8px; border:1px solid rgba(255,255,255,0.04); color:#81C784; font-weight:600; }
.stInfo{ background: rgba(30,60,30,0.18); border-left:4px solid #81C784; }
.stDownloadButton > button{ background: linear-gradient(90deg,#5FD068 0%,#2D7A3E 100%); border-radius:8px; color:#fff; font-weight:700; }
.sidebar .css-1d391kg{ padding-top: 1rem; }
hr{ height:1px; background: linear-gradient(90deg,transparent, rgba(255,255,255,0.06), transparent); border:none; margin:1.5rem 0; }
.stMarkdown a.anchor-link, .stMarkdown .anchor-link, .anchor-link { display: none !important; }
.stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a { display: none !important; }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

col_logo, col_title = st.columns([0.5, 4.5], gap="large")
with col_logo:
    st.markdown("""<div style="background:linear-gradient(135deg,#2D7A3E 0%,#1B4D2B 100%); width:80px; height:80px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:40px;">💹</div>""", unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <h1>Automated Equity Research & Valuation Engine</h1>
    <p style="color:#6B7280; margin-top:4px;">Professional-grade fundamental analysis for informed investment decisions</p>
    """, unsafe_allow_html=True)

st.markdown("")
st.markdown('<hr>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<h3 style="color:#2D7A3E;">Configuration</h3>', unsafe_allow_html=True)
    
    ticker_input = st.text_input('Ticker Symbol', value='AAPL', placeholder='e.g., AAPL, MSFT', key='ticker_input', label_visibility='collapsed')
    
    if ticker_input and not re.match(r'^[A-Za-z0-9.=-]+$', ticker_input):
        st.warning("⚠️ Invalid characters in ticker. Use letters, numbers, dots, or hyphens only.")
        st.stop()
    
    country_code = st.selectbox('Market Region', ['US','IN','GB','DE','FR','CA','JP'], index=0)
    
    st.markdown("#### Settings")
    use_usd = st.checkbox("Convert to USD", value=False)
    
    st.markdown('---')
    # REPLACED: use_container_width=True is deprecated for buttons too? No, mostly dataframe/charts.
    # But for safety and consistency we'll stick to use_container_width which is fine for buttons in current versions
    # or remove it if problematic. Sticking to use_container_width as it is valid for buttons.
    run_btn = st.button('▶ Generate Research Report', type='primary', use_container_width=True)
    
    if st.session_state.get('analysis_done', False):
        st.markdown("")
        try:
            summ = st.session_state['summary']
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                pd.DataFrame([summ]).drop(columns=['fetched_info', 'yf_object'], errors='ignore').astype(str).to_excel(writer, sheet_name='Summary', index=False)
                
                yf_obj_export = yf.Ticker(summ['ticker_used'])
                if not yf_obj_export.financials.empty: 
                    clean_financial_df(yf_obj_export.financials).to_excel(writer, sheet_name='Income_Statement')
                if not yf_obj_export.balance_sheet.empty: 
                    clean_financial_df(yf_obj_export.balance_sheet).to_excel(writer, sheet_name='Balance_Sheet')
                if not yf_obj_export.cashflow.empty: 
                    clean_financial_df(yf_obj_export.cashflow).to_excel(writer, sheet_name='Cash_Flow')
                
                for sheet in writer.sheets.values():
                    for column in sheet.columns:
                        max_length = 0
                        column = [cell for cell in column]
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except: pass
                        adjusted_width = (max_length + 2)
                        sheet.column_dimensions[column[0].column_letter].width = adjusted_width

            output.seek(0)
            st.download_button('📥 Download Excel Report', data=output, file_name=f"{summ['ticker_used']}_full_report.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
        except Exception as e:
            st.warning("Export requires 'openpyxl'. Please install it.")

    st.markdown("---")
    st.markdown(
        """<div style='text-align: center; color: #6B7280; font-size: 12px; margin-top: 10px;'>
            Developed by<br><strong style='color: #81C784; font-size: 14px;'>Sai Advaith Parimisetti</strong>
            <br><br><em style='font-size: 10px;'>Disclaimer: This tool is for educational purposes only and does not constitute financial advice.</em>
        </div>""", unsafe_allow_html=True)

# --- APP LOGIC ---

if run_btn:
    clean_ticker = ticker_input.strip().upper()
    if clean_ticker in TICKER_CORRECTIONS:
        original = clean_ticker
        clean_ticker = TICKER_CORRECTIONS[clean_ticker]
        st.toast(f"ℹ️ Auto-corrected '{original}' to '{clean_ticker}'")
    
    with st.spinner(f"📊 Analyzing {clean_ticker}..."):
        summary = run_playbook_for_ticker(clean_ticker, country_code)
        if summary:
            rate = 1.0
            if use_usd and summary['price_currency'] != 'USD':
                rate = get_currency_rate(summary['price_currency'], 'USD')
            
            st.session_state['summary'] = summary
            st.session_state['usd_rate'] = rate
            st.session_state['use_usd'] = use_usd
            st.session_state['analysis_done'] = True
            st.rerun()
        else:
            st.session_state['analysis_done'] = False

if st.session_state.get('analysis_done', False) and 'summary' in st.session_state:
    summary = st.session_state['summary']
    rate = st.session_state.get('usd_rate', 1.0)
    using_usd = st.session_state.get('use_usd', False)

    yf_obj = yf.Ticker(summary['ticker_used'])
    summary['yf_object'] = yf_obj 

    disp_price = summary['price'] * rate if using_usd and summary['price'] else summary['price']
    disp_mcap = summary['marketCap'] * rate if using_usd and summary['marketCap'] else summary['marketCap']
    disp_ccy = "USD" if using_usd else summary['price_currency']

    st.markdown('<h3 style="color:#1B4D2B;">Key Metrics</h3>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4, gap='medium')
    with col1:
        st.metric('💰 Current Price', human_readable_price(disp_price, disp_ccy))
    with col2:
        st.metric('📈 Market Cap', human_readable_marketcap(disp_mcap, disp_ccy))
    with col3:
        underval_pct = min(100, int((summary['undervalued_score'] / 40) * 100))
        st.metric('🎯 Undervalued', f"{summary['undervalued_score']}/40", f"{underval_pct}%")
    with col4:
        multibag_pct = min(100, int((summary['multibagger_score'] / 50) * 100))
        st.metric('🚀 Multibagger', f"{summary['multibagger_score']}/50", f"{multibag_pct}%")

    st.markdown('')
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visual Report", "📈 Interactive Chart", "📋 Fundamental Analysis", "⚖️ Peer Comparison"])
    
    with tab1:
        st.markdown('<h3 style="color:#1B4D2B;">Visual Analysis Report</h3>', unsafe_allow_html=True)
        fig = render_infographic(summary, convert_usd=using_usd, rate=rate)
        if fig:
            st.pyplot(fig, use_container_width=True)
            fn = f"{summary['ticker_used']}_equity_research.png"
            img = io.BytesIO()
            fig.savefig(img, format='png', bbox_inches='tight', dpi=150)
            img.seek(0)
            plt.close(fig)
            st.markdown('<hr>', unsafe_allow_html=True)
            col_d1, col_d2, col_d3 = st.columns([1,1.5,1])
            with col_d2:
                st.download_button('📥 Download High-Resolution Report', data=img, file_name=fn, mime='image/png', use_container_width=True)

    with tab2:
        st.markdown('<h3 style="color:#1B4D2B;">Advanced Technical Chart</h3>', unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            chart_period = st.selectbox("Timeframe", ["1y", "2y", "5y", "10y", "max"], index=1, key="chart_period_box")
        with col_c2:
            indicators = st.multiselect("Overlays", ["SMA 50", "SMA 200", "Bollinger Bands"], default=["SMA 50"], key="chart_indicators_box")
        with col_c3:
            oscillator = st.selectbox("Bottom Panel", ["Volume", "RSI", "MACD"], index=0, key="chart_oscillator_box")

        try:
            hist = yf_obj.history(period=chart_period)
            if not hist.empty:
                if "SMA 50" in indicators: hist['SMA50'] = hist['Close'].rolling(window=50).mean()
                if "SMA 200" in indicators: hist['SMA200'] = hist['Close'].rolling(window=200).mean()
                if "Bollinger Bands" in indicators:
                    hist['MA20'] = hist['Close'].rolling(window=20).mean()
                    hist['STD20'] = hist['Close'].rolling(window=20).std()
                    hist['BB_Upper'] = hist['MA20'] + (hist['STD20'] * 2)
                    hist['BB_Lower'] = hist['MA20'] - (hist['STD20'] * 2)

                row_heights = [0.7, 0.3]
                fig_plotly = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights, subplot_titles=(f"{summary['ticker_used']} Price", oscillator))

                fig_plotly.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)

                if "SMA 50" in indicators: fig_plotly.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color='orange', width=1.5), name='SMA 50'), row=1, col=1)
                if "SMA 200" in indicators: fig_plotly.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color='blue', width=1.5), name='SMA 200'), row=1, col=1)
                if "Bollinger Bands" in indicators:
                    fig_plotly.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), showlegend=False), row=1, col=1)
                    fig_plotly.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name='BB'), row=1, col=1)

                if oscillator == "Volume":
                    fig_plotly.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color='teal', name='Volume'), row=2, col=1)
                elif oscillator == "RSI":
                    hist['RSI'] = calculate_rsi(hist['Close'])
                    fig_plotly.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
                    fig_plotly.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig_plotly.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                elif oscillator == "MACD":
                    macd, sig, hist_macd = calculate_macd(hist['Close'])
                    fig_plotly.add_trace(go.Scatter(x=hist.index, y=macd, line=dict(color='cyan', width=1.5), name='MACD'), row=2, col=1)
                    fig_plotly.add_trace(go.Scatter(x=hist.index, y=sig, line=dict(color='orange', width=1.5), name='Signal'), row=2, col=1)
                    fig_plotly.add_trace(go.Bar(x=hist.index, y=hist_macd, marker_color='gray', name='Hist'), row=2, col=1)

                fig_plotly.update_layout(template='plotly_dark', height=700, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
                # REPLACED: use_container_width=True with key argument for proper container fitting in new Streamlit versions
                st.plotly_chart(fig_plotly, use_container_width=True)
            else:
                st.warning("No price history available.")
        except Exception as e:
            st.error(f"Error creating chart: {e}")

    with tab3:
        st.markdown('<h3 style="color:#1B4D2B;">Deep Dive Analysis</h3>', unsafe_allow_html=True)
        info = summary['fetched_info']

        subtab_info, subtab_fin, subtab_rec, subtab_news = st.tabs(["🏢 Profile", "💵 Financial Statements", "🎯 Analyst Ratings", "📰 News"])
        
        with subtab_info:
            col_info_1, col_info_2 = st.columns([2, 1])
            with col_info_1:
                st.markdown("#### Business Summary")
                st.info(info.get('longBusinessSummary', "No summary available."))
            with col_info_2:
                st.markdown("#### Key Data")
                st.table(pd.DataFrame({
                    "Metric": ["Sector", "Industry", "Employees", "Website"],
                    "Value": [info.get('sector', 'N/A'), info.get('industry', 'N/A'), info.get('fullTimeEmployees', 'N/A'), info.get('website', 'N/A')]
                }).astype(str))

        with subtab_fin:
            st.markdown("#### Financial Statements (Annual)")
            fin_type = st.selectbox("Select Statement", ["Income Statement", "Balance Sheet", "Cash Flow"])
            try:
                # REPLACED: use_container_width=True with width argument for dataframes
                if fin_type == "Income Statement": 
                    st.dataframe(clean_financial_df(yf_obj.financials), width=None) # Default responsive
                elif fin_type == "Balance Sheet": 
                    st.dataframe(clean_financial_df(yf_obj.balance_sheet), width=None)
                elif fin_type == "Cash Flow": 
                    st.dataframe(clean_financial_df(yf_obj.cashflow), width=None)
            except: st.warning("Financial data unavailable.")

        with subtab_rec:
            st.markdown("#### Analyst Consensus")
            
            rec_mean = info.get('recommendationMean')
            rec_key = info.get('recommendationKey', '').replace('_', ' ').title() 
            
            if rec_mean:
                col_gauge, col_text = st.columns([2, 1])
                with col_gauge:
                    # Logic: Yahoo 1=Strong Buy, 5=Strong Sell
                    # Reversal: 6 - score makes 5=Strong Buy (Right side)
                    plot_value = 6 - rec_mean

                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = plot_value, 
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"Consensus: {rec_key}", 'font': {'size': 24, 'color': '#AEE7B1'}},
                        number = {'font': {'size': 40, 'color': 'white'}, 'suffix': " Score"}, 
                        gauge = {
                            'axis': {
                                'range': [1, 5], 
                                'tickwidth': 1, 
                                'tickcolor': "white", 
                                'ticktext': ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy'], 
                                'tickvals': [1, 2, 3, 4, 5]
                            },
                            'bar': {'color': "white", 'thickness': 0.02}, 
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 0,
                            'steps': [
                                {'range': [1, 1.8], 'color': "#FF0000"},    
                                {'range': [1.8, 2.6], 'color': "#FF8C00"},  
                                {'range': [2.6, 3.4], 'color': "#FFD700"},  
                                {'range': [3.4, 4.2], 'color': "#88E338"},  
                                {'range': [4.2, 5], 'color': "#00C805"}     
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': plot_value
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor = "rgba(0,0,0,0)", 
                        font = {'color': "white", 'family': "Inter"}, 
                        height=350, 
                        margin=dict(l=30, r=30, t=80, b=20) 
                    )
                    # REPLACED: use_container_width=True for chart
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col_text:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 20px; margin-top: 60px;">
                        <div style="font-size: 14px; color: #888;">Number of Analysts</div>
                        <div style="font-size: 24px; font-weight: bold; color: #E6F0EA;">{info.get('numberOfAnalystOpinions', 'N/A')}</div>
                        <br>
                        <div style="font-size: 14px; color: #888;">Actual Yahoo Score</div>
                        <div style="font-size: 18px; font-weight: bold; color: #AEE7B1;">{rec_mean}</div>
                        <div style="font-size: 10px; color: #666;">(1.0 = Strong Buy)</div>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.info("No sufficient analyst data to generate a gauge.")

            target = info.get('targetMeanPrice')
            current = info.get('regularMarketPrice')
            if target and current:
                delta = ((target - current) / current) * 100
                st.markdown("---")
                st.metric("Mean Analyst Price Target", human_readable_price(target, summary['price_currency']), f"{delta:.2f}% Upside")

        with subtab_news:
            # FIXED NEWS SECTION with robust key checks
            news_items = []
            
            try:
                yf_news = yf_obj.news
                if yf_news:
                    for item in yf_news:
                        title = item.get('title')
                        link = item.get('link')
                        if not title or not link:
                            continue
                        
                        publisher = item.get('publisher', 'Yahoo Finance')
                        pub_time = item.get('providerPublishTime')
                        
                        if pub_time:
                            pub_date_str = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d')
                        else:
                            pub_date_str = "Recent"
                            
                        news_items.append({
                            'title': title,
                            'link': link,
                            'publisher': publisher,
                            'date': pub_date_str,
                            'source': 'Yahoo'
                        })
            except Exception:
                pass
            
            if not news_items:
                news_items = fetch_news_fallback(summary['ticker_used'])
            
            if news_items:
                for item in news_items[:10]:
                    safe_title = html.escape(item.get('title', ''))
                    safe_pub = html.escape(item.get('publisher', 'Unknown'))
                    link = item.get('link', '#')
                    date_str = item.get('date') or item.get('providerPublishTime') or "Recent"
                    
                    border_color = "#2D7A3E" if item.get('source') == 'Yahoo' else "#F59E0B"
                    link_color = "#AEE7B1" if item.get('source') == 'Yahoo' else "#FCD34D"
                    
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid {border_color};">
                        <a href="{link}" target="_blank" style="color: {link_color}; font-weight: bold; text-decoration: none; font-size: 16px;">
                            {safe_title}
                        </a>
                        <div style="color: #aaa; font-size: 12px; margin-top: 4px;">
                            {date_str} • {safe_pub}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent news found for this ticker.")

    with tab4:
        st.markdown('<h3 style="color:#1B4D2B;">Peer Comparison</h3>', unsafe_allow_html=True)
        peer_input = st.text_input("Enter Peer Tickers (comma separated)", placeholder="e.g. MSFT, GOOGL, META")
        compare_btn = st.button("Compare Peers")
        
        if compare_btn and peer_input:
            peers = [x.strip().upper() for x in peer_input.split(',') if x.strip()]
            peers.insert(0, summary['ticker_used']) 
            
            def fetch_peer_data(p_ticker):
                try:
                    p_obj = yf.Ticker(p_ticker)
                    p_info = p_obj.info
                    if p_info and 'regularMarketPrice' in p_info:
                        return {
                            "Ticker": p_ticker,
                            "Price": p_info.get('regularMarketPrice'),
                            "Market Cap": format_large_number(p_info.get('marketCap')),
                            "P/E Ratio": p_info.get('trailingPE'),
                            "Forward P/E": p_info.get('forwardPE'),
                            "P/B Ratio": p_info.get('priceToBook'),
                            "ROE": p_info.get('returnOnEquity'),
                            "Rev Growth": p_info.get('revenueGrowth'),
                            "Div Yield": p_info.get('dividendYield')
                        }
                except:
                    return None
            
            comparison_data = []
            with st.spinner(f"Fetching data for {len(peers)} tickers..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = list(executor.map(fetch_peer_data, peers))
                comparison_data = [r for r in results if r is not None]

            if comparison_data:
                comp_df = pd.DataFrame(comparison_data).set_index("Ticker").transpose()
                # REPLACED: use_container_width=True with width argument
                st.dataframe(comp_df, width=None)
            else:
                st.warning("No data found for peers.")

else:
    col_land_1, col_land_2 = st.columns([1.6,1], gap='large')
    with col_land_1:
        st.markdown('''
        <div style="padding:2rem 0;">
            <h3 style="color:#1B4D2B; font-size:24px;">Welcome to Your Research Hub</h3>
            <p style="color:#6B7280; font-size:16px; line-height:1.8;">Analyze any stock with institutional-grade research tools. Our engine evaluates:</p>
            <ul style="color:#6B7280; font-size:15px; margin-bottom:1.5rem;">
                <li><strong>Valuation & Growth Scoring</strong>: Proprietary "Undervalued" and "Multibagger" algorithms.</li>
                <li><strong>Interactive Technicals</strong>: RSI, MACD, Bollinger Bands, and Volume analysis.</li>
                <li><strong>Deep Fundamental Data</strong>: Access Income Statements, Balance Sheets, and Cash Flows.</li>
                <li><strong>Peer Benchmarking</strong>: Compare key ratios against competitors instantly.</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    with col_land_2:
        st.markdown('''
        <div style="background:linear-gradient(135deg,#EBF8F0 0%,#E8F0ED 100%); border:1px solid #2D7A3E; border-radius:12px; padding:2rem; text-align:center;">
            <div style="font-size:64px; margin-bottom:1rem;">💹</div>
            <h4 style="color:#1B4D2B; margin:0 0 1rem 0; font-size:18px;">Ready to Research?</h4>
            <p style="color:#6B7280; margin:0; font-size:14px; line-height:1.6;">Start by entering a stock ticker in the sidebar.</p>
        </div>
        ''', unsafe_allow_html=True)
