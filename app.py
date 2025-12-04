import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import traceback
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import io

# Optional PDF export support
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

sns.set_style("whitegrid")

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

def human_readable_marketcap(value, currency_code=None, show_usd_equiv=False):
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

def fetch_info_with_variants(base_ticker, country_code):
    tried = []
    variants = get_yf_ticker_variants(base_ticker, country_code)
    for variant in variants:
        try:
            t = yf.Ticker(variant)
            info = t.info
            if info and (info.get("regularMarketPrice") is not None or info.get("symbol")):
                return t, info, variant
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

    details.update({"pe": pe, "pb": pb, "roe": roe, "roic": roic, "debtToEquity": dte,
                    "revenueGrowth": rev_growth, "dividendYield": div_yield})

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

    details.update({"marketCap": market_cap, "eps_growth": eps_growth, "roe": roe, "roic": roic,
                    "payoutRatio": payout, "revenueGrowth": rev_growth, "grossMargins": gross, "operatingMargins": opm})

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

def run_playbook_for_ticker(ticker_input, country_code=""):
    try:
        yf_obj, info, used_variant = fetch_info_with_variants(ticker_input, country_code)
    except Exception as e:
        st.error(f"Could not fetch data for {ticker_input}: {e}")
        return None

    screener_facts = {}
    if country_code and country_code.upper() == "IN":
        screener_facts = fetch_screener_in(ticker_input)

    u_score, u_details = score_undervalued_from_info(info)
    m_score, m_details = score_multibagger_from_info(info)

    decisions = {
        "undervalued_pass": u_score >= 25,
        "multibagger_pass": m_score >= 30
    }

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
        "screener_facts": screener_facts,
        "decisions": decisions,
        "fetched_info": info,
        "as_of": datetime.utcnow().isoformat() + "Z"
    }
    return summary

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
    plt.xticks(angles[:-1], labels, color='#333', size=10)
    ax.set_rlabel_position(0)
    ax.plot(angles, vals, linewidth=2, linestyle='solid', color='#2b7fc1')
    ax.fill(angles, vals, color='#2b7fc1', alpha=0.25)
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title, y=1.08, fontsize=12)

def fmt_pct_val(x):
    if x is None: return "n/a"
    try: return f"{x*100:.1f}%"
    except: return str(x)

def render_infographic(playbook_result):
    if playbook_result is None:
        return None

    s = playbook_result
    company = s.get('company') or s.get('ticker_used')
    ticker_used = s.get('ticker_used')
    price = s.get('price')
    price_ccy = s.get('price_currency')
    mcap = s.get('marketCap')
    mcap_ccy = s.get('marketCap_currency')
    sector = s.get('sector') or "N/A"

    u_score = s.get('undervalued_score') or 0
    m_score = s.get('multibagger_score') or 0
    u_pct = max(0, min(1, u_score / 40.0))
    m_pct = max(0, min(1, m_score / 50.0))

    u_details = s.get('undervalued_details', {})
    m_details = s.get('multibagger_details', {})

    checklist = [
        ("Price < Intrinsic", s['decisions']['undervalued_pass']),
        ("Debt/Equity < 0.5", (u_details.get('debtToEquity') is not None and u_details.get('debtToEquity') < 0.5)),
        ("ROE > 15%", (u_details.get('roe') is not None and u_details.get('roe') > 0.15)),
        ("Rev growth > 5%", (u_details.get('revenueGrowth') is not None and u_details.get('revenueGrowth') > 0.05)),
        ("Small-cap < $2B", (m_details.get('marketCap') is not None and m_details.get('marketCap') < 2e9)),
        ("EPS Accel > 20%", (m_details.get('eps_growth') is not None and m_details.get('eps_growth') > 0.2)),
    ]

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
    ax_header.text(0.01, 0.08, f"Report generated: {datetime.utcnow().strftime('%Y-%m-%d')}", fontsize=9, color='#666')

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
            ax_price.plot(hist.index, hist['Close'], label='Close', lw=1.2)
            ma50 = hist['Close'].rolling(50).mean()
            ma200 = hist['Close'].rolling(200).mean()
            ax_price.plot(hist.index, ma50, label='MA50', ls='--', lw=0.9)
            ax_price.plot(hist.index, ma200, label='MA200', ls='-.', lw=0.9)
            ax_price.set_title(f"Price History (12mo) — {ticker_used}")
            ax_price.legend()
        else:
            ax_price.text(0.5, 0.5, "Price history not available", ha='center', va='center')
    except:
        ax_price.text(0.5, 0.5, "Price history fetch failed", ha='center', va='center')

    plt.suptitle(f"{company} — Playbook Infographic", fontsize=16, fontweight='bold', y=0.995)
    return fig

# ==========================================
# 2. STREAMLIT APP UI — BEAUTIFIED
# ==========================================

# Update page config with clear title and icon
st.set_page_config(page_title="Automated Equity Research & Valuation Engine", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Custom CSS (typography, spacing, subtle accents)
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
.stApp{font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;}
.stApp { background: linear-gradient(180deg,#071014 0%,#0f1724 100%); }
/* Dark theme with green accents */
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
/* Hide Streamlit heading anchor link icon that appears on hover */
.stMarkdown a.anchor-link, .stMarkdown .anchor-link, .anchor-link { display: none !important; }
/* Also hide 'copy link' icon that may appear next to headings */
.stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a { display: none !important; }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Header
col_logo, col_title = st.columns([0.5, 4.5], gap="large")
with col_logo:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#2D7A3E 0%,#1B4D2B 100%); width:80px; height:80px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:40px;">💹</div>
    """, unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <h1>Automated Equity Research & Valuation Engine</h1>
    <p style="color:#6B7280; margin-top:4px;">Professional-grade fundamental analysis for informed investment decisions</p>
    """, unsafe_allow_html=True)

st.markdown("")
st.markdown('<hr>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h3 style="color:#2D7A3E;">Configuration</h3>', unsafe_allow_html=True)
    st.markdown('<label style="color:#AEE7B1; font-weight:700; margin-bottom:6px; display:block;">🔍 Ticker Symbol</label>', unsafe_allow_html=True)
    ticker_input = st.text_input('', value='AAPL', placeholder='e.g., AAPL, MSFT, RELIANCE', key='ticker_input', label_visibility='collapsed')
    country_code = st.selectbox('Market Region', ['US','IN','GB','DE','FR','CA','JP'], index=0)
    st.markdown('---')
    run_btn = st.button('▶ Generate Research Report', type='primary', use_container_width=True)
    st.markdown('')
    st.markdown('''
        <div class="stInfo" style="padding:12px; border-radius:8px;">
            <strong style="color:#1B4D2B; display:block; margin-bottom:6px;">💡 Quick Tips</strong>
            <small style="color:#2D7A3E;">• Use exact ticker symbols<br>• Results refresh daily<br>• Download high-res charts</small>
        </div>
    ''', unsafe_allow_html=True)

# Main
if run_btn:
    with st.spinner(f"📊 Analyzing {ticker_input}..."):
        summary = run_playbook_for_ticker(ticker_input, country_code)
        if summary:
            st.markdown('<h3 style="color:#1B4D2B;">Key Metrics</h3>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4, gap='medium')
            with col1:
                st.metric('💰 Current Price', human_readable_price(summary['price'], summary['price_currency']))
            with col2:
                st.metric('📈 Market Cap', human_readable_marketcap(summary['marketCap'], summary['marketCap_currency']))
            with col3:
                underval_pct = min(100, int((summary['undervalued_score'] / 40) * 100))
                st.metric('🎯 Undervalued', f"{summary['undervalued_score']}/40", f"{underval_pct}%")
            with col4:
                multibag_pct = min(100, int((summary['multibagger_score'] / 50) * 100))
                st.metric('🚀 Multibagger', f"{summary['multibagger_score']}/50", f"{multibag_pct}%")

            st.markdown('')
            st.markdown('<h3 style="color:#1B4D2B;">Visual Analysis Report</h3>', unsafe_allow_html=True)
            fig = render_infographic(summary)
            if fig:
                st.pyplot(fig, use_container_width=True)
                fn = f"{ticker_input}_equity_research.png"
                img = io.BytesIO()
                fig.savefig(img, format='png', bbox_inches='tight', dpi=150)
                img.seek(0)
                st.markdown('<hr>', unsafe_allow_html=True)
                col_d1, col_d2, col_d3 = st.columns([1,1.5,1])
                with col_d2:
                    st.download_button('📥 Download High-Resolution Report', data=img, file_name=fn, mime='image/png', use_container_width=True)
else:
    # Landing
    col_land_1, col_land_2 = st.columns([1.6,1], gap='large')
    with col_land_1:
        st.markdown('''
        <div style="padding:2rem 0;">
            <h3 style="color:#1B4D2B; font-size:24px;">Welcome to Your Research Hub</h3>
            <p style="color:#6B7280; font-size:16px; line-height:1.8;">Analyze any stock with institutional-grade research tools. Our engine evaluates:</p>
            <div style="background:#fff; border:1px solid #E5E7EB; border-radius:8px; padding:1.5rem; margin-top:1rem;">
                <div style="display:flex; align-items:start; margin-bottom:1rem;"><span style="color:#2D7A3E; font-size:20px; margin-right:1rem;">✓</span><div><p style="color:#1B4D2B; font-weight:600; margin:0 0 0.3rem 0;">Valuation Analysis</p><p style="color:#6B7280; font-size:14px; margin:0;">Is the stock trading below intrinsic value?</p></div></div>
                <div style="display:flex; align-items:start; margin-bottom:1rem;"><span style="color:#2D7A3E; font-size:20px; margin-right:1rem;">✓</span><div><p style="color:#1B4D2B; font-weight:600; margin:0 0 0.3rem 0;">Growth Potential</p><p style="color:#6B7280; font-size:14px; margin:0;">Does it have characteristics of a 10x multibagger?</p></div></div>
                <div style="display:flex; align-items:start;"><span style="color:#2D7A3E; font-size:20px; margin-right:1rem;">✓</span><div><p style="color:#1B4D2B; font-weight:600; margin:0 0 0.3rem 0;">Financial Health</p><p style="color:#6B7280; font-size:14px; margin:0;">Visualize Moat, Growth, and Balance Sheet strength</p></div></div>
            </div>
            <p style="color:#6B7280; font-size:13px; line-height:1.6; margin-top:1rem;"><strong>Getting Started:</strong> Select a ticker symbol and market region in the sidebar, then click "Generate Research Report" to begin your analysis.</p>
        </div>
        ''', unsafe_allow_html=True)
    with col_land_2:
        st.markdown('''
        <div style="background:linear-gradient(135deg,#EBF8F0 0%,#E8F0ED 100%); border:1px solid #2D7A3E; border-radius:12px; padding:2rem; text-align:center;">
            <div style="font-size:64px; margin-bottom:1rem;">💹</div>
            <h4 style="color:#1B4D2B; margin:0 0 1rem 0; font-size:18px;">Ready to Research?</h4>
            <p style="color:#6B7280; margin:0; font-size:14px; line-height:1.6;">Start by entering a stock ticker in the sidebar configuration panel to unlock comprehensive equity analysis.</p>
        </div>
        ''', unsafe_allow_html=True)