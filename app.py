import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import traceback
# THREAD SAFETY FIX: Import Figure directly
from matplotlib.figure import Figure 
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures

# PDF Report Generation Imports
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

sns.set_style("whitegrid")

# SECTOR BENCHMARKS FOR VALUATION & EXIT MULTIPLES
SECTOR_PE_MAP = {
    "Technology": 25, "Financial Services": 12, "Healthcare": 20,
    "Consumer Cyclical": 18, "Industrials": 18, "Energy": 10,
    "Utilities": 15, "Real Estate": 35, "Basic Materials": 15,
    "Communication Services": 20
}

COUNTRY_SUFFIX_MAP = {
    "IN": [".NS", ".BO"], "US": [""], "GB": [".L"], "DE": [".DE"],
    "FR": [".PA"], "CA": [".TO"], "AU": [".AX"], "HK": [".HK"],
    "JP": [".T"], "SG": [".SI"], "CH": [".SW"], "NL": [".AS"],
    "SE": [".ST"], "IT": [".MI"],
}

_CURRENCY_SYMBOLS = {
    "INR": "₹", "USD": "$", "EUR": "€", "GBP": "£",
    "JPY": "¥", "HKD": "HK$", "AUD": "A$", "CAD": "C$",
    "SGD": "S$", "CHF": "CHF ",
}

def _get_currency_symbol(currency_code):
    if not currency_code: return ""
    return _CURRENCY_SYMBOLS.get(currency_code.upper(), currency_code.upper() + " ")

# CACHED: Currency rates
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

# CACHED: Risk Free Rate
@st.cache_data(ttl=86400, show_spinner=False)
def get_risk_free_rate():
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1] / 100.0
    except:
        pass
    return 0.042 # Institutional anchor (4.2%)

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
        try:
            cr = value / 1e7
            return f"₹{cr:,.2f} Cr"
        except: return str(value)
    else:
        return human_readable_generic(value, currency)

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
            
            # ATTEMPT 1: Standard info fetch (often rate-limited on Cloud)
            try:
                info = t.info
                # Check for critical data
                if info and info.get("regularMarketPrice") is not None:
                    return info, variant
            except:
                info = {}

            # ATTEMPT 2: Fallback to fast_info (API-based, more robust)
            # If standard info failed or returned None, try fast_info
            if not info or info.get("regularMarketPrice") is None:
                fast = t.fast_info
                if fast and hasattr(fast, 'last_price') and fast.last_price is not None:
                    # Reconstruct a minimal info dict
                    info = {
                        "regularMarketPrice": fast.last_price,
                        "marketCap": fast.market_cap,
                        "currency": fast.currency,
                        "symbol": variant,
                        "shortName": variant, # Fallback
                        "longName": variant,  # Fallback
                        "sector": "Unknown",  # Cannot fetch via fast_info
                        "industry": "Unknown",
                        "beta": 1.0,          # Conservative default
                        "trailingPE": None,
                        "returnOnEquity": 0.15 # Conservative default for models
                    }
                    return info, variant
                    
            tried.append(variant)
        except Exception:
            tried.append(variant)
            
    raise RuntimeError(f"No usable yfinance data found. Tried: {tried}")

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
    # Improved fallback using robust Google News RSS parsing
    news_items = []
    try:
        # Clean ticker for search (remove suffixes for better news results)
        search_ticker = ticker.split('.')[0]
        query = f"{search_ticker} stock finance"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.content, features="xml")
        items = soup.findAll('item')
        
        for item in items[:5]: 
            try:
                title = item.title.text if item.title else "No Title"
                link = item.link.text if item.link else "#"
                pub_date = item.pubDate.text if item.pubDate else ""
                
                # Format date
                date_str = "Recent"
                if pub_date:
                    try:
                        dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                        date_str = dt.strftime("%Y-%m-%d")
                    except:
                        pass
                
                # Deduplication check
                if any(x['link'] == link for x in news_items):
                    continue
                    
                news_items.append({
                    'title': title, 
                    'link': link, 
                    'publisher': 'Google News', 
                    'providerPublishTime': date_str
                })
            except: continue
    except Exception: 
        pass
    return news_items

# --- FINANCIAL MODELING HELPERS (INSTITUTIONAL GRADE) ---
def calculate_wacc_institutional(info, financials, balance_sheet):
    """
    Calculates Dual WACC (Base & Stress):
    1. Base Case: Normalized inputs (Beta ~1.1-1.3, ERP 5.25%).
    2. Stress Case: High inputs (Beta capped 1.6, higher risk).
    """
    try:
        # 1. Inputs & Normalization
        raw_beta = info.get('beta')
        if raw_beta is None: raw_beta = 1.0
        
        # Blume Adjustment (Standardize)
        adj_beta = (raw_beta * 0.67) + (1.0 * 0.33)
        
        # Beta Scenarios
        beta_stress = min(adj_beta, 1.60) # Capped at 1.6 for stress
        beta_base = min(adj_beta, 1.25)   # Normalized to ~1.25 for Base Case (Megacap/Moat standard)

        # Macro Inputs
        rf = get_risk_free_rate()
        rf = max(0.035, min(rf, 0.050)) # Clamp 3.5% - 5.0%
        
        erp = 0.0525
        
        # Cost of Equity
        ke_stress = rf + (beta_stress * erp)
        ke_base = rf + (beta_base * erp)

        # 2. Capital Structure
        market_cap = info.get('marketCap', 0)
        total_debt = 0
        
        if not balance_sheet.empty:
            for col in ['Total Debt', 'Long Term Debt', 'LongTermDebt']:
                if col in balance_sheet.index:
                    total_debt = balance_sheet.loc[col].iloc[0]
                    break

        if market_cap is None or market_cap == 0:
            market_cap = 1 
            
        total_val = market_cap + total_debt
        w_e = market_cap / total_val
        w_d = total_debt / total_val

        # 3. Cost of Debt
        interest_expense = 0
        if not financials.empty:
             for col in ['Interest Expense', 'Interest Expense Non Operating']:
                if col in financials.index:
                    interest_expense = abs(financials.loc[col].iloc[0])
                    break
        
        tax_rate = 0.21
        if not financials.empty and 'Tax Provision' in financials.index and 'Pretax Income' in financials.index:
            try:
                taxes = financials.loc['Tax Provision'].iloc[0]
                pretax = financials.loc['Pretax Income'].iloc[0]
                if pretax != 0:
                    eff_tax = taxes / pretax
                    if 0.15 < eff_tax < 0.30: tax_rate = eff_tax
            except: pass

        if total_debt > 0 and interest_expense > 0:
            cost_debt_pre = (interest_expense / total_debt)
            cost_debt_pre = min(cost_debt_pre, 0.10) 
        else:
            cost_debt_pre = rf + 0.015 

        cost_debt_after_tax = cost_debt_pre * (1 - tax_rate)

        # 4. WACC Calculation (Dual)
        wacc_stress = (w_e * ke_stress) + (w_d * cost_debt_after_tax)
        wacc_base = (w_e * ke_base) + (w_d * cost_debt_after_tax)
        
        # Clamp Logic
        wacc_stress = max(0.08, min(wacc_stress, 0.14)) # Stress range 8-14%
        wacc_base = max(0.07, min(wacc_base, 0.11))     # Base range 7-11%

        details = {
            "Risk Free Rate": rf,
            "Beta (Base/Stress)": f"{beta_base:.2f} / {beta_stress:.2f}",
            "ERP": erp,
            "Cost of Equity (Base)": ke_base,
            "WACC (Base)": wacc_base,
            "WACC (Stress)": wacc_stress
        }
        return wacc_base, wacc_stress, details
    except Exception:
        return 0.10, 0.125, {"Note": "Fallback"}

def calculate_auto_growth(info):
    """
    Calculates Explicit Period Growth (Years 1-5).
    """
    try:
        roe = info.get('returnOnEquity', 0.15)
        if roe is None: roe = 0.15
        payout = info.get('payoutRatio', 0.0)
        if payout is None: payout = 0.0
        retention = 1 - payout
        
        growth = roe * retention
        growth = max(0.05, min(growth, 0.25)) # Cap at 25%
        return growth
    except:
        return 0.10

def calculate_normalized_fcf(yf_obj, reported_fcf):
    """
    SURGICAL UPGRADE: Normalize Starting FCF.
    Uses 3-year average FCF margin on TTM Revenue.
    """
    try:
        financials = yf_obj.financials
        cashflow = yf_obj.cashflow
        
        if financials.empty or cashflow.empty:
            return reported_fcf

        if 'Total Revenue' in financials.index:
            rev_history = financials.loc['Total Revenue']
        elif 'TotalRevenue' in financials.index:
            rev_history = financials.loc['TotalRevenue']
        else:
            return reported_fcf

        common_cols = rev_history.index.intersection(cashflow.columns)
        margins = []
        for col in common_cols[:3]: 
            try:
                ocf = 0
                if 'Operating Cash Flow' in cashflow.index:
                    ocf = cashflow.loc['Operating Cash Flow'][col]
                elif 'Total Cash From Operating Activities' in cashflow.index:
                    ocf = cashflow.loc['Total Cash From Operating Activities'][col]
                
                capex = 0
                if 'Capital Expenditure' in cashflow.index:
                    capex = cashflow.loc['Capital Expenditure'][col]
                
                fcf = ocf + capex if capex < 0 else ocf - capex
                rev = rev_history[col]
                if rev > 0: margins.append(fcf / rev)
            except: pass
        
        if not margins: return reported_fcf
            
        avg_margin = sum(margins) / len(margins)
        ttm_rev = rev_history.iloc[0]
        normalized_fcf = ttm_rev * avg_margin
        
        return max(reported_fcf, normalized_fcf)
    except Exception:
        return reported_fcf

# --- RISK METRICS ---
def calculate_beta(stock_hist, market_hist):
    stock_returns = stock_hist['Close'].pct_change().dropna()
    market_returns = market_hist['Close'].pct_change().dropna()
    common_dates = stock_returns.index.intersection(market_returns.index)
    stock_returns = stock_returns.loc[common_dates]
    market_returns = market_returns.loc[common_dates]
    if len(stock_returns) < 30: return None
    covariance = np.cov(stock_returns, market_returns)[0][1]
    variance = np.var(market_returns)
    return covariance / variance

def calculate_var(stock_hist, confidence_level=0.95):
    returns = stock_hist['Close'].pct_change().dropna()
    if returns.empty: return None
    return np.percentile(returns, (1 - confidence_level) * 100)

# --- SCORING LOGIC ---
def score_undervalued_from_info(info):
    score = 0
    details = {}
    sector = info.get("sector", "Unknown")
    pe_benchmark = SECTOR_PE_MAP.get(sector, 20)

    pe = safe_numeric(info.get("trailingPE"))
    pb = safe_numeric(info.get("priceToBook"))
    roe = safe_numeric(info.get("returnOnEquity"))
    roic = safe_numeric(info.get("returnOnInvestment")) or safe_numeric(info.get("returnOnAssets")) or roe
    dte = safe_numeric(info.get("debtToEquity"))
    rev_growth = safe_numeric(info.get("revenueGrowth"))
    div_yield = safe_numeric(info.get("dividendYield"))
    
    details.update({"pe": pe, "pb": pb, "roe": roe, "roic": roic, "debtToEquity": dte, "revenueGrowth": rev_growth, "dividendYield": div_yield, "sector_pe_benchmark": pe_benchmark})

    if pe and pe > 0:
        if pe < (pe_benchmark * 0.6): score += 8
        elif pe < (pe_benchmark * 0.9): score += 4
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
        st.error(f"❌ Data Fetch Error: {e}") 
        return None

    yf_obj = yf.Ticker(used_variant)
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
        "as_of": datetime.utcnow().isoformat() + "Z"
    }
    return summary

# --- PDF GENERATOR (COMPLETE & INTERPRETABLE) ---
def generate_comprehensive_pdf(summary, dcf_data, risk_data, visual_fig, risk_fig, dcf_charts):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # STYLES
    styles = getSampleStyleSheet()
    styleN = styles['Normal']
    styleH = styles['Heading2']
    styleN.fontSize = 10
    styleN.leading = 14
    
    # --- HELPER: DRAW WATERMARK ---
    def draw_watermark(c):
        c.saveState()
        c.setFont("Helvetica-Bold", 60)
        c.setFillColorRGB(0.9, 0.9, 0.9) 
        c.setFillAlpha(0.3) 
        c.translate(width/2, height/2)
        c.rotate(45)
        c.drawCentredString(0, 0, "Sai Advaith Parimisetti")
        c.restoreState()

    # --- HELPER: DRAW FOOTER ---
    def draw_footer(c):
        c.saveState()
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.grey)
        c.drawCentredString(width/2, 20, "EDUCATIONAL PURPOSE ONLY. NOT FINANCIAL ADVICE. DATA GENERATED ALGORITHMICALLY.")
        c.restoreState()

    # --- HELPER: DRAW HEADER STRIP ---
    def draw_header_strip(c, title):
        c.setFillColorRGB(0.05, 0.2, 0.4) # Navy Blue
        c.rect(0, height-60, width, 60, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 24)
        c.drawString(30, height-40, title)
        c.setFont("Helvetica", 12)
        c.drawString(width-200, height-40, f"{summary['ticker_used']} | {datetime.now().strftime('%Y-%m-%d')}")

    # --- HELPER: DRAW INSIGHT TEXT BOX ---
    def draw_insight_box(c, x, y, w, h, title, text):
        # Draw Box
        c.setStrokeColor(colors.lightgrey)
        c.setFillColor(colors.whitesmoke)
        c.rect(x, y, w, h, fill=1)
        # Draw Title
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x+10, y+h-15, title)
        # Draw Text
        p = Paragraph(text, styleN)
        p.wrapOn(c, w-20, h-30)
        p.drawOn(c, x+10, y+h-25-p.height)

    # ---------------------------------------------------------
    # PAGE 1: EXECUTIVE SUMMARY
    # ---------------------------------------------------------
    draw_watermark(c)
    draw_header_strip(c, f"Equity Research: {summary['company']}")
    
    # 1. Main Infographic
    if visual_fig:
        img_buffer = io.BytesIO()
        visual_fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=120)
        img_buffer.seek(0)
        img = ImageReader(img_buffer)
        c.drawImage(img, 30, height - 500, width=550, height=420, preserveAspectRatio=True)
    
    # 2. Key Metrics Strip
    y_metrics = height - 530
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, y_metrics, "Key Fundamental Metrics")
    
    # Prepare Data for Table
    details = summary['undervalued_details']
    m_details = summary['multibagger_details']
    
    metrics_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Value Score", f"{summary['undervalued_score']}/40", "Growth Score", f"{summary['multibagger_score']}/50"],
        ["P/E Ratio", f"{details.get('pe', 'N/A')}", "ROE", f"{fmt_pct_val(details.get('roe'))}"],
        ["Sector Avg P/E", f"{details.get('sector_pe_benchmark', 'N/A')}", "Revenue Growth", f"{fmt_pct_val(details.get('revenueGrowth'))}"],
        ["Debt/Equity", f"{details.get('debtToEquity', 'N/A')}", "Gross Margin", f"{fmt_pct_val(m_details.get('grossMargins'))}"]
    ]
    
    t_metrics = Table(metrics_data, colWidths=[130, 130, 130, 130])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0E1117')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0,1), (-1,-1), colors.white),
    ]))
    t_metrics.wrapOn(c, width, height)
    t_metrics.drawOn(c, 40, y_metrics - 120)

    # 3. Analyst Verdict (Auto-Generated)
    # FIX 1: Recommendation Language ("Screens as...")
    verdict = "Screens as Undervalued (Model-Based)" if summary['undervalued_score'] > 25 else "Screens as Fairly Valued / Overvalued (Model-Based)"
    verdict_text = f"Based on the algorithmic scoring, {summary['company']} <b>{verdict}</b> relative to its historicals and sector peers. The company has a Fundamental Score of {summary['undervalued_score']}/40."
    
    # FIX 1: "Algorithmic Verdict" -> "Model Output Summary"
    draw_insight_box(c, 40, y_metrics - 220, 530, 80, "Model Output Summary", verdict_text)

    draw_footer(c)
    c.showPage()
    
    # ---------------------------------------------------------
    # PAGE 2: VALUATION DEEP DIVE (CHARTS & INSIGHTS)
    # ---------------------------------------------------------
    draw_watermark(c)
    draw_header_strip(c, "Deep Dive Valuation Analysis")
    
    current_y = height - 100
    
    # A. WATERFALL CHART
    if dcf_charts and 'waterfall' in dcf_charts:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(30, current_y, "1. Valuation Bridge (Source of Value)")
        current_y -= 220
        
        # Draw Chart
        w_data = dcf_charts['waterfall']
        fig_w = Figure(figsize=(6, 3), dpi=100)
        ax_w = fig_w.subplots()
        vals = w_data['values']
        labels = w_data['labels']
        bottoms = [0, vals[0], 0]
        ax_w.bar(labels, vals, bottom=bottoms, color=['#2b7fc1', '#d93f3f', '#2ea043'])
        ax_w.set_ylabel("Value")
        ax_w.grid(axis='y', alpha=0.3)
        
        img_buf = io.BytesIO()
        fig_w.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        c.drawImage(ImageReader(img_buf), 30, current_y, width=350, height=175, preserveAspectRatio=True)
        
        # Draw Insight
        # FIX 3: "Intrinsic Value" -> "Model-Implied Intrinsic Value"
        term_pct = (vals[1] / vals[2]) * 100 if vals[2] > 0 else 0
        insight_text = f"This chart breaks down the Model-Implied Intrinsic Value. Notice that <b>{term_pct:.1f}%</b> of the value comes from the Terminal Value. This indicates the valuation is highly dependent on long-term stability rather than short-term cash flows."
        draw_insight_box(c, 400, current_y + 20, 180, 140, "Analyst Commentary", insight_text)
        
        current_y -= 50

    # B. SENSITIVITY HEATMAP
    if dcf_charts and 'sensitivity' in dcf_charts:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(30, current_y, "2. Sensitivity Analysis (WACC vs Growth)")
        current_y -= 220
        
        # Draw Chart
        s_data = dcf_charts['sensitivity']
        df_sens = s_data['matrix']
        fig_h = Figure(figsize=(6, 3), dpi=100)
        ax_h = fig_h.subplots()
        sns.heatmap(df_sens, annot=True, fmt=".0f", cmap="RdYlGn", ax=ax_h, cbar=False)
        ax_h.set_xlabel("Terminal Growth")
        ax_h.set_ylabel("WACC")
        
        img_buf = io.BytesIO()
        fig_h.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        c.drawImage(ImageReader(img_buf), 30, current_y, width=350, height=175, preserveAspectRatio=True)
        
        # Draw Insight
        insight_text = "The Heatmap shows the theoretical share price under different economic conditions. Green areas represent optimistic scenarios (Low WACC / High Growth), while Red areas indicate downside risk if interest rates rise or growth slows."
        draw_insight_box(c, 400, current_y + 20, 180, 140, "Risk Assessment", insight_text)
        
        current_y -= 50

    # C. METHODOLOGY COMPARISON
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, current_y, "3. Methodology Face-off")
    
    if dcf_data:
        # Create Comparison Chart manually since it wasn't in state
        # Simplified bar chart
        fig_c = Figure(figsize=(6, 2), dpi=100)
        ax_c = fig_c.subplots()
        methods = ["Perpetual Growth", "Exit Multiple"]
        
        try:
            curr_price = summary['price']
            intr_val = float(dcf_data['intrinsic_value'].replace(summary['price_currency'], '').replace(',',''))
            
            ax_c.barh(['Current Price', 'Intrinsic Value'], [curr_price, intr_val], color=['grey', '#2ea043'])
            ax_c.set_xlim(0, max(curr_price, intr_val)*1.2)
            
            img_buf = io.BytesIO()
            fig_c.savefig(img_buf, format='png', bbox_inches='tight')
            img_buf.seek(0)
            c.drawImage(ImageReader(img_buf), 30, current_y - 120, width=350, height=100, preserveAspectRatio=True)
            
            diff = intr_val - curr_price
            direction = "Discount" if diff > 0 else "Premium"
            insight_text = f"The stock is trading at a <b>{abs(diff):.2f} {summary['price_currency']} {direction}</b> to its fair value. A wide gap suggests a potential margin of safety."
            draw_insight_box(c, 400, current_y - 120, 180, 100, "Price Action", insight_text)
            
        except: pass

    draw_footer(c)
    c.showPage()

    # ---------------------------------------------------------
    # PAGE 3: RISK REPORT
    # ---------------------------------------------------------
    draw_watermark(c)
    draw_header_strip(c, "Risk Management Profile")
    
    # Risk Metrics Table
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, height - 100, "Quantitative Risk Metrics (1 Year)")
    
    if risk_data:
        r_data = [
            ["Metric", "Value", "Interpretation"],
            ["Beta", risk_data.get('beta', 'N/A'), "Volatility relative to Market"],
            ["VaR (95%)", risk_data.get('var', 'N/A'), "Max expected daily loss (95% conf)"],
            ["Max Drawdown", risk_data.get('drawdown', 'N/A'), "Worst peak-to-trough drop"]
        ]
        t2 = Table(r_data, colWidths=[100, 100, 300])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0E1117')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('GRID', (0,0), (-1,-1), 1, colors.grey)
        ]))
        t2.wrapOn(c, width, height)
        t2.drawOn(c, 30, height - 200)
        
        # Risk Histogram
        c.drawString(30, height - 250, "Return Distribution Histogram")
        if risk_fig:
            r_img_buffer = io.BytesIO()
            risk_fig.savefig(r_img_buffer, format='png', bbox_inches='tight', dpi=120)
            r_img_buffer.seek(0)
            c.drawImage(ImageReader(r_img_buffer), 30, height - 550, width=500, height=250, preserveAspectRatio=True)
            
            hist_insight = """
            <b>Interpreting the Histogram:</b><br/>
            - <b>Fat Tails:</b> If the curve extends far left/right, the stock has high "Tail Risk" (prone to extreme crashes or rallies).<br/>
            - <b>Peak:</b> A tall, narrow peak indicates stability. A flat, wide curve indicates unpredictability.<br/>
            - <b>VaR Line:</b> The red line marks the danger zone. Losses beyond this point happen less than 5% of the time.
            """
            draw_insight_box(c, 30, height - 680, 500, 100, "Volatility Analysis", hist_insight)

    draw_footer(c)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

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
    ax.text(0.5, 0.65, title, ha='center', va='center', fontsize=12 if not small else 10, color='#333', transform=ax.transAxes)
    ax.text(0.5, 0.35, value, ha='center', va='center', fontsize=18 if not small else 14, fontweight='bold', color='#111', transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.12, subtitle, ha='center', va='center', fontsize=9 if not small else 8, color='#666', transform=ax.transAxes)

def gauge_bar(ax, pct, label, bgcolor='#f3f3f3'):
    ax.axis('off')
    ax.add_patch(patches.Rectangle((0, 0.25), 1, 0.5, color=bgcolor, zorder=0, ec='none', alpha=1.0))
    pct_clamped = max(0.0, min(1.0, pct))
    cmap_map = sns.color_palette("RdYlGn", as_cmap=True)
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
    
    fig = Figure(constrained_layout=True, figsize=(14,10))
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
    ax_header.text(0.01, 0.7, company, fontsize=20, fontweight='bold', color='#111', transform=ax_header.transAxes)
    ax_header.text(0.01, 0.48, f"{ticker_used}   |   Sector: {sector}", fontsize=11, color='#444', transform=ax_header.transAxes)
    ax_header.text(0.01, 0.28, f"Price: {human_readable_price(price, price_ccy)}    MarketCap: {human_readable_marketcap(mcap, mcap_ccy)}", fontsize=11, color='#333', transform=ax_header.transAxes)
    ax_header.text(0.01, 0.08, f"Report generated: {datetime.utcnow().strftime('%Y-%m-%d')} | Analysis by Sai Advaith Parimisetti", fontsize=9, color='#666', transform=ax_header.transAxes)
    ax_header.text(0.01, -0.15, "Disclaimer: For informational purposes only. Not financial advice.", fontsize=7, color='#999', ha='left', transform=ax_header.transAxes)

    ax_right.axis('off')
    ax_right.add_patch(patches.Rectangle((0.02, 0.12), 0.96, 0.76, color='#fafafa', ec='#e6e6e6'))
    ax_right.text(0.5, 0.78, "Core Snapshot", ha='center', fontsize=11, fontweight='bold', transform=ax_right.transAxes)
    pe_bench = u_details.get('sector_pe_benchmark', 20)
    ax_right.text(0.07, 0.62, f"P/E: {u_details.get('pe') if u_details.get('pe') else 'n/a'} (Sector Avg: {pe_bench})", fontsize=10, transform=ax_right.transAxes)
    ax_right.text(0.07, 0.50, f"P/B: {u_details.get('pb') if u_details.get('pb') else 'n/a'}", fontsize=10, transform=ax_right.transAxes)
    ax_right.text(0.07, 0.38, f"ROE: {fmt_pct_val(u_details.get('roe'))}", fontsize=10, transform=ax_right.transAxes)
    ax_right.text(0.07, 0.26, f"Rev Growth: {fmt_pct_val(u_details.get('revenueGrowth'))}", fontsize=10, transform=ax_right.transAxes)

    draw_kpi(ax_kpi_1, "Price", human_readable_price(price, price_ccy))
    draw_kpi(ax_kpi_2, "Market Cap", human_readable_marketcap(mcap, mcap_ccy), small=True)
    # FIX 1: Recommendation Language
    draw_kpi(ax_kpi_3, "Value Score", f"{u_score}/40", subtitle=f"{int(u_pct*100)}%")
    draw_kpi(ax_kpi_4, "Growth Score", f"{m_score}/50", subtitle=f"{int(m_pct*100)}%")
    gauge_bar(ax_gauge_u, u_pct, "Value Score")
    gauge_bar(ax_gauge_m, m_pct, "Growth Score")

    valuation_comp = 0.0
    pe_val = u_details.get('pe')
    if pe_val and pe_val > 0: valuation_comp = max(0, min(1, (pe_bench*1.5 - pe_val) / (pe_bench*1.5)))
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

    # FIX 4: Pass/Fail Language
    checklist = [
        ("Price < Model Value", s['decisions']['undervalued_pass']),
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
        ax_pos.text(0.07, y_top-0.05, label, va='center', ha='left', color='white', fontsize=10, transform=ax_pos.transAxes)

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

    fig.suptitle(f"{company} — Playbook Infographic", fontsize=16, fontweight='bold', y=0.995)
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
    country_code = st.selectbox('Market Region', ['US','IN','GB','DE','FR','CA','JP'], index=0)
    
    st.markdown("#### Settings")
    use_usd = st.checkbox("Convert to USD", value=False)
    
    st.markdown('---')
    run_btn = st.button('▶ Generate Research Report', type='primary', use_container_width=True)
    
    # --- EXPORT HUB (UPDATED) ---
    if st.session_state.get('analysis_done', False):
        st.markdown("### 📥 Export Hub")
        
        # 1. Expanded Excel Export
        try:
            summ = st.session_state['summary']
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Basic Sheets
                pd.DataFrame([summ]).drop(columns=['fetched_info', 'yf_object'], errors='ignore').astype(str).to_excel(writer, sheet_name='Summary', index=False)
                yf_obj_export = yf.Ticker(summ['ticker_used'])
                if not yf_obj_export.financials.empty: clean_financial_df(yf_obj_export.financials).to_excel(writer, sheet_name='Income_Statement')
                if not yf_obj_export.balance_sheet.empty: clean_financial_df(yf_obj_export.balance_sheet).to_excel(writer, sheet_name='Balance_Sheet')
                if not yf_obj_export.cashflow.empty: clean_financial_df(yf_obj_export.cashflow).to_excel(writer, sheet_name='Cash_Flow')
                
                # New Data Sheets (DCF & Risk)
                if st.session_state.get('dcf_data'):
                    pd.DataFrame([st.session_state['dcf_data']]).to_excel(writer, sheet_name='DCF_Model')
                if st.session_state.get('risk_data'):
                    pd.DataFrame([st.session_state['risk_data']]).to_excel(writer, sheet_name='Risk_Metrics')
                
                # Formatting
                for sheet in writer.sheets.values():
                    for column in sheet.columns:
                        max_length = 0
                        column = [cell for cell in column]
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length: max_length = len(str(cell.value))
                            except: pass
                        sheet.column_dimensions[column[0].column_letter].width = max_length + 2

            output.seek(0)
            st.download_button('📥 Download Full Excel', data=output, file_name=f"{summ['ticker_used']}_full_report.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
        except Exception as e:
            st.warning("Export requires 'openpyxl'. Please install it.")

        # 2. PDF Export
        if REPORTLAB_AVAILABLE:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                # Retrieve chart data from session state
                pdf_data = generate_comprehensive_pdf(
                    st.session_state['summary'], 
                    st.session_state.get('dcf_data'), 
                    st.session_state.get('risk_data'),
                    st.session_state.get('fig_visual'),
                    st.session_state.get('fig_risk'),
                    st.session_state.get('dcf_charts_data') 
                )
                st.download_button("⬇️ Download PDF", pdf_data, f"{st.session_state['summary']['ticker_used']}_Report.pdf", "application/pdf", use_container_width=True)
        else:
            st.warning("Install 'reportlab' to enable PDF export.")

    st.markdown("---")
    # FIX 2 & 5: Strengthen Disclaimer & No Personalization
    st.markdown("""
    <div style='font-size: 11px; color: #888; line-height: 1.4;'>
    <strong>Legal Disclaimer</strong><br>
    This platform provides automated financial analysis and valuation tools for educational and informational purposes only. It does not constitute investment advice, financial advice, or a recommendation to buy, sell, or hold any security.<br><br>
    All outputs are generated algorithmically based on publicly available data and user-defined assumptions. No representation is made regarding accuracy, completeness, or future performance. Users are solely responsible for their investment decisions.<br><br>
    This platform does not provide personalized or suitability-based recommendations.
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        """<div style='text-align: center; color: #6B7280; font-size: 12px; margin-top: 10px;'>
            Developed by<br><strong style='color: #81C784; font-size: 14px;'>Sai Advaith Parimisetti</strong>
        </div>""", unsafe_allow_html=True)

# --- APP LOGIC ---

if run_btn:
    summary = run_playbook_for_ticker(ticker_input, country_code)
    if summary:
        rate = 1.0
        if use_usd and summary['price_currency'] != 'USD':
            rate = get_currency_rate(summary['price_currency'], 'USD')
        
        st.session_state['summary'] = summary
        st.session_state['usd_rate'] = rate
        st.session_state['use_usd'] = use_usd
        st.session_state['analysis_done'] = True
        # Reset data containers
        st.session_state['dcf_data'] = None
        st.session_state['risk_data'] = None
        st.session_state['fig_visual'] = None
        st.session_state['fig_risk'] = None
        st.session_state['dcf_charts_data'] = None
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
        # FIX 1: Recommendation Language (Value Score)
        st.metric('🎯 Value Score', f"{summary['undervalued_score']}/40")
    with col4:
        # FIX 1: Recommendation Language (Growth Score)
        st.metric('🚀 Growth Score', f"{summary['multibagger_score']}/50")

    st.markdown('')
    
    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Visual Report", "🛠️ Intrinsic Value (DCF)", "⚖️ Risk Analysis", "📈 Interactive Chart", "📋 Fundamental Analysis", "⚖️ Peer Comparison"])
    
    # 1. VISUAL REPORT
    with tab1:
        st.markdown('<h3 style="color:#1B4D2B;">Visual Analysis Report</h3>', unsafe_allow_html=True)
        fig = render_infographic(summary, convert_usd=using_usd, rate=rate)
        if fig:
            st.pyplot(fig, use_container_width=True)
            st.session_state['fig_visual'] = fig # SAVE FOR PDF
            
            # Simple Image Download
            fn = f"{summary['ticker_used']}_equity_research.png"
            img = io.BytesIO()
            Canvas = FigureCanvasAgg(fig)
            Canvas.print_png(img)
            img.seek(0)
            st.markdown('<hr>', unsafe_allow_html=True)
            col_d1, col_d2, col_d3 = st.columns([1,1.5,1])
            with col_d2:
                st.download_button('📥 Download High-Resolution Report', data=img, file_name=fn, mime='image/png', use_container_width=True)

    # 2. INTRINSIC VALUE (INSTITUTIONAL UPGRADE 5.0: FINAL POLISH & NARRATIVE)
    with tab2:
        st.markdown('<h3 style="color:#1B4D2B;">Institutional Valuation (DCF) - Conservative Framework</h3>', unsafe_allow_html=True)
        
        # 1. FETCH DATA REQUIRED FOR ALL MODELS
        fcf_available = False
        latest_fcf = 0
        shares = yf_obj.info.get('sharesOutstanding')
        try:
            cf = yf_obj.cashflow
            bs = yf_obj.balance_sheet
            fin = yf_obj.financials
            if not cf.empty:
                possible_names = ['Free Cash Flow', 'Free Cash Flow', 'FreeCashFlow']
                for name in possible_names:
                    if name in cf.index:
                        latest_fcf = cf.loc[name].iloc[0]
                        fcf_available = True
                        break
        except: pass

        if fcf_available and shares and not bs.empty and not fin.empty:
            if using_usd and summary['price_currency'] != 'USD':
                calc_fcf = latest_fcf * rate
            else:
                calc_fcf = latest_fcf

            # FIX 1: Normalize Starting FCF
            normalized_starting_fcf = calculate_normalized_fcf(yf_obj, calc_fcf)

            # 2. PERFORM DUAL CALCULATIONS (Base vs Stress)
            wacc_base, wacc_stress, wacc_details = calculate_wacc_institutional(yf_obj.info, fin, bs)
            
            # Growth Inputs
            explicit_growth = calculate_auto_growth(yf_obj.info)
            terminal_growth = 0.025 # GDP Capped
            
            # POLISH 1: Calculate Implied Reinvestment & ROIC
            payout = yf_obj.info.get('payoutRatio', 0.0)
            if payout is None: payout = 0.0
            
            # Reinvestment Rate = 1 - Payout Ratio.
            # We enforce a floor of 5% reinvestment to avoid division by zero errors in high payout scenarios
            implied_reinvestment = max(0.05, 1 - payout) 
            
            # Implied ROIC = Growth / Reinvestment Rate
            # This checks: "Is the growth realistic given how much they are reinvesting?"
            implied_roic = explicit_growth / implied_reinvestment

            # FIX 2: Slower Fade (10 Years)
            future_fcf = []
            current_val = normalized_starting_fcf
            
            # Years 1-5 (Explicit)
            for i in range(1, 6):
                current_val = current_val * (1 + explicit_growth)
                future_fcf.append(current_val)
                
            # Years 6-10 (Slower Fade to Bridge)
            fade_target = max(terminal_growth, explicit_growth * 0.5) 
            growth_decay = (explicit_growth - fade_target) / 5
            current_growth = explicit_growth
            
            for i in range(6, 11):
                current_growth -= growth_decay
                current_val = current_val * (1 + current_growth)
                future_fcf.append(current_val)
            
            # Discounting (10 Years) using BASE WACC
            discount_factors = [1 / ((1 + wacc_base) ** i) for i in range(1, 11)]
            dcf_vals = [f * d for f, d in zip(future_fcf, discount_factors)]
            sum_pv_fcf = sum(dcf_vals)

            # --- SUB TABS ---
            subtab_auto, subtab_sens, subtab_exit = st.tabs(["🤖 Auto-WACC & Growth", "🌡️ Sensitivity Matrix", "🚪 Exit Multiple"])

            # -----------------------------------------------------------
            # CALCULATIONS HOISTED TO MAIN SCOPE FOR NARRATIVE GENERATION
            # -----------------------------------------------------------
            
            # A. DCF (Base Case)
            tv_perp = (future_fcf[-1] * (1 + terminal_growth)) / (wacc_base - terminal_growth)
            pv_tv_perp = tv_perp / ((1 + wacc_base) ** 10)
            equity_val_perp = sum_pv_fcf + pv_tv_perp
            share_price_perp = equity_val_perp / shares
            upside_perp = (share_price_perp - disp_price) / disp_price

            # B. Exit Multiple (Base Case)
            ebitda = 0
            if 'EBITDA' in fin.index: ebitda = fin.loc['EBITDA'].iloc[0]
            share_price_exit = 0
            
            if ebitda > 0:
                ebitda_y10 = ebitda
                current_g_exit = explicit_growth
                decay_exit = (explicit_growth - fade_target) / 5
                for _ in range(5): ebitda_y10 *= (1 + explicit_growth)
                for _ in range(5): 
                    current_g_exit -= decay_exit
                    ebitda_y10 *= (1 + current_g_exit)
                
                sector = summary.get('sector', 'Technology')
                base_pe = SECTOR_PE_MAP.get(sector, 20)
                sector_mult = max(10, min(base_pe, 30))
                
                tv_exit = ebitda_y10 * sector_mult
                pv_tv_exit = tv_exit / ((1 + wacc_base) ** 10)
                equity_val_exit = sum_pv_fcf + pv_tv_exit
                share_price_exit = equity_val_exit / shares
            
            # C. Stress Case DCF
            tv_stress = (future_fcf[-1] * (1 + terminal_growth)) / (wacc_stress - terminal_growth)
            pv_tv_stress = tv_stress / ((1 + wacc_stress) ** 10)
            stress_dcf_vals = [f * (1/((1+wacc_stress)**(i+1))) for i, f in enumerate(future_fcf)]
            share_price_stress = (sum(stress_dcf_vals) + pv_tv_stress) / shares

            # POLISH 2: Valuation Narrative
            narrative = ""
            if share_price_exit > share_price_perp * 1.3:
                narrative = f"**Insight:** The Market-Implied Value ({human_readable_price(share_price_exit, disp_ccy)}) is significantly higher than the Conservative DCF. This suggests the market is pricing in a 'longer moat' (slower fade) or higher terminal margins than our base case assumes."
            elif share_price_perp > share_price_exit * 1.1:
                narrative = f"**Insight:** The Conservative DCF ({human_readable_price(share_price_perp, disp_ccy)}) is notably higher than the Exit Multiple valuation. This implies the stock may be fundamentally undervalued even if market multiples compress."
            else:
                narrative = "**Insight:** Both valuation methods (DCF & Exit Multiple) are converging, suggesting the current valuation is well-supported by both fundamental cash flows and market comparables."

            # A. AUTO-ASSISTED DCF DISPLAY
            with subtab_auto:
                st.markdown(f"##### 🤖 Institutional DCF Model (Conservative Base Case)")
                st.caption(f"Valuation using **Base WACC ({wacc_base:.1%})** and **Normalized Cash Flows**.")
                
                c_main1, c_main2 = st.columns([1, 2])
                with c_main1:
                    # FIX 3: "Intrinsic Value" -> "Model-Implied Intrinsic Value"
                    st.metric("Model-Implied Intrinsic Value (Base)", human_readable_price(share_price_perp, disp_ccy), f"{upside_perp*100:.1f}%")
                    st.caption(f"Downside (Stress Case): {human_readable_price(share_price_stress, disp_ccy)}")
                with c_main2:
                    # ROIC Logic for display
                    roic_color = "green" if implied_roic > wacc_base else "red"
                    roic_msg = "Value Creation" if implied_roic > wacc_base else "Value Destruction"
                    
                    st.info(f"""
                    **Economic Reality Check:**
                    - **Implied ROIC**: :{roic_color}[**{implied_roic:.1%}**] vs WACC {wacc_base:.1%} ({roic_msg})
                    - **Reinvestment Rate**: {implied_reinvestment:.1%} of earnings retained.
                    - **Growth Fade**: {explicit_growth:.1%} (5Y) → {fade_target:.1%} (10Y)
                    """)
                
                st.markdown(f"> {narrative}")
                
                # SAVE DCF DATA FOR EXPORT
                st.session_state['dcf_data'] = {
                    "intrinsic_value": human_readable_price(share_price_perp, disp_ccy),
                    "upside": f"{upside_perp*100:.2f}%",
                    "wacc": f"{wacc_base:.2%}",
                    "growth_rate": f"{explicit_growth:.2%}",
                    "fcf_start": f"{normalized_starting_fcf:,.2f}"
                }

                # NEW CHART: Valuation Bridge (Waterfall)
                bridge_data = pd.DataFrame({
                    "Component": ["PV of Cash Flows (1-10Y)", "PV of Terminal Value", "Total Equity Value"],
                    "Value": [sum_pv_fcf, pv_tv_perp, equity_val_perp]
                })
                
                if st.session_state.get('dcf_charts_data') is None:
                    st.session_state['dcf_charts_data'] = {}
                
                st.session_state['dcf_charts_data']['waterfall'] = {
                    'labels': ["PV of Cash Flows (1-10Y)", "Terminal Value (PV)", "Total Equity"],
                    'values': [sum_pv_fcf, pv_tv_perp, equity_val_perp]
                }

                fig_bridge = go.Figure(go.Bar(
                    x=bridge_data['Component'],
                    y=bridge_data['Value'],
                    marker_color=['#00C9FF', '#FF00FF', '#00C805'],
                    text=[human_readable_generic(v, disp_ccy) for v in bridge_data['Value']],
                    textposition='auto'
                ))
                fig_bridge.update_layout(title="Valuation Bridge: Where does the value come from?", template="plotly_dark", height=400, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_bridge, use_container_width=True)

            # B. SENSITIVITY ANALYSIS
            with subtab_sens:
                st.markdown("##### 🌡️ Valuation Heatmap (Base WACC Center)")
                st.caption("Sensitivity of Intrinsic Value to long-term economic assumptions.")
                
                wacc_range = [wacc_base - 0.01, wacc_base - 0.005, wacc_base, wacc_base + 0.005, wacc_base + 0.01]
                growth_range = [0.015, 0.020, 0.025, 0.030, 0.035]
                
                data = []
                for w in wacc_range:
                    row = []
                    for g in growth_range:
                        if w <= g: val = 0 
                        else:
                            cell_tv = (future_fcf[-1] * (1 + g)) / (w - g)
                            cell_pv_tv = cell_tv / ((1 + w) ** 10)
                            cell_dcf_vals = [f * (1/((1+w)**(i+1))) for i, f in enumerate(future_fcf)]
                            cell_sum_pv = sum(cell_dcf_vals)
                            val = (cell_sum_pv + cell_pv_tv) / shares
                        row.append(val)
                    data.append(row)
                
                if st.session_state.get('dcf_charts_data'):
                    st.session_state['dcf_charts_data']['sensitivity'] = {
                        'matrix': pd.DataFrame(data, index=[f"{w:.1%}" for w in wacc_range], columns=[f"{g:.1%}" for g in growth_range]),
                        'rows': wacc_range,
                        'cols': growth_range
                    }

                df_sens = pd.DataFrame(data, index=[f"WACC {w:.1%}" for w in wacc_range], columns=[f"Term Growth {g:.1%}" for g in growth_range])
                st.dataframe(df_sens.style.format(lambda x: f"{_get_currency_symbol(disp_ccy)}{x:,.2f}").background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)

                fig_heat = go.Figure(data=go.Heatmap(
                    z=data,
                    x=[f"{g:.1%}" for g in growth_range],
                    y=[f"{w:.1%}" for w in wacc_range],
                    colorscale='RdYlGn',
                    texttemplate="%{z:.0f}",
                    hoverongaps=False
                ))
                fig_heat.update_layout(title="Sensitivity Heatmap (Interactive)", xaxis_title="Terminal Growth (GDP Linked)", yaxis_title="WACC", template="plotly_dark", height=400)
                st.plotly_chart(fig_heat, use_container_width=True)

            # C. EXIT MULTIPLE
            with subtab_exit:
                st.markdown("##### 🚪 Market-Implied Value (Exit Multiple)")
                st.caption("Valuation assuming sale at a dynamic sector-based multiple.")
                
                if ebitda > 0:
                    upside_exit = (share_price_exit - disp_price) / disp_price
                    
                    c_ex1, c_ex2 = st.columns(2)
                    with c_ex1:
                        # FIX 3: "Intrinsic Value" -> "Model-Implied Value"
                        st.metric("Market-Implied Value", human_readable_price(share_price_exit, disp_ccy), f"{upside_exit*100:.1f}%")
                    with c_ex2:
                        st.write(f"Assumed Exit Multiple: **{sector_mult}x**")
                        st.write(f"Projected Year 10 EBITDA: {human_readable_generic(ebitda_y10, disp_ccy)}")
                        
                    comp_df = pd.DataFrame({
                        "Method": ["Conservative DCF", "Market-Implied Value"],
                        "Share Price": [share_price_perp, share_price_exit]
                    })
                    fig_comp = go.Figure(go.Bar(
                        x=comp_df["Method"],
                        y=comp_df["Share Price"],
                        marker_color=['#00C9FF', '#FFD700'],
                        text=[human_readable_price(p, disp_ccy) for p in comp_df["Share Price"]],
                        textposition='auto'
                    ))
                    fig_comp.add_hline(y=disp_price, line_dash="dash", line_color="red", annotation_text="Current Price")
                    fig_comp.update_layout(title="Methodology Comparison", template="plotly_dark", height=400)
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                else:
                    st.warning("Negative or missing EBITDA. Cannot perform Exit Multiple analysis.")

        else:
            st.warning("Insufficient financial data (FCF, Debt, or Shares) to auto-calculate valuation.")

    # 3. RISK ANALYSIS (UPDATED WITH INTERPRETATIONS)
    with tab3:
        st.markdown('<h3 style="color:#1B4D2B;">Risk Analysis</h3>', unsafe_allow_html=True)
        
        bench_ticker = "^GSPC" 
        if country_code == "IN": bench_ticker = "^NSEI" 
        
        with st.spinner("Calculating Risk Metrics..."):
            try:
                end_date = datetime.now()
                start_date = end_date - pd.Timedelta(days=365)
                stock_hist = yf_obj.history(start=start_date, end=end_date)
                bench_hist = yf.Ticker(bench_ticker).history(start=start_date, end=end_date)
                
                if not stock_hist.empty and not bench_hist.empty:
                    beta = calculate_beta(stock_hist, bench_hist)
                    var_95 = calculate_var(stock_hist, 0.95)
                    max_drawdown = (stock_hist['Close'].min() - stock_hist['Close'].max()) / stock_hist['Close'].max()
                    
                    # SAVE RISK DATA FOR EXPORT
                    st.session_state['risk_data'] = {
                        "beta": f"{beta:.2f}" if beta else "N/A",
                        "var": f"{var_95*100:.2f}%" if var_95 else "N/A",
                        "drawdown": f"{max_drawdown*100:.2f}%"
                    }

                    # METRICS ROW
                    c_r1, c_r2, c_r3 = st.columns(3)
                    with c_r1:
                        st.metric("Beta", f"{beta:.2f}" if beta else "N/A")
                        if beta:
                            interp = "High Volatility (Aggressive)" if beta > 1.2 else "Low Volatility (Defensive)" if beta < 0.8 else "Market correlated"
                            st.caption(f"**Inference:** {interp}")
                    with c_r2:
                        var_val = var_95 * 100 if var_95 else 0
                        st.metric("VaR (95%)", f"{var_val:.2f}%")
                        st.caption(f"**Meaning:** 95% confidence you won't lose more than {abs(var_val):.1f}% in a single day.")
                    with c_r3:
                        st.metric("Max Drawdown (1Y)", f"{max_drawdown*100:.2f}%")
                        st.caption("**Meaning:** The worst drop from peak to bottom this year.")

                    # DISTRIBUTION CHART
                    st.markdown("#### 📉 Volatility Distribution")
                    rets = stock_hist['Close'].pct_change().dropna()
                    
                    # Matplotlib Figure for both Display AND PDF
                    fig_dist = Figure(figsize=(10, 4))
                    ax_dist = fig_dist.subplots()
                    sns.histplot(rets, kde=True, ax=ax_dist, color="orange", bins=50)
                    ax_dist.axvline(var_95, color='red', linestyle='--', label=f'VaR 95%: {var_val:.2f}%')
                    ax_dist.set_title("Daily Returns Distribution (Fat tails = Higher Risk)")
                    ax_dist.legend()
                    st.pyplot(fig_dist)
                    
                    st.session_state['fig_risk'] = fig_dist # SAVE FOR PDF
                    
                    st.info("""
                    **How to interpret this graph:**
                    - **The Peak:** Most daily price changes happen here (usually near 0%).
                    - **The Width:** A wider curve means the stock is wild and volatile. A narrow curve means it's stable.
                    - **The Red Line (VaR):** This is your 'danger zone'. Returns to the left of this line represent the worst 5% of trading days.
                    """)
                    
                else:
                    st.warning("Insufficient historical data for risk metrics.")
            except Exception as e:
                st.error(f"Risk calc error: {e}")

    # 4. TECHNICALS
    with tab4:
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
                st.plotly_chart(fig_plotly, use_container_width=True)
            else:
                st.warning("No price history available.")
        except Exception as e:
            st.error(f"Error creating chart: {e}")

    with tab5:
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
                }))

        with subtab_fin:
            st.markdown("#### Financial Statements (Annual)")
            fin_type = st.selectbox("Select Statement", ["Income Statement", "Balance Sheet", "Cash Flow"])
            try:
                if fin_type == "Income Statement": 
                    st.dataframe(clean_financial_df(yf_obj.financials), use_container_width=True)
                elif fin_type == "Balance Sheet": 
                    st.dataframe(clean_financial_df(yf_obj.balance_sheet), use_container_width=True)
                elif fin_type == "Cash Flow": 
                    st.dataframe(clean_financial_df(yf_obj.cashflow), use_container_width=True)
            except: st.warning("Financial data unavailable.")

        with subtab_rec:
            st.markdown("#### Analyst Consensus")
            
            rec_mean = info.get('recommendationMean')
            rec_key = info.get('recommendationKey', '').replace('_', ' ').title() 
            
            if rec_mean:
                col_gauge, col_text = st.columns([2, 1])
                with col_gauge:
                    plot_value = 6 - rec_mean
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = plot_value, 
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"Consensus: {rec_key}", 'font': {'size': 24, 'color': '#AEE7B1'}},
                        number = {'font': {'size': 40, 'color': 'white'}, 'suffix': " Score"}, 
                        gauge = {
                            'axis': {'range': [1, 5], 'tickwidth': 1, 'tickcolor': "white", 'ticktext': ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy'], 'tickvals': [1, 2, 3, 4, 5]},
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
                            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': plot_value}
                        }
                    ))
                    fig_gauge.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "white", 'family': "Inter"}, height=350, margin=dict(l=30, r=30, t=80, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col_text:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 20px; margin-top: 60px;">
                        <div style="font-size: 14px; color: #888;">Number of Analysts</div>
                        <div style="font-size: 24px; font-weight: bold; color: #E6F0EA;">{info.get('numberOfAnalystOpinions', 'N/A')}</div>
                        <br>
                        <div style="font-size: 14px; color: #888;">Actual Yahoo Score</div>
                        <div style="font-size: 18px; font-weight: bold; color: #AEE7B1;">{rec_mean}</div>
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
            # IMPROVED NEWS LOGIC with rigorous None checking
            valid_news_count = 0
            
            # 1. Try Yahoo Finance News
            try:
                yf_news = yf_obj.news
                if yf_news and isinstance(yf_news, list):
                    for item in yf_news[:5]:
                        if not isinstance(item, dict): continue
                        
                        # Strict checking for title
                        title = item.get('title')
                        if not title or title.strip() == "" or title.lower() == "none": continue
                        
                        # Filter out common Yahoo boilerplate
                        if "yahoo finance" in title.lower() and len(title) < 20: continue
                        
                        link = item.get('link', '#')
                        publisher = item.get('publisher', 'Yahoo Finance')
                        
                        # Format Date
                        pub_time_raw = item.get('providerPublishTime')
                        pub_time_str = "Recent"
                        if pub_time_raw:
                            try:
                                pub_time_str = datetime.fromtimestamp(pub_time_raw).strftime('%Y-%m-%d')
                            except: pass
                            
                        st.markdown(f"""<div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #2D7A3E;"><a href="{link}" target="_blank" style="color: #AEE7B1; font-weight: bold; text-decoration: none;">{title}</a><div style="color: #888; font-size: 12px;">{pub_time_str} • {publisher}</div></div>""", unsafe_allow_html=True)
                        valid_news_count += 1
            except: pass
            
            # 2. Fallback if YF failed or returned nothing valid
            if valid_news_count == 0:
                fallback_news = fetch_news_fallback(summary['ticker_used'])
                if fallback_news:
                    for item in fallback_news:
                        title = item.get('title')
                        if not title or title.lower() == "no title": continue
                        
                        st.markdown(f"""<div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #F59E0B;"><a href="{item['link']}" target="_blank" style="color: #FCD34D; font-weight: bold; text-decoration: none;">{title}</a><div style="color: #888; font-size: 12px;">{item['providerPublishTime']} • {item['publisher']}</div></div>""", unsafe_allow_html=True)
                        valid_news_count += 1
            
            if valid_news_count == 0:
                st.info("No recent news found for this ticker.")

    with tab6:
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
                st.dataframe(comp_df, use_container_width=True)
            else:
                st.warning("No data found for peers.")

else:
    # Landing
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
                <li><strong>NEW: DCF Calculator</strong>: 3 Models (Auto, Matrix, Exit Multiple).</li>
                <li><strong>NEW: Risk Metrics</strong>: Interpretable Risk & Volatility Analysis.</li>
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
