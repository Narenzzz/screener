import streamlit as st
import pandas as pd
import numpy as np
import math
import time
import re
import requests
from bs4 import BeautifulSoup
import yfinance as yf

st.set_page_config(page_title="Institutional All-in-One Nifty Screener", layout="wide")

st.title("🛡️ Institutional All-in-One Stock Screener & Valuation Engine")
st.markdown("Upload your NSE ticker list (`.txt` or `.csv`). This engine extracts audited financial statements, computes Multi-Model Intrinsic Values, tracks 5Q Shareholding trends, and categorizes stocks into **Aggressive Compounders vs. Defensive Anchors** via Beta ($\beta$).")

# --- SIDEBAR: CONTROLS & UPLOAD ---
st.sidebar.header("1. Upload Ticker List")
ticker_file = st.sidebar.file_uploader("Upload Ticker List (.txt or .csv)", type=["txt", "csv"])

st.sidebar.header("2. Global Valuation Assumptions")
default_wacc = st.sidebar.slider("Assumed WACC (r %)", 8.0, 18.0, 12.0, 0.5)
default_term_g = st.sidebar.slider("Terminal Growth Rate ($g_{term}$ %)", 2.0, 6.0, 4.0, 0.25)

st.sidebar.header("3. Institutional & Valuation Filters")
min_roce = st.sidebar.slider("Min ROCE (%)", 0.0, 50.0, 15.0, 1.0)
max_de = st.sidebar.slider("Max Debt-to-Equity", 0.0, 2.0, 0.5, 0.1)
max_pe = st.sidebar.slider("Max Trailing P/E", 5.0, 100.0, 35.0, 1.0)
max_peg = st.sidebar.slider("Max PEG Ratio", 0.1, 5.0, 2.5, 0.1)
selected_moats = st.sidebar.multiselect("Filter Moat Grades", ["A+", "A", "B", "C", "D"], default=["A+", "A", "B"])

st.sidebar.header("4. Growth Persona Filter (Beta)")
selected_personas = st.sidebar.multiselect(
    "Filter by Investment Persona",
    [
        "🚀 Aggressive Compounder", 
        "🛡️ Defensive Anchor", 
        "🟡 Fairly Priced Compounder", 
        "⚠️ High-Beta Cyclical / Speculative",
        "🔴 Priced for Perfection", 
        "🛑 Weak Moat / Value Trap"
    ],
    default=["🚀 Aggressive Compounder", "🛡️ Defensive Anchor"]
)

# --- HELPER: SAFE STATEMENT EXTRACTION ---
def get_statement_value(df, possible_keys, default=0.0):
    if df is None or df.empty:
        return default
    for key in possible_keys:
        if key in df.index:
            val = df.loc[key].dropna()
            if not val.empty:
                return float(val.iloc[0])
    return default

# --- HELPER: 5-QUARTER SHAREHOLDING TREND EXTRACTOR ---
def classify_5q_trend(series_vals, threshold=0.25):
    if not series_vals or len(series_vals) < 2:
        return "N/A"
    window = series_vals[-5:] if len(series_vals) >= 5 else series_vals
    start_val = window[0]
    end_val = window[-1]
    diff = round(end_val - start_val, 2)
    
    if diff > threshold:
        return f"📈 Rising ({end_val:.1f}% | +{diff:.2f}%)"
    elif diff < -threshold:
        return f"📉 Decline ({end_val:.1f}% | {diff:.2f}%)"
    else:
        return f"➖ Flat ({end_val:.1f}% | {diff:+.2f}%)"

def fetch_5q_shareholding_trends(nse_symbol, yf_info=None):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    
    for mode in ["consolidated/", ""]:
        url = f"https://www.screener.in/company/{nse_symbol}/{mode}"
        try:
            resp = requests.get(url, headers=headers, timeout=6)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            sh_sec = soup.find("section", id="shareholding")
            if not sh_sec:
                continue
            table = sh_sec.find("table", class_="data-table")
            if not table:
                continue
                
            promoters, fiis, diis, public = [], [], [], []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if not cells or len(cells) < 3:
                    continue
                row_title = re.sub(r'[\+\-\s]+$', '', cells[0]).strip().lower()
                
                nums = []
                for c in cells[1:]:
                    c_clean = c.replace('%', '').replace(',', '').strip()
                    try:
                        nums.append(float(c_clean))
                    except ValueError:
                        pass
                if not nums:
                    continue
                    
                if "promoter" in row_title:
                    promoters = nums
                elif "fii" in row_title or "foreign" in row_title:
                    fiis = nums
                elif "dii" in row_title or "domestic" in row_title:
                    diis = nums
                elif "public" in row_title or "retail" in row_title:
                    public = nums
                    
            inst = []
            if fiis and diis and len(fiis) == len(diis):
                inst = [round(f + d, 2) for f, d in zip(fiis, diis)]
            elif fiis:
                inst = fiis
            elif diis:
                inst = diis
                
            if promoters or inst or public:
                return (
                    classify_5q_trend(promoters),
                    classify_5q_trend(inst),
                    classify_5q_trend(public)
                )
        except Exception:
            continue
            
    if yf_info:
        insiders = (yf_info.get("heldPercentInsiders") or 0.0) * 100.0
        inst_pct = (yf_info.get("heldPercentInstitutions") or 0.0) * 100.0
        if insiders > 0 or inst_pct > 0:
            retail_pct = max(0.0, 100.0 - insiders - inst_pct)
            return (
                f"Snapshot: {insiders:.1f}%",
                f"Snapshot: {inst_pct:.1f}%",
                f"Snapshot: {retail_pct:.1f}%"
            )
            
    return "N/A", "N/A", "N/A"

# --- CORE INSTITUTIONAL LOGIC FUNCTIONS ---
def calculate_moat_grade(roce, roe, de):
    if roce >= 20.0 and roe >= 18.0 and de <= 0.30:
        return "A+"
    elif roce >= 15.0 and roe >= 14.0 and de <= 0.60:
        return "A"
    elif roce >= 12.0 and de <= 1.0:
        return "B"
    elif roce >= 8.0:
        return "C"
    else:
        return "D"

def solve_reverse_dcf(cmp, fcf_0, wacc, term_g, shares):
    if not cmp or cmp <= 0 or not fcf_0 or fcf_0 <= 0 or not shares or shares <= 0:
        return 10.0
    
    cf_per_share = fcf_0 / shares
    if cf_per_share <= 0:
        return 5.0
        
    low, high = -0.30, 1.00
    best_g = 10.0
    
    for _ in range(50):
        mid = (low + high) / 2.0
        pv = 0.0
        current_cf = cf_per_share
        for yr in range(1, 11):
            current_cf *= (1.0 + mid)
            pv += current_cf / ((1.0 + wacc) ** yr)
        
        terminal_val = (current_cf * (1.0 + term_g)) / (wacc - term_g) if wacc > term_g else current_cf * 10.0
        pv_terminal = terminal_val / ((1.0 + wacc) ** 10)
        total_iv = pv + pv_terminal
        
        if total_iv > cmp:
            high = mid
        else:
            low = mid
        best_g = mid * 100.0
    return round(max(-20.0, min(80.0, best_g)), 2)

def calculate_forward_dcf_iv(fcf_0, shares, growth_rate, wacc, term_g):
    if fcf_0 <= 0 or shares <= 0:
        return 0.0
    cf_per_share = fcf_0 / shares
    pv = 0.0
    current_cf = cf_per_share
    for yr in range(1, 11):
        current_cf *= (1.0 + growth_rate)
        pv += current_cf / ((1.0 + wacc) ** yr)
    terminal_val = (current_cf * (1.0 + term_g)) / (wacc - term_g) if wacc > term_g else current_cf * 10.0
    pv_terminal = terminal_val / ((1.0 + wacc) ** 10)
    return pv + pv_terminal

def compute_buy_zones(cmp, dcf_iv, graham_val, pe_iv, moat_grade):
    valid_vals = [v for v in [dcf_iv, graham_val, pe_iv] if v and v > 0]
    anchor = np.median(valid_vals) if valid_vals else cmp
    
    sip_zone = (anchor * 0.75, anchor * 0.90)
    garp_zone = (anchor * 0.90, anchor * 1.10)
    deep_value = (0.0, anchor * 0.75)
    
    in_sip = sip_zone[0] <= cmp <= sip_zone[1]
    in_garp = garp_zone[0] <= cmp <= garp_zone[1]
    in_deep = cmp <= deep_value[1]
    
    return anchor, sip_zone, garp_zone, (in_sip or in_garp or in_deep)

def classify_growth_persona(moat, roce, roe, de, cmp, anchor, implied_g1, beta):
    """
    Categorizes quality stocks at reasonable prices based on beta:
    - High Beta (>= 1.05) + Moat A+/A/B + Fair Price -> Aggressive Compounder
    - Low Beta (< 1.05) + Moat A+/A/B + Fair Price -> Defensive Anchor
    """
    is_quality = moat in ["A+", "A", "B"] and roce >= 12.0 and de <= 0.8
    is_fair_or_undervalued = cmp <= (anchor * 1.15) and implied_g1 <= 16.0

    if is_quality and is_fair_or_undervalued:
        if beta >= 1.05:
            return "🚀 Aggressive Compounder"
        else:
            return "🛡️ Defensive Anchor"
    elif is_quality and implied_g1 <= 20.0:
        return "🟡 Fairly Priced Compounder"
    elif beta > 1.10 and (roce < 12.0 or de > 0.8):
        return "⚠️ High-Beta Cyclical / Speculative"
    elif implied_g1 > 20.0 or cmp > (anchor * 1.30):
        return "🔴 Priced for Perfection"
    else:
        return "🛑 Weak Moat / Value Trap"

# --- EXECUTION FLOW ---
if ticker_file is not None:
    if ticker_file.name.endswith('.csv'):
        temp_df = pd.read_csv(ticker_file)
        col_cand = [c for c in temp_df.columns if 'ticker' in c.lower() or 'symbol' in c.lower() or 'name' in c.lower()]
        raw_tickers = temp_df[col_cand[0]].dropna().tolist() if col_cand else temp_df.iloc[:, 0].dropna().tolist()
    else:
        string_data = ticker_file.getvalue().decode("utf-8")
        raw_tickers = [line.strip() for line in string_data.splitlines() if line.strip()]

    st.write(f"📥 Loaded **{len(raw_tickers)}** ticker symbols from your file.")

    if st.button("🚀 Fetch Audited Statements, Beta, 5Q Trends & Screen", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total = len(raw_tickers)
        
        for idx, t in enumerate(raw_tickers):
            base_symbol = str(t).upper().strip().replace(".NS", "")
            clean_t = base_symbol + ".NS"
                
            status_text.text(f"Extracting ({idx+1}/{total}): {base_symbol}...")
            progress_bar.progress((idx + 1) / total)
            
            try:
                stock = yf.Ticker(clean_t)
                info = stock.info
                
                inc_stmt = stock.financials
                bal_sheet = stock.balance_sheet
                cf_stmt = stock.cashflow
                
                name = info.get("longName", base_symbol)
                sector = info.get("sector", "N/A")
                industry = info.get("industry", "N/A")
                
                cmp = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
                eps = info.get("trailingEps") or 0.0
                pe = info.get("trailingPE") or (cmp / eps if eps > 0 else 0.0)
                shares = info.get("sharesOutstanding") or 0.0
                
                # --- BETA VALUE EXTRACTION ---
                # Fallback to 1.0 (market neutral) if beta is not reported by exchange
                beta = info.get("beta")
                beta = round(float(beta), 2) if beta is not None else 1.00
                
                if cmp == 0 or shares == 0:
                    continue
                
                # Financial statements parsing
                ebit = get_statement_value(inc_stmt, ["EBIT", "Operating Income", "Total Operating Income As Reported"])
                net_income = get_statement_value(inc_stmt, ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"])
                if net_income == 0:
                    net_income = info.get("netIncomeToCommon") or (eps * shares)
                if ebit == 0:
                    ebit = info.get("ebitda", net_income * 1.35) * 0.85
                
                total_equity = get_statement_value(bal_sheet, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"])
                if total_equity == 0:
                    bvps = info.get("bookValue") or 0.0
                    total_equity = bvps * shares if bvps > 0 else 1.0
                else:
                    bvps = total_equity / shares
                    
                total_debt = get_statement_value(bal_sheet, ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"])
                if total_debt == 0:
                    total_debt = info.get("totalDebt") or 0.0
                    
                total_assets = get_statement_value(bal_sheet, ["Total Assets"])
                current_liab = get_statement_value(bal_sheet, ["Current Liabilities", "Total Current Liabilities"])
                
                if total_assets > 0 and current_liab > 0 and (total_assets - current_liab) > 0:
                    capital_employed = total_assets - current_liab
                else:
                    capital_employed = total_equity + total_debt
                    
                roce = round((ebit / capital_employed) * 100.0, 2) if capital_employed > 0 else 0.0
                roe = round((net_income / total_equity) * 100.0, 2) if total_equity > 0 else 0.0
                de_ratio = round(total_debt / total_equity, 2) if total_equity > 0 else 0.0
                
                fcf = get_statement_value(cf_stmt, ["Free Cash Flow"])
                if fcf == 0:
                    ocf = get_statement_value(cf_stmt, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
                    capex = get_statement_value(cf_stmt, ["Capital Expenditure", "Purchase Of PPE"])
                    if ocf != 0:
                        fcf = ocf + capex if capex < 0 else ocf - capex
                    else:
                        fcf = info.get("freeCashflow") or (net_income * 0.70)
                        
                moat = calculate_moat_grade(roce, roe, de_ratio)
                
                wacc_dec = default_wacc / 100.0
                term_g_dec = default_term_g / 100.0
                implied_g1 = solve_reverse_dcf(cmp, fcf, wacc_dec, term_g_dec, shares)
                sustainable_g = min(0.18, max(0.06, (roe / 100.0) * 0.65)) if roe > 0 else 0.08
                
                dcf_iv = calculate_forward_dcf_iv(fcf, shares, sustainable_g, wacc_dec, term_g_dec)
                if dcf_iv <= 0 and eps > 0:
                    dcf_iv = eps * 15.0
                
                graham_val = math.sqrt(22.5 * eps * bvps) if (eps > 0 and bvps > 0) else 0.0
                fair_pe_mult = 22.0 if moat == "A+" else (18.0 if moat == "A" else (15.0 if moat == "B" else 10.0))
                pe_iv = eps * fair_pe_mult if eps > 0 else 0.0
                
                valid_ivs = [v for v in [graham_val, dcf_iv, pe_iv] if v > 0]
                blended_iv = np.mean(valid_ivs) if valid_ivs else cmp
                
                peg = info.get("pegRatio")
                if not peg or peg <= 0:
                    peg = (pe / (sustainable_g * 100.0)) if (pe > 0 and sustainable_g > 0) else 0.0
                
                anchor, sip_z, garp_z, is_in_zone = compute_buy_zones(cmp, dcf_iv, graham_val, pe_iv, moat)
                
                # --- GROWTH PERSONA CLASSIFICATION (BETA + QUALITY + VALUATION) ---
                persona_verdict = classify_growth_persona(moat, roce, roe, de_ratio, cmp, anchor, implied_g1, beta)
                
                # Shareholding trend scrape
                prom_trend, inst_trend, retail_trend = fetch_5q_shareholding_trends(base_symbol, info)
                time.sleep(0.15)
                    
                results.append({
                    "Ticker": base_symbol,
                    "Company Name": name,
                    "Growth Persona": persona_verdict,
                    "Beta (β)": beta,
                    "Sector": sector,
                    "Industry": industry,
                    "Moat Grade": moat,
                    "CMP (₹)": round(cmp, 2),
                    "Trailing P/E": round(pe, 2),
                    "PEG Ratio": round(peg, 2),
                    "ROCE (%)": roce,
                    "ROE (%)": roe,
                    "Debt/Equity": de_ratio,
                    "Graham IV (₹)": round(graham_val, 2),
                    "DCF IV (₹)": round(dcf_iv, 2),
                    "P/E IV (₹)": round(pe_iv, 2),
                    "Blended IV (₹)": round(blended_iv, 2),
                    "Implied Growth (g₁ %)": implied_g1,
                    "Valuation Anchor (₹)": round(anchor, 2),
                    "In Active Buy Zone": "✅ Yes" if is_in_zone else "❌ No",
                    "Institutional Trend (5Q)": inst_trend,
                    "Retail Holding Trend (5Q)": retail_trend,
                    "Promoter Holding Trend (5Q)": prom_trend
                })
            except Exception:
                continue

        status_text.text("✅ Multi-factor screener run complete!")
        
        if results:
            res_df = pd.DataFrame(results)
            st.session_state["screened_df"] = res_df

    # Display results if available in session state
    if "screened_df" in st.session_state:
        res_df = st.session_state["screened_df"]
        
        filtered_df = res_df[
            (res_df["ROCE (%)"] >= min_roce) &
            (res_df["Debt/Equity"] <= max_de) &
            ((res_df["Trailing P/E"] <= max_pe) & (res_df["Trailing P/E"] > 0)) &
            ((res_df["PEG Ratio"] <= max_peg) | (res_df["PEG Ratio"] == 0.0)) &
            (res_df["Moat Grade"].isin(selected_moats)) &
            (res_df["Growth Persona"].isin(selected_personas))
        ]
        
        st.subheader(f"📊 Screening Results Summary ({len(filtered_df)} of {len(res_df)} stocks matched)")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Analyzed", len(res_df))
        col2.metric("Matching Shortlist", len(filtered_df))
        col3.metric("🚀 Aggressive Compounders", len(res_df[res_df["Growth Persona"] == "🚀 Aggressive Compounder"]))
        col4.metric("🛡️ Defensive Anchors", len(res_df[res_df["Growth Persona"] == "🛡️ Defensive Anchor"]))
        
        # Frozen Ticker & Company Name columns
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", pinned=True),
                "Company Name": st.column_config.TextColumn("Company Name", pinned=True),
                "Growth Persona": st.column_config.TextColumn("Growth Persona", pinned=True),
            }
        )
        
        csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Shortlist (CSV)",
            data=csv_bytes,
            file_name="institutional_screener_persona_shortlist.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Please upload your ticker file in the sidebar to begin.")
