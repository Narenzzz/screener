import streamlit as st
import pandas as pd
import numpy as np
import math
import yfinance as yf

st.set_page_config(page_title="Institutional All-in-One Nifty 500 Screener", layout="wide")

st.title("🛡️ Institutional All-in-One Nifty 500 Screener & Valuation Engine")
st.markdown("Upload a plain text or CSV file containing your ticker symbols (e.g., `RELIANCE`, `TCS`). The app will fetch live fundamentals, compute Moat grades, run Reverse DCF $g_1$ valuations, and categorize your stocks into GARP & Quality buckets instantly.")

# --- SIDEBAR: CONTROLS & UPLOAD ---
st.sidebar.header("1. Upload Ticker List")
ticker_file = st.sidebar.file_uploader("Upload Ticker List (.txt or .csv)", type=["txt", "csv"])

st.sidebar.header("2. Global Valuation Assumptions")
default_wacc = st.sidebar.slider("Assumed WACC (r %)", 8.0, 18.0, 12.0, 0.5)
default_term_g = st.sidebar.slider("Terminal Growth Rate ($g_{term}$ %)", 2.0, 6.0, 4.0, 0.25)

st.sidebar.header("3. Institutional Filters")
min_roce = st.sidebar.slider("Min ROCE (%)", 0.0, 50.0, 15.0, 1.0)
max_de = st.sidebar.slider("Max Debt-to-Equity", 0.0, 2.0, 0.5, 0.1)
selected_moats = st.sidebar.multiselect("Filter Moat Grades", ["A+", "A", "B", "C", "D"], default=["A+", "A", "B"])

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
        pv = 0
        current_cf = cf_per_share
        for yr in range(1, 11):
            current_cf *= (1.0 + mid)
            pv += current_cf / ((1.0 + wacc) ** yr)
        
        terminal_val = (current_cf * (1.0 + term_g)) / (wacc - term_g) if wacc > term_g else current_cf * 10
        pv_terminal = terminal_val / ((1.0 + wacc) ** 10)
        total_iv = pv + pv_terminal
        
        if total_iv > cmp:
            high = mid
        else:
            low = mid
        best_g = mid * 100.0
    return round(max(-20.0, min(80.0, best_g)), 2)

def compute_buy_zones(cmp, dcf_iv, graham_val, moat_grade):
    valid_vals = [v for v in [dcf_iv, graham_val] if v and v > 0]
    anchor = np.median(valid_vals) if valid_vals else cmp
    
    sip_zone = (anchor * 0.75, anchor * 0.90)
    garp_zone = (anchor * 0.90, anchor * 1.10)
    deep_value = (0.0, anchor * 0.75)
    
    in_sip = sip_zone[0] <= cmp <= sip_zone[1]
    in_garp = garp_zone[0] <= cmp <= garp_zone[1]
    in_deep = cmp <= deep_value[1]
    
    return anchor, sip_zone, garp_zone, in_sip or in_garp or in_deep

# --- EXECUTION FLOW ---
if ticker_file is not None:
    # Read tickers from uploaded text or csv file
    if ticker_file.name.endswith('.csv'):
        temp_df = pd.read_csv(ticker_file)
        # Try finding a column with ticker names
        col_cand = [c for c in temp_df.columns if 'ticker' in c.lower() or 'symbol' in c.lower() or 'name' in c.lower()]
        raw_tickers = temp_df[col_cand[0]].dropna().tolist() if col_cand else temp_df.iloc[:, 0].dropna().tolist()
    else:
        string_data = ticker_file.getvalue().decode("utf-8")
        raw_tickers = [line.strip() for line in string_data.splitlines() if line.strip()]

    st.write(f"📥 Loaded **{len(raw_tickers)}** ticker symbols from your file.")

    if st.button("🚀 Fetch Data & Run Institutional Screening", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total = len(raw_tickers)
        
        for idx, t in enumerate(raw_tickers):
            clean_t = t.upper().strip()
            if not clean_t.endswith(".NS"):
                clean_t += ".NS"
                
            status_text.text(f"Processing ({idx+1}/{total}): {clean_t}...")
            progress_bar.progress((idx + 1) / total)
            
            try:
                stock = yf.Ticker(clean_t)
                info = stock.info
                
                name = info.get("longName", clean_t.replace(".NS", ""))
                cmp = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
                eps = info.get("trailingEps", 0.0)
                shares = info.get("sharesOutstanding", 0.0)
                
                if cmp == 0 or shares == 0:
                    continue
                
                total_cash = info.get("totalCash", 0.0)
                total_debt = info.get("totalDebt", 0.0)
                total_equity = info.get("bookValue", 1.0) * shares
                de_ratio = (total_debt / total_equity) if total_equity > 0 else 0.2
                
                roce = (info.get("returnOnCapitalEmployed", 0.15) or 0.15) * 100.0
                roe = (info.get("returnOnEquity", 0.12) or 0.12) * 100.0
                fcf = info.get("freeCashflow") or (info.get("netIncomeToCommon", 1000000.0) * 0.6)
                
                # Apply Institutional Logic
                moat = calculate_moat_grade(roce, roe, de_ratio)
                implied_g1 = solve_reverse_dcf(cmp, fcf, default_wacc / 100.0, default_term_g / 100.0, shares)
                
                dcf_iv = max(cmp * 0.8, eps * 15 * (1 + implied_g1/100))
                graham_val = math.sqrt(22.5 * max(1.0, eps) * max(1.0, eps * 1.1)) if eps > 0 else cmp
                
                anchor, sip_z, garp_z, is_in_zone = compute_buy_zones(cmp, dcf_iv, graham_val, moat)
                
                if moat == "A+" and implied_g1 <= 15.0:
                    verdict = "💎 ELITE COMPOUNDING BUY"
                elif implied_g1 <= 12.0:
                    verdict = "🟢 ATTRACTIVE GARP"
                elif implied_g1 <= 20.0:
                    verdict = "🟡 FAIRLY PRICED"
                else:
                    verdict = "🔴 PRICED FOR PERFECTION / AVOID"
                    
                results.append({
                    "Ticker": clean_t.replace(".NS", ""),
                    "Company Name": name,
                    "Moat Grade": moat,
                    "CMP (₹)": round(cmp, 2),
                    "ROCE (%)": round(roce, 2),
                    "ROE (%)": round(roe, 2),
                    "Debt/Equity": round(de_ratio, 2),
                    "Implied Growth (g₁ %)": implied_g1,
                    "Valuation Anchor (₹)": round(anchor, 2),
                    "SIP Zone Start (₹)": round(sip_z[0], 2),
                    "SIP Zone End (₹)": round(sip_z[1], 2),
                    "In Active Buy Zone": "✅ Yes" if is_in_zone else "❌ No",
                    "Institutional Verdict": verdict
                })
            except Exception:
                continue

        status_text.text("✅ Analysis complete!")
        
        if results:
            res_df = pd.DataFrame(results)
            st.session_state["screened_df"] = res_df
        else:
        # Fallback safeguard if live fetch returns empty dictionary elements
            st.warning("No data retrieved. Please check your symbols.")

    # Display results if available in session state
    if "screened_df" in st.session_state:
        res_df = st.session_state["screened_df"]
        
        filtered_df = res_df[
            (res_df["ROCE (%)"] >= min_roce) &
            (res_df["Debt/Equity"] <= max_de) &
            (res_df["Moat Grade"].isin(selected_moats))
        ]
        
        st.subheader(f"📊 Screening Results Summary ({len(filtered_df)} of {len(res_df)} stocks matched)")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Successfully Analyzed", len(res_df))
        col2.metric("Matching Filters", len(filtered_df))
        col3.metric("A+ Moat Count", len(res_df[res_df["Moat Grade"] == "A+"]))
        col4.metric("In Buy Zone", len(filtered_df[filtered_df["In Active Buy Zone"] == "✅ Yes"]))
        
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        
        csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Institutional Shortlist (CSV)",
            data=csv_bytes,
            file_name="institutional_allinone_shortlist.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Please upload a text or CSV file containing your list of stock symbols in the sidebar to begin.")
    
    with st.expander("📌 How to structure your input file:"):
        st.markdown("""
        Create a simple text file (`tickers.txt`) where each line is just the NSE symbol:
        ```text
        RELIANCE
        TCS
        INFY
        HDFCBANK
        ICICIBANK
        ```
        The app will automatically append `.NS`, pull live metrics, grade their moats, run Reverse DCF, and filter them based on your settings!
        """)
