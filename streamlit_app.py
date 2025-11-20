import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import numpy as np

# 設定網頁寬度與標題
st.set_page_config(layout="wide", page_title="S&P 500 Market Dashboard")

# ==========================================
# 1. 核心數據定義 (完整 S&P 500 成分股)
# ==========================================
# 為了準確計算市場廣度，這裡包含了主要成分股清單
# (註：為了程式碼長度與維護性，這裡列出各板塊主要代表，若需 100% 精確的全體 503 檔，可持續擴充此列表)

RAW_SECTOR_DATA = {
    'XLB (原物料)': [
        {'Symbol': 'LIN'}, {'Symbol': 'NEM'}, {'Symbol': 'SHW'}, {'Symbol': 'ECL'},
        {'Symbol': 'NUE'}, {'Symbol': 'FCX'}, {'Symbol': 'DD'}, {'Symbol': 'VMC'},
        {'Symbol': 'MLM'}, {'Symbol': 'APD'}, {'Symbol': 'CTVA'}, {'Symbol': 'IP'},
        {'Symbol': 'STLD'}, {'Symbol': 'PPG'}, {'Symbol': 'SW'}, {'Symbol': 'AMCR'},
        {'Symbol': 'DOW'}, {'Symbol': 'PKG'}, {'Symbol': 'IFF'}, {'Symbol': 'AVY'},
        {'Symbol': 'CF'}, {'Symbol': 'BALL'}, {'Symbol': 'LYB'}, {'Symbol': 'ALB'},
        {'Symbol': 'MOS'}, {'Symbol': 'EMN'},
    ],
    'XLC (通訊)': [
        {'Symbol': 'META'}, {'Symbol': 'GOOGL'}, {'Symbol': 'GOOG'}, {'Symbol': 'WBD'},
        {'Symbol': 'NFLX'}, {'Symbol': 'EA'}, {'Symbol': 'TTWO'}, {'Symbol': 'DIS'},
        {'Symbol': 'VZ'}, {'Symbol': 'CMCSA'}, {'Symbol': 'TMUS'}, {'Symbol': 'T'},
        {'Symbol': 'LYV'}, {'Symbol': 'CHTR'}, {'Symbol': 'TTD'}, {'Symbol': 'OMC'},
        {'Symbol': 'TKO'}, {'Symbol': 'FOXA'}, {'Symbol': 'NWSA'}, {'Symbol': 'IPG'},
        {'Symbol': 'FOX'}, {'Symbol': 'MTCH'}, {'Symbol': 'PSKY'}, {'Symbol': 'NWS'},
    ],
    'XLE (能源)': [
        {'Symbol': 'XOM'}, {'Symbol': 'CVX'}, {'Symbol': 'COP'}, {'Symbol': 'WMB'},
        {'Symbol': 'MPC'}, {'Symbol': 'EOG'}, {'Symbol': 'PSX'}, {'Symbol': 'SLB'},
        {'Symbol': 'VLO'}, {'Symbol': 'KMI'}, {'Symbol': 'BKR'}, {'Symbol': 'OKE'},
        {'Symbol': 'TRGP'}, {'Symbol': 'EQT'}, {'Symbol': 'OXY'}, {'Symbol': 'FANG'},
        {'Symbol': 'EXE'}, {'Symbol': 'HAL'}, {'Symbol': 'DVN'}, {'Symbol': 'TPL'},
        {'Symbol': 'CTRA'}, {'Symbol': 'APA'},
    ],
    'XLF (金融)': [
        {'Symbol': 'BRK-B'}, {'Symbol': 'JPM'}, {'Symbol': 'V'}, {'Symbol': 'MA'},
        {'Symbol': 'BAC'}, {'Symbol': 'WFC'}, {'Symbol': 'GS'}, {'Symbol': 'MS'},
        {'Symbol': 'AXP'}, {'Symbol': 'C'}, {'Symbol': 'SCHW'}, {'Symbol': 'BLK'},
        {'Symbol': 'SPGI'}, {'Symbol': 'COF'}, {'Symbol': 'PGR'}, {'Symbol': 'HOOD'},
        {'Symbol': 'BX'}, {'Symbol': 'CB'}, {'Symbol': 'CME'}, {'Symbol': 'MMC'},
        {'Symbol': 'ICE'}, {'Symbol': 'KKR'}, {'Symbol': 'COIN'}, {'Symbol': 'BK'},
        {'Symbol': 'MCO'}, {'Symbol': 'USB'}, {'Symbol': 'PNC'}, {'Symbol': 'AON'},
        {'Symbol': 'AJG'}, {'Symbol': 'PYPL'}, {'Symbol': 'TRV'}, {'Symbol': 'APO'},
        {'Symbol': 'TFC'}, {'Symbol': 'AFL'}, {'Symbol': 'ALL'}, {'Symbol': 'AMP'},
        {'Symbol': 'MSCI'}, {'Symbol': 'MET'}, {'Symbol': 'AIG'}, {'Symbol': 'SQ'},
        {'Symbol': 'NDAQ'}, {'Symbol': 'FI'}, {'Symbol': 'PRU'}, {'Symbol': 'HIG'},
        {'Symbol': 'STT'}, {'Symbol': 'FIS'}, {'Symbol': 'ACGL'}, {'Symbol': 'WTW'},
        {'Symbol': 'IBKR'}, {'Symbol': 'MTB'}, {'Symbol': 'RJF'}, {'Symbol': 'FITB'},
        {'Symbol': 'SYF'}, {'Symbol': 'NTRS'}, {'Symbol': 'CBOE'}, {'Symbol': 'HBAN'},
        {'Symbol': 'CINF'}, {'Symbol': 'BRO'}, {'Symbol': 'TROW'}, {'Symbol': 'CFG'},
        {'Symbol': 'RF'}, {'Symbol': 'WRB'}, {'Symbol': 'GPN'}, {'Symbol': 'CPAY'},
        {'Symbol': 'PFG'}, {'Symbol': 'L'}, {'Symbol': 'KEY'}, {'Symbol': 'EG'},
        {'Symbol': 'JKHY'}, {'Symbol': 'IVZ'}, {'Symbol': 'GL'}, {'Symbol': 'AIZ'},
        {'Symbol': 'FDS'}, {'Symbol': 'ERIE'}, {'Symbol': 'BEN'},
    ],
    'XLI (工業)': [
        {'Symbol': 'GE'}, {'Symbol': 'CAT'}, {'Symbol': 'RTX'}, {'Symbol': 'UBER'},
        {'Symbol': 'BA'}, {'Symbol': 'GEV'}, {'Symbol': 'ETN'}, {'Symbol': 'HON'},
        {'Symbol': 'UNP'}, {'Symbol': 'DE'}, {'Symbol': 'ADP'}, {'Symbol': 'LMT'},
        {'Symbol': 'PH'}, {'Symbol': 'TT'}, {'Symbol': 'MMM'}, {'Symbol': 'GD'},
        {'Symbol': 'HWM'}, {'Symbol': 'NOC'}, {'Symbol': 'EMR'}, {'Symbol': 'JCI'},
        {'Symbol': 'TDG'}, {'Symbol': 'WM'}, {'Symbol': 'UPS'}, {'Symbol': 'PWR'},
        {'Symbol': 'CSX'}, {'Symbol': 'ITW'}, {'Symbol': 'CTAS'}, {'Symbol': 'NSC'},
        {'Symbol': 'CMI'}, {'Symbol': 'AXON'}, {'Symbol': 'URI'}, {'Symbol': 'FDX'},
        {'Symbol': 'LHX'}, {'Symbol': 'PCAR'}, {'Symbol': 'CARR'}, {'Symbol': 'FAST'},
        {'Symbol': 'RSG'}, {'Symbol': 'AME'}, {'Symbol': 'GWW'}, {'Symbol': 'ROK'},
        {'Symbol': 'DAL'}, {'Symbol': 'PAYX'}, {'Symbol': 'CPRT'}, {'Symbol': 'XYL'},
        {'Symbol': 'OTIS'}, {'Symbol': 'EME'}, {'Symbol': 'WAB'}, {'Symbol': 'UAL'},
        {'Symbol': 'VRSK'}, {'Symbol': 'IR'}, {'Symbol': 'EFX'}, {'Symbol': 'BR'},
        {'Symbol': 'ODFL'}, {'Symbol': 'HUBB'}, {'Symbol': 'DOV'}, {'Symbol': 'VLTO'},
        {'Symbol': 'LDOS'}, {'Symbol': 'J'}, {'Symbol': 'PNR'}, {'Symbol': 'SNA'},
        {'Symbol': 'FTV'}, {'Symbol': 'LUV'}, {'Symbol': 'EXPD'}, {'Symbol': 'LII'},
        {'Symbol': 'CHRW'}, {'Symbol': 'ROL'}, {'Symbol': 'TXT'}, {'Symbol': 'ALLE'},
        {'Symbol': 'MAS'}, {'Symbol': 'IEX'}, {'Symbol': 'JBHT'}, {'Symbol': 'BLDR'},
        {'Symbol': 'NDSN'}, {'Symbol': 'HII'}, {'Symbol': 'DAY'}, {'Symbol': 'SWK'},
        {'Symbol': 'GNRC'}, {'Symbol': 'PAYC'}, {'Symbol': 'AOS'},
    ],
    'XLK (科技)': [
        {'Symbol': 'NVDA'}, {'Symbol': 'MSFT'}, {'Symbol': 'AAPL'}, {'Symbol': 'AVGO'},
        {'Symbol': 'PLTR'}, {'Symbol': 'AMD'}, {'Symbol': 'ORCL'}, {'Symbol': 'IBM'},
        {'Symbol': 'CSCO'}, {'Symbol': 'MU'}, {'Symbol': 'CRM'}, {'Symbol': 'LRCX'},
        {'Symbol': 'QCOM'}, {'Symbol': 'NOW'}, {'Symbol': 'AMAT'}, {'Symbol': 'INTU'},
        {'Symbol': 'INTC'}, {'Symbol': 'APP'}, {'Symbol': 'APH'}, {'Symbol': 'ANET'},
        {'Symbol': 'KLAC'}, {'Symbol': 'ACN'}, {'Symbol': 'TXN'}, {'Symbol': 'PANW'},
        {'Symbol': 'ADBE'}, {'Symbol': 'CRWD'}, {'Symbol': 'ADI'}, {'Symbol': 'CDNS'},
        {'Symbol': 'SNPS'}, {'Symbol': 'MSI'}, {'Symbol': 'TEL'}, {'Symbol': 'GLW'},
        {'Symbol': 'ADSK'}, {'Symbol': 'STX'}, {'Symbol': 'FTNT'}, {'Symbol': 'MPWR'},
        {'Symbol': 'NXPI'}, {'Symbol': 'DDOG'}, {'Symbol': 'WDAY'}, {'Symbol': 'DELL'},
        {'Symbol': 'WDC'}, {'Symbol': 'ROP'}, {'Symbol': 'FICO'}, {'Symbol': 'CTSH'},
        {'Symbol': 'MCHP'}, {'Symbol': 'HPE'}, {'Symbol': 'KEYS'}, {'Symbol': 'TER'},
        {'Symbol': 'SMCI'}, {'Symbol': 'HPQ'}, {'Symbol': 'FSLR'}, {'Symbol': 'TDY'},
        {'Symbol': 'JBL'}, {'Symbol': 'PTC'}, {'Symbol': 'NTAP'}, {'Symbol': 'ON'},
        {'Symbol': 'TYL'}, {'Symbol': 'CDW'}, {'Symbol': 'VRSN'}, {'Symbol': 'IT'},
        {'Symbol': 'TRMB'}, {'Symbol': 'GDDY'}, {'Symbol': 'FFIV'}, {'Symbol': 'GEN'},
        {'Symbol': 'ZBRA'}, {'Symbol': 'SWKS'}, {'Symbol': 'AKAM'}, {'Symbol': 'EPAM'},
    ],
    'XLP (必需消費)': [
        {'Symbol': 'WMT'}, {'Symbol': 'COST'}, {'Symbol': 'PG'}, {'Symbol': 'KO'},
        {'Symbol': 'PM'}, {'Symbol': 'PEP'}, {'Symbol': 'MO'}, {'Symbol': 'MDLZ'},
        {'Symbol': 'CL'}, {'Symbol': 'MNST'}, {'Symbol': 'TGT'}, {'Symbol': 'KR'},
        {'Symbol': 'KMB'}, {'Symbol': 'KDP'}, {'Symbol': 'SYY'}, {'Symbol': 'ADM'},
        {'Symbol': 'KVUE'}, {'Symbol': 'HSY'}, {'Symbol': 'GIS'}, {'Symbol': 'EL'},
        {'Symbol': 'K'}, {'Symbol': 'DG'}, {'Symbol': 'KHC'}, {'Symbol': 'CHD'},
        {'Symbol': 'DLTR'}, {'Symbol': 'STZ'}, {'Symbol': 'MKC'}, {'Symbol': 'TSN'},
        {'Symbol': 'CLX'}, {'Symbol': 'BG'}, {'Symbol': 'SJM'}, {'Symbol': 'LW'},
        {'Symbol': 'CAG'}, {'Symbol': 'TAP'}, {'Symbol': 'HRL'}, {'Symbol': 'CPB'},
        {'Symbol': 'BF-B'},
    ],
    'XLRE (房地產)': [
        {'Symbol': 'WELL'}, {'Symbol': 'PLD'}, {'Symbol': 'AMT'}, {'Symbol': 'EQIX'},
        {'Symbol': 'SPG'}, {'Symbol': 'PSA'}, {'Symbol': 'O'}, {'Symbol': 'DLR'},
        {'Symbol': 'CBRE'}, {'Symbol': 'CCI'}, {'Symbol': 'VTR'}, {'Symbol': 'VICI'},
        {'Symbol': 'EXR'}, {'Symbol': 'IRM'}, {'Symbol': 'CSGP'}, {'Symbol': 'AVB'},
        {'Symbol': 'EQR'}, {'Symbol': 'SBAC'}, {'Symbol': 'WY'}, {'Symbol': 'ESS'},
        {'Symbol': 'INVH'}, {'Symbol': 'MAA'}, {'Symbol': 'KIM'}, {'Symbol': 'DOC'},
        {'Symbol': 'REG'}, {'Symbol': 'HST'}, {'Symbol': 'CPT'}, {'Symbol': 'BXP'},
        {'Symbol': 'UDR'}, {'Symbol': 'ARE'}, {'Symbol': 'FRT'},
    ],
    'XLU (公用事業)': [
        {'Symbol': 'NEE'}, {'Symbol': 'CEG'}, {'Symbol': 'SO'}, {'Symbol': 'DUK'},
        {'Symbol': 'AEP'}, {'Symbol': 'VST'}, {'Symbol': 'SRE'}, {'Symbol': 'D'},
        {'Symbol': 'EXC'}, {'Symbol': 'XEL'}, {'Symbol': 'ETR'}, {'Symbol': 'PEG'},
        {'Symbol': 'WEC'}, {'Symbol': 'ED'}, {'Symbol': 'PCG'}, {'Symbol': 'NRG'},
        {'Symbol': 'DTE'}, {'Symbol': 'AEE'}, {'Symbol': 'ATO'}, {'Symbol': 'ES'},
        {'Symbol': 'PPL'}, {'Symbol': 'CNP'}, {'Symbol': 'AWK'}, {'Symbol': 'FE'},
        {'Symbol': 'CMS'}, {'Symbol': 'EIX'}, {'Symbol': 'NI'}, {'Symbol': 'EVRG'},
        {'Symbol': 'LNT'}, {'Symbol': 'PNW'}, {'Symbol': 'AES'},
    ],
    'XLV (醫療)': [
        {'Symbol': 'LLY'}, {'Symbol': 'JNJ'}, {'Symbol': 'ABBV'}, {'Symbol': 'UNH'},
        {'Symbol': 'ABT'}, {'Symbol': 'MRK'}, {'Symbol': 'TMO'}, {'Symbol': 'ISRG'},
        {'Symbol': 'AMGN'}, {'Symbol': 'BSX'}, {'Symbol': 'GILD'}, {'Symbol': 'PFE'},
        {'Symbol': 'DHR'}, {'Symbol': 'SYK'}, {'Symbol': 'MDT'}, {'Symbol': 'VRTX'},
        {'Symbol': 'CVS'}, {'Symbol': 'MCK'}, {'Symbol': 'BMY'}, {'Symbol': 'CI'},
        {'Symbol': 'HCA'}, {'Symbol': 'ELV'}, {'Symbol': 'REGN'}, {'Symbol': 'COR'},
        {'Symbol': 'ZTS'}, {'Symbol': 'BDX'}, {'Symbol': 'IDXX'}, {'Symbol': 'EW'},
        {'Symbol': 'A'}, {'Symbol': 'CAH'}, {'Symbol': 'RMD'}, {'Symbol': 'IQV'},
        {'Symbol': 'GEHC'}, {'Symbol': 'HUM'}, {'Symbol': 'MTD'}, {'Symbol': 'DXCM'},
        {'Symbol': 'STE'}, {'Symbol': 'PODD'}, {'Symbol': 'BIIB'}, {'Symbol': 'LH'},
        {'Symbol': 'WST'}, {'Symbol': 'WAT'}, {'Symbol': 'ZBH'}, {'Symbol': 'DGX'},
        {'Symbol': 'CNC'}, {'Symbol': 'HOLX'}, {'Symbol': 'INCY'}, {'Symbol': 'COO'},
        {'Symbol': 'UHS'}, {'Symbol': 'VTRS'}, {'Symbol': 'BAX'}, {'Symbol': 'RVTY'},
        {'Symbol': 'SOLV'}, {'Symbol': 'TECH'}, {'Symbol': 'ALGN'}, {'Symbol': 'CRL'},
        {'Symbol': 'MOH'}, {'Symbol': 'MRNA'}, {'Symbol': 'HSIC'}, {'Symbol': 'DVA'},
    ],
    'XLY (非必需消費)': [
        {'Symbol': 'AMZN'}, {'Symbol': 'TSLA'}, {'Symbol': 'HD'}, {'Symbol': 'MCD'},
        {'Symbol': 'BKNG'}, {'Symbol': 'TJX'}, {'Symbol': 'LOW'}, {'Symbol': 'DASH'},
        {'Symbol': 'SBUX'}, {'Symbol': 'ORLY'}, {'Symbol': 'NKE'}, {'Symbol': 'RCL'},
        {'Symbol': 'GM'}, {'Symbol': 'AZO'}, {'Symbol': 'HLT'}, {'Symbol': 'MAR'},
        {'Symbol': 'ABNB'}, {'Symbol': 'CMG'}, {'Symbol': 'ROST'}, {'Symbol': 'F'},
        {'Symbol': 'EBAY'}, {'Symbol': 'DHI'}, {'Symbol': 'YUM'}, {'Symbol': 'GRMN'},
        {'Symbol': 'CCL'}, {'Symbol': 'TSCO'}, {'Symbol': 'LEN'}, {'Symbol': 'EXPE'},
        {'Symbol': 'WSM'}, {'Symbol': 'TPR'}, {'Symbol': 'PHM'}, {'Symbol': 'ULTA'},
        {'Symbol': 'DRI'}, {'Symbol': 'NVR'}, {'Symbol': 'APTV'}, {'Symbol': 'LULU'},
        {'Symbol': 'LVS'}, {'Symbol': 'GPC'}, {'Symbol': 'BBY'}, {'Symbol': 'DPZ'},
        {'Symbol': 'RL'}, {'Symbol': 'DECK'}, {'Symbol': 'HAS'}, {'Symbol': 'WYNN'},
        {'Symbol': 'NCLH'}, {'Symbol': 'POOL'}, {'Symbol': 'LKQ'}, {'Symbol': 'KMX'},
        {'Symbol': 'MGM'}, {'Symbol': 'MHK'},
    ]
}

@st.cache_data(ttl=3600)
def parse_sector_data():
    tickers = []
    sector_map = {}
    for sector, stocks in RAW_SECTOR_DATA.items():
        for stock in stocks:
            sym = stock['Symbol']
            tickers.append(sym)
            sector_map[sym] = sector
    return list(set(tickers)), sector_map

# ==========================================
# 2. 數據下載與計算
# ==========================================

@st.cache_data(ttl=3600)
def get_market_data(tickers):
    all_tickers = tickers + ['^GSPC', 'TLT', '^VIX'] 
    try:
        # 下載 2 年歷史數據以確保長天期指標計算無誤
        data = yf.download(all_tickers, period="2y", group_by='ticker', threads=True, auto_adjust=True)
        return data
    except Exception as e:
        st.error(f"數據下載錯誤: {e}")
        return pd.DataFrame()

def calculate_market_indicators(data, tickers):
    # 1. 準備基準數據
    sp500 = data['^GSPC']['Close']
    tlt = data['TLT']['Close']
    vix = data['^VIX']['Close']
    benchmark_idx = sp500.index
    
    # 2. 建立個股矩陣
    valid_tickers = [t for t in tickers if t in data]
    close_df = pd.DataFrame({t: data[t]['Close'] for t in valid_tickers}).reindex(benchmark_idx)
    high_df = pd.DataFrame({t: data[t]['High'] for t in valid_tickers}).reindex(benchmark_idx)
    low_df = pd.DataFrame({t: data[t]['Low'] for t in valid_tickers}).reindex(benchmark_idx)
    
    # A. 市場廣度 (MA60)
    ma60_df = close_df.rolling(window=60).mean()
    above_ma60 = (close_df > ma60_df)
    valid_counts = ma60_df.notna().sum(axis=1)
    above_counts = above_ma60.sum(axis=1)
    breadth_pct = (above_counts / valid_counts * 100).fillna(0)
    
    # B. 52 週新高/新低比率 (Highs / Lows Ratio) - S&P 500 Version
    # 這是您特別指定的「S&P 500 版」貪婪恐懼指標
    roll_max_252 = high_df.rolling(window=252).max()
    roll_min_252 = low_df.rolling(window=252).min()
    new_highs = (high_df >= roll_max_252).sum(axis=1)
    new_lows = (low_df <= roll_min_252).sum(axis=1)
    
    # 避免除以 0：若新低為 0，設分母為 1
    safe_lows = new_lows.replace(0, 1) 
    nh_nl_ratio = new_highs / safe_lows
    
    # C. 騰落指標 (A/D Line 20MA)
    daily_change = close_df.diff()
    advancing = (daily_change > 0).sum(axis=1)
    declining = (daily_change < 0).sum(axis=1)
    net_adv_dec = advancing - declining
    ad_ma20 = net_adv_dec.rolling(window=20).mean()
    
    # D. 資產強弱 (報酬率差值)
    sp500_ret_20 = sp500.pct_change(20) * 100
    tlt_ret_20 = tlt.pct_change(20) * 100
    strength_diff = sp500_ret_20 - tlt_ret_20
    
    # E. VIX
    vix_ma50 = vix.rolling(window=50).mean()

    lookback = 130
    return {
        'dates': sp500.index[-lookback:],
        'sp500': sp500.iloc[-lookback:],
        'breadth_pct': breadth_pct.iloc[-lookback:],
        'nh_nl_ratio': nh_nl_ratio.iloc[-lookback:], 
        'ad_ma20': ad_ma20.iloc[-lookback:],
        'strength_diff': strength_diff.iloc[-lookback:],
        'vix': vix.iloc[-lookback:],
        'vix_ma50': vix_ma50.iloc[-lookback:]
    }

def get_latest_snapshot(data, tickers):
    results = []
    for ticker in tickers:
        try:
            if ticker not in data: continue
            df = data[ticker]
            df = df.dropna(subset=['Close', 'Volume'])
            if df.empty or len(df) < 252: continue 
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            change_pct = ((curr['Close'] - prev['Close']) / prev['Close']) * 100
            turnover = curr['Close'] * curr['Volume']
            ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
            bias_20 = ((curr['Close'] - ma20) / ma20) * 100
            volatility = ((curr['High'] - curr['Low']) / prev['Close']) * 100
            avg_vol_20 = df['Volume'].iloc[-22:-2].mean()
            r_vol = curr['Volume'] / avg_vol_20 if avg_vol_20 > 0 else 0
            
            # 52週高低
            high_52w = df['High'].tail(252).max()
            low_52w = df['Low'].tail(252).min()
            
            results.append({
                'Ticker': ticker,
                'Close': curr['Close'],
                'Change %': change_pct,
                'Turnover': turnover,
                'Bias 20(%)': bias_20,
                'Volatility': volatility,
                'RVol': r_vol,
                '52W High': high_52w,
                '52W Low': low_52w
            })
        except:
            continue
    return pd.DataFrame(results)

# ==========================================
# 3. 視覺化與 Streamlit 呈現
# ==========================================

def main():
    st.title("📊 S&P 500 Advanced Market Dashboard")
    st.write(f"Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.info("首次載入可能需要 30-60 秒下載完整成分股數據，請稍候...")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner('Downloading Market Data (Full S&P 500)...'):
        tickers, sector_map = parse_sector_data()
        full_data = get_market_data(tickers)
    
    if full_data.empty:
        st.error("Failed to download data.")
        return

    with st.spinner('Calculating Indicators...'):
        mkt = calculate_market_indicators(full_data, tickers)
        df_snapshot = get_latest_snapshot(full_data, tickers)
        df_snapshot['Sector'] = df_snapshot['Ticker'].map(sector_map)
        
        # 計算各產業今日平均漲跌幅 (Today's Average Sector Change)
        sector_perf = df_snapshot.groupby('Sector')['Change %'].mean().sort_values(ascending=False)

    # 繪圖 Layout
    fig = make_subplots(
        rows=10, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.08, 0.08, 0.08, 0.08],
        specs=[
            [{"colspan": 2, "secondary_y": True}, None], # R1
            [{"colspan": 2, "secondary_y": True}, None], # R2
            [{"colspan": 2, "secondary_y": True}, None], # R3
            [{"colspan": 2, "secondary_y": True}, None], # R4
            [{"colspan": 2, "secondary_y": True}, None], # R5
            [{"colspan": 2, "secondary_y": False}, None],# R6: Sector Perf
            [{"type": "table"}, {"type": "table"}],
            [{"type": "table"}, {"type": "table"}],
            [{"type": "table"}, {"type": "table"}],
            [{"type": "table"}, {"type": "table"}]
        ],
        vertical_spacing=0.06,
        subplot_titles=(
            "市場廣度：站上 60MA 比例 vs S&P 500",
            "市場內部：52週新高/新低 家數比率 (Highs/Lows Ratio) - S&P 500 Version",
            "市場動能：20日平均淨上漲家數 (Net Adv-Dec) vs S&P 500",
            "資產強弱：(S&P500 20日報酬 - TLT 20日報酬) 差值 (折線圖)",
            "恐慌指數：VIX vs 50日均線",
            "各產業今日平均漲跌幅 (Today's Sector Performance)",
            "1. 漲幅最強 10 檔", "2. 跌幅最重 10 檔",
            "3. 高波動度", "6. 正乖離過大 (>MA20)",
            "7. 負乖離過大 (<MA20)", "4. 爆量上漲",
            "5. 爆量下跌", ""
        )
    )

    x_axis = mkt['dates']

    # R1: Breadth
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", line=dict(color='black', width=1)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['breadth_pct'], name="% > MA60", line=dict(color='blue', width=2), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'), row=1, col=1, secondary_y=True)

    # R2: NH/NL Ratio (Line Chart)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['nh_nl_ratio'], name="Highs/Lows Ratio", line=dict(color='green', width=2)), row=2, col=1, secondary_y=True)
    fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1, secondary_y=True)

    # R3: A/D Line
    ad_colors = ['green' if v >= 0 else 'red' for v in mkt['ad_ma20']]
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=x_axis, y=mkt['ad_ma20'], name="20MA Net Adv-Dec", marker_color=ad_colors, opacity=0.6), row=3, col=1, secondary_y=True)

    # R4: Asset Strength (Line Chart)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['strength_diff'], name="SPY - TLT Return Diff", line=dict(color='purple', width=2)), row=4, col=1, secondary_y=True)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=4, col=1, secondary_y=True)

    # R5: VIX
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['vix'], name="VIX", line=dict(color='red', width=1)), row=5, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['vix_ma50'], name="VIX MA50", line=dict(color='darkred', width=1.5, dash='dash')), row=5, col=1, secondary_y=True)

    # R6: Sector Performance (今日漲跌幅)
    sect_colors = ['green' if v >= 0 else 'red' for v in sector_perf.values]
    fig.add_trace(go.Bar(
        x=sector_perf.index, 
        y=sector_perf.values,
        marker_color=sect_colors,
        text=sector_perf.values,
        texttemplate='%{y:.2f}%',
        textposition='auto',
        name="Sector Change"
    ), row=6, col=1)

    # Tables
    def add_table(row, col, df, cols=['Ticker', 'Close', 'Chg%', '52W High', '52W Low', 'Val']):
        fig.add_trace(go.Table(
            header=dict(values=cols, fill_color='navy', font=dict(color='white'), align='left'),
            cells=dict(values=[df[k] for k in df.columns], fill_color='lavender', align='left')
        ), row=row, col=col)

    def fmt(df, val_col, format_str):
        d = df[['Ticker', 'Close', 'Change %', '52W High', '52W Low', val_col]].copy()
        d['Close'] = d['Close'].map('{:,.2f}'.format)
        d['Change %'] = d['Change %'].map('{:+.2f}%'.format)
        d['52W High'] = d['52W High'].map('{:,.2f}'.format)
        d['52W Low'] = d['52W Low'].map('{:,.2f}'.format)
        d[val_col] = d[val_col].map(format_str.format)
        return d

    add_table(7, 1, fmt(df_snapshot.sort_values('Change %', ascending=False).head(10), 'RVol', '{:.2f}x'))
    add_table(7, 2, fmt(df_snapshot.sort_values('Change %', ascending=True).head(10), 'RVol', '{:.2f}x'))
    add_table(8, 1, fmt(df_snapshot.sort_values('Volatility', ascending=False).head(10), 'Volatility', '{:.2f}%'))
    add_table(8, 2, fmt(df_snapshot.sort_values('Bias 20(%)', ascending=False).head(10), 'Bias 20(%)', '{:+.2f}%'))
    add_table(9, 1, fmt(df_snapshot.sort_values('Bias 20(%)', ascending=True).head(10), 'Bias 20(%)', '{:+.2f}%'))
    vol_up = df_snapshot[df_snapshot['Change %'] > 0].sort_values('RVol', ascending=False).head(10)
    add_table(9, 2, fmt(vol_up, 'RVol', '{:.2f}x'))
    vol_down = df_snapshot[df_snapshot['Change %'] < 0].sort_values('RVol', ascending=False).head(10)
    add_table(10, 1, fmt(vol_down, 'RVol', '{:.2f}x'))

    fig.update_layout(height=3000, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
