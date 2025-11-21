import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import numpy as np

st.set_page_config(layout="wide", page_title="S&P 500 Pro Market Dashboard")

# ==========================================
# 1. 核心數據定義 (完整 S&P 500 成分股)
# ==========================================

SECTOR_ETF_MAP = {
    'XLB (原物料)': 'XLB', 'XLC (通訊)': 'XLC', 'XLE (能源)': 'XLE',
    'XLF (金融)': 'XLF', 'XLI (工業)': 'XLI', 'XLK (科技)': 'XLK',
    'XLP (必需消費)': 'XLP', 'XLRE (房地產)': 'XLRE', 'XLU (公用事業)': 'XLU',
    'XLV (醫療)': 'XLV', 'XLY (非必需消費)': 'XLY'
}

# 為了確保廣度指標 (TRIN, A/D Line) 的準確性，這裡列出完整的 S&P 500 成分股清單
# 由於資料量大，這裡使用一個較大的代表性清單
RAW_SECTOR_DATA = {
    'XLB': ['LIN', 'NEM', 'SHW', 'ECL', 'NUE', 'FCX', 'DD', 'VMC', 'MLM', 'APD', 'CTVA', 'IP', 'STLD', 'PPG', 'SW', 'AMCR', 'DOW', 'PKG', 'IFF', 'AVY', 'CF', 'BALL', 'LYB', 'ALB', 'MOS', 'EMN'],
    'XLC': ['META', 'GOOGL', 'GOOG', 'WBD', 'NFLX', 'EA', 'TTWO', 'DIS', 'VZ', 'CMCSA', 'TMUS', 'T', 'LYV', 'CHTR', 'TTD', 'OMC', 'TKO', 'FOXA', 'NWSA', 'IPG', 'FOX', 'MTCH', 'PSKY', 'NWS'],
    'XLE': ['XOM', 'CVX', 'COP', 'WMB', 'MPC', 'EOG', 'PSX', 'SLB', 'VLO', 'KMI', 'BKR', 'OKE', 'TRGP', 'EQT', 'OXY', 'FANG', 'EXE', 'HAL', 'DVN', 'TPL', 'CTRA', 'APA'],
    'XLF': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'C', 'SCHW', 'BLK', 'SPGI', 'COF', 'PGR', 'HOOD', 'BX', 'CB', 'CME', 'MMC', 'ICE', 'KKR', 'COIN', 'BK', 'MCO', 'USB', 'PNC', 'AON', 'AJG', 'PYPL', 'TRV', 'APO', 'TFC', 'AFL', 'ALL', 'AMP', 'MSCI', 'MET', 'AIG', 'SQ', 'NDAQ', 'FI', 'PRU', 'HIG', 'STT', 'FIS', 'ACGL', 'WTW', 'IBKR', 'MTB', 'RJF', 'FITB', 'SYF', 'NTRS', 'CBOE', 'HBAN', 'CINF', 'BRO', 'TROW', 'CFG', 'RF', 'WRB', 'GPN', 'CPAY', 'PFG', 'L', 'KEY', 'EG', 'JKHY', 'IVZ', 'GL', 'AIZ', 'FDS', 'ERIE', 'BEN'],
    'XLI': ['GE', 'CAT', 'RTX', 'UBER', 'BA', 'GEV', 'ETN', 'HON', 'UNP', 'DE', 'ADP', 'LMT', 'PH', 'TT', 'MMM', 'GD', 'HWM', 'NOC', 'EMR', 'JCI', 'TDG', 'WM', 'UPS', 'PWR', 'CSX', 'ITW', 'CTAS', 'NSC', 'CMI', 'AXON', 'URI', 'FDX', 'LHX', 'PCAR', 'CARR', 'FAST', 'RSG', 'AME', 'GWW', 'ROK', 'DAL', 'PAYX', 'CPRT', 'XYL', 'OTIS', 'EME', 'WAB', 'UAL', 'VRSK', 'IR', 'EFX', 'BR', 'ODFL', 'HUBB', 'DOV', 'VLTO', 'LDOS', 'J', 'PNR', 'SNA', 'FTV', 'LUV', 'EXPD', 'LII', 'CHRW', 'ROL', 'TXT', 'ALLE', 'MAS', 'IEX', 'JBHT', 'BLDR', 'NDSN', 'HII', 'DAY', 'SWK', 'GNRC', 'PAYC', 'AOS'],
    'XLK': ['NVDA', 'MSFT', 'AAPL', 'AVGO', 'PLTR', 'AMD', 'ORCL', 'IBM', 'CSCO', 'MU', 'CRM', 'LRCX', 'QCOM', 'NOW', 'AMAT', 'INTU', 'INTC', 'APP', 'APH', 'ANET', 'KLAC', 'ACN', 'TXN', 'PANW', 'ADBE', 'CRWD', 'ADI', 'CDNS', 'SNPS', 'MSI', 'TEL', 'GLW', 'ADSK', 'STX', 'FTNT', 'MPWR', 'NXPI', 'DDOG', 'WDAY', 'DELL', 'WDC', 'ROP', 'FICO', 'CTSH', 'MCHP', 'HPE', 'KEYS', 'TER', 'SMCI', 'HPQ', 'FSLR', 'TDY', 'JBL', 'PTC', 'NTAP', 'ON', 'TYL', 'CDW', 'VRSN', 'IT', 'TRMB', 'GDDY', 'FFIV', 'GEN', 'ZBRA', 'SWKS', 'AKAM', 'EPAM'],
    'XLP': ['WMT', 'COST', 'PG', 'KO', 'PM', 'PEP', 'MO', 'MDLZ', 'CL', 'MNST', 'TGT', 'KR', 'KMB', 'KDP', 'SYY', 'ADM', 'KVUE', 'HSY', 'GIS', 'EL', 'K', 'DG', 'KHC', 'CHD', 'DLTR', 'STZ', 'MKC', 'TSN', 'CLX', 'BG', 'SJM', 'LW', 'CAG', 'TAP', 'HRL', 'CPB', 'BF-B'],
    'XLRE': ['WELL', 'PLD', 'AMT', 'EQIX', 'SPG', 'PSA', 'O', 'DLR', 'CBRE', 'CCI', 'VTR', 'VICI', 'EXR', 'IRM', 'CSGP', 'AVB', 'EQR', 'SBAC', 'WY', 'ESS', 'INVH', 'MAA', 'KIM', 'DOC', 'REG', 'HST', 'CPT', 'BXP', 'UDR', 'ARE', 'FRT'],
    'XLU': ['NEE', 'CEG', 'SO', 'DUK', 'AEP', 'VST', 'SRE', 'D', 'EXC', 'XEL', 'ETR', 'PEG', 'WEC', 'ED', 'PCG', 'NRG', 'DTE', 'AEE', 'ATO', 'ES', 'PPL', 'CNP', 'AWK', 'FE', 'CMS', 'EIX', 'NI', 'EVRG', 'LNT', 'PNW', 'AES'],
    'XLV': ['LLY', 'JNJ', 'ABBV', 'UNH', 'ABT', 'MRK', 'TMO', 'ISRG', 'AMGN', 'BSX', 'GILD', 'PFE', 'DHR', 'SYK', 'MDT', 'VRTX', 'CVS', 'MCK', 'BMY', 'CI', 'HCA', 'ELV', 'REGN', 'COR', 'ZTS', 'BDX', 'IDXX', 'EW', 'A', 'CAH', 'RMD', 'IQV', 'GEHC', 'HUM', 'MTD', 'DXCM', 'STE', 'PODD', 'BIIB', 'LH', 'WST', 'WAT', 'ZBH', 'DGX', 'CNC', 'HOLX', 'INCY', 'COO', 'UHS', 'VTRS', 'BAX', 'RVTY', 'SOLV', 'TECH', 'ALGN', 'CRL', 'MOH', 'MRNA', 'HSIC', 'DVA'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'BKNG', 'TJX', 'LOW', 'DASH', 'SBUX', 'ORLY', 'NKE', 'RCL', 'GM', 'AZO', 'HLT', 'MAR', 'ABNB', 'CMG', 'ROST', 'F', 'EBAY', 'DHI', 'YUM', 'GRMN', 'CCL', 'TSCO', 'LEN', 'EXPE', 'WSM', 'TPR', 'PHM', 'ULTA', 'DRI', 'NVR', 'APTV', 'LULU', 'LVS', 'GPC', 'BBY', 'DPZ', 'RL', 'DECK', 'HAS', 'WYNN', 'NCLH', 'POOL', 'LKQ', 'KMX', 'MGM', 'MHK']
}

@st.cache_data(ttl=3600)
def parse_sector_data():
    tickers = []
    sector_map = {}
    for sec, stocks in RAW_SECTOR_DATA.items():
        for s in stocks:
            tickers.append(s)
            sector_map[s] = sec
    return list(set(tickers)), sector_map

# ==========================================
# 2. 數據下載與計算
# ==========================================

@st.cache_data(ttl=3600)
def get_market_data(tickers):
    # 增加 ^VIX3M 用於計算期限結構
    sector_etfs = list(SECTOR_ETF_MAP.values())
    all_tickers = tickers + ['^GSPC', 'TLT', '^VIX', '^VIX3M'] + sector_etfs
    try:
        # 下載 2 年數據
        data = yf.download(all_tickers, period="2y", group_by='ticker', threads=True, auto_adjust=True)
        return data
    except Exception as e:
        st.error(f"數據下載錯誤: {e}")
        return pd.DataFrame()

def calculate_market_indicators(data, tickers):
    sp500 = data['^GSPC']['Close']
    tlt = data['TLT']['Close']
    vix = data['^VIX']['Close']
    vix3m = data['^VIX3M']['Close'] 
    
    benchmark_idx = sp500.index
    valid_tickers = [t for t in tickers if t in data]
    
    # 建立矩陣
    close_df = pd.DataFrame({t: data[t]['Close'] for t in valid_tickers}).reindex(benchmark_idx)
    high_df = pd.DataFrame({t: data[t]['High'] for t in valid_tickers}).reindex(benchmark_idx)
    low_df = pd.DataFrame({t: data[t]['Low'] for t in valid_tickers}).reindex(benchmark_idx)
    volume_df = pd.DataFrame({t: data[t]['Volume'] for t in valid_tickers}).reindex(benchmark_idx)
    
    # A. 市場廣度 (MA60)
    ma60_df = close_df.rolling(window=60).mean()
    above_ma60 = (close_df > ma60_df)
    valid_counts = ma60_df.notna().sum(axis=1)
    above_counts = above_ma60.sum(axis=1)
    breadth_pct = (above_counts / valid_counts * 100).fillna(0)
    
    # B. 累積淨新高
    roll_max_252 = high_df.rolling(window=252).max()
    roll_min_252 = low_df.rolling(window=252).min()
    new_highs = (high_df >= roll_max_252).sum(axis=1)
    new_lows = (low_df <= roll_min_252).sum(axis=1)
    net_nh_nl = new_highs - new_lows
    cum_net_highs = net_nh_nl.cumsum()
    
    # C. VIX 期限結構
    vix_term_structure = vix / vix3m
    
    # D. 資產強弱
    sp500_ret = sp500.pct_change(20) * 100
    tlt_ret = tlt.pct_change(20) * 100
    strength_diff = sp500_ret - tlt_ret

    # E. TRIN (Arms Index)
    daily_change = close_df.diff()
    up_mask = daily_change > 0
    down_mask = daily_change < 0
    advancing_issues = up_mask.sum(axis=1)
    declining_issues = down_mask.sum(axis=1)
    advancing_volume = (volume_df * up_mask).sum(axis=1)
    declining_volume = (volume_df * down_mask).sum(axis=1)
    
    ad_ratio = advancing_issues / declining_issues.replace(0, 1)
    vol_ratio = advancing_volume / declining_volume.replace(0, 1)
    trin = ad_ratio / vol_ratio
    
    lookback = 130
    return {
        'dates': sp500.index[-lookback:],
        'sp500': sp500.iloc[-lookback:],
        'breadth_pct': breadth_pct.iloc[-lookback:],
        'cum_net_highs': cum_net_highs.iloc[-lookback:], 
        'vix_term': vix_term_structure.iloc[-lookback:], 
        'strength_diff': strength_diff.iloc[-lookback:],
        'vix': vix.iloc[-lookback:],
        'trin': trin.iloc[-lookback:]
    }

def calculate_rrg_data(data):
    sp500 = data['^GSPC']['Close']
    rrg_data = []
    
    for name, ticker in SECTOR_ETF_MAP.items():
        if ticker in data:
            sector_close = data[ticker]['Close']
            rs = sector_close / sp500
            rs_trend = rs.rolling(window=10).mean()
            rs_mean = rs_trend.rolling(window=60).mean()
            rs_std = rs_trend.rolling(window=60).std()
            x_val = ((rs_trend - rs_mean) / rs_std).iloc[-1]
            x_val_prev = ((rs_trend - rs_mean) / rs_std).iloc[-10]
            y_val = x_val - x_val_prev
            
            # 產業今日漲跌 (ETF)
            df = data[ticker]
            df = df.dropna(subset=['Close'])
            if len(df) >= 2:
                curr = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2]
                chg = ((curr - prev) / prev) * 100
            else:
                chg = 0
            
            rrg_data.append({
                'Sector': name.split(' ')[0],
                'X': x_val,
                'Y': y_val,
                'Change': chg
            })
    return pd.DataFrame(rrg_data)

def get_sector_performance(data):
    """計算 ETF 的今日漲跌幅"""
    sector_changes = {}
    for name, ticker in SECTOR_ETF_MAP.items():
        try:
            if ticker in data:
                df = data[ticker]
                df = df.dropna(subset=['Close'])
                if len(df) >= 2:
                    curr = df['Close'].iloc[-1]
                    prev = df['Close'].iloc[-2]
                    change = ((curr - prev) / prev) * 100
                    sector_changes[name] = change
        except:
            continue
    return pd.Series(sector_changes).sort_values(ascending=False)

def get_latest_snapshot_with_strategy(data, tickers):
    results = []
    for ticker in tickers:
        try:
            if ticker not in data: continue
            df = data[ticker]
            df = df.dropna(subset=['Close', 'Volume'])
            if df.empty or len(df) < 252: continue
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            close = float(curr['Close'])
            
            change_pct = ((close - prev['Close']) / prev['Close']) * 100
            turnover = close * float(curr['Volume'])
            
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            ma150 = df['Close'].rolling(150).mean().iloc[-1]
            ma200 = df['Close'].rolling(200).mean().iloc[-1]
            
            high_52w = df['High'].tail(252).max()
            low_52w = df['Low'].tail(252).min()
            
            # 策略 1: 趨勢模板
            trend_score = 0
            if close > ma50 > ma150 > ma200: trend_score += 1
            if close > low_52w * 1.3: trend_score += 1
            if close > high_52w * 0.75: trend_score += 1
            is_super_trend = (trend_score == 3)
            
            # 策略 2: 口袋支點
            is_pocket_pivot = False
            if change_pct > 0:
                last_10 = df.iloc[-11:-1]
                down_days = last_10[last_10['Close'] < last_10['Open']]
                if not down_days.empty:
                    max_down_vol = down_days['Volume'].max()
                    if curr['Volume'] > max_down_vol:
                        is_pocket_pivot = True
                elif curr['Volume'] > last_10['Volume'].max():
                     is_pocket_pivot = True

            # RVol
            avg_vol_20 = df['Volume'].iloc[-22:-2].mean()
            r_vol = curr['Volume'] / avg_vol_20 if avg_vol_20 > 0 else 0
            
            # Bias & Volatility (For Basic Scanners)
            ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
            bias_20 = ((close - ma20) / ma20) * 100
            volatility = ((curr['High'] - curr['Low']) / prev['Close']) * 100

            results.append({
                'Ticker': ticker,
                'Close': close,
                'Change %': change_pct,
                'Turnover': turnover,
                'RVol': r_vol,
                '52W High': high_52w,
                '52W Low': low_52w,
                'Super Trend': is_super_trend,
                'Pocket Pivot': is_pocket_pivot,
                'Bias 20(%)': bias_20,
                'Volatility': volatility
            })
        except:
            continue
    return pd.DataFrame(results)

# ==========================================
# 3. 視覺化
# ==========================================

def main():
    st.title("📊 S&P 500 Pro Market Dashboard (策略升級版)")
    st.write(f"Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner('Downloading & Calculating (Full S&P 500 Strategies)...'):
        tickers, sector_map = parse_sector_data()
        full_data = get_market_data(tickers)
        
        if full_data.empty:
            st.error("Data download failed.")
            return

        mkt = calculate_market_indicators(full_data, tickers)
        df_snapshot = get_latest_snapshot_with_strategy(full_data, tickers)
        df_snapshot['Sector'] = df_snapshot['Ticker'].map(sector_map)
        
        rrg_df = calculate_rrg_data(full_data)
        sector_perf = get_sector_performance(full_data)

    # Layout
    # Row 1-6: Charts (R6 is TRIN)
    # Row 7: Sector Perf
    # Row 8-11: Scanners
    fig = make_subplots(
        rows=11, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.07, 0.07, 0.07, 0.07],
        specs=[
            [{"colspan": 2, "secondary_y": True}, None], # R1: Breadth
            [{"colspan": 2, "secondary_y": True}, None], # R2: Cumul NH/NL
            [{"colspan": 2, "secondary_y": True}, None], # R3: VIX Term
            [{"colspan": 2, "secondary_y": True}, None], # R4: Strength Diff
            [{"colspan": 2, "type": "scatter"}, None],   # R5: RRG
            [{"colspan": 2, "secondary_y": True}, None], # R6: TRIN (New)
            [{"colspan": 2, "secondary_y": False}, None],# R7: Sector Perf
            [{"type": "table"}, {"type": "table"}],      # R8: Trend/Pocket
            [{"type": "table"}, {"type": "table"}],      # R9
            [{"type": "table"}, {"type": "table"}],      # R10
            [{"type": "table"}, {"type": "table"}]       # R11
        ],
        vertical_spacing=0.05,
        subplot_titles=(
            "市場廣度：站上 60MA 比例 vs S&P 500",
            "市場趨勢：累積淨新高線 (Cumulative Net Highs)",
            "恐慌結構：VIX / VIX3M 比率 (>1.0 恐慌)",
            "資產強弱：(SPY - TLT) 20日報酬差值",
            "動態板塊輪動 (RRG Proxy)",
            "量價結構：TRIN (阿姆斯指數) - (>2.0 恐慌清洗, <0.5 極度貪婪)",
            "各產業 ETF 今日漲跌幅",
            "🔥 超級趨勢股 (Minervini Trend Template)", "💎 口袋支點爆量 (Pocket Pivot)",
            "1. 漲幅最強", "2. 跌幅最重",
            "3. 高波動度", "4. 爆量上漲",
            "5. 爆量下跌", ""
        )
    )

    x_axis = mkt['dates']

    # R1: Breadth
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", line=dict(color='black', width=1)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['breadth_pct'], name="% > MA60", line=dict(color='blue', width=2), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="比例 (%)", range=[0, 100], secondary_y=True, row=1, col=1)


    # R2: Cumul Net Highs
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['cum_net_highs'], name="Cumul Net Highs", line=dict(color='green', width=2)), row=2, col=1, secondary_y=True)

    # R3: VIX Term
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['vix_term'], name="VIX/VIX3M", line=dict(color='red', width=2)), row=3, col=1, secondary_y=True)
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", row=3, col=1, secondary_y=True)

    # R4: Asset Strength
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['strength_diff'], name="SPY - TLT Diff", line=dict(color='purple', width=2)), row=4, col=1, secondary_y=True)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=4, col=1, secondary_y=True)

    # R5: RRG
    fig.add_trace(go.Scatter(
        x=rrg_df['X'], y=rrg_df['Y'], mode='markers+text', text=rrg_df['Sector'],
        textposition='top center',
        marker=dict(size=20, color=rrg_df['Change'], colorscale='RdYlGn', showscale=True, colorbar=dict(title="Today %", len=0.2, y=0.5)),
        name="Sectors"
    ), row=5, col=1)
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray", row=5, col=1)
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", row=5, col=1)

    # R6: TRIN (New)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=6, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['trin'], name="TRIN", line=dict(color='orange', width=2)), row=6, col=1, secondary_y=True)
    fig.add_hline(y=1.0, line_dash="solid", line_color="gray", row=6, col=1, secondary_y=True)
    fig.add_hline(y=2.0, line_dash="dot", line_color="red", annotation_text="Panic (>2.0)", row=6, col=1, secondary_y=True)
    fig.add_hline(y=0.5, line_dash="dot", line_color="green", annotation_text="Greed (<0.5)", row=6, col=1, secondary_y=True)
    fig.update_yaxes(range=[0, 3], secondary_y=True, row=6, col=1)

    # R7: Sector Perf (ETF)
    sect_colors = ['green' if v >= 0 else 'red' for v in sector_perf.values]
    fig.add_trace(go.Bar(
        x=sector_perf.index, y=sector_perf.values, marker_color=sect_colors,
        text=sector_perf.values, texttemplate='%{y:.2f}%', textposition='auto', name="Sector Change"
    ), row=7, col=1)

    # --- Tables ---
    def add_table(row, col, df, cols):
        fig.add_trace(go.Table(
            header=dict(values=cols, fill_color='navy', font=dict(color='white'), align='left'),
            cells=dict(values=[df[k] for k in df.columns], fill_color='lavender', align='left')
        ), row=row, col=col)

    def fmt(df, val_col=None, fmt_str='{:.2f}'):
        d = df.copy()
        d['Close'] = d['Close'].map('{:,.2f}'.format)
        d['Change %'] = d['Change %'].map('{:+.2f}%'.format)
        d['52W High'] = d['52W High'].map('{:,.2f}'.format)
        d['52W Low'] = d['52W Low'].map('{:,.2f}'.format)
        if val_col and val_col in d.columns and fmt_str:
            d[val_col] = d[val_col].map(fmt_str.format)
        return d

    # R8: Strategies
    df_super = df_snapshot[df_snapshot['Super Trend'] == True].sort_values('RVol', ascending=False).head(10)
    cols_strat = ['Ticker', 'Close', 'Change %', 'RVol', '52W High', '52W Low']
    add_table(8, 1, fmt(df_super[cols_strat], 'RVol', '{:.2f}x'), cols_strat)
    
    df_pocket = df_snapshot[df_snapshot['Pocket Pivot'] == True].sort_values('Change %', ascending=False).head(10)
    add_table(8, 2, fmt(df_pocket[cols_strat], 'RVol', '{:.2f}x'), cols_strat)

    # R9-R11: Basic Scanners
    cols_basic = ['Ticker', 'Close', 'Change %', '52W High', '52W Low', 'Val']
    
    gainer_df = df_snapshot.sort_values('Change %', ascending=False).head(10)[['Ticker','Close','Change %','52W High','52W Low','RVol']]
    gainer_df.columns = ['Ticker','Close','Change %','52W High','52W Low','Val']
    add_table(9, 1, fmt(gainer_df, 'Val', '{:.2f}x'), cols_basic)
    
    loser_df = df_snapshot.sort_values('Change %', ascending=True).head(10)[['Ticker','Close','Change %','52W High','52W Low','RVol']]
    loser_df.columns = ['Ticker','Close','Change %','52W High','52W Low','Val']
    add_table(9, 2, fmt(loser_df, 'Val', '{:.2f}x'), cols_basic)
    
    high_vol = df_snapshot.sort_values('Volatility', ascending=False).head(10)[['Ticker','Close','Change %','52W High','52W Low','Volatility']]
    high_vol.columns = ['Ticker','Close','Change %','52W High','52W Low','Val']
    add_table(10, 1, fmt(high_vol, 'Val', '{:.2f}%'), cols_basic)
    
    vol_up = df_snapshot[df_snapshot['Change %'] > 0].sort_values('RVol', ascending=False).head(10)[['Ticker','Close','Change %','52W High','52W Low','RVol']]
    vol_up.columns = ['Ticker','Close','Change %','52W High','52W Low','Val']
    add_table(10, 2, fmt(vol_up, 'Val', '{:.2f}x'), cols_basic)
    
    vol_down = df_snapshot[df_snapshot['Change %'] < 0].sort_values('RVol', ascending=False).head(10)[['Ticker','Close','Change %','52W High','52W Low','RVol']]
    vol_down.columns = ['Ticker','Close','Change %','52W High','52W Low','Val']
    add_table(11, 1, fmt(vol_down, 'Val', '{:.2f}x'), cols_basic)

    fig.update_layout(height=3500, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
