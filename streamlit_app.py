import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import numpy as np

# ==========================================
# 1. 核心數據定義 (S&P 500 完整成分股)
# ==========================================

# 板塊 ETF 映射表
SECTOR_ETF_MAP = {
    'XLB (原物料)': 'XLB', 'XLC (通訊)': 'XLC', 'XLE (能源)': 'XLE',
    'XLF (金融)': 'XLF', 'XLI (工業)': 'XLI', 'XLK (科技)': 'XLK',
    'XLP (必需消費)': 'XLP', 'XLRE (房地產)': 'XLRE', 'XLU (公用事業)': 'XLU',
    'XLV (醫療)': 'XLV', 'XLY (非必需消費)': 'XLY'
}

# 產業名稱中文化對照表 (用於個股分類顯示)
SECTOR_NAME_MAP = {
    'XLB': '原物料 (XLB)', 'XLC': '通訊 (XLC)', 'XLE': '能源 (XLE)',
    'XLF': '金融 (XLF)', 'XLI': '工業 (XLI)', 'XLK': '科技 (XLK)',
    'XLP': '必需消費 (XLP)', 'XLRE': '房地產 (XLRE)', 'XLU': '公用事業 (XLU)',
    'XLV': '醫療 (XLV)', 'XLY': '非必需消費 (XLY)'
}

# 完整 S&P 500 成分股清單 (為了廣度指標精確性)
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

def parse_sector_data():
    tickers = []
    sector_map = {}
    for sec, stocks in RAW_SECTOR_DATA.items():
        for s in stocks:
            tickers.append(s)
            sector_map[s] = SECTOR_NAME_MAP.get(sec, sec) # 使用中文產業名稱映射
    return list(set(tickers)), sector_map

# ==========================================
# 2. 數據下載與計算
# ==========================================

def get_full_historical_data(tickers):
    sector_etfs = list(SECTOR_ETF_MAP.values())
    # 增加下載: Sector ETFs, HYG, VIX3M
    all_tickers = tickers + ['^GSPC', 'TLT', '^VIX', '^VIX3M', 'HYG'] + sector_etfs
    print(f"正在下載 {len(all_tickers)} 檔標的歷史數據 (Period: 2y)...")
    
    try:
        data = yf.download(all_tickers, period="2y", group_by='ticker', threads=True, auto_adjust=True)
        return data
    except Exception as e:
        print(f"數據下載錯誤: {e}")
        return pd.DataFrame()

def calculate_market_indicators(data, tickers):
    print("正在運算進階市場指標...")
    
    sp500 = data['^GSPC']['Close']
    tlt = data['TLT']['Close']
    hyg = data['HYG']['Close']
    vix = data['^VIX']['Close']
    vix3m = data['^VIX3M']['Close']
    
    benchmark_idx = sp500.index
    valid_tickers = [t for t in tickers if t in data]
    
    close_df = pd.DataFrame({t: data[t]['Close'] for t in valid_tickers}).reindex(benchmark_idx)
    high_df = pd.DataFrame({t: data[t]['High'] for t in valid_tickers}).reindex(benchmark_idx)
    low_df = pd.DataFrame({t: data[t]['Low'] for t in valid_tickers}).reindex(benchmark_idx)
    volume_df = pd.DataFrame({t: data[t]['Volume'] for t in valid_tickers}).reindex(benchmark_idx)
    
    # A. 廣度 (MA60)
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
    
    # D. 資產強弱 (SPY vs TLT)
    sp500_ret = sp500.pct_change(20) * 100
    tlt_ret = tlt.pct_change(20) * 100
    strength_diff = sp500_ret - tlt_ret

    # E. TRIN
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

    # F. 平均近 20 日上漲-下跌家數
    net_advances = advancing_issues - declining_issues
    ad_ma20 = net_advances.rolling(window=20).mean()

    # G. 風險偏好: 高收益債/公債 (HYG/TLT)
    hyg_tlt_ratio = hyg / tlt
    
    lookback = 130
    return {
        'dates': sp500.index[-lookback:],
        'sp500': sp500.iloc[-lookback:],
        'breadth_pct': breadth_pct.iloc[-lookback:],
        'cum_net_highs': cum_net_highs.iloc[-lookback:], 
        'vix_term': vix_term_structure.iloc[-lookback:], 
        'strength_diff': strength_diff.iloc[-lookback:],
        'vix': vix.iloc[-lookback:],
        'trin': trin.iloc[-lookback:],
        'ad_ma20': ad_ma20.iloc[-lookback:],
        'hyg_tlt': hyg_tlt_ratio.iloc[-lookback:]
    }

def calculate_rrg_data(data):
    sp500 = data['^GSPC']['Close']
    rrg_data = []
    for name, ticker in SECTOR_ETF_MAP.items():
        # 提取中文簡稱: "XLB (原物料)" -> "原物料"
        short_name = name.split('(')[1].strip(')') if '(' in name else name
        
        if ticker in data:
            sector_close = data[ticker]['Close']
            # 相對強度計算 (RRG 邏輯簡化版)
            rs = sector_close / sp500
            rs_trend = rs.rolling(window=10).mean()
            rs_mean = rs_trend.rolling(window=60).mean()
            rs_std = rs_trend.rolling(window=60).std()
            
            x_val = ((rs_trend - rs_mean) / rs_std).iloc[-1]
            x_val_prev = ((rs_trend - rs_mean) / rs_std).iloc[-10]
            y_val = x_val - x_val_prev
            
            df = data[ticker]
            df = df.dropna(subset=['Close'])
            chg = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
            
            rrg_data.append({'Sector': short_name, 'X': x_val, 'Y': y_val, 'Change': chg})
    return pd.DataFrame(rrg_data)

def get_sector_performance(data):
    sector_changes = {}
    for name, ticker in SECTOR_ETF_MAP.items():
        # 格式化顯示: "科技 (XLK)"
        short_name = name.split('(')[1].replace(')', '') + ' (' + name.split(' ')[0] + ')'
        try:
            if ticker in data:
                df = data[ticker]
                df = df.dropna(subset=['Close'])
                if len(df) >= 2:
                    curr = df['Close'].iloc[-1]
                    prev = df['Close'].iloc[-2]
                    change = ((curr - prev) / prev) * 100
                    sector_changes[short_name] = change
        except:
            continue
    return pd.Series(sector_changes).sort_values(ascending=False)

def get_latest_snapshot(data, tickers, sector_map):
    results = []
    print("正在提取最新選股數據...")
    
    for ticker in tickers:
        try:
            if ticker not in data: continue
            df = data[ticker]
            df = df.dropna(subset=['Close', 'Volume'])
            if df.empty or len(df) < 252: continue
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            close = float(curr['Close'])
            prev_close = float(prev['Close'])
            volume = float(curr['Volume'])
            
            change_pct = ((close - prev_close) / prev_close) * 100
            turnover = close * volume
            
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            ma150 = df['Close'].rolling(150).mean().iloc[-1]
            ma200 = df['Close'].rolling(200).mean().iloc[-1]
            high_52w = df['High'].tail(252).max()
            low_52w = df['Low'].tail(252).min()
            
            # 策略 1: 超級趨勢
            trend_score = 0
            if close > ma50 > ma150 > ma200: trend_score += 1
            if close > low_52w * 1.3: trend_score += 1
            if close > high_52w * 0.75: trend_score += 1
            is_super_trend = (trend_score == 3)
            
            # 策略 2: Pocket Pivot
            is_pocket_pivot = False
            if change_pct > 0:
                last_10 = df.iloc[-11:-1]
                down_days = last_10[last_10['Close'] < last_10['Open']]
                if not down_days.empty:
                    if curr['Volume'] > down_days['Volume'].max(): is_pocket_pivot = True
                elif curr['Volume'] > last_10['Volume'].max(): is_pocket_pivot = True
            
            # 其他指標
            avg_vol_20 = df['Volume'].iloc[-22:-2].mean()
            r_vol = volume / avg_vol_20 if avg_vol_20 > 0 else 0
            ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
            bias_20 = ((close - ma20) / ma20) * 100
            high = float(curr['High'])
            low = float(curr['Low'])
            volatility = ((high - low) / prev_close) * 100
            
            results.append({
                'Ticker': ticker,
                'Sector': sector_map.get(ticker, ''),
                'Close': close,
                'Change %': change_pct,
                'Turnover': turnover,
                'Bias 20(%)': bias_20,
                'Volatility': volatility,
                'RVol': r_vol,
                '52W High': high_52w,
                '52W Low': low_52w,
                'Super Trend': is_super_trend,
                'Pocket Pivot': is_pocket_pivot
            })
        except:
            continue
    return pd.DataFrame(results)

# ==========================================
# 3. 視覺化 (Final Layout)
# ==========================================

def generate_dashboard(mkt, df_snapshot, rrg_df, sector_perf):
    
    x_axis = mkt['dates']
    
    # 計算 S&P 500 Y軸動態範圍 (Zoom In)
    sp500_min = mkt['sp500'].min()
    sp500_max = mkt['sp500'].max()
    padding = (sp500_max - sp500_min) * 0.05 
    sp500_range = [sp500_min - padding, sp500_max + padding]

    fig = make_subplots(
        rows=11, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.12, 0.12, 0.12, 0.12, 0.15, 0.1, 0.1, 0.07, 0.07, 0.07, 0.07],
        specs=[
            [{"colspan": 2, "secondary_y": True}, None], # R1: Breadth
            [{"colspan": 2, "secondary_y": True}, None], # R2: Net Advances
            [{"colspan": 2, "secondary_y": True}, None], # R3: Cumul Net Highs
            [{"colspan": 2, "secondary_y": True}, None], # R4: TRIN
            [{"colspan": 2, "secondary_y": True}, None], # R5: VIX Term
            [{"colspan": 2, "secondary_y": False}, None],# R6: Asset Strength
            [{"colspan": 2, "secondary_y": True}, None], # R7: HYG/TLT
            [{"type": "scatter", "colspan": 2}, None],   # R8: RRG
            [{"type": "bar", "colspan": 2}, None],       # R9: Sector Perf
            [{"type": "table"}, {"type": "table"}],      # R10
            [{"type": "table"}, {"type": "table"}]       # R11
        ],
        # 修正 Layout: 我們將 RRG 和 Sector Perf 移到下面，並且將 Table 移到最後
        # 為了符合您的要求 (類似那斯達克的順序)
        # 實際上 Plotly Subplots 很難動態插入標題，我們在 HTML 輸出時可能無法像 Streamlit 那樣分段
        # 但這裡我們盡力排列整齊
        
        # 重新設計:
        # R1: Breadth
        # R2: Net Advances (New)
        # R3: Cumul NH/NL
        # R4: TRIN
        # R5: VIX Term
        # R6: Asset Strength
        # R7: HYG/TLT (New)
        # R8: RRG
        # R9: Sector Perf
        # R10: Strategy Tables
        # R11: Scanner Tables
        vertical_spacing=0.04,
        subplot_titles=(
            "一、大盤健康度：市場廣度 (>50% 多頭)", 
            "市場動能：平均近 20 日淨上漲家數 (>0 多頭)",
            "市場趨勢：累積淨新高線 (向上=多頭)",
            "量價結構：TRIN (>2.0 恐慌, <0.5 貪婪)",
            "二、風險控管：VIX 期限結構 (>1.0 恐慌)",
            "資產強弱：SPY - TLT 20日報酬差",
            "風險偏好：HYG / TLT 比率 (向上=Risk On)",
            "三、板塊輪動：動態 RRG (右上領先)",
            "各產業 ETF 今日漲跌幅",
            "四、強勢股：🔥 超級趨勢股 (Super Trend)", "💎 口袋支點爆量",
            "🚀 漲幅最強 Top 10", "💥 爆量上漲 Top 10"
        )
    )
    
    # --- R1: Breadth ---
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", line=dict(color='black', width=1)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['breadth_pct'], name="% > MA60", line=dict(color='blue', width=2), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'), row=1, col=1, secondary_y=True)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=1, col=1, secondary_y=True)
    fig.update_yaxes(range=sp500_range, secondary_y=False, row=1, col=1)
    fig.update_yaxes(range=[0, 100], secondary_y=True, row=1, col=1)

    # --- R2: Net Advances (MA20) ---
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=x_axis, y=mkt['ad_ma20'], name="Net Adv MA20", marker_color=['green' if v>0 else 'red' for v in mkt['ad_ma20']]), row=2, col=1, secondary_y=True)
    fig.add_hline(y=0, line_color="gray", row=2, col=1, secondary_y=True)
    fig.update_yaxes(range=sp500_range, secondary_y=False, row=2, col=1)

    # --- R3: Cumul NH/NL ---
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['cum_net_highs'], name="Cumul Net Highs", line=dict(color='green', width=2)), row=3, col=1, secondary_y=True)
    fig.update_yaxes(range=sp500_range, secondary_y=False, row=3, col=1)

    # --- R4: TRIN ---
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['trin'], name="TRIN", line=dict(color='orange', width=2)), row=4, col=1, secondary_y=True)
    fig.add_hline(y=2.0, line_dash="dot", line_color="red", row=4, col=1, secondary_y=True)
    fig.add_hline(y=0.5, line_dash="dot", line_color="green", row=4, col=1, secondary_y=True)
    fig.update_yaxes(range=sp500_range, secondary_y=False, row=4, col=1)
    fig.update_yaxes(range=[0, 3], secondary_y=True, row=4, col=1)

    # --- R5: VIX Term ---
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['vix_term'], name="VIX/VIX3M", line=dict(color='red', width=2)), row=5, col=1, secondary_y=True)
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", row=5, col=1, secondary_y=True)
    fig.update_yaxes(range=sp500_range, secondary_y=False, row=5, col=1)

    # --- R6: Asset Strength ---
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=6, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['strength_diff'], name="SPY-TLT Diff", line=dict(color='purple', width=2)), row=6, col=1, secondary_y=True)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=6, col=1, secondary_y=True)
    fig.update_yaxes(range=sp500_range, secondary_y=False, row=6, col=1)

    # --- R7: HYG/TLT ---
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=7, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['hyg_tlt'], name="HYG/TLT Ratio", line=dict(color='orange', width=2)), row=7, col=1, secondary_y=True)
    fig.update_yaxes(range=sp500_range, secondary_y=False, row=7, col=1)

    # --- R8: RRG ---
    fig.add_trace(go.Scatter(
        x=rrg_df['X'], y=rrg_df['Y'], mode='markers+text', text=rrg_df['Sector'],
        textposition='top center',
        marker=dict(size=20, color=rrg_df['Change'], colorscale='RdYlGn', showscale=True, colorbar=dict(title="Today %", len=0.5)),
        name="Sectors"
    ), row=8, col=1)
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray", row=8, col=1)
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", row=8, col=1)
    # 象限標籤
    fig.add_annotation(x=2, y=2, text="領先", showarrow=False, font=dict(size=16, color="green"), opacity=0.5, row=8, col=1)
    fig.add_annotation(x=2, y=-2, text="轉弱", showarrow=False, font=dict(size=16, color="orange"), opacity=0.5, row=8, col=1)
    fig.add_annotation(x=-2, y=-2, text="落後", showarrow=False, font=dict(size=16, color="red"), opacity=0.5, row=8, col=1)
    fig.add_annotation(x=-2, y=2, text="改善", showarrow=False, font=dict(size=16, color="blue"), opacity=0.5, row=8, col=1)

    # --- R9: Sector Perf ---
    sect_colors = ['green' if v >= 0 else 'red' for v in sector_perf.values]
    fig.add_trace(go.Bar(
        x=sector_perf.index, y=sector_perf.values, marker_color=sect_colors,
        text=sector_perf.values, texttemplate='%{y:.2f}%', textposition='auto', name="Sector Change"
    ), row=9, col=1)

    # --- R10 & R11: Tables ---
    def add_table(row, col, df, cols):
        fig.add_trace(go.Table(
            header=dict(values=cols, fill_color='navy', font=dict(color='white'), align='left'),
            cells=dict(values=[df[k] for k in df.columns], fill_color='lavender', align='left')
        ), row=row, col=col)

    def fmt(df, val_col, format_str):
        d = df[['Ticker', 'Sector', 'Close', 'Change %', '52W High', '52W Low', val_col]].copy()
        d['Close'] = d['Close'].map('{:,.2f}'.format)
        d['Change %'] = d['Change %'].map('{:+.2f}%'.format)
        d['52W High'] = d['52W High'].map('{:,.2f}'.format)
        d['52W Low'] = d['52W Low'].map('{:,.2f}'.format)
        d[val_col] = d[val_col].map(format_str.format)
        return d

    cols_basic = ['Ticker', 'Sector', 'Close', 'Change %', '52W High', '52W Low', 'Val']
    
    # Super Trend
    df_super = df_snapshot[df_snapshot['Super Trend'] == True].sort_values('RVol', ascending=False).head(10)
    add_table(10, 1, fmt(df_super, 'RVol', '{:.2f}x'), cols_basic)
    
    # Pocket Pivot
    df_pocket = df_snapshot[df_snapshot['Pocket Pivot'] == True].sort_values('Change %', ascending=False).head(10)
    add_table(10, 2, fmt(df_pocket, 'RVol', '{:.2f}x'), cols_basic)
    
    # Top Gainers
    gainer_df = df_snapshot.sort_values('Change %', ascending=False).head(10)
    add_table(11, 1, fmt(gainer_df, 'RVol', '{:.2f}x'), cols_basic)
    
    # Vol Up
    vol_up = df_snapshot[df_snapshot['Change %'] > 0].sort_values('RVol', ascending=False).head(10)
    add_table(11, 2, fmt(vol_up, 'RVol', '{:.2f}x'), cols_basic)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.update_layout(
        title_text=f"S&P 500 Advanced Market Analysis (Generated: {current_time})", 
        height=3500, 
        template="plotly_white", 
        showlegend=False
    )
    return fig

if __name__ == "__main__":
    tickers, sector_map = parse_sector_data()
    full_data = get_full_historical_data(tickers)
    if not full_data.empty:
        mkt = calculate_market_indicators(full_data, tickers)
        snap = get_latest_snapshot(full_data, tickers, sector_map)
        rrg_df = calculate_rrg_data(full_data)
        sector_perf = get_sector_performance(full_data)
        
        fig = generate_dashboard(mkt, snap, rrg_df, sector_perf)
        fig.write_html("SP500_Advanced_Report.html", auto_open=True)
        print("\n報告生成完畢！")
    else:
        print("數據下載失敗。")
