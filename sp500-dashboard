import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import numpy as np

# 設定網頁寬度
st.set_page_config(layout="wide", page_title="S&P 500 Market Dashboard")

# ==========================================
# 1. 核心數據定義
# ==========================================

RAW_SECTOR_DATA = {
    'XLB (原物料)': ['LIN', 'NEM', 'SHW', 'ECL', 'NUE', 'FCX', 'DD', 'PPG', 'DOW', 'APD', 'CTVA', 'ALB'],
    'XLC (通訊)': ['META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'VZ', 'CMCSA', 'TMUS', 'T', 'WBD'],
    'XLE (能源)': ['XOM', 'CVX', 'COP', 'WMB', 'MPC', 'EOG', 'SLB', 'VLO', 'KMI', 'OXY'],
    'XLF (金融)': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'C', 'BLK', 'SPGI', 'CB', 'CME'],
    'XLI (工業)': ['GE', 'CAT', 'RTX', 'UBER', 'BA', 'GEV', 'ETN', 'HON', 'UNP', 'DE', 'ADP', 'LMT', 'UPS', 'MMM'],
    'XLK (科技)': ['NVDA', 'MSFT', 'AAPL', 'AVGO', 'PLTR', 'AMD', 'ORCL', 'IBM', 'CSCO', 'MU', 'QCOM', 'AMAT', 'INTC', 'TXN'],
    'XLP (必需消費)': ['WMT', 'COST', 'PG', 'KO', 'PM', 'PEP', 'MO', 'MDLZ', 'CL', 'TGT'],
    'XLRE (房地產)': ['PLD', 'AMT', 'EQIX', 'SPG', 'PSA', 'O', 'DLR', 'CCI', 'VICI', 'CSGP'],
    'XLU (公用事業)': ['NEE', 'CEG', 'SO', 'DUK', 'AEP', 'VST', 'SRE', 'D', 'EXC', 'XEL'],
    'XLV (醫療)': ['LLY', 'JNJ', 'ABBV', 'UNH', 'ABT', 'MRK', 'TMO', 'ISRG', 'AMGN', 'PFE', 'GILD', 'CVS'],
    'XLY (非必需消費)': ['AMZN', 'TSLA', 'HD', 'MCD', 'BKNG', 'TJX', 'LOW', 'SBUX', 'NKE', 'GM']
}

@st.cache_data(ttl=3600)
def parse_sector_data():
    tickers = []
    sector_map = {}
    for sector, stocks in RAW_SECTOR_DATA.items():
        for stock in stocks:
            tickers.append(stock)
            sector_map[stock] = sector
    return list(set(tickers)), sector_map

# ==========================================
# 2. 數據下載與計算
# ==========================================

@st.cache_data(ttl=3600)
def get_market_data(tickers):
    all_tickers = tickers + ['^GSPC', 'TLT', '^VIX'] 
    try:
        # 下載近 2 年數據
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
    
    # B. 創 52 週新高/新低 (NH-NL)
    roll_max_252 = high_df.rolling(window=252).max()
    roll_min_252 = low_df.rolling(window=252).min()
    new_highs = (high_df >= roll_max_252).sum(axis=1)
    new_lows = (low_df <= roll_min_252).sum(axis=1)
    net_new_highs = new_highs - new_lows
    
    # C. 騰落指標 (A/D Line 20MA)
    daily_change = close_df.diff()
    advancing = (daily_change > 0).sum(axis=1)
    declining = (daily_change < 0).sum(axis=1)
    net_adv_dec = advancing - declining
    ad_ma20 = net_adv_dec.rolling(window=20).mean()
    
    # D. 資產強弱 (20日報酬率相減)
    # 邏輯：SP500_Ret - TLT_Ret
    sp500_ret_20 = sp500.pct_change(20) * 100
    tlt_ret_20 = tlt.pct_change(20) * 100
    
    strength_diff = sp500_ret_20 - tlt_ret_20
    
    # E. VIX
    vix_ma50 = vix.rolling(window=50).mean()

    lookback = 130 # 半年
    return {
        'dates': sp500.index[-lookback:],
        'sp500': sp500.iloc[-lookback:],
        'breadth_pct': breadth_pct.iloc[-lookback:],
        'net_nh_nl': net_new_highs.iloc[-lookback:],
        'ad_ma20': ad_ma20.iloc[-lookback:],
        'strength_diff': strength_diff.iloc[-lookback:], # 新邏輯
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
            if df.empty or len(df) < 60: continue
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            change_pct = ((curr['Close'] - prev['Close']) / prev['Close']) * 100
            turnover = curr['Close'] * curr['Volume']
            ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
            bias_20 = ((curr['Close'] - ma20) / ma20) * 100
            volatility = ((curr['High'] - curr['Low']) / prev['Close']) * 100
            avg_vol_20 = df['Volume'].iloc[-22:-2].mean()
            r_vol = curr['Volume'] / avg_vol_20 if avg_vol_20 > 0 else 0
            
            results.append({
                'Ticker': ticker,
                'Close': curr['Close'],
                'Change %': change_pct,
                'Turnover': turnover,
                'Bias 20(%)': bias_20,
                'Volatility': volatility,
                'RVol': r_vol
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
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner('Downloading Market Data...'):
        tickers, sector_map = parse_sector_data()
        full_data = get_market_data(tickers)
    
    if full_data.empty:
        st.error("Failed to download data.")
        return

    with st.spinner('Calculating Indicators...'):
        mkt = calculate_market_indicators(full_data, tickers)
        df_snapshot = get_latest_snapshot(full_data, tickers)
        df_snapshot['Sector'] = df_snapshot['Ticker'].map(sector_map)
        
        # 計算產業平均漲跌幅
        sector_perf = df_snapshot.groupby('Sector')['Change %'].mean().sort_values(ascending=False)

    # 繪圖 Layout
    # R1-R5: Charts
    # R6: Sector Bar Chart (New)
    # R7-R10: Scanners
    fig = make_subplots(
        rows=10, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.08, 0.08, 0.08, 0.08],
        specs=[
            [{"colspan": 2, "secondary_y": True}, None], # R1
            [{"colspan": 2, "secondary_y": True}, None], # R2
            [{"colspan": 2, "secondary_y": True}, None], # R3
            [{"colspan": 2, "secondary_y": True}, None], # R4: Asset Strength (Diff)
            [{"colspan": 2, "secondary_y": True}, None], # R5
            [{"colspan": 2, "secondary_y": False}, None],# R6: Sector Perf (New)
            [{"type": "table"}, {"type": "table"}],
            [{"type": "table"}, {"type": "table"}],
            [{"type": "table"}, {"type": "table"}],
            [{"type": "table"}, {"type": "table"}]
        ],
        vertical_spacing=0.06,
        subplot_titles=(
            "市場廣度：站上 60MA 比例 vs S&P 500",
            "市場內部：淨 52 週新高家數 (折線圖)",
            "市場動能：20日平均淨上漲家數 (Net Adv-Dec) vs S&P 500",
            "資產強弱：(S&P500 20日報酬 - TLT 20日報酬) 差值 (Positive = Risk On)",
            "恐慌指數：VIX vs 50日均線",
            "各產業今日平均漲跌幅 (Sector Performance)",
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

    # R2: NH-NL (Line)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['net_nh_nl'], name="Net Highs-Lows", line=dict(color='green', width=2)), row=2, col=1, secondary_y=True)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1, secondary_y=True)

    # R3: A/D Line
    ad_colors = ['green' if v >= 0 else 'red' for v in mkt['ad_ma20']]
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=x_axis, y=mkt['ad_ma20'], name="20MA Net Adv-Dec", marker_color=ad_colors, opacity=0.6), row=3, col=1, secondary_y=True)

    # R4: Asset Strength (Diff) - Updated to Difference
    diff_colors = ['green' if v >= 0 else 'red' for v in mkt['strength_diff']]
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=x_axis, y=mkt['strength_diff'], name="SPY - TLT Return Diff", marker_color=diff_colors, opacity=0.6), row=4, col=1, secondary_y=True)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=4, col=1, secondary_y=True)

    # R5: VIX
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['sp500'], name="S&P 500", showlegend=False, line=dict(color='black', width=1)), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['vix'], name="VIX", line=dict(color='red', width=1)), row=5, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x_axis, y=mkt['vix_ma50'], name="VIX MA50", line=dict(color='darkred', width=1.5, dash='dash')), row=5, col=1, secondary_y=True)

    # R6: Sector Performance (New)
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
    def add_table(row, col, df, cols=['Ticker', 'Close', 'Chg%', 'Val']):
        fig.add_trace(go.Table(
            header=dict(values=cols, fill_color='navy', font=dict(color='white'), align='left'),
            cells=dict(values=[df[k] for k in df.columns], fill_color='lavender', align='left')
        ), row=row, col=col)

    def fmt(df, val_col, format_str):
        d = df[['Ticker', 'Close', 'Change %', val_col]].copy()
        d['Close'] = d['Close'].map('{:,.2f}'.format)
        d['Change %'] = d['Change %'].map('{:+.2f}%'.format)
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
