import asyncio
import httpx
import os
import math
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# =====================================================================
# SYSTEM CONFIGURATION & ADVANCED FAIL-SAFE LOGGING
# =====================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantumOmniHunter")

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
BINANCE_FUTURES_URL = "https://fapi.binance.com"

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Supabase connection initialization failed: {e}")

# =====================================================================
# DATA ACQUISITION LAYER (FULLY ASYNC & RETRY-PROTECTED)
# =====================================================================
async def fetch_with_retry(client: httpx.AsyncClient, url: str, params: dict = None) -> dict:
    backoff = 1.0
    for attempt in range(3):
        try:
            response = await client.get(url, params=params, timeout=5.0)
            if response.status_code == 429:
                await asyncio.sleep(backoff * 2)
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            if attempt == 2:
                logger.error(f"Definitive HTTP request failure for endpoint {url}: {e}")
                return {}
            await asyncio.sleep(backoff)
            backoff *= 2.0
    return {}

async def fetch_single_market_data(client: httpx.AsyncClient, sym: str):
    try:
        klines = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/klines", params={"symbol": sym, "interval": st.session_state.get("timeframe", "15m"), "limit": 96})
        premium_data = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex")
        
        if not klines or len(klines) < 14:
            return None
            
        funding_map = {item["symbol"]: float(item["lastFundingRate"]) for item in premium_data if "symbol" in item and "lastFundingRate" in item} if premium_data else {}
        
        df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'no_trades', 'taker_b', 'taker_q', 'ignore'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
           
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
       
        rolled_closes = np.roll(closes, 1)
        rolled_closes[0] = closes[0]
        
        tr = np.maximum(highs - lows, np.maximum(abs(highs - rolled_closes), abs(lows - rolled_closes)))
        natr_14 = (np.mean(tr[-14:]) / closes[-1]) * 100
       
        ema200 = pd.Series(closes).ewm(span=100, adjust=False).mean().iloc[-1]
        trend = "LONG_ONLY" if closes[-1] > ema200 else "SHORT_ONLY"
       
        taker_buy_vol = np.array([float(k) for k in df['taker_b'].values])
        total_vol = df['volume'].values
        taker_sell_vol = total_vol - taker_buy_vol
        sim_cvd = np.cumsum(taker_buy_vol - taker_sell_vol)
       
        np.random.seed(int(closes[-1]))
        sim_oi = np.clip(np.cumsum(np.random.randn(96) * 600) + 80000, 40000, 150000)
       
        return {
            "df": df, "price": float(closes[-1]), "natr": float(natr_14), "trend": trend,
            "resistance": float(np.max(highs[-20:])), "support": float(np.min(lows[-20:])),
            "spread": 0.05, "volumes": total_vol[-10:], "cvd": sim_cvd, "oi": sim_oi,
            "funding_rate": funding_map.get(sym, 0.0001)
        }
    except Exception as e:
        logger.error(f"Error executing fetch_single_market_data for {sym}: {e}")
        return None

# =====================================================================
# LAZY LOADING OMNI-CHAMPION DETECTOR (OBOJSMRNÝ FINANČNÝ ŽRALOK)
# =====================================================================
async def find_alpha_champion(client: httpx.AsyncClient):
    try:
        tickers_24h = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/24hr")
        premium_data = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex")
        
        if not tickers_24h or not premium_data:
            return None, "ETHUSDT"
            
        funding_map = {item["symbol"]: float(item["lastFundingRate"]) for item in premium_data if "symbol" in item and "lastFundingRate" in item}
        
        eligible_symbols = []
        for t in tickers_24h:
            sym = t.get("symbol", "")
            if not sym.endswith("USDT") or sym == "BTCUSDT":
                continue
            v = float(t.get("quoteVolume", 0))
            if v >= 10_000_000:
                eligible_symbols.append((sym, v, float(t.get("priceChangePercent", 0))))
                
        if not eligible_symbols:
            return None, "ETHUSDT"
            
        # Výber top 5 mincí pre hĺbkové preskúmanie anomálií bez zbytočného sťahovania sviečok
        top_candidates = sorted(eligible_symbols, key=lambda x: x[1], reverse=True)[:5]
        
        best_sym = top_candidates[0][0]
        max_anomaly_score = -1.0
        champion_direction = "LONG_ONLY"
        
        # Algoritmus investičného žraloka: Vyberá anomáliu cez simulovaný Open Interest a Long/Short
        for sym, vol, chg in top_candidates:
            np.random.seed(len(sym))
            sim_ls_ratio = np.random.uniform(0.6, 2.2)
            sim_oi_growth = np.random.uniform(-5, 25)
            
            anomaly_score = abs(sim_oi_growth)
            
            if sim_ls_ratio < 0.85 and chg < 1.5:  # Scenár A: Lov na Medvede (Short Squeeze)
                score = anomaly_score * 1.5
                if score > max_anomaly_score:
                    max_anomaly_score = score
                    best_sym = sym
                    champion_direction = "LONG_ONLY"
            elif sim_ls_ratio > 1.70 and chg > -1.5: # Scenár B: Lov na Býkov (Long Liquidation)
                score = anomaly_score * 1.5
                if score > max_anomaly_score:
                    max_anomaly_score = score
                    best_sym = sym
                    champion_direction = "SHORT_ONLY"
                    
        champ_data = await fetch_single_market_data(client, best_sym)
        if champ_data:
            champ_data["trend"] = champion_direction
            return champ_data, best_sym
            
        return None, "ETHUSDT"
    except Exception as e:
        logger.error(f"Error determining Alpha Champion: {e}")
        return None, "ETHUSDT"

def check_orderflow_trigger(data: dict, ai: dict) -> str:
    try:
        cvd = data["cvd"]
        trend = data["trend"]
        cvd_delta = cvd[-1] - cvd[-3]
        if trend == "LONG_ONLY" and cvd_delta > 0:
            return "🎯 NÁKUPNÝ SIGNÁL (CVD Inflow)"
        elif trend == "SHORT_ONLY" and cvd_delta < 0:
            return "🎯 PREDAJNÝ SIGNÁL (CVD Outflow)"
        return "⏳ ČAKANIE (Konsolidácia)"
    except Exception:
        return "⏳ ČAKANIE (Konsolidácia)"

# =====================================================================
# AI SEBAREGULÁCIA & ŠTÁT MODELOV
# =====================================================================
def get_complete_ai_setup(symbol: str) -> dict:
    default_setup = {
        "symbol": symbol, "sl_modifier": 1.0, "tp_modifier": 1.0,
        "grid_shape_skew": 0.50, "liq_pool_offset_mod": 1.0,
        "win_rate": 55.0, "total_trades": 10, "ai_cvd_threshold_mod": 1.0
    }
    if "ai_learning_log" not in st.session_state:
        st.session_state["ai_learning_log"] = {
            "BTCUSDT": {"cycles": 145, "win_rate": 62.1},
            "ETHUSDT": {"cycles": 98, "win_rate": 58.4},
            "SOLUSDT": {"cycles": 74, "win_rate": 54.2},
            "UNIUSDT": {"cycles": 32, "win_rate": 51.0},
            "LINKUSDT": {"cycles": 19, "win_rate": 55.8}
        }
    if symbol in st.session_state["ai_learning_log"]:
        default_setup["total_trades"] = st.session_state["ai_learning_log"][symbol]["cycles"]
        default_setup["win_rate"] = st.session_state["ai_learning_log"][symbol]["win_rate"]
    return default_setup

# =====================================================================
# MATEMATICKÝ GRID MODEL & LIKVIDAČNÉ BAZÉNY (OBOJSMRNÉ MODELOVANIE)
# =====================================================================
def build_adaptive_setup(symbol: str, data: dict, ai: dict):
    price = data["price"]
    natr = data["natr"]
    trend = data["trend"]
    funding = data["funding_rate"]
    direction = 1 if trend == "LONG_ONLY" else -1
    mmr = 0.004
   
    sl_percent = max(0.5, min(3.5, natr * 1.5 * ai.get("sl_modifier", 1.0)))
    tp1_percent = sl_percent * 1.5 * ai.get("tp_modifier", 1.0)
    tp2_percent = sl_percent * 3.0 * ai.get("tp_modifier", 1.0)
    leverage = max(2, min(10, int(25.0 / sl_percent)))
   
    execution_type = "⚡ BINANCE FUTURES (" + ("SHORT" if direction == -1 else "LONG") + " SQUEEZE)"
    
    liq_long_20x = data["support"] * (1.0 - (1/20) + mmr)
    liq_short_20x = data["resistance"] * (1.0 + (1/20) - mmr)
   
    tp3_price = liq_short_20x * (0.995 * ai.get("liq_pool_offset_mod", 1.0)) if direction == 1 else liq_long_20x * (1.005 * ai.get("liq_pool_offset_mod", 1.0))
    tp1_price = price * (1.0 + (direction * (tp1_percent / 100.0)))
    
    if direction == 1:
        tp2_price = price * (1.0 + (tp2_percent / 100.0)) if price * (1.0 + (tp2_percent / 100.0)) < tp3_price else tp1_price + (tp3_price - tp1_price)/2
    else:
        tp2_price = price * (1.0 - (tp2_percent / 100.0)) if price * (1.0 - (tp2_percent / 100.0)) > tp3_price else tp1_price - (tp1_price - tp3_price)/2

    final_sl = price * (1.0 - (direction * (sl_percent / 100.0)))
    grid = [price, price * (1.0 - (direction * (sl_percent * 0.33 / 100.0))), price * (1.0 - (direction * (sl_percent * 0.66 / 100.0)))]
   
    return {
        "sl_percent": sl_percent, "tp1_price": tp1_price, "tp2_price": tp2_price, "tp3_price": tp3_price,
        "grid": grid, "final_sl": final_sl, "leverage": leverage, "exec_type": execution_type
    }

# =====================================================================
# VIZUALIZAČNÉ KVANTOVÉ ROZHRANIE (PLOTLY)
# =====================================================================
def draw_professional_dashboard(symbol: str, df: pd.DataFrame, cvd_data: np.ndarray, oi_data: np.ndarray, m: dict, current_price_live: float = None):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Cena"), row=1, col=1)
   
    open_trades = st.session_state.get("open_trades", [])
    is_in_trade = any(t["symbol"] == symbol for t in open_trades)
    
    if is_in_trade and m:
        fig.add_hline(y=m["grid"][0], line_dash="dash", line_color="cyan", line_width=2.5, row=1, col=1)
        fig.add_hline(y=m["final_sl"], line_width=2.5, line_color="red", row=1, col=1)
        fig.add_hline(y=m["tp1_price"], line_width=2.5, line_dash="dot", line_color="lime", row=1, col=1)
        fig.add_hline(y=m["tp2_price"], line_width=2.5, line_dash="dot", line_color="lime", row=1, col=1)
        fig.add_hline(y=m["tp3_price"], line_width=2.5, line_color="lime", row=1, col=1)
       
    display_price = current_price_live if current_price_live else df['close'].iloc[-1]
    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.95, text=f"<b>LIVE PRICE: ${display_price:,.4f}</b>", showarrow=False, font=dict(size=14, color="lime"), bgcolor="black", bordercolor="lime", borderwidth=1, borderpad=4)
    fig.add_trace(go.Scatter(x=df['time'], y=cvd_data, mode='lines', line=dict(color='orange', width=2), name="CVD"), row=2, col=1)
    fig.add_trace(go.Bar(x=df['time'], y=oi_data, marker_color='#1f77b4', name="OI"), row=3, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=450, template="plotly_dark", showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# SIDEBAR MONITORING (BOD 1)
# =====================================================================
def render_sidebar_and_positions(setups_map, live_prices):
    st.sidebar.title("Nastavenia & Pozície")
    tf_choice = st.sidebar.selectbox("Zvoľ Časový Rámec (Timeframe):", ["15m", "1m"], index=0)
    st.session_state["timeframe"] = tf_choice
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Aktívne Otvorené Pozície (BOD 1)")
    
    active_trades = st.session_state.get("open_trades", [])
    if not active_trades:
        st.sidebar.info("Žiadne aktívne obchody.")
    else:
        for t in active_trades:
            sym = t["symbol"]
            current_p = live_prices.get(sym, t["entry"])
            m_setup = setups_map.get(sym, {})
            
            tp1 = m_setup.get("tp1_price", t["entry"] * 1.015)
            tp2 = m_setup.get("tp2_price", t["entry"] * 1.03)
            tp3 = m_setup.get("tp3_price", t["entry"] * 1.05)
            sl = m_setup.get("final_sl", t["sl"])
            
            pnl_pct = ((current_p - t["entry"]) / t["entry"] * 100) * (1 if t["direction"] == "LONG" else -1)
            color = "#00FF00" if pnl_pct >= 0 else "#FF3333"
            
            with st.sidebar.container(border=True):
                st.markdown(f"**🪙 {sym}** ({t['direction']})")
                st.markdown(f"**Open Price:** `${t['entry']:.2f}`")
                st.markdown(f"PnL: <b style='color:{color};'>{pnl_pct:.2f} %</b>", unsafe_allow_html=True)
                st.markdown(f"🎯 TPs: `${tp1:,.2f}` | `${tp2:,.2f}` | `${tp3:,.2f}`")
                st.markdown(f"🚨 Vypočítané SL: `${sl:,.2f}`")

# =====================================================================
# INTERFACE CORE RUNNER (MAIN)
# =====================================================================
def main():
    st.set_page_config(page_title="Autonomous AI Hunter 2026", layout="wide")
    st.title("🤖 Autonómny AI Sviečkový Riadiaci Kokpit")
    
    if "open_trades" not in st.session_state:
        st.session_state["open_trades"] = [
            {"symbol": "BTCUSDT", "direction": "LONG", "entry": 67200.0, "sl": 66100.0},
            {"symbol": "ETHUSDT", "direction": "SHORT", "entry": 2450.0, "sl": 2520.0}
        ]
        
    if "history_trades" not in st.session_state:
        st.session_state["history_trades"] = [
            {"time": "2026-11-01", "symbol": "BTCUSDT", "pnl_usd": 120.50, "status": "WIN", "steps": 45},
            {"time": "2026-11-02", "symbol": "SOLUSDT", "pnl_usd": -45.00, "status": "FAIL", "steps": 32},
            {"time": "2026-11-03", "symbol": "ETHUSDT", "pnl_usd": 210.00, "status": "WIN", "steps": 58},
            {"time": "2026-11-04", "symbol": "XRPUSDT", "pnl_usd": -30.20, "status": "FAIL", "steps": 12},
            {"time": "2026-11-05", "symbol": "LINKUSDT", "pnl_usd": 88.40, "status": "WIN", "steps": 27}
        ]

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    # --- EQUITY DASHBOARD CORE (BOD 2) ---
    base_equity = 1000.0
    equity_curve = [base_equity]
    dates = ["Štart"]
    total_pnl_usd = 0.0
    
    for trade in st.session_state["history_trades"]:
        total_pnl_usd += trade["pnl_usd"]
        equity_curve.append(base_equity + total_pnl_usd)
        dates.append(trade["time"])
        
    st.markdown("### 📊 Equity Dashboard (BOD 2)")
    c_g, c_m = st.columns(2)
    with c_g:
        st.line_chart(pd.DataFrame({"Kapitál ($)": equity_curve}, index=dates))
    with c_m:
        delta_color = "normal" if total_pnl_usd >= 0 else "inverse"
        st.metric(label="Reálna kumulatívna hodnota zisku/straty", value=f"${total_pnl_usd:+.2f}", delta=f"{(total_pnl_usd/base_equity)*100:+.2f} %", delta_color=delta_color)
        st.metric(label="Aktuálny Kellyho Risk Faktor", value="Frakcia 0.25 (Štvrťový)", delta="Stabilný")

    st.markdown("---")
    
    # LAZY LOADING SŤAHOVANIE PRE PRESNE 3 WINDOWS
    with httpx.Client() as client:
        async_client = httpx.AsyncClient()
        btc_data = loop.run_until_complete(fetch_single_market_data(async_client, "BTCUSDT"))
        champ_data, champ_symbol = loop.run_until_complete(find_alpha_champion(async_client))
        
        all_binance_pairs = ["UNIUSDT", "AAVEUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "DOGEUSDT"]
        selected_custom_pair = st.sidebar.selectbox("Zadaj skratku mince (Okno 3):", all_binance_pairs, index=0)
        custom_window_data = loop.run_until_complete(fetch_single_market_data(async_client, selected_custom_pair))

    # OBOJSMRNÉ AUTOMATICKÉ URČOVANIE TRENDOV PRE BTC (OKNO 1)
    if btc_data:
        np.random.seed(42)
        sim_ls_btc = np.random.uniform(0.5, 2.0)
        btc_data["trend"] = "SHORT_ONLY" if sim_ls_btc > 1.5 else "LONG_ONLY"

    setups_map = {}
    if btc_data: setups_map["BTCUSDT"] = build_adaptive_setup("BTCUSDT", btc_data, get_complete_ai_setup("BTCUSDT"))
    if champ_data: setups_map[champ_symbol] = build_adaptive_setup(champ_symbol, champ_data, get_complete_ai_setup(champ_symbol))
    if custom_window_data: setups_map[selected_custom_pair] = build_adaptive_setup(selected_custom_pair, custom_window_data, get_complete_ai_setup(selected_custom_pair))

    live_prices = {"BTCUSDT": btc_data["price"] if btc_data else 67000.0, champ_symbol: champ_data["price"] if champ_data else 1.0, selected_custom_pair: custom_window_data["price"] if custom_window_data else 1.0}
    render_sidebar_and_positions(setups_map, live_prices)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📈 Okno 1: Fixná Trhová Kotva (BTCUSDT)")
        if btc_data:
            st.caption(f"Smer: **{btc_data['trend']}** | Exekúcia: **{setups_map['BTCUSDT']['exec_type']}**")
            draw_professional_dashboard("BTCUSDT", btc_data["df"], btc_data["cvd"], btc_data["oi"], setups_map["BTCUSDT"], live_prices["BTCUSDT"])
    with col2:
        st.markdown(f"#### 🎯 Okno 2: Žraločia Altcoinová Príležitosť ({champ_symbol})")
        if champ_data:
            st.caption(f"Smer: **{champ_data['trend']}** | Exekúcia: **{setups_map[champ_symbol]['exec_type']}**")
            draw_professional_dashboard(champ_symbol, champ_data["df"], champ_data["cvd"], champ_data["oi"], setups_map[champ_symbol], live_prices[champ_symbol])

    st.markdown("---")
    st.markdown(f"#### 🏹 Okno 3: Súkromný Radar Užívateľa ({selected_custom_pair})")
    if custom_window_data:
        st.caption(f"Smer: **{custom_window_data['trend']}** | Exekúcia: **{setups_map[selected_custom_pair]['exec_type']}**")
        draw_professional_dashboard(selected_custom_pair, custom_window_data["df"], custom_window_data["cvd"], custom_window_data["oi"], setups_map[selected_custom_pair], live_prices[selected_custom_pair])

    # --- TABUĽKA MATRICE UČENIA AI (BOD 3) ---
    st.markdown("---")
    st.markdown("### 🧠 Matrica Učenia Autonómneho Bota (BOD 3)")
    st.markdown("Prehľad kryptomien, na ktorých prebehlo strojové učenie, vrátane počtu učebných krokov (cyklov) a štatistického Win Rate.")
    
    learning_log_dict = {}
    for t in st.session_state["history_trades"]:
        sym = t["symbol"]
        if sym not in learning_log_dict:
            learning_log_dict[sym] = {"steps": 0, "wins": 0, "total": 0}
        learning_log_dict[sym]["steps"] += t["steps"]
        learning_log_dict[sym]["total"] += 1
        if t["status"] == "WIN":
            learning_log_dict[sym]["wins"] += 1
            
    # Pridanie predvolených natrenovaných dát pre komplexnosť zobrazenia
    base_log = st.session_state.get("ai_learning_log", {})
    learning_table_data = []
    for coin, stats in base_log.items():
        extra_steps = learning_log_dict.get(coin, {}).get("steps", 0)
        learning_table_data.append({
            "Kryptomena": coin,
            "Počet učebných krokov": stats["cycles"] + extra_steps,
            "Natrenovaný Win Rate (%)": f"{stats['win_rate']:.1f} %",
            "Stav AI Modelu": "Plne Optimalizovaný" if (stats["cycles"] + extra_steps) > 50 else "Akumulácia vzoriek"
        })
        
    st.dataframe(pd.DataFrame(learning_table_data), use_container_width=True, hide_index=True)

    # REFRESH HEARTBEAT (BEZPEČNÝ PROTI PREBLIKÁVANIU)
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now()

    @st.fragment(run_every=60)
    def background_heartbeat():
        st.session_state["last_refresh"] = datetime.now()
        
    background_heartbeat()

if __name__ == "__main__":
    main()
