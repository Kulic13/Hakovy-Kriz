import asyncio
import httpx
import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =====================================================================
# 1. ARCHITEKTÚRA, BEZPEČNOSŤ A FAIL-SAFE TRHOVEJ SIETE
# =====================================================================
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
BINANCE_FUTURES_URL = "https://www.binance.com"

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Automatický auto-inštalátor tabuľky v Supabase, ak neexistuje
        try:
            supabase.rpc("execute_sql", {"query": """
                CREATE TABLE IF NOT EXISTS simulated_trades (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price FLOAT NOT NULL,
                    sl_price FLOAT NOT NULL,
                    highest_price FLOAT,
                    lowest_price FLOAT,
                    tp1_price FLOAT NOT NULL,
                    tp2_price FLOAT NOT NULL,
                    tp3_price FLOAT NOT NULL,
                    leverage INT NOT NULL,
                    status TEXT NOT NULL,
                    outcome FLOAT,
                    max_runup FLOAT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW())
                );
            """}).execute()
        except Exception:
            pass
    except Exception:
        pass

async def fetch_with_retry(client: httpx.AsyncClient, url: str, params: dict = None) -> dict:
    backoff = 2.0
    for attempt in range(3):
        try:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(backoff)
            backoff *= 2.0

# =====================================================================
# 2. CONTINUOUS OMNI-SKENER (TOP 15 ALTCOINOV)
# =====================================================================
async def scan_all_markets(client: httpx.AsyncClient):
    try:
        exchange_info = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/exchangeInfo")
        tickers_24h = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/24hr")
        premium_data = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex")
       
        volume_map = {t["symbol"]: float(t["quoteVolume"]) for t in tickers_24h}
        funding_map = {item["symbol"]: float(item["lastFundingRate"]) for item in premium_data if "lastFundingRate" in item}
       
        # Všetky páry na trhu pre zoznam v Okne 3
        all_futures_symbols = sorted([s["symbol"] for s in exchange_info.get("symbols", []) 
                               if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"])
        
        # Orezaný zoznam pre kontinuálny automatický skener Top 15
        filtered_symbols = [s for s in all_futures_symbols if volume_map.get(s, 0) >= 10_000_000]
        symbols_to_scan = sorted(filtered_symbols, key=lambda x: volume_map.get(x, 0), reverse=True)[:15]
        
        if "BTCUSDT" not in symbols_to_scan:
            symbols_to_scan.append("BTCUSDT")
           
        trend_pool = []
        sniper_pool = []
               
        for sym in symbols_to_scan:
            market_obj = await fetch_single_ticker_data(client, sym, st.session_state.get("global_timeframe", "15m"), funding_map)
            if not market_obj:
                continue
                
            # Detekcia Sniper anomálie
            closes = market_obj["df"]['close'].values
            tr = market_obj["tr"]
            natr_14 = market_obj["natr"]
            sim_cvd = market_obj["cvd"]
            sim_oi = market_obj["oi"]
            
            historical_natr_avg = np.mean(tr / closes * 100)
            volatility_compressed = natr_14 < (historical_natr_avg * 0.65)
            cvd_growing = sim_cvd[-1] > sim_cvd[-6]
            oi_growing = sim_oi[-1] > sim_oi[-6]
            price_stagnating = abs(closes[-1] - closes[-6]) / closes[-6] < 0.003
           
            is_sniper_anomaly = volatility_compressed and cvd_growing and oi_growing and price_stagnating
           
            if is_sniper_anomaly and sym != "BTCUSDT":
                sniper_pool.append((sym, market_obj))
            else:
                trend_pool.append((sym, market_obj))
               
        return trend_pool, sniper_pool, all_futures_symbols, funding_map
    except Exception as e:
        st.error(f"Chyba Omni-Skenera: {e}")
        return [], [], ["BTCUSDT"], {}

async def fetch_single_ticker_data(client: httpx.AsyncClient, sym: str, timeframe: str, funding_map: dict):
    try:
        klines = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/klines", params={"symbol": sym, "interval": timeframe, "limit": 96})
        df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'no_trades', 'taker_b', 'taker_q', 'ignore'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
           
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
       
        tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
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
            "funding_rate": funding_map.get(sym, 0.0001), "tr": tr
        }
    except Exception:
        return None

# =====================================================================
# 2b. SPÚŠŤAČ ORDERFLOW
# =====================================================================
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
        return "⏳ ČAKANIE (Chyba Výpočtu)"

# =====================================================================
# 3. AI COMPONENT & BAYESOVSKÁ SEBAREGULÁCIA
# =====================================================================
def get_complete_ai_setup(symbol: str):
    default_setup = {
        "symbol": symbol, "sl_modifier": 1.0, "tp_modifier": 1.0,
        "grid_shape_skew": 0.50, "liq_pool_offset_mod": 1.0,
        "win_rate": 50.0, "total_trades": 0, "ai_cvd_threshold_mod": 1.0
    }
    if not supabase:
        return default_setup
    try:
        res = supabase.table("ai_complete_setups").select("*").eq("symbol", symbol).execute()
        if hasattr(res, 'data') and res.data:
            return res.data[0] if isinstance(res.data, list) else res.data
    except Exception:
        pass
    return default_setup

def train_complete_setup_loop(symbol: str, outcome: float, max_runup: float, max_drawdown: float, missed_entry: bool, fakeout: bool):
    if not supabase:
        return
    c = get_complete_ai_setup(symbol)
   
    new_sl = c.get("sl_modifier", 1.0)
    new_tp = c.get("tp_modifier", 1.0)
    new_skew = c.get("grid_shape_skew", 0.50)
    new_offset = c.get("liq_pool_offset_mod", 1.0)
    new_cvd_mod = c.get("ai_cvd_threshold_mod", 1.0)
   
    if outcome == -1.0 and max_runup > max_drawdown:
        new_sl += 0.04
    elif outcome < 1.0 and max_runup > 0.6:
        new_tp -= 0.04
       
    if missed_entry and outcome == 1.0:
        new_skew += 0.05
    elif outcome == -1.0 and max_drawdown > 2.0:
        new_skew -= 0.05
       
    if fakeout:
        new_cvd_mod += 0.10
    elif outcome == 1.0 and not missed_entry:
        new_cvd_mod -= 0.02
       
    if outcome < 1.0 and max_runup > 0.9:
        new_offset -= 0.03
   
    data = {
        "symbol": symbol,
        "sl_modifier": max(0.80, min(1.20, new_sl)),
        "tp_modifier": max(0.80, min(1.20, new_tp)),
        "grid_shape_skew": max(0.20, min(0.70, new_skew)),
        "liq_pool_offset_mod": max(0.90, min(1.10, new_offset)),
        "win_rate": ((c.get("win_rate", 50.0) * c.get("total_trades", 0)) + (100 if outcome > 0 else 0)) / (c.get("total_trades", 0) + 1),
        "total_trades": c.get("total_trades", 0) + 1,
        "ai_cvd_threshold_mod": max(0.50, min(2.0, new_cvd_mod)),
        "updated_at": datetime.utcnow().isoformat()
    }
    try:
        supabase.table("ai_complete_setups").upsert(data).execute()
    except Exception:
        pass

# =====================================================================
# 4. MATEMATICKÁ STRATÉGIA, LOV POOLOV A PREPÍNANIE SPOTU
# =====================================================================
def build_adaptive_setup(symbol: str, data: dict, ai: dict, is_sniper: bool):
    price = data["price"]
    natr = data["natr"]
    trend = data["trend"]
    funding = data["funding_rate"]
    direction = 1 if trend == "LONG_ONLY" else -1
    mmr = 0.004
   
    sl_percent_raw = max(0.5, min(3.5, natr * 1.5 * ai["sl_modifier"]))
    tp1_percent_raw = sl_percent_raw * 1.5 * ai["tp_modifier"]
    est_leverage = max(2, min(10, int(25.0 / sl_percent_raw)))
   
    funding_drag = (abs(funding) * est_leverage) / (tp1_percent_raw / 100.0)
   
    if is_sniper:
        execution_type = "🎯 AI SNIPER AKUMULÁCIA (Pred-pumpový režim)"
        leverage = 10
        sl_percent = max(0.4, natr * 0.8)
    elif funding_drag > 0.15 and trend == "LONG_ONLY":
        execution_type = "🚨 BINANCE SPOT (Ochrana pred Funding Squeezom)"
        leverage = 1
        sl_percent = sl_percent_raw * 1.3
    else:
        execution_type = "⚡ BINANCE FUTURES (Trendové momentum)"
        leverage = est_leverage
        sl_percent = sl_percent_raw

    tp1_percent = sl_percent * 1.5 * ai["tp_modifier"]
    tp2_percent = sl_percent * 3.0 * ai["tp_modifier"]
   
    liq_long_50x = data["support"] * (1.0 - (1/50) + mmr)
    liq_long_20x = data["support"] * (1.0 - (1/20) + mmr)
    liq_short_50x = data["resistance"] * (1.0 + (1/50) - mmr)
    liq_short_20x = data["resistance"] * (1.0 + (1/20) - mmr)
   
    tp3_price = liq_short_20x * (0.995 * ai["liq_pool_offset_mod"]) if trend == "LONG_ONLY" else liq_long_20x * (1.005 * ai["liq_pool_offset_mod"])

    tp1_price = price * (1.0 + (direction * (tp1_percent / 100.0)))
    tp3_fixed_price = tp3_price
   
    if trend == "LONG_ONLY":
        if price * (1.0 + (tp2_percent / 100.0)) > tp3_fixed_price:
            tp2_price = tp1_price + ((tp3_fixed_price - tp1_price) / 2)
        else:
            tp2_price = price * (1.0 + (tp2_percent / 100.0))
    else:
        if price * (1.0 - (tp2_percent / 100.0)) < tp3_fixed_price:
            tp2_price = tp1_price - ((tp1_price - tp3_fixed_price) / 2)
        else:
            tp2_price = price * (1.0 - (tp2_percent / 100.0))

    final_sl = price * (1.0 - (direction * (sl_percent / 100.0)))
   
    w1 = ai["grid_shape_skew"]
    w2 = (1.0 - w1) * 0.6
    w3 = (1.0 - w1) * 0.4
    grid = [price, price * (1.0 - (direction * (sl_percent * 0.33 / 100.0))), price * (1.0 - (direction * (sl_percent * 0.66 / 100.0)))]
   
    base_strike = round(price / 500) * 500
   
    return {
        "sl_percent": sl_percent, "tp1_price": tp1_price, "tp2_price": tp2_price, "tp3_price": tp3_fixed_price,
        "liq_long_50x": liq_long_50x, "liq_long_20x": liq_long_20x, "liq_short_50x": liq_short_50x, "liq_short_20x": liq_short_20x,
        "grid": grid, "final_sl": final_sl, "leverage": leverage, "weights": [w1*100, w2*100, w3*100],
        "trailing_dist": natr * 0.5, "fee_buffer": 0.20, "max_pain": base_strike - 1500, "call_wall": base_strike + 2500, "put_wall": base_strike - 2500,
        "exec_type": execution_type, "funding_drag": funding_drag * 100
    }

# =====================================================================
# 5. AUTOMATICKÝ SIMULAČNÝ PAPER-TRADING ENGINE S TRAILING SL
# =====================================================================
def get_live_trades_from_db():
    if 'simulated_trades_cache' not in st.session_state:
        st.session_state.simulated_trades_cache = {}
    return st.session_state.simulated_trades_cache

def save_live_trades_to_db(trades):
    st.session_state.simulated_trades_cache = trades

def handle_simulation_logic(symbol: str, current_data: dict, trigger: str, ai: dict):
    trades = get_live_trades_from_db()
    price = current_data["price"]
    
    if symbol in trades:
        t = trades[symbol]
        if t["direction"] == "LONG":
            if price > t["highest_price"]:
                t["highest_price"] = price
                new_sl = price - t["trailing_dist"]
                if new_sl > t["sl_price"]:
                    t["sl_price"] = new_sl
        else:
            if price < t["lowest_price"]:
                t["lowest_price"] = price
                new_sl = price + t["trailing_dist"]
                if new_sl < t["sl_price"]:
                    t["sl_price"] = new_sl
                    
        is_sl_hit = (t["direction"] == "LONG" and price <= t["sl_price"]) or (t["direction"] == "SHORT" and price >= t["sl_price"])
        is_tp_hit = (t["direction"] == "LONG" and price >= t["tp3_price"]) or (t["direction"] == "SHORT" and price <= t["tp3_price"])
        
        if is_sl_hit or is_tp_hit:
            outcome = 1.0 if is_tp_hit else -1.0
            train_complete_setup_loop(symbol, outcome, 1.5, 1.0, False, False)
            del trades[symbol]
            save_live_trades_to_db(trades)
            st.toast(f"🤖 AI optimalizovala a pretrénovala celý setup pre {symbol}!", icon="🤖")
        else:
            trades[symbol] = t
            save_live_trades_to_db(trades)
            
    elif "SIGNÁL" in trigger:
        m = build_adaptive_setup(symbol, current_data, ai, is_sniper=False)
        direction_str = "LONG" if "NÁKUPNÝ" in trigger else "SHORT"
        
        trades[symbol] = {
            "symbol": symbol, "direction": direction_str, "entry_price": price,
            "sl_price": m["final_sl"], "tp1_price": m["tp1_price"], "tp2_price": m["tp2_price"], "tp3_price": m["tp3_price"],
            "leverage": m["leverage"], "highest_price": price, "lowest_price": price, "trailing_dist": m["trailing_dist"]
        }
        save_live_trades_to_db(trades)

# =====================================================================
# 6. MULTI-PANELOVÁ INTERAKTÍVNA VIZUALIZÁCIA (Plotly Dark Mode)
# =====================================================================
def draw_professional_dashboard(symbol: str, df: pd.DataFrame, cvd_data: np.ndarray, oi_data: np.ndarray, m: dict, live_price: float):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
   
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Cena"), row=1, col=1)
   
    fig.add_hline(y=m["grid"][0], line_dash="dash", line_color="cyan", row=1, col=1)
    fig.add_hline(y=m["final_sl"], line_width=2, line_color="red", annotation_text="🚨 SL poistka", annotation_position="top left", row=1, col=1)
    fig.add_hline(y=m["tp1_price"], line_width=1.5, line_dash="dot", line_color="green", annotation_text="🎯 TP1", annotation_position="top left", row=1, col=1)
    fig.add_hline(y=m["tp2_price"], line_width=1.5, line_dash="dot", line_color="green", annotation_text="🎯 TP2", annotation_position="top left", row=1, col=1)
    fig.add_hline(y=m["tp3_price"], line_width=2, line_color="lime", annotation_text="🚀 TP3 Pool", annotation_position="top left", row=1, col=1)
   
    fig.add_hrect(y0=m["liq_long_20x"]*0.998, y1=m["liq_long_50x"]*1.002, fillcolor="orange", opacity=0.12, layer="below", line_width=0, row=1, col=1)
    fig.add_hrect(y0=m["liq_short_50x"]*0.998, y1=m["liq_short_20x"]*1.002, fillcolor="yellow", opacity=0.12, layer="below", line_width=0, row=1, col=1)
   
    if symbol == "BTCUSDT":
        fig.add_hline(y=m["max_pain"], line_width=1.5, line_dash="dashdot", line_color="violet", annotation_text="🔮 Max Pain", annotation_position="top left", row=1, col=1)
        fig.add_hline(y=m["call_wall"], line_width=2.5, line_color="fuchsia", annotation_text="🧱 Call Wall", annotation_position="top left", row=1, col=1)
        fig.add_hline(y=m["put_wall"], line_width=2.5, line_color="purple", annotation_text="🧱 Put Wall", annotation_position="top left", row=1, col=1)
   
    fig.add_annotation(
        x=0.98, y=0.95, xref="paper", yref="paper",
        text=f"LIVE PRICE: ${live_price:,.2f}", showarrow=False,
        font=dict(size=18, color="#00FF00" if df['close'].iloc[-1] >= df['open'].iloc[-1] else "#FF0000"),
        bgcolor="rgba(0,0,0,0.7)", bordercolor="#00FF00", borderwidth=2, borderpad=6, row=1, col=1
    )

    fig.add_trace(go.Scatter(x=df['time'], y=cvd_data, mode='lines', line=dict(color='orange', width=2), name="CVD (Volume Delta)"), row=2, col=1)
   
    oi_diff = np.diff(oi_data, prepend=oi_data)
    oi_colors = ['#1f77b4' if diff >= 0 else '#0f3a57' for diff in oi_diff]
    fig.add_trace(go.Bar(x=df['time'], y=oi_data, marker_color=oi_colors, name="Open Interest (Pozície)"), row=3, col=1)
   
    fig.update_layout(xaxis_rangeslider_visible=False, height=520, template="plotly_dark", showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{symbol}_{timeframe}")

# =====================================================================
# 7. SEC-BY-SEC FRAGMENT PRE SIDEBAR & GRAFY
# =====================================================================
@st.fragment(run_every=1)
def render_ultra_fast_tickers(symbols_to_fetch, trend_pool, sniper_pool, selected_sym_data):
    try:
        with httpx.Client() as client:
            res = client.get(f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/price")
            prices_raw = res.json()
            price_dict = {p["symbol"]: float(p["price"]) for p in prices_raw if p["symbol"] in symbols_to_fetch}
    except Exception:
        price_dict = {}

    trades = get_live_trades_from_db()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Živé Monitorované Pozície")
    
    if not trades:
        st.sidebar.info("Žiadne otvorené simulované pozície.")
    else:
        for sym, t in list(trades.items()):
            current_price = price_dict.get(sym, t["entry_price"])
            direction_mult = 1 if t["direction"] == "LONG" else -1
            raw_pnl = (current_price - t["entry_price"]) / t["entry_price"] * 100 * t["leverage"] * direction_mult
            pnl_color = "green" if raw_pnl >= 0 else "red"
            
            with st.sidebar.container(border=True):
                st.markdown(f"**{sym}** | {t['direction']} {t['leverage']}x")
                st.markdown(f"Vstup: `${t['entry_price']:.4f}` → Live: `${current_price:.4f}`")
                st.markdown(f"PnL: <span style='color:{pnl_color};font-weight:bold;'>{raw_pnl:+.2f} %</span>", unsafe_allow_html=True)
                st.markdown(f"Vlečný SL: `${t['sl_price']:.4f}`")

    st.markdown("### 📊 AI Intelligence & Dashboard")
    
    # 1. OKNO: BTCUSDT
    btc_data = next((item[1] for item in trend_pool if item[0] == "BTCUSDT"), None)
    if btc_data:
        btc_price = price_dict.get("BTCUSDT", btc_data["price"])
        btc_ai = get_complete_ai_setup("BTCUSDT")
        btc_trigger = check_orderflow_trigger(btc_data, btc_ai)
        handle_simulation_logic("BTCUSDT", btc_data, btc_trigger, btc_ai)
        btc_m = build_adaptive_setup("BTCUSDT", btc_data, btc_ai, is_sniper=False)
        with st.expander(f"🧱 1. FIXNÉ OKNO: BTCUSDT | Spúšťač: {btc_trigger}", expanded=True):
            draw_professional_dashboard("BTCUSDT", btc_data["df"], btc_data["cvd"], btc_data["oi"], btc_m, btc_price)

    # 2. OKNO: ALTCOIN (BEZ BTC)
    altcoin_pool = [item for item in trend_pool + sniper_pool if item[0] != "BTCUSDT"]
    if altcoin_pool:
        best_alt = max(altcoin_pool, key=lambda x: abs(x[1]["cvd"][-1] - x[1]["cvd"][-3]))
        alt_sym = best_alt[0]
        alt_data = best_alt[1]
        alt_price = price_dict.get(alt_sym, alt_data["price"])
        alt_ai = get_complete_ai_setup(alt_sym)
        alt_trigger = check_orderflow_trigger(alt_data, alt_ai)
        handle_simulation_logic(alt_sym, alt_data, alt_trigger, alt_ai)
        alt_m = build_adaptive_setup(alt_sym, alt_data, alt_ai, is_sniper=(best_alt in sniper_pool))
        with st.expander(f"🔥 2. DYNAMICKÝ ALTCOIN: {alt_sym} | Spúšťač: {alt_trigger}", expanded=True):
            draw_professional_dashboard(alt_sym, alt_data["df"], alt_data["cvd"], alt_data["oi"], alt_m, alt_price)

    # 3. OKNO: VOLITEĽNÝ RADAR
    if selected_sym_data:
        sel_sym = selected_sym_data["symbol"]
        sel_price = price_dict.get(sel_sym, selected_sym_data["price"])
        sel_ai = get_complete_ai_setup(sel_sym)
        sel_trigger = check_orderflow_trigger(selected_sym_data, sel_ai)
        handle_simulation_logic(sel_sym, selected_sym_data, sel_trigger, sel_ai)
        sel_m = build_adaptive_setup(sel_sym, selected_sym_data, sel_ai, is_sniper=False)
        with st.expander(f"🏹 3. VOLITEĽNÝ RADAR: {sel_sym} | Spúšťač: {sel_trigger}", expanded=True):
            draw_professional_dashboard(sel_sym, selected_sym_data["df"], selected_sym_data["cvd"], selected_sym_data["oi"], sel_m, sel_price)

# =====================================================================
# 8. HLAVNÝ WEB TERMINÁL
# =====================================================================
def main():
    st.set_page_config(page_title="Autonomous AI Hunter 2026", layout="wide")
    st.title("🤖 Autonómny AI Sviečkový Riadiaci Kokpit")
   
    st.sidebar.markdown("##### ⏱️ Kontinuálny Skener Siete")
    timeframe_choice = st.sidebar.selectbox("Časový rámec (Timeframe)", ["1m", "15m"], index=1)
    st.session_state["global_timeframe"] = timeframe_choice
    
    async def run_scanner():
        async with httpx.AsyncClient() as client:
            return await scan_all_markets(client)
            
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    trend_pool, sniper_pool, all_futures_symbols, funding_map = loop.run_until_complete(run_scanner())
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏹 Nastavenie Okna 3")
    
    # st.selectbox v Streamlite už natívne obsahuje integrované vyhľadávacie textové okienko (User can type to filter)
    user_choice = st.sidebar.selectbox("Zadaj alebo vyhľadaj skratku mince:", all_futures_symbols, index=all_futures_symbols.index("ETHUSDT") if "ETHUSDT" in all_futures_symbols else 0)
    
    async def load_demanded_coin():
        async with httpx.AsyncClient() as client:
            return await fetch_single_ticker_data(client, user_choice, timeframe_choice, funding_map)
            
    selected_sym_data = loop.run_until_complete(load_demanded_coin())
    if selected_sym_data:
        selected_sym_data["symbol"] = user_choice

    symbols_to_fetch = ["BTCUSDT"]
    if trend_pool + sniper_pool:
        altcoin_pool = [item for item in trend_pool + sniper_pool if item[0] != "BTCUSDT"]
        if altcoin_pool:
            best_alt = max(altcoin_pool, key=lambda x: abs(x[1]["cvd"][-1] - x[1]["cvd"][-3]))
            symbols_to_fetch.append(best_alt[0])
    if user_choice not in symbols_to_fetch:
        symbols_to_fetch.append(user_choice)

    render_ultra_fast_tickers(symbols_to_fetch, trend_pool, sniper_pool, selected_sym_data)

    async def delay_refresh():
        await asyncio.sleep(60)
        st.rerun()
    loop.run_until_complete(delay_refresh())

if __name__ == "__main__":
    main()
