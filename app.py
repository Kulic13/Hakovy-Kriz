import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# LOGGING
# =====================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantumOmniHunter")

# =====================================================================
# CONFIG & SECRETS
# =====================================================================
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
BINANCE_FUTURES_URL = "https://fapi.binance.com"
BINANCE_OPTIONS_URL = "https://eapi.binance.com"

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase pripojené úspešne")
    except Exception as e:
        logger.error(f"Chyba inicializácie Supabase: {e}")

# =====================================================================
# NETWORK LAYER - s exponenciálnym backoffom
# =====================================================================
async def fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: Optional[dict] = None,
    max_retries: int = 4
) -> dict | list:
    backoff = 0.6
    for attempt in range(max_retries):
        try:
            resp = await client.get(url, params=params, timeout=6.0)
            if resp.status_code == 429:
                await asyncio.sleep(backoff * (2 ** attempt))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Definitívne zlyhanie požiadavky na {url}: {e}")
                return {}
            await asyncio.sleep(backoff * (2 ** attempt))
    return {}

# =====================================================================
# OPTIONS - iba BTC weekly contracts
# =====================================================================
async def calculate_options_metrics(client: httpx.AsyncClient) -> dict:
    try:
        exchange_info = await fetch_with_retry(client, f"{BINANCE_OPTIONS_URL}/eapi/v1/exchangeInfo")
        if not exchange_info or "symbols" not in exchange_info:
            return {}

        now = datetime.now(timezone.utc)
        weekly_symbols = []

        for s in exchange_info["symbols"]:
            if s.get("baseAsset") != "BTC" or s.get("status") != "TRADING":
                continue
            parts = s["symbol"].split("-")
            if len(parts) < 4:
                continue
            try:
                exp_str = parts[1]  # formát YYMMDD
                exp_date = datetime.strptime(exp_str, "%y%m%d").replace(tzinfo=timezone.utc)
                days_to_exp = (exp_date - now).days
                if 0 <= days_to_exp <= 8:  # iba weekly
                    weekly_symbols.append(s["symbol"])
            except Exception:
                continue

        if not weekly_symbols:
            return {}

        open_interest_data = await fetch_with_retry(client, f"{BINANCE_OPTIONS_URL}/eapi/v1/openInterest")
        if not open_interest_data:
            return {}

        oi_map = {item["symbol"]: float(item.get("sumOpenInterest", 0)) for item in open_interest_data}

        calls_oi = {}
        puts_oi = {}

        for sym in weekly_symbols:
            parts = sym.split("-")
            try:
                strike = float(parts[2])
                opt_type = parts[3]
                oi = oi_map.get(sym, 0.0)
                if opt_type == "C":
                    calls_oi[strike] = calls_oi.get(strike, 0.0) + oi
                elif opt_type == "P":
                    puts_oi[strike] = puts_oi.get(strike, 0.0) + oi
            except Exception:
                continue

        if not calls_oi and not puts_oi:
            return {}

        call_wall = max(calls_oi, key=calls_oi.get) if calls_oi else None
        put_wall = max(puts_oi, key=puts_oi.get) if puts_oi else None

        all_strikes = sorted(set(list(calls_oi.keys()) + list(puts_oi.keys())))
        min_pain = float("inf")
        max_pain = all_strikes[0]

        for k_j in all_strikes:
            total_pain = 0.0
            for k_i in all_strikes:
                call_pain = max(0.0, k_j - k_i) * calls_oi.get(k_i, 0.0)
                put_pain = max(0.0, k_i - k_j) * puts_oi.get(k_i, 0.0)
                total_pain += (call_pain + put_pain)
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain = k_j

        return {
            "max_pain": max_pain,
            "call_wall": call_wall,
            "put_wall": put_wall
        }
    except Exception as e:
        logger.error(f"Chyba výpočtu opčných metrík: {e}")
        return {}

# =====================================================================
# SHARK ALPHA MATRIX - výber šampióna
# =====================================================================
async def scan_market_for_champion(client: httpx.AsyncClient) -> str:
    try:
        tickers_24h = await fetch_with_retry(client, f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/24hr")
        if not tickers_24h:
            return "ETHUSDT"

        candidates = [
            t for t in tickers_24h
            if "symbol" in t and t["symbol"].endswith("USDT") and t["symbol"] != "BTCUSDT"
            and float(t.get("quoteVolume", 0)) >= 10_000_000
        ]

        if not candidates:
            return "ETHUSDT"

        top_candidates = sorted(candidates, key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)[:5]

        best_score = -1.0
        champion_symbol = "ETHUSDT"

        for item in top_candidates:
            sym = item["symbol"]
            oi_res = await fetch_with_retry(
                client,
                f"{BINANCE_FUTURES_URL}/fapi/v1/openInterest",
                params={"symbol": sym}
            )
            if not oi_res or "openInterest" not in oi_res:
                continue

            oi_curr = float(oi_res["openInterest"])
            price_change = abs(float(item.get("priceChangePercent", 0)))
            score = (oi_curr * float(item.get("lastPrice", 1))) / max(0.1, price_change)

            if score > best_score:
                best_score = score
                champion_symbol = sym

        return champion_symbol
    except Exception as e:
        logger.error(f"Chyba skenovania šampióna: {e}")
        return "ETHUSDT"

# =====================================================================
# SŤAHOVANIE SVIEČOK - IBA REÁLNE DÁTA Z BINANCE (bez fallbacku)
# =====================================================================
async def fetch_candle_data(client: httpx.AsyncClient, sym: str, timeframe: str) -> Optional[dict]:
    try:
        klines = await fetch_with_retry(
            client,
            f"{BINANCE_FUTURES_URL}/fapi/v1/klines",
            params={"symbol": sym, "interval": timeframe, "limit": 96}
        )

        if not klines or len(klines) < 20:
            logger.error(f"Nedostatočné reálne dáta z Binance pre {sym}")
            return None

        df = pd.DataFrame(klines, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "q_vol", "no_trades", "taker_b", "taker_q", "ignore"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        # True Range + NATR
        rolled_closes = np.roll(closes, 1)
        rolled_closes[0] = closes[0]
        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - rolled_closes), np.abs(lows - rolled_closes)))
        natr_14 = (np.mean(tr[-14:]) / closes[-1]) * 100

        # Trend podľa EMA 100
        ema100 = pd.Series(closes).ewm(span=100, adjust=False).mean().iloc[-1]
        trend = "LONG_ONLY" if closes[-1] > ema100 else "SHORT_ONLY"

        # Aktivácia obchodu podľa volatility
        is_active_trade = True if natr_14 > 1.1 else False

        # Likvidity 20x
        mmr = 0.004
        liq_long_20x = float(np.min(lows[-20:])) * (1.0 - (1 / 20) + mmr)
        liq_short_20x = float(np.max(highs[-20:])) * (1.0 + (1 / 20) - mmr)

        return {
            "df": df,
            "price": float(closes[-1]),
            "natr": float(natr_14),
            "trend": trend,
            "resistance": float(np.max(highs[-20:])),
            "support": float(np.min(lows[-20:])),
            "liq_long_20x": liq_long_20x,
            "liq_short_20x": liq_short_20x,
            "is_active_trade": is_active_trade
        }
    except Exception as e:
        logger.error(f"Chyba pri sťahovaní klines pre {sym}: {e}")
        return None

# =====================================================================
# AI LEARNING - načítanie a ukladanie stavu
# =====================================================================
def get_complete_ai_setup(symbol: str) -> dict:
    default_setup = {
        "symbol": symbol,
        "win_rate": 0.0,
        "total_trades": 0,
        "sl_modifier": 1.0,
        "tp_modifier": 1.0,
        "wins": 0,
        "losses": 0,
        "avg_win_r": 1.5,
        "avg_loss_r": 1.0,
        "expectancy": 0.0
    }
    if not supabase:
        return default_setup
    try:
        res = supabase.table("ai_model_state").select("*").eq("symbol", symbol).execute()
        if res and hasattr(res, "data") and res.data and len(res.data) > 0:
            data_row = res.data[0]
            wins = float(data_row.get("wins", 0))
            losses = float(data_row.get("losses", 0))
            total = wins + losses
            win_rate = (wins / total * 100.0) if total > 0 else 0.0
            return {
                "symbol": symbol,
                "win_rate": win_rate,
                "total_trades": int(total),
                "sl_modifier": float(data_row.get("sl_modifier", 1.0)),
                "tp_modifier": float(data_row.get("tp_modifier", 1.0)),
                "wins": int(wins),
                "losses": int(losses),
                "avg_win_r": float(data_row.get("avg_win_r", 1.5)),
                "avg_loss_r": float(data_row.get("avg_loss_r", 1.0)),
                "expectancy": float(data_row.get("expectancy", 0.0))
            }
    except Exception as e:
        logger.warning(f"Nepodarilo sa načítať AI stav pre {symbol}: {e}")
    return default_setup

def save_ai_state(state: dict):
    if not supabase:
        return
    try:
        supabase.table("ai_model_state").upsert(state).execute()
    except Exception as e:
        logger.error(f"Chyba pri ukladaní AI stavu: {e}")

# =====================================================================
# KELLY CRITERION - matematicky korektné
# =====================================================================
def calculate_kelly_fraction(win_prob: float, avg_win_r: float, avg_loss_r: float) -> float:
    """
    Full Kelly: f* = (p * b - q) / b
    kde b = avg_win_r / avg_loss_r
    """
    if avg_loss_r <= 0 or win_prob <= 0 or win_prob >= 1:
        return 0.0
    b = avg_win_r / avg_loss_r
    q = 1.0 - win_prob
    f = (win_prob * b - q) / b
    return max(0.0, f)

def calculate_fractional_kelly(win_prob: float, avg_win_r: float, avg_loss_r: float, fraction: float = 0.35) -> float:
    full = calculate_kelly_fraction(win_prob, avg_win_r, avg_loss_r)
    return full * fraction

def position_size_usd(equity: float, kelly_f: float, max_risk_pct: float = 0.02) -> float:
    kelly_risk = equity * kelly_f
    max_risk = equity * max_risk_pct
    return min(kelly_risk, max_risk)

# =====================================================================
# ADAPTIVE SETUP
# =====================================================================
def build_adaptive_setup(symbol: str, data: dict, ai: dict) -> dict:
    price = data["price"]
    natr = data["natr"]
    trend = data["trend"]
    direction = 1 if trend == "LONG_ONLY" else -1

    sl_percent = max(0.4, min(3.2, natr * 1.35 * ai.get("sl_modifier", 1.0)))
    tp1_percent = sl_percent * 1.6 * ai.get("tp_modifier", 1.0)
    tp2_percent = sl_percent * 2.8 * ai.get("tp_modifier", 1.0)

    calc_leverage = int(22.0 / sl_percent)
    actual_lev = max(2, min(12, calc_leverage))
    exec_type = f"FUTURES {actual_lev}x"

    tp1_price = price * (1.0 + (direction * (tp1_percent / 100.0)))
    tp2_price = price * (1.0 + (direction * (tp2_percent / 100.0)))
    tp3_price = data["liq_short_20x"] * 0.995 if trend == "LONG_ONLY" else data["liq_long_20x"] * 1.005

    final_sl = price * (1.0 - (direction * (sl_percent / 100.0)))
    grid = [price, price * (1.0 - (direction * (sl_percent * 0.33 / 100.0)))]

    return {
        "sl_percent": sl_percent,
        "tp1_price": tp1_price,
        "tp2_price": tp2_price,
        "tp3_price": tp3_price,
        "final_sl": final_sl,
        "grid": grid,
        "leverage": actual_lev,
        "exec_type": exec_type
    }

# =====================================================================
# KRESLENIE GRAFOV - reálne candlestick + entry/SL/TP len pri aktívnom obchode
# =====================================================================
def draw_professional_dashboard(symbol: str, data: dict, m: dict, option_lines: dict = None, is_active: bool = False):
    df = data["df"]

    fig = go.Figure()

    # Reálny candlestick z Binance
    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Cena",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350"
    ))

    # Entry + SL + TP1 + TP2 + TP3 - zobrazia sa IBA ak je obchod aktívny
    if is_active and m:
        fig.add_hline(
            y=m["grid"][0],
            line_dash="dash",
            line_color="#00E5FF",
            line_width=2,
            annotation_text="ENTRY",
            annotation_position="top left"
        )
        fig.add_hline(
            y=m["final_sl"],
            line_color="#FF1744",
            line_width=2.5,
            annotation_text="SL",
            annotation_position="bottom left"
        )
        fig.add_hline(
            y=m["tp1_price"],
            line_dash="dot",
            line_color="#76FF03",
            line_width=1.8,
            annotation_text="TP1",
            annotation_position="top right"
        )
        fig.add_hline(
            y=m["tp2_price"],
            line_dash="dot",
            line_color="#64DD17",
            line_width=1.8,
            annotation_text="TP2",
            annotation_position="top right"
        )
        fig.add_hline(
            y=m["tp3_price"],
            line_color="#00C853",
            line_width=2.2,
            annotation_text="TP3",
            annotation_position="top right"
        )

    # Options hladiny - iba pre BTCUSDT (weekly contracts)
    if option_lines and symbol == "BTCUSDT":
        if option_lines.get("max_pain"):
            fig.add_hline(
                y=option_lines["max_pain"],
                line_dash="dashdot",
                line_color="#E040FB",
                line_width=2,
                annotation_text="Max Pain",
                annotation_position="top left"
            )
        if option_lines.get("call_wall"):
            fig.add_hline(
                y=option_lines["call_wall"],
                line_color="#FF00FF",
                line_width=2.5,
                annotation_text="Call Wall",
                annotation_position="top left"
            )
        if option_lines.get("put_wall"):
            fig.add_hline(
                y=option_lines["put_wall"],
                line_color="#7C4DFF",
                line_width=2.5,
                annotation_text="Put Wall",
                annotation_position="bottom left"
            )

    fig.update_layout(
        height=480,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(t=30, b=20, l=10, r=10),
        title=dict(text=f"{symbol}", font=dict(size=14)),
        yaxis=dict(side="right")
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# HLAVNÝ CYKLUS
# =====================================================================
def main():
    st.set_page_config(page_title="Quantum Omni Hunter 2026 – Pro", layout="wide")
    st.title("Autonómny AI Sviečkový Riadiaci Kokpit – Pro Edition")

    # Inicializácia session state pre timeframe
    if "tf_w1" not in st.session_state:
        st.session_state["tf_w1"] = "15m"
    if "tf_w2" not in st.session_state:
        st.session_state["tf_w2"] = "15m"
    if "tf_w3" not in st.session_state:
        st.session_state["tf_w3"] = "15m"

    if "monthly_trades" not in st.session_state:
        st.session_state["monthly_trades"] = []
        st.session_state["current_month"] = datetime.now().month

    if datetime.now().month != st.session_state["current_month"]:
        st.session_state["monthly_trades"] = []
        st.session_state["current_month"] = datetime.now().month

    # Asynchrónne načítanie dát
    async def load_data():
        async with httpx.AsyncClient() as client:
            opt_lines = await calculate_options_metrics(client)
            champ_pair = await scan_market_for_champion(client)

            btc_data = await fetch_candle_data(client, "BTCUSDT", st.session_state["tf_w1"])
            champ_data = await fetch_candle_data(client, champ_pair, st.session_state["tf_w2"])

            all_pairs = ["ETHUSDT", "SOLUSDT", "UNIUSDT", "LINKUSDT", "AAVEUSDT", "DOGEUSDT"]
            return opt_lines, champ_pair, btc_data, champ_data, all_pairs

    opt_lines, champ_pair, btc_data, champ_data, all_pairs = asyncio.run(load_data())

    selected_custom = st.sidebar.selectbox(
        "Zadaj alebo vyhľadaj skratku mince (Okno 3):",
        all_pairs,
        index=1
    )

    # Načítanie dát pre Okno 3 (po výbere)
    async def load_custom():
        async with httpx.AsyncClient() as client:
            return await fetch_candle_data(client, selected_custom, st.session_state["tf_w3"])

    custom_data = asyncio.run(load_custom())

    # Build setups
    setups = {}
    if btc_data:
        setups["BTCUSDT"] = build_adaptive_setup("BTCUSDT", btc_data, get_complete_ai_setup("BTCUSDT"))
    if champ_data:
        setups[champ_pair] = build_adaptive_setup(champ_pair, champ_data, get_complete_ai_setup(champ_pair))
    if custom_data:
        setups[selected_custom] = build_adaptive_setup(selected_custom, custom_data, get_complete_ai_setup(selected_custom))

    # =====================================================================
    # SIDEBAR - Monitor pozícií
    # =====================================================================
    st.sidebar.markdown("### Monitor Otvorených Pozícií")

    for s_name, label_window, d_source in [
        ("BTCUSDT", "Okno 1 (BTC)", btc_data),
        (champ_pair, f"Okno 2 ({champ_pair})", champ_data),
        (selected_custom, f"Okno 3 ({selected_custom})", custom_data)
    ]:
        if not d_source:
            continue

        is_active = d_source["is_active_trade"]
        m_s = setups.get(s_name, {})
        live_price = d_source["price"]
        ai_info = get_complete_ai_setup(s_name)

        with st.sidebar.container(border=True):
            st.markdown(f"**{label_window}**")
            st.markdown(f"* Aktuálna trhová cena: `{live_price:,.2f} USD`")

            if is_active:
                win_prob = ai_info["win_rate"] / 100.0 if ai_info["win_rate"] > 0 else 0.55
                kelly_f = calculate_fractional_kelly(
                    win_prob,
                    ai_info.get("avg_win_r", 1.5),
                    ai_info.get("avg_loss_r", 1.0),
                    fraction=0.35
                )
                allocated_usd = position_size_usd(100.0, kelly_f)

                entry_price = m_s.get("grid", [live_price])[0]
                direction = 1 if d_source["trend"] == "LONG_ONLY" else -1
                live_pnl_percent = ((live_price - entry_price) / entry_price) * 100.0 * direction

                st.markdown(f"* Exekúcia: `{m_s.get('exec_type', 'SPOT')}`")
                st.markdown(f"* Alokovaná Kelly marža: `${allocated_usd:.2f} USD`")
                st.markdown(f"* Vstupná cena: `${entry_price:,.2f}`")
                st.markdown(f"TP1: ${m_s.get('tp1_price', 0):,.2f} | TP2: ${m_s.get('tp2_price', 0):,.2f} | TP3: ${m_s.get('tp3_price', 0):,.2f}")
                st.markdown(f"SL: ${m_s.get('final_sl', 0):,.2f}")

                if live_pnl_percent >= 0:
                    st.markdown(f'* Live PnL: <span style="color:lime; font-weight:bold;">+{live_pnl_percent:.2f} %</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'* Live PnL: <span style="color:red; font-weight:bold;">{live_pnl_percent:.2f} %</span>', unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:orange; font-weight:bold;'>SKENOVANIE SIETE</span>", unsafe_allow_html=True)
                st.caption("Systém monitoruje anomálie a čaká na vygenerovanie optimálneho vstupu modelom.")

    # =====================================================================
    # VÝKONNOSTNÁ VRSTVA
    # =====================================================================
    st.markdown("### Kvantitatívna Výkonnostná Vrstva (Štart od 6. Novembra 2026)")

    c_equity, c_matrix = st.columns(2)

    with c_equity:
        st.markdown("#### Equity Dashboard (Počiatočný kapitál: $100)")
        equity_curve = [100.0]
        timeline = ["Štart"]

        for idx, t in enumerate(st.session_state["monthly_trades"]):
            equity_curve.append(100.0 + sum([x["pnl_usd"] for x in st.session_state["monthly_trades"][:idx + 1]]))
            timeline.append(t.get("date_label", f"Trade {idx+1}"))

        df_eq = pd.DataFrame({"Konto Balance ($)": equity_curve}, index=timeline)
        st.line_chart(df_eq)

    with c_matrix:
        st.markdown("#### Matrica Výkonu a Učenia AI modelov")
        matrix_rows = []
        for c in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "UNIUSDT"]:
            ai_setup = get_complete_ai_setup(c)
            matrix_rows.append({
                "Kryptomena": c,
                "Učebné kroky (AI)": ai_setup["total_trades"],
                "Win Rate": f"{ai_setup['win_rate']:.1f} %",
                "Expectancy": f"{ai_setup.get('expectancy', 0):.3f} R"
            })
        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Mesačný register zrealizovaných obchodov")
    if st.session_state["monthly_trades"]:
        st.dataframe(
            pd.DataFrame(st.session_state["monthly_trades"])[["Pair", "Result", "PnL %"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption("Žiadne zrealizované obchody za tento kalendárny mesiac.")

    st.markdown("---")

    # =====================================================================
    # OPERAČNÉ OKNÁ - všetky 3 pod seba + vlastný timeframe
    # =====================================================================
    st.markdown("### Autonómne Sviečkové Operačné Okná")

    # ---------- OKNO 1: BTCUSDT ----------
    st.markdown("---")
    c_head1, c_tf1 = st.columns([0.75, 0.25])
    with c_head1:
        st.markdown("#### Okno 1: Fixná Trhová Kotva (BTCUSDT)")
    with c_tf1:
        st.session_state["tf_w1"] = st.radio(
            "Timeframe",
            ["15m", "1m"],
            index=0 if st.session_state["tf_w1"] == "15m" else 1,
            horizontal=True,
            key="rad_w1"
        )

    if btc_data:
        draw_professional_dashboard(
            "BTCUSDT",
            btc_data,
            setups.get("BTCUSDT"),
            opt_lines,
            btc_data["is_active_trade"]
        )
    else:
        st.error("Nepodarilo sa načítať reálne dáta z Binance pre Okno 1")

    # ---------- OKNO 2: Šampión ----------
    st.markdown("---")
    c_head2, c_tf2 = st.columns([0.75, 0.25])
    with c_head2:
        st.markdown(f"#### Okno 2: Šampión Anomálií ({champ_pair})")
    with c_tf2:
        st.session_state["tf_w2"] = st.radio(
            "Timeframe",
            ["15m", "1m"],
            index=0 if st.session_state["tf_w2"] == "15m" else 1,
            horizontal=True,
            key="rad_w2"
        )

    if champ_data:
        draw_professional_dashboard(
            champ_pair,
            champ_data,
            setups.get(champ_pair),
            None,
            champ_data["is_active_trade"]
        )
    else:
        st.error("Nepodarilo sa načítať reálne dáta z Binance pre Okno 2")

    # ---------- OKNO 3: Vlastný výber ----------
    st.markdown("---")
    c_head3, c_tf3 = st.columns([0.75, 0.25])
    with c_head3:
        st.markdown(f"#### Okno 3: Súkromný Radar ({selected_custom})")
    with c_tf3:
        st.session_state["tf_w3"] = st.radio(
            "Timeframe",
            ["15m", "1m"],
            index=0 if st.session_state["tf_w3"] == "15m" else 1,
            horizontal=True,
            key="rad_w3"
        )

    if custom_data:
        draw_professional_dashboard(
            selected_custom,
            custom_data,
            setups.get(selected_custom),
            None,
            custom_data["is_active_trade"]
        )
    else:
        st.error("Nepodarilo sa načítať reálne dáta z Binance pre Okno 3")

    # Heartbeat
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now()

    @st.fragment(run_every=60)
    def background_heartbeat():
        st.session_state["last_refresh"] = datetime.now()

    background_heartbeat()


if __name__ == "__main__":
    main()
