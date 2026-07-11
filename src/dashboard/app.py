"""Streamlit dashboard: 投資組合回測 + 單一標的策略回測 (TPEx 上櫃/興櫃).

Run with:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.backtest.engine import Backtester
from src.backtest.portfolio import run_portfolio
from src.backtest.strategy import BollingerBand, MovingAverageCross, RSIThreshold
from src.crawler.client import TpexClient
from src.crawler.update import sync_daily_range, sync_index_range
from src.storage import db

st.set_page_config(page_title="櫃買市場回測系統", layout="wide")

MARKET_LABEL = {"otc": "上櫃", "esb": "興櫃"}
# 嚴格限定櫃買市場（上櫃、興櫃）掛牌之證券；display 名稱區隔股票與各類商品
CATEGORY_DISPLAY = {"上櫃": "上櫃股票", "興櫃": "興櫃股票", "ETF": "ETF", "債券ETF": "債券ETF", "ETN": "ETN"}
CATEGORY_ORDER = ["上櫃股票", "興櫃股票", "ETF", "債券ETF", "ETN"]


@st.cache_resource
def get_conn():
    conn = db.get_conn()
    db.init_db(conn)
    return conn


# ---------------------------------------------------------------- portfolio


def _ensure_index_data(conn, start: date, end: date) -> pd.Series:
    bench = db.load_index_series(conn, start.isoformat(), end.isoformat())
    if bench.empty or bench.index.max().date() < end - timedelta(days=7):
        with st.spinner("正在補抓櫃買指數資料..."):
            try:
                sync_index_range(TpexClient(), conn, start, end)
                bench = db.load_index_series(conn, start.isoformat(), end.isoformat())
            except Exception as exc:  # noqa: BLE001
                st.warning(f"櫃買指數抓取失敗（{exc}），將不顯示大盤比較。")
    return bench


def portfolio_tab(conn) -> None:
    companies = db.load_companies(conn)
    if companies.empty:
        st.warning("尚無標的清單，請先在終端機執行 `python cli.py sync-companies`。")
        return

    # 嚴格限定櫃買市場掛牌證券（上櫃、興櫃），並以顯示名稱區隔股票／ETF／ETN／債券ETF
    companies = companies[companies["category"].isin(CATEGORY_DISPLAY)].copy()
    companies["cat_display"] = companies["category"].map(CATEGORY_DISPLAY)

    if "pf_list" not in st.session_state:
        st.session_state.pf_list = []  # [{code, market, name, cat, weight}]
    pf_list = st.session_state.pf_list

    st.subheader("投資組合設定")
    st.caption("先選類別，再從下拉選單選擇標的後按「加入」。範圍限櫃買市場（上櫃、興櫃）掛牌之證券。")
    c_cat, c_sec, c_add = st.columns([1.4, 3, 0.9])
    cat = c_cat.selectbox("類別", CATEGORY_ORDER, key="pf_cat")
    sub = companies[companies["cat_display"] == cat].sort_values("code")
    options = (sub["code"] + "  " + sub["name"]).tolist()
    picked = c_sec.selectbox("標的", options, key=f"pf_pick_{cat}", placeholder="下拉選擇或輸入搜尋")
    c_add.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if c_add.button("＋ 加入", use_container_width=True, key="pf_add") and picked:
        code = picked.split(" ")[0]
        row = sub[sub["code"] == code].iloc[0]
        if not any(x["code"] == code and x["market"] == row["market"] for x in pf_list):
            pf_list.append({"code": code, "market": row["market"], "name": row["name"], "cat": row["cat_display"], "weight": 0.0})
            equal = round(100 / len(pf_list), 2)
            for x in pf_list:
                x["weight"] = equal
            st.rerun()

    if not pf_list:
        st.info("尚未加入標的。")
        return

    weight_df = pd.DataFrame(
        {
            "標的": [f"{x['code']} {x['name']}" for x in pf_list],
            "類別": [x["cat"] for x in pf_list],
            "權重": [x["weight"] for x in pf_list],
            "移除": [False] * len(pf_list),
        }
    )
    edited = st.data_editor(
        weight_df,
        column_config={
            "標的": st.column_config.TextColumn(disabled=True),
            "類別": st.column_config.TextColumn(disabled=True),
            "權重": st.column_config.NumberColumn("權重 (%)", min_value=0.0, max_value=100.0, step=1.0),
            "移除": st.column_config.CheckboxColumn("移除"),
        },
        hide_index=True,
        use_container_width=True,
        key=f"pf_editor_{len(pf_list)}_{hash(tuple(x['code'] for x in pf_list)) & 0xFFFF}",
    )
    for x, w in zip(pf_list, edited["權重"]):
        x["weight"] = float(w)
    if edited["移除"].any():
        st.session_state.pf_list = [x for x, rm in zip(pf_list, edited["移除"]) if not rm]
        st.rerun()
    total_w = edited["權重"].sum()
    if abs(total_w - 100) > 0.5:
        st.caption(f"目前權重合計 {total_w:.1f}%，計算時會自動按比例正規化為 100%。")

    c1, c2, c3, c4 = st.columns(4)
    start_date = c1.date_input("開始日期", value=date.today() - timedelta(days=365), key="pf_start")
    end_date = c2.date_input("結束日期", value=date.today(), key="pf_end")
    capital = c3.number_input("起始資金 (NTD)", value=1_000_000, step=100_000, key="pf_capital")
    rebalance_label = c4.selectbox("再平衡頻率", ["不再平衡（買進持有）", "每月", "每季"], key="pf_rebalance")
    rebalance = {"不再平衡（買進持有）": "none", "每月": "monthly", "每季": "quarterly"}[rebalance_label]

    if not st.button("執行組合回測", type="primary", key="pf_run"):
        return

    # -------- load prices
    price_map: dict[str, pd.Series] = {}
    display_name: dict[str, str] = {}
    cat_of: dict[str, str] = {}
    weights_dict: dict[str, float] = {}
    missing = []
    for x in pf_list:
        s = db.load_price_series(conn, x["code"], x["market"], start_date.isoformat(), end_date.isoformat())["close"].dropna()
        label = f"{x['code']} {x['name']}（{x['cat']}）"
        if s.empty:
            missing.append(label)
        else:
            price_map[x["code"]] = s
            display_name[x["code"]] = label
            cat_of[x["code"]] = x["cat"]
            weights_dict[x["code"]] = x["weight"]
    if missing:
        st.error(
            "以下標的在資料庫中查無此區間資料，請先透過下方「資料更新」抓取，或調整日期：\n\n- "
            + "\n- ".join(missing)
        )
    if len(price_map) < 1:
        _data_update_expander(conn, start_date, end_date)
        return

    prices = pd.DataFrame(price_map)
    weights = pd.Series(weights_dict, dtype=float)

    bench = _ensure_index_data(conn, start_date, end_date)

    try:
        result = run_portfolio(prices, weights, initial_capital=capital, rebalance=rebalance, benchmark=bench)
    except ValueError as exc:
        st.error(str(exc))
        return

    m = result.metrics
    st.subheader("組合績效")
    cols = st.columns(6)
    cols[0].metric("組合報酬率", f"{m['total_return']:.2%}")
    bench_ret = m.get("benchmark_return")
    cols[1].metric("大盤報酬率（櫃買指數）", f"{bench_ret:.2%}" if pd.notna(bench_ret) else "-")
    excess = m.get("excess_return")
    cols[2].metric("超額報酬", f"{excess:+.2%}" if pd.notna(excess) else "-")
    cols[3].metric("年化報酬率 (CAGR)", f"{m['cagr']:.2%}" if pd.notna(m.get("cagr")) else "-")
    cols[4].metric("Sharpe", f"{m['sharpe']:.2f}" if pd.notna(m.get("sharpe")) else "-")
    cols[5].metric("最大回撤", f"{m['max_drawdown']:.2%}")

    # -------- risk metrics
    st.subheader("風險指標")
    rcols = st.columns(6)
    rcols[0].metric("年化波動度", f"{m['annual_volatility']:.2%}" if pd.notna(m.get("annual_volatility")) else "-")
    rcols[1].metric("Beta（對櫃買指數）", f"{m['beta']:.3f}" if pd.notna(m.get("beta")) else "-")
    rcols[2].metric("Alpha（年化）", f"{m['alpha_annual']:+.2%}" if pd.notna(m.get("alpha_annual")) else "-")
    rcols[3].metric("Sortino", f"{m['sortino']:.2f}" if pd.notna(m.get("sortino")) else "-")
    rcols[4].metric("日 VaR (95%)", f"{m['var95']:.2%}" if pd.notna(m.get("var95")) else "-")
    rcols[5].metric("下行波動（年化）", f"{m['downside_dev']:.2%}" if pd.notna(m.get("downside_dev")) else "-")

    reg = m.get("regression")
    if reg:
        with st.expander(f"市場模型迴歸（組合日報酬 ~ 櫃買指數日報酬，n={reg['n']}，R²={reg['r2']:.3f}）"):
            reg_df = pd.DataFrame(
                [
                    {
                        "": r["name"],
                        "Coef": f"{r['coef']:.4f}",
                        "Std err": f"{r['se']:.4f}",
                        "t": f"{r['t']:.3f}" if pd.notna(r["t"]) else "-",
                        "P>|t|": f"{r['p']:.3f}" if pd.notna(r["p"]) else "-",
                        "[0.025": f"{r['lo']:.4f}",
                        "0.975]": f"{r['hi']:.4f}",
                    }
                    for r in reg["rows"]
                ]
            )
            st.dataframe(reg_df, hide_index=True, use_container_width=True)
            st.caption("Beta > 1 表示波動大於大盤；Alpha 為經市場風險調整後的超額報酬（截距×252 為年化值）。P 值以常態近似計算。")

    # -------- pie + equity curves side by side
    pie_col, curve_col = st.columns([2, 3])
    norm_w = weights / weights.sum()
    pie_df = pd.DataFrame(
        {
            "標的": [display_name[c] for c in norm_w.index],
            "類別": [cat_of[c] for c in norm_w.index],
            "權重": norm_w.values,
        }
    )
    with pie_col:
        fig_pie = px.pie(
            pie_df, values="權重", names="標的", hover_data=["類別"], hole=0.35,
            title="組合配置",
        )
        fig_pie.update_traces(textinfo="percent+label", textposition="inside")
        fig_pie.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

        cat_df = pie_df.groupby("類別", as_index=False)["權重"].sum()
        fig_cat = px.pie(cat_df, values="權重", names="類別", hole=0.35, title="類別占比")
        fig_cat.update_traces(textinfo="percent+label")
        fig_cat.update_layout(height=320)
        st.plotly_chart(fig_cat, use_container_width=True)

    with curve_col:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity, name="投資組合", line=dict(color="#F58518", width=2)))
        if result.benchmark_equity is not None:
            fig.add_trace(go.Scatter(
                x=result.benchmark_equity.index, y=result.benchmark_equity,
                name="櫃買指數（同額投入）", line=dict(color="#4C78A8", dash="dash"),
            ))
        fig.update_layout(height=420, title="權益曲線 vs 大盤", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        running_max = result.equity.cummax()
        drawdown = result.equity / running_max - 1
        dd_fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", line=dict(color="#D62728")))
        dd_fig.update_layout(height=320, title="組合回撤", yaxis_tickformat=".0%")
        st.plotly_chart(dd_fig, use_container_width=True)

    # -------- per-asset table
    st.subheader("各標的明細")
    detail = pd.DataFrame(
        {
            "標的": [display_name[c] for c in prices.columns],
            "類別": [cat_of[c] for c in prices.columns],
            "權重": [f"{norm_w[c]:.1%}" for c in prices.columns],
            "期初價": prices.iloc[0].values,
            "期末價": prices.iloc[-1].values,
            "期間報酬率": [f"{result.asset_returns[c]:+.2%}" for c in prices.columns],
        }
    )
    st.dataframe(detail, hide_index=True, use_container_width=True)
    st.caption(
        f"實際回測區間：{result.equity.index[0].date()} ~ {result.equity.index[-1].date()}"
        "（自所有標的皆有價格的第一天起算）。興櫃以最後成交價計算；組合回測未計入交易成本。"
    )

    _data_update_expander(conn, start_date, end_date)


def _data_update_expander(conn, start_date: date, end_date: date) -> None:
    with st.expander("資料更新（呼叫櫃買中心 API）"):
        st.caption("抓取上櫃＋興櫃全市場此區間的日成交資料與櫃買指數。每個交易日約兩次請求，時間視區間長短。")
        if st.button("更新此區間資料", key=f"pf_sync_{start_date}_{end_date}"):
            client = TpexClient()
            companies = db.load_companies(conn)
            progress = st.progress(0.0, text="同步中...")
            for i, market in enumerate(("otc", "esb")):
                valid = set(companies.loc[companies["market"] == market, "code"])
                sync_daily_range(client, conn, market, start_date, end_date, valid_codes=valid or None, show_progress=False)
                progress.progress((i + 1) / 3, text=f"{MARKET_LABEL[market]} 完成")
            sync_index_range(client, conn, start_date, end_date)
            progress.progress(1.0, text="完成")
            st.success("資料更新完成，請重新執行回測。")


# ---------------------------------------------------------------- single stock


def build_strategy(kind: str, params: dict):
    if kind == "ma_cross":
        return MovingAverageCross(short_window=params["short"], long_window=params["long"])
    if kind == "rsi":
        return RSIThreshold(period=params["period"], lower=params["lower"], upper=params["upper"])
    if kind == "bollinger":
        return BollingerBand(window=params["window"], num_std=params["num_std"])
    raise ValueError(kind)


def single_stock_tab(conn) -> None:
    with st.sidebar:
        st.header("單一標的策略回測")
        market = st.selectbox("市場", ["otc", "esb"], format_func=lambda m: MARKET_LABEL[m])
        companies = db.load_companies(conn, market)

        if companies.empty:
            st.warning("尚無標的清單，請先在終端機執行：\n\n`python cli.py sync-companies`")
            code = st.text_input("股票代號", value="")
        else:
            options = (companies["code"] + "  " + companies["name"] + "（" + companies["category"] + "）").tolist()
            choice = st.selectbox("股票", options)
            code = choice.split(" ")[0] if choice else ""

        default_start = date.today() - timedelta(days=365 * 2)
        start_date = st.date_input("開始日期", value=default_start)
        end_date = st.date_input("結束日期", value=date.today())

        st.divider()
        strategy_kind = st.selectbox(
            "策略類型",
            ["ma_cross", "rsi", "bollinger"],
            format_func=lambda k: {"ma_cross": "均線交叉", "rsi": "RSI 門檻", "bollinger": "布林通道"}[k],
        )
        params = {}
        if strategy_kind == "ma_cross":
            params["short"] = st.slider("短期均線", 2, 60, 5)
            params["long"] = st.slider("長期均線", 5, 240, 20)
        elif strategy_kind == "rsi":
            params["period"] = st.slider("RSI 週期", 2, 60, 14)
            params["lower"] = st.slider("買進門檻", 0, 50, 30)
            params["upper"] = st.slider("賣出門檻", 50, 100, 70)
        elif strategy_kind == "bollinger":
            params["window"] = st.slider("窗口天數", 5, 120, 20)
            params["num_std"] = st.slider("標準差倍數", 0.5, 4.0, 2.0, step=0.1)

        st.divider()
        capital = st.number_input("起始資金 (NTD)", value=1_000_000, step=100_000)
        commission_pct = st.number_input("手續費率 (%)", value=0.1425, format="%.4f")
        discount = st.slider("手續費折數", 0.1, 1.0, 1.0)
        tax_pct = st.number_input("證交稅率 (%，僅賣出課徵)", value=0.3, format="%.3f")
        slippage_bp = st.number_input("滑價 (基點 bp)", value=0.0, step=1.0)

        run_clicked = st.button("執行回測", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("在左側側欄選擇標的與策略後，點擊「執行回測」。")
        return

    if not code:
        st.error("請輸入或選擇股票代號。")
        return

    price_df = db.load_price_series(conn, code, market, start_date.isoformat(), end_date.isoformat())
    if price_df.empty:
        st.error("資料庫內查無此標的在此區間的資料，請先於「投資組合」分頁的資料更新區塊抓取資料。")
        return

    strategy = build_strategy(strategy_kind, params)
    bt = Backtester(
        initial_capital=capital,
        commission_rate=commission_pct / 100,
        commission_discount=discount,
        tax_rate=tax_pct / 100,
        slippage_bp=slippage_bp,
    )
    result = bt.run(price_df, strategy)

    name = companies.loc[companies["code"] == code, "name"].values if not companies.empty else []
    title = f"{code} {name[0] if len(name) else ''}　{MARKET_LABEL[market]}　策略：{strategy.name}"
    st.subheader(title)

    m = result.metrics
    cols = st.columns(6)
    cols[0].metric("總報酬率", f"{m.get('total_return', float('nan')):.2%}")
    cols[1].metric("年化報酬率 (CAGR)", f"{m.get('cagr', float('nan')):.2%}")
    cols[2].metric("年化波動度", f"{m.get('annual_volatility', float('nan')):.2%}")
    cols[3].metric("Sharpe", f"{m.get('sharpe', float('nan')):.2f}")
    cols[4].metric("最大回撤", f"{m.get('max_drawdown', float('nan')):.2%}")
    cols[5].metric("交易次數", f"{m.get('num_trades', 0):d}")
    cols2 = st.columns(6)
    cols2[0].metric("勝率", f"{m.get('win_rate', float('nan')):.1%}" if pd.notna(m.get("win_rate")) else "-")
    pf = m.get("profit_factor")
    cols2[1].metric("獲利因子", f"{pf:.2f}" if pd.notna(pf) else "-")
    avg_tr = m.get("avg_trade_return")
    cols2[2].metric("平均每筆報酬", f"{avg_tr:.2%}" if pd.notna(avg_tr) else "-")

    pos_change = result.positions.diff().fillna(result.positions.iloc[0])
    buy_dates = pos_change[pos_change > 0].index
    sell_dates = pos_change[pos_change < 0].index

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.6, 0.4], subplot_titles=("收盤價與買賣點", "權益曲線"),
    )
    fig.add_trace(go.Scatter(x=price_df.index, y=price_df["close"], name="收盤價", line=dict(color="#4C78A8")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=buy_dates, y=price_df.loc[buy_dates, "close"], mode="markers", name="買進",
        marker=dict(symbol="triangle-up", size=10, color="#2CA02C"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sell_dates, y=price_df.loc[sell_dates, "close"], mode="markers", name="賣出",
        marker=dict(symbol="triangle-down", size=10, color="#D62728"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.equity_curve.index, y=result.equity_curve, name="權益", line=dict(color="#F58518")), row=2, col=1)
    fig.update_layout(height=700, legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    running_max = result.equity_curve.cummax()
    drawdown = result.equity_curve / running_max - 1
    dd_fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", line=dict(color="#D62728")))
    dd_fig.update_layout(height=250, title="回撤", yaxis_tickformat=".0%")
    st.plotly_chart(dd_fig, use_container_width=True)

    st.subheader("交易明細")
    if result.trades.empty:
        st.write("此區間內策略未產生任何交易。")
    else:
        trades_display = result.trades.copy()
        trades_display["return"] = trades_display["return"].map(lambda x: f"{x:.2%}")
        trades_display["open"] = trades_display["open"].map(lambda x: "持有中（未平倉）" if x else "已平倉")
        trades_display = trades_display.rename(columns={"open": "狀態"})
        st.dataframe(trades_display, use_container_width=True)


def main() -> None:
    conn = get_conn()
    st.title("台灣櫃買市場（上櫃 / 興櫃）回測系統")
    tab_portfolio, tab_single = st.tabs(["投資組合回測", "單一標的策略回測"])
    with tab_portfolio:
        portfolio_tab(conn)
    with tab_single:
        single_stock_tab(conn)


if __name__ == "__main__":
    main()
