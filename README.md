# 台灣櫃買市場（上櫃／興櫃）回測系統

抓取證券櫃檯買賣中心（TPEx）公開資料，建立本地資料庫，並提供 Streamlit 儀表板：

- **投資組合回測**：同時選擇多檔標的（上櫃股票／興櫃股票／ETF／債券ETF／ETN 可混搭），
  自訂權重與再平衡頻率，顯示配置圓餅圖（含類別占比）、組合報酬率、大盤（櫃買指數）
  報酬率與超額報酬、權益曲線對比、回撤，以及各標的類別與期間報酬明細。
- **單一標的策略回測**：均線交叉／RSI／布林通道策略，含台灣交易成本設定與交易明細。

## 安裝

```bash
pip install -r requirements.txt
```

## 使用流程

1. 初始化資料庫並抓取公司清單：

   ```bash
   python cli.py sync-companies
   ```

2. 抓取歷史日成交資料（依市場分別執行，第一次會抓取起訖區間內每個交易日的
   全市場資料再篩選存檔，時間視區間長短而定）：

   ```bash
   python cli.py sync-quotes --market otc --start 2023-01-01 --end 2026-07-11
   python cli.py sync-quotes --market esb --start 2023-01-01 --end 2026-07-11
   ```

   之後若只想補到最新，可省略 `--start`（會從上次同步的日期繼續）：

   ```bash
   python cli.py sync-quotes --market otc
   ```

3. 抓取櫃買指數（投資組合分頁的大盤比較用；儀表板執行回測時若缺資料也會自動補抓）：

   ```bash
   python cli.py sync-index --start 2023-01-01
   ```

4. 查看目前資料庫涵蓋範圍：

   ```bash
   python cli.py status
   ```

5. 啟動儀表板：

   ```bash
   streamlit run src/dashboard/app.py
   ```

   在左側選擇市場、股票、資料區間、策略與交易成本，點「執行回測」即可看到
   價格走勢與買賣點、權益曲線、回撤、績效指標與交易明細。若資料庫內該區間
   尚無資料，可在儀表板的「資料更新」區塊直接觸發抓取。

## 蒙地卡羅回測分析網頁（React + FastAPI）

五頁式單頁應用：①標的與參數設定（市場類別→產業分類→個股三層連動下拉、
chip 複選）②回測結果（權益曲線、指標卡、Coef/Std err/t/P>|t|/CI 迴歸統計表、
逐筆交易明細＋CSV 匯出）③蒙地卡羅設定（reshuffle／bootstrap、次數、進度條）
④結果視覺化（回撤分布直方圖、Spaghetti 疊圖、百分位數表、風險低估判定）
⑤多標的比較表。

```bash
pip install fastapi "uvicorn[standard]"
uvicorn backend.main:app --port 8600 --app-dir .   # 於 tpex-backtest 目錄執行
# 開啟 http://localhost:8600/
```

- 後端模組：`backend/tpex_data.py`（清單／價格，僅上櫃＋興櫃）、
  `backend/backtest.py`（交易型回測：均線交叉／RSI、單筆風險%＝停損、槓桿、
  OLS 迴歸）、`backend/monte_carlo.py`（reshuffle／bootstrap 模擬）。
- 前端：`frontend/index.html`（React 18 + Tailwind + Plotly，CDN 載入，
  Babel 需鎖 7.x——Babel 8 的 JSX 預設 automatic runtime 會使 inline script 失效）。
- 單筆風險%的語意：該筆交易的權益停損門檻（收盤價檢查）；
  蒙地卡羅的「原始最大回撤」以交易序列計，與日線口徑的差異在頁面上有註明。

## 部署蒙地卡羅網頁到 Render（公開網址，任何人可用）

已備妥 `render.yaml`（Blueprint）與 `requirements-deploy.txt`。步驟：

1. 在 GitHub 建立 repo（private 亦可），推送本資料夾：
   ```bash
   git remote add origin https://github.com/<你的帳號>/<repo>.git
   git push -u origin master
   ```
2. 到 <https://render.com> 用 GitHub 登入 → **New + → Blueprint** →
   選擇該 repo → **Apply**。Render 會自動讀取 `render.yaml` 建立服務。
3. 部署完成後得到 `https://tpex-mc-backtest.onrender.com` 之類的公開網址。

注意：
- 免費方案閒置 15 分鐘會休眠，下次開啟需等約 30–60 秒喚醒。
- 行情資料是隨 repo 提交的快照（`data/tpex.sqlite3`）；要更新，
  在本機執行 `python cli.py sync-quotes ...` 後重新 commit + push，Render 會自動重新部署。

## 部署到 Streamlit Community Cloud（讓其他人線上使用）

1. 把本資料夾推上 GitHub（`data/tpex.sqlite3` 一併提交，作為預載資料；
   使用者也可在網頁上用「資料更新」按鈕即時抓取櫃買中心資料）。
2. 到 <https://share.streamlit.io> 用 GitHub 帳號登入 → New app →
   選擇該 repo，主檔案填 `streamlit_app.py` → Deploy。
3. 部署完成後會得到一個公開網址，任何人都能開啟操作。

注意：Cloud 上的檔案系統是暫時性的——網頁上「資料更新」抓的資料在應用程式
重啟後會消失。要永久擴充歷史資料，請在本機同步後重新 commit `data/tpex.sqlite3`。

## 輸出成可分享的網頁

`export_web.py` 會把資料庫內的行情資料連同 JavaScript 版回測引擎打包成單一
HTML 檔（`web/index.html`），任何人用瀏覽器開啟即可操作，不需安裝 Python：

```bash
python export_web.py                # 匯出資料庫內全部區間
python export_web.py --start 2025-01-01 --end 2026-07-11
```

網頁內容為匯出當下的靜態快照；同步新資料後重新執行匯出（並重新發佈）即可更新。
模板在 `web/template.html`，兩個分頁（投資組合／單一標的策略）的計算邏輯與
Python 引擎一致。

## 資料來源與限制

- 上櫃（OTC）與興櫃（ESB／興櫃）資料皆來自 `https://www.tpex.org.tw` 的公開
  查詢頁面（`src/crawler/daily_quotes.py` 中列有詳細端點說明），非官方文件化
  API，若網站改版可能需要調整。
- **興櫃沒有真正的「開盤價」**：興櫃採議價／推薦證券商制度，資料只提供當日
  最高、最低、最後成交價與日均價，因此 `daily_quotes` 中興櫃的 `open` 欄位
  一律為空，回測一律以「最後成交價」（`close`）作為交易執行價。上櫃則有完整
  開高低收。
- 交易成本預設值為台灣一般個人戶概略行情（手續費 0.1425%、賣出證交稅
  0.3%），可在儀表板調整；興櫃實際手續費另有規範，使用前請自行確認。
- 回測引擎為單一標的、多空僅做多（不含放空），訊號於隔日執行以避免未來
  函數；請勿將回測結果視為投資建議。

## 專案結構

```
cli.py                     命令列工具：初始化 DB、同步標的清單／日成交資料／櫃買指數
src/crawler/                爬蟲：client.py（HTTP）、company_list.py（含 ETF/ETN 類別）、
                            daily_quotes.py、market_index.py（櫃買指數）、update.py
src/storage/db.py           SQLite 讀寫（companies / daily_quotes / index_quotes / crawl_state）
src/backtest/                策略（strategy.py）、單標的引擎（engine.py）、
                            投資組合引擎（portfolio.py）、績效指標（metrics.py）
src/dashboard/app.py         Streamlit 儀表板（投資組合＋單一標的雙分頁）
data/tpex.sqlite3            SQLite 資料庫（首次執行 cli.py 會自動建立）
```
