# QFF-TSM Spread Backtest

這個專案下載並整理 TAIFEX QFF、Binance TSMUSDT.P、BitoPro USDT/TWD 的 1m 資料，計算 TradingView Pine 公式對應的 spread 與 rolling z-score，並用簡單事件式回測框架測試 QFF/TSM 配對交易策略。

## 資料來源

所有時間戳都轉成 Taipei time (`+08:00`)。

- **QFF**
  - 來源：TAIFEX tick 歷史資料。
  - 腳本：`scripts/build_qff1_1m.py`
  - 輸出：`data/processed/qff1_1m.csv`
  - 處理：將 tick 聚合成 1m OHLCV，作為 QFF1 front-month 連續資料。

- **TSMUSDT.P**
  - 來源：Binance USD-M futures，透過 `ccxt` 下載。
  - symbol：`TSM/USDT:USDT`
  - 腳本：`scripts/download_binance_tsmusdtp_1m.py`
  - 輸出：`data/processed/binance_tsmusdtp_1m_taipei.csv`

- **USDT/TWD**
  - 來源：BitoPro，透過 `ccxt` 下載。
  - symbol：`USDT/TWD`
  - 腳本：`scripts/download_bitopro_usdttwd_1m.py`
  - 輸出：`data/processed/bitopro_usdttwd_1m_taipei.csv`

目前 QFF session 回測資料範圍為 `2026-05-08 17:25:00+08:00` 到 `2026-06-22 13:44:00+08:00`，共約 `29,909` 根 1m bars。

## 指標計算

Spread 計算腳本：

```powershell
& 'D:\Users\miniconda3\condabin\conda.bat' run -n Quant python scripts/calculate_qff_tsm_spread_1m.py
```

計算方式：

```text
qff_close_filled = QFF close, forward-filled only inside QFF trading sessions
tsm_twd_fair = tsm_close * usdttwd_close / 5
spread = (tsm_twd_fair - qff_close_filled) / (tsm_twd_fair + qff_close_filled) * 200
```

資料以 QFF trading session 為主，非 trading session 全部裁掉。Session 內若 QFF 沒有該分鐘 bar，才用上一根 QFF close 補齊並標記 `qff_was_filled=True`；TSM 與 USDT/TWD 若缺分鐘則視為資料異常，不自動補值。

Z-score 計算腳本：

```powershell
& 'D:\Users\miniconda3\condabin\conda.bat' run -n Quant python scripts/calculate_spread_zscore_1m.py
```

計算方式：

```text
spread_mean_997 = rolling_mean(spread, window=997, min_periods=997)
spread_std_997  = rolling_std(spread, window=997, min_periods=997, ddof=0)
spread_zscore   = (spread - spread_mean_997) / spread_std_997
```

前 `996` 筆 QFF session 觀測為 warmup，`zscore_valid = False`。

## 交易策略

回測腳本：

```powershell
& 'D:\Users\miniconda3\condabin\conda.bat' run -n Quant python scripts/backtest_pair_strategy_1m.py
```

預設參數：

```text
entry_z = 2.0
exit_z = 0.0
leg_notional_twd = 1,000,000
initial_capital_twd = 2,000,000
max_entry_delay_minutes = 15
```

Entry:

- 只在 `entry_allowed=True` 且 `zscore_valid=True` 的分鐘評估。
- 若目前空手且 `z > entry_z`，建立 `short TSM / long QFF` 訊號。
- 若目前空手且 `z < -entry_z`，建立 `long TSM / short QFF` 訊號。
- 訊號成立後，在下一個 `entry_allowed=True` 分鐘用 open 成交。
- 若成交時間與訊號時間差超過 `max_entry_delay_minutes`，取消該訊號。
- 成交分鐘不重新驗證 z-score。

Exit:

- `short TSM / long QFF`：若 `z < -exit_z`，建立平倉訊號。
- `long TSM / short QFF`：若 `z > exit_z`，建立平倉訊號。
- z-score 與 time-stop 平倉訊號成立後，在下一個 `close_allowed=True` 分鐘用 open 成交。
- 若持倉進入每週最後一段可交易 session 且沒有觸發 z-score exit，會在該 session 最後一個 `close_allowed=True` 分鐘用 close 強制平倉，`exit_reason = friday_session_end`。
- 若資料結束仍有持倉，最後一根強制平倉。

QFF 交易時段：

- 日盤：`08:45` 到 `13:45`
- 夜盤：`17:25` 到次日 `05:00`
- 週五夜盤與每週最後一段可交易 session 只允許平倉，不允許開倉。
- 每週最後 session 由資料中的下一個 `close_allowed=True` 分鐘是否跨 ISO week 判斷，因此週五假日或沒有週五夜盤時，會改由週末前最後一段實際可交易 session 生效。

## 部位與成本

QFF 先用 entry fill bar 的 open 轉成實際可交易整數口數，再用 QFF 實際名目金額對齊 TSM 腿：

```text
raw_qff_contracts = leg_notional_twd / (entry_qff_open_filled * qff_contract_multiplier)
qff_contracts = floor(raw_qff_contracts + 0.5)
actual_leg_notional_twd = abs(qff_contracts) * qff_contract_multiplier * entry_qff_open_filled
tsm_units = actual_leg_notional_twd / entry_tsm_twd_fair_open
```

若 `qff_contracts == 0`，該次 entry 取消。

預設成本：

```text
tsm_fee_bps = 5.0
qff_fee_per_contract_twd = 88.0
qff_tax_rate = 0.00002
qff_contract_multiplier = 100
```

QFF 單邊手續費 88 TWD/口，來回 88×2 = 176；加上來回交易稅（每邊約 5，合計約 10），單口來回總交易成本約 88×2 + 10 = 186 TWD。

手續費與交易稅都是單邊成本，entry 與 exit 各扣一次：

```text
tsm_fee_twd = abs(tsm_units) * fill_tsm_twd_fair * tsm_fee_bps / 10000
qff_fee_twd = abs(qff_contracts) * qff_fee_per_contract_twd
qff_tax_twd = abs(qff_contracts) * round(qff_price * qff_contract_multiplier * qff_tax_rate)
net_pnl_twd = gross_pnl_twd - total_fee_twd
```

目前不計入 FX 換匯成本、TSM funding、滑價、保證金利息或券商額外手續費。

## 輸出

回測輸出：

- `data/processed/qff_tsm_pair_backtest_equity_1m_qff_session.csv`
- `data/processed/qff_tsm_pair_backtest_trades_qff_session.csv`
- `data/processed/qff_tsm_pair_backtest_summary_qff_session.json`

目前 QFF session 預設輸出的最新結果（QFF 手續費 88/口，fee as-of 2026-06-30）：

```text
trades = 90
net_pnl_twd = 172,453.68
return_pct = 8.6227%
max_drawdown_twd = -38,486.57
qff_forward_filled_session_minutes = 6,328
zscore_valid_rows = 29,412
```

主力回測已改用 15m 資料（見 `qff_tsm_parameter_grid_report_15m.html`）；15m grid 最佳組態 w33 / entry 2.0 / exit 0.5 在新手續費下為 net 273,923、return 13.70%、Sharpe 6.10。

大型 raw/processed CSV 資料預設不進 Git；需要重建時依序重跑下載、spread、z-score、backtest 腳本。舊的連續 1m 檔案保留，新 QFF session 補值版輸出使用 `_qff_session` 後綴。
