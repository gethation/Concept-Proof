# CCF/UMC 週末規則分析

**結論：CCF/UMC 應移除週末的兩條限制**（週五 session 進場禁令、該 session 最後
一根 bar 強制平倉）。實測淨利 +19.7%，**最大回撤完全不變**。

QFF/TSM 維持原樣 —— 那條規則對它是必要的。

日期：2026-07-24。資料：1m / window 2500 / entry_z 1.5 / exit_z 0.0 /
29 個 UMC RTH session（2026-06-09 → 07-23）。

---

## 這條規則原本在防什麼

週末強平是從 QFF/TSM 繼承來的，理由寫在 `lux_trader/core/calendar.py`：

> Binance TSM 永續 **24/7 交易**，而 QFF 週末凍結。持倉過週末等於裸露一條腿 ——
> TSM 那邊價格繼續動，QFF 這邊完全無法反應。

**CCF/UMC 沒有這個結構。** TAIFEX 與 NYSE 週末都關，兩腿同時凍結，避險比例
完整保持。所以移除它**不是放寬風控，是移除一條套錯標的的規則**。

---

## 三個變體的實測

| 政策 | 交易 | 淨利 | 報酬 | 虧損 | 最大回撤 | 曝險 |
|---|---:|---:|---:|---:|---:|---:|
| `flat`（現行，兩條都保留） | 12 | 181,054 | 9.05% | 1 | −25,586 | 20.2% |
| `no-entry`（只移除強制平倉） | 11 | 169,365 | 8.47% | 0 | −25,586 | 28.8% |
| **`none`（兩條都移除）** | **16** | **216,716** | **10.84%** | **0** | **−25,586** | 47.8% |

**三者的最大回撤完全相同。** 多承擔 3 個週末持倉，沒有讓最壞情況變差。

### 跨取樣頻率的獨立佐證

同樣的比較先前已在 5m 與 15m 上做過（`data/processed/*_weekendoff.json`、
`*_noforceclose.json`）。三個取樣頻率、三組獨立資料，結論一致：

| 取樣 | 基準 | 只移除強平 | **兩條都移除** | 最大回撤 |
|---|---:|---:|---:|---:|
| 15m w104 | 267,060 | 276,972 | **361,935 (+35.5%)** | 三者皆 −54,454 |
| 5m w500 | 181,931 | 196,888 | **222,762 (+22.4%)** | 三者皆 −23,645 |
| 1m w2500 | 181,054 | 169,365 | **216,716 (+19.7%)** | 三者皆 −25,586 |

**「兩條都移除」在每個頻率都是最佳，而最大回撤在每一組內都完全不變。**
這不是單一資料集的巧合。

「只移除強平」則不穩定：5m/15m 小幅改善，1m 反而變差（原因見下方分解）——
再次說明強平的去留本身影響不大，真正的成本在進場禁令。

重現方式：

```powershell
$conda = 'D:\Users\miniconda3\condabin\conda.bat'
foreach ($p in @('flat','no-entry','none')) {
    $tag = $p -replace '-',''
    & $conda run -n Quant python scripts/calculate_ccf_umc_spread_1m.py `
        --weekend-policy $p --out "data/processed/wk_${tag}_spread.csv"
    & $conda run -n Quant python scripts/calculate_spread_zscore_1m.py `
        --spread-path "data/processed/wk_${tag}_spread.csv" `
        --out "data/processed/wk_${tag}_zscore.csv" --window 2500
    & $conda run -n Quant python scripts/backtest_pair_strategy_1m.py `
        --input "data/processed/wk_${tag}_zscore.csv" `
        --qff-ohlcv data/processed/ccf1_1m_cumulative.csv `
        --tsm-ohlcv data/processed/umc_1m_cumulative.csv `
        --usdttwd-ohlcv "data/processed/wk_${tag}_spread_fx.csv" `
        --entry-z 1.5 --exit-z 0.0 --leg-notional-twd 1000000 `
        --qff-contract-multiplier 2000 --qff-fee-per-contract-twd 88 --tsm-fee-bps 2.5 `
        --summary-out "data/processed/wk_${tag}_summary.json" `
        --trades-out "data/processed/wk_${tag}_trades.csv"
}
```

---

## 貢獻分解：幾乎全部來自解除**進場禁令**

`none` 比 `flat` 多做的 5 筆交易：

| 進場 | 淨利 | 持倉 |
|---|---:|---:|
| 2026-06-26 | 26,322 | 180 min |
| 2026-07-02 | 2,576 | 173 min |
| 2026-07-02 | 5,451 | 5,406 min |
| 2026-07-09 | 8,397 | 5,759 min |
| 2026-07-17 | 4,605 | 291 min |
| **合計** | **47,351** | |

**其中 3 筆持倉不到 5 小時，根本不會碰到週末。** 它們只是剛好落在「該週最後一個
session」就被無差別禁止。

這是這條規則最大的代價：**它禁掉的不是「會跨週末的交易」，而是那個 session 裡的
所有訊號。** 一個週五早上出現、當天下午就收斂的訊號，跟週末毫無關係，卻同樣被擋。

### `no-entry` 為什麼反而更差

差別只在一筆交易：`2026-06-22` 那筆賺 29,564。

- `flat`：週五強制平倉 → 系統回到空手 → **週一開盤立刻接到新訊號**
- `no-entry`：沒有強制平倉，舊部位持有到週一才出場 → **錯過那個訊號**

**這是巧合，不是機制。** 強制平倉剛好把系統「重置」成可進場狀態。不應據此保留它 ——
同樣的巧合下次可能反向發生。

---

## 那筆唯一的虧損就是強制平倉造成的

`flat` 變體裡唯一的虧損：

```
進場       2026-06-18 00:10   z = −2.07
出場       2026-06-19 03:59   z = −1.02      ← 出場目標是 0，根本沒收斂
exit_reason  friday_session_end
毛利        +234
手續費      −1,015
淨利        −781
```

**毛利是正的。** 它被趕出場時價差還在收斂途中，扣掉手續費就變虧損。

移除限制後，同一筆持有到 z 收斂，變成 **+17,095**。

---

## 仍然存在的風險

**1. 週末跳空未被模型化。** 兩腿都凍結代表沒有裸腿，但週一開盤時價差可能已跳到
不利位置，而你在週末完全無法反應。29 個 session 只涵蓋 **3 個週末**，樣本量遠不足
以宣稱「週末跳空不是問題」。這 3 次都獲利（+17,095 / +5,451 / +8,397），但那是
3 個樣本。

**2. 曝險從 20% 升到 48%。** 資金佔用時間翻倍以上。CCF 與 QFF 共用同一個富邦帳戶，
所以這會直接提高保證金紅線的觸發機率 —— 兩個 pair 同時持倉的時間變長了。

**3. 16 筆交易、0 虧損不可信。** 這個「完美紀錄」比先前的 11 勝 1 敗**更**可疑，
它只說明這 29 個 session 裡沒有出現過不利走勢，不代表策略不會虧。

---

## 建議

1. **CCF/UMC 採用 `none`**，QFF/TSM 保持 `flat`
2. 實作上，週末規則必須是 **per-pair 設定**，不能是全域常數
3. **樣本累積後重驗** —— 每日跑 `accumulate_taifex_1m.py --product CCF`，
   三個月後樣本會變三倍，屆時特別看跨週末那幾筆是否仍然無害

---

## 相關檔案

| 檔案 | 用途 |
|---|---|
| `scripts/calculate_ccf_umc_spread_1m.py` | 1m spread，含 `--weekend-policy` |
| `scripts/calculate_spread_zscore_1m.py` | 滾動 z-score（沿用既有腳本）|
| `scripts/backtest_pair_strategy_1m.py` | 回測（沿用既有腳本）|
| `scripts/accumulate_taifex_1m.py` | CCF 1m 累積（30 天滾動視窗，需每日執行）|
| `data/processed/ccf1_1m_cumulative.csv` | CCF 1m |
| `data/processed/umc_1m_cumulative.csv` | UMC 1m（IBKR）|
