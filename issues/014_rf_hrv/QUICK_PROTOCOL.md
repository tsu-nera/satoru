


# 共鳴周波数測定プロトコル（簡易版）

**所要時間**: 約25分
**測定回数**: 6回（各3分間）

---

## 測定順序（ランダム化済み）

**この順序で測定してください**

| Trial | Rate (bpm) | Inhale (s) | Exhale (s) |
|------:|-----------:|-----------:|-----------:|
| 1 | 5.0 | 4.8 | 7.2 |
| 2 | 3.5 | 6.9 | 10.3 |
| 3 | 6.0 | 4.0 | 6.0 |
| 4 | 4.5 | 5.3 | 8.0 |
| 5 | 5.5 | 4.4 | 6.5 |
| 6 | 4.0 | 6.0 | 9.0 |

---

## 測定手順

### 準備
- 静かな環境
- 測定前5分間安静にする
- Elite HRVアプリとセンサー準備

### 各トライアル
1. Elite HRVで該当する呼吸レートを設定
2. **3分間**呼吸ガイドに従う
3. データを保存（`YYYY-MM-DD HH-MM-SS.txt`）
4. **1分間**休憩
5. 次のトライアルへ

### データ保存
- 保存先: `issues/014_rf_hrv/data/`
- ファイル名とトライアル番号をメモ

---

## 分析

測定完了後、以下を実行：

```bash
# rf_hrv_analysis.pyのtrials変数を以下に更新

trials = [
    {'trial': 1, git 'rate': 5.0, 'file': 'YYYY-MM-DD HH-MM-SS.txt',
     'inhale': 4.8, 'exhale': 7.2},
    {'trial': 2, 'rate': 3.5, 'file': 'YYYY-MM-DD HH-MM-SS.txt',
     'inhale': 6.9, 'exhale': 10.3},
    {'trial': 3, 'rate': 6.0, 'file': 'YYYY-MM-DD HH-MM-SS.txt',
     'inhale': 4.0, 'exhale': 6.0},
    {'trial': 4, 'rate': 4.5, 'file': 'YYYY-MM-DD HH-MM-SS.txt',
     'inhale': 5.3, 'exhale': 8.0},
    {'trial': 5, 'rate': 5.5, 'file': 'YYYY-MM-DD HH-MM-SS.txt',
     'inhale': 4.4, 'exhale': 6.5},
    {'trial': 6, 'rate': 4.0, 'file': 'YYYY-MM-DD HH-MM-SS.txt',
     'inhale': 6.0, 'exhale': 9.0},
]

# 分析実行
python issues/014_rf_hrv/rf_hrv_analysis.py
```

---

## 注意事項

- **3分測定**: 通常より短いため精度は粗い（スクリーニング用）
- **ランダム順序**: 必ず上記の順序で測定
- 低レート（3.5-4.5 bpm）はゆっくり、無理しない
