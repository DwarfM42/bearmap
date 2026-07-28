# higuma — 北海道ヒグマ出没情報の時空間分析

北海道ヒグマの出没情報から、ヒグマの相対的な出没・空間利用パターンと人間側の観測・通報努力をどこまで分離できるか、
分離した上で秋季食物資源の豊凶・気象と対応づけられるかを検証するプロジェクト。

詳細な研究計画・フェーズ定義は [PROJECT_SPEC.md](PROJECT_SPEC.md) を参照。

## 現在のフェーズ

- [x] P0: 先行研究サーベイ → [SURVEY.md](SURVEY.md)（判定: GO）
- [x] P1: データ取得可能性の確認 → [DATA_SOURCES.md](DATA_SOURCES.md)（判定: 継続可）
- [x] P2: データ取得・パネル構築（札幌市のみ） → [P2_GATE.md](P2_GATE.md)（P2d判定: 札幌市高解像度／全道低解像度を別系列化、全道統合は保留）
- [x] P3: 記述統計・観測過程診断（札幌市のみ） → [P3_DESCRIPTIVE.md](P3_DESCRIPTIVE.md)（H1判定保留、H4部分的支持、H6曜日差は支持・原因は判定不能）
- [x] P4a: 共変量取得・断層診断・分析仕様固定 → [reports/P4A_SUMMARY.md](reports/P4A_SUMMARY.md) / [P4_MODEL_PLAN.md](P4_MODEL_PLAN.md)
- [x] P4b: ベースラインモデル・残差診断 → [P4_RESULTS.md](P4_RESULTS.md)（H1支持[識別問題あり]、H2支持、H6不支持[1条件1種別のみ限定的]）
- [ ] P5: 食物資源・気象との突合（人間の承認待ち）
- [ ] P6: レポート
