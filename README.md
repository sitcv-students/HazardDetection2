# HazardDetection2 — エゴセントリック映像による危険検知のサーベイ

一人称（egocentric）映像を使う歩行支援システムで、危険を検知する研究のサーベイ集。
物体検出ではなく、「これから危険が起きるか」「意図した経路を安全に進み続けられるか」を問う研究を対象にしている。

## ファイル構成

| ファイル | 内容 |
|---|---|
| [survey_hazard_detection.md](survey_hazard_detection.md) | 入門。主要論文の一覧、推奨する読む順番、研究の流れ（英語・日本語） |
| [survey_scene_centric.md](survey_scene_centric.md) | **シーン中心**: 映像全体について、事故がいつ起きるかを予測する（accident anticipation、2016–2026） |
| [survey_risk_region.md](survey_risk_region.md) | **危険領域**: 危険が**どこに**あるか（危険物体の特定、リスクヒートマップ、注意予測、VLM による根拠付き推論） |
| [survey_affordance_centric.md](survey_affordance_centric.md) | **アフォーダンス中心**: 移動できる空間が失われる・変化することを危険とみなす（traversability、freespace/occupancy forecasting、将来経路予測） |
| [oss.md](oss.md) | 関連する OSS・プロジェクトページへのリンク |

各サーベイは同じ章立てになっている: 定義とスコープ → 問題設定と評価指標 → データセット → 手法の系譜 → 主要論文の要約 → 歩行者視点への適用 → オープン問題 → 参考文献。

## 三つのパラダイム

| 問い | パラダイム | サーベイ |
|---|---|---|
| 近いうちに危険が起きるか？ いつ起きるか？ | シーン中心 | `survey_scene_centric.md` |
| 危険はどこにあるか？ | 危険領域 | `survey_risk_region.md` |
| 意図した経路を安全に進み続けられるか？ | アフォーダンス中心 | `survey_affordance_centric.md` |

## 注記

- `survey_scene_centric.md` と `survey_risk_region.md` は、引用文献の書誌情報を arXiv・CVF・Crossref などで確認している（各冒頭の「検証方針」を参照）。
- **注意**: `survey_hazard_detection.md` の推奨論文（Ego-Alert、EgoPATH、"Learning to Anticipate Traffic Accidents from Egocentric Video"）は、実在を確認できていない。検証結果は `survey_risk_region.md` §8 と `survey_scene_centric.md` 付録を参照。
- 論文 PDF（`pdfs/`）は `.gitignore` によりリポジトリに含めていない。
