# 危険領域の予測（Risk Region Prediction / Risk Localization）サーベイ — 2026年11月時点

> 対象: 「危険が**どこに**あるか」を出力する研究（危険物体の特定、リスクヒートマップ、注意予測のリスク代理利用、危険領域セグメンテーション、物体レベルの事故予測、VLM/LLM による根拠付き危険推論）。
> 想定読者: エゴセントリック映像による歩行支援システムの危険検出を研究するメンバー。
> 検証方針: 本文で引用した文献はすべて arXiv の abs ページ（タイトル・第一著者・投稿日・Comments/Journal-ref 欄）、CVF/IEEE/ACM ページ、またはローカル PDF の書誌表記で存在を確認した。採録先は確認できた範囲でのみ記載し、確認できないものは「arXiv」とした。ベンチマーク数値は原論文アブストラクトで確認できたものだけを載せている（データセット規模など）。手法間の性能比較数値は、評価条件の違いが大きいため載せていない。

---

## 1. 定義とスコープ

### 1.1 「危険領域の予測」とは

入力（主にエゴ視点の映像）から、**近い将来に自己（車両・歩行者）の安全を脅かす、またはその行動を変えさせる空間的な要素**の位置を推定するタスク群である。出力の形によって次のように分かれる。

| 出力の形 | 例 |
|---|---|
| 物体 ID・BBox（危険物体 / 重要物体） | Li et al. IROS 2020、RiskBench、DRAMA |
| 物体ごとのスコア・ランキング | Rank2Tell、HOIST、TOI |
| 画素レベルのヒートマップ（注意 / リスク） | DR(eye)VE、BDD-A、DADA-2000、W³DA |
| 画素レベルのマスク（障害物・異常） | Lost and Found、SegmentMeIfYouCan |
| BEV 上のリスク場 | Potential Field ROI、MC-Risk |
| 自然言語＋グラウンディング（BBox） | DRAMA キャプション、UniDrive、NuRisk |

### 1.2 「危険」の定義の流派

1. **衝突ベース**: 将来衝突に関与する物体が危険（事故予測系。DSA/DAD、UString、DoTA、MM-AU など）。
2. **ドライバー中心（行動変化ベース）**: ドライバーの行動（停止・減速・回避）を引き起こした物体が危険（Li et al. 2020 "Who Make Drivers Stop?"、DROID、RiskBench の BCP 系）。「衝突は危険の一つの源にすぎない」とする立場。
3. **重要度ベース**: 自車の制御判断に影響する物体が重要（Gao et al. 2019、Li J. et al. 2022、HOIST、Rank2Tell）。
4. **人間注意ベース**: ドライバーが見る場所が危険・重要（DR(eye)VE、BDD-A、DADA）。事故映像を見た観察者の注視を集めた BDD-A/DADA は「危険領域」の代理ラベルとして使われることが多い。
5. **言語・推論ベース**: 人や VLM が「何がなぜ危険か」を言語化し、その対象を BBox で根拠付ける（DRAMA、DHPR、NuRisk、PotentialRiskQA）。

### 1.3 スコープ外（対比として簡単に触れるだけ）

フレーム・シーン単位で「事故が起きるか」の確率だけを出す**シーンレベル事故予測**はスコープ外である。例: TOP（Zhao et al. NeurIPS 2025）、RiskProp（Zou et al. CVPR 2026）、Progressive Risk Estimation（Hicsonmez et al. NeurIPS 2026）、Predict and Resist（Liu et al. AAAI 2026）、BADAS / BADAS-2.0、Nexar Collision Prediction。これらは時間方向の「いつ」は高精度化が進む一方、「どこ」は出力しないか、注意マップを副産物として可視化する程度にとどまる。ローカル PDF の RiskProp は関連研究で「物体検出器や注意マップで空間的根拠付けを強める流れ」に触れているが、自身は空間的出力を評価していない。

---

## 2. 問題設定の分類と評価指標

### 2.1 タスク分類

| 分類 | 入力 | 出力 | 教師 | 代表 |
|---|---|---|---|---|
| (A) 危険物体の特定（Risky Object ID / ROI） | 映像＋検出/追跡 BBox | 危険物体 ID（各時刻） | 行動ラベルのみ（弱教師）〜物体ラベル | Li 2020、DROID、RAIST、PF-ROI、RiskBench、RAID、Conformal Risk Tube、RiskWorld |
| (A') 重要物体推定（Object Importance） | 映像＋BBox（＋目的・経路） | 物体ごとの重要度／二値 | 人手の重要度ラベル | Gao 2019、Zhang 2020、Li J. 2022、HOIST、TOI、IDD-X、Rank2Tell、Ling 2026 |
| (B) リスクヒートマップ / リスク場 | 画像・映像 or BEV | 画素 / グリッド単位のリスク | 注視・衝突・軌道予測 | Zeng 2017（risky region）、PF-ROI、MC-Risk、Unified Risk Map |
| (C) 注意をリスクとして使う | 映像 | 注視ヒートマップ | 視線計測 | DR(eye)VE、BDD-A、DADA-2000、MEDIRL、SAGE-Net、W³DA |
| (D) 危険領域のセグメンテーション | 画像（ステレオ） | 障害物 / 異常マスク | 画素ラベル | Lost and Found、Fishyscapes、SMIYC、PEDESTRIAN（検出）、GuideTWSI |
| (E) BBox レベルの事故予測（空間的説明付き） | 映像＋物体特徴 | 事故確率＋関与物体 / 注意 | 事故時刻＋関与物体 BBox | DSTA、DRIVE、AM-Net（ROL）、WWW（AOLA）、DoTA（STAUC）、MM-AU、CARA |
| (F) 根拠付き VLM / LLM 危険推論 | 画像・映像＋プロンプト | 言語説明＋BBox / 物体 ID | QA・キャプション・BBox | DRAMA、DRAMA-X、DHPR、NuRisk、INSIGHT、COOOL、UniDrive、What's Hidden Matters |

### 2.2 評価指標

- **物体レベル（A, A', E）**: フレーム単位の Precision / Recall / F1、物体単位の AP / mAP、ランキング指標。RiskBench は空間的定位・時間的予測・意思決定支援の 3 観点で評価し、意思決定への影響を見る **PIC（Progressive Increasing Cost）** を導入した。PF-ROI は「空間的・時間的一貫性」も評価している。WWW（Liao et al. MM 2024）は事故関与物体の定位精度 **AOLA** を、DoTA（Yao et al.）は空間・時間を合わせた **STAUC** を提案した。
- **ヒートマップ（B, C）**: 顕著性指標 KLD、CC、NSS、SIM、AUC-J/AUC-B。注意予測を危険領域の代理として使う場合も、通常はこれらで評価する。
- **セグメンテーション（D）**: 画素 AUPRC、FPR95、成分単位の sIoU / PPV / F1（SMIYC）。
- **時間方向（E）**: AP、mTTA（平均事故予測時間）、TTA@R80。ただし「どこ」の評価ではない。
- **言語＋定位（F）**: BBox の IoU / mIoU（DRAMA）、BLEU / METEOR / ROUGE / CIDEr、多肢選択 QA の正答率、GPT-as-judge。COOOL はフレームごとの危険一致率（正しく予測した危険数 / max(既知危険数, 予測数)）で採点する。

**課題**: 指標がタスクごとにばらばらで、特に「早く」かつ「正しい物体を」当てる性能を同時に測る指標は STAUC、AOLA、PIC、Conformal Risk Tube の被覆保証などに分かれていて、統一されていない。

---

## 3. データセット

| データセット | 年 / 会議 | 視点・ドメイン | 空間ラベル | 規模（論文記載） | URL |
|---|---|---|---|---|---|
| HDD | CVPR 2018 | 車載・実走行（米国） | ドライバー行動＋原因（stimulus）。Li 2020 / DROID が ROI 評価に使用 | 104 時間 | https://arxiv.org/abs/1811.02307 |
| DR(eye)VE | TPAMI | 車載・実走行 | 実ドライバーの注視マップ | 74 シーン（W³DA 表より） | https://arxiv.org/abs/1705.03854 |
| BDD-A | ACCV 2018 | 車載・危機的場面 | 観察者の注視マップ（実験室） | 1,435（W³DA 表より） | https://arxiv.org/abs/1711.06406 |
| DADA-2000 | ITSC 2019 投稿 | 車載・事故映像 | 注視マップ＋事故区間 | 2,000 本 | https://arxiv.org/abs/1904.12634 |
| EyeCar | ICCV 2021（MEDIRL） | 事故が起きやすい場面 | 注視 | — | https://arxiv.org/abs/1912.07773 |
| MAAD | ICCVW 2021 | 運転映像の観察 | 注視＋attended awareness | 24.5 時間、23 名 | https://arxiv.org/abs/2110.08610 |
| CoCAtt | arXiv 2021 | 運転（手動 / 自動） | 注視＋注意散漫 / 意図ラベル | — | https://arxiv.org/abs/2111.10014 |
| W³DA | ICCV 2025 | 通常・危機・事故 | 注視＋「何を」「なぜ」 | 3,548 シーン、69,980 キーフレーム | https://arxiv.org/abs/2506.23088 |
| Epic Fail (EF) | CVPR 2017 | 事故のバイラル動画 | 危険領域 | 3,000 本 | https://arxiv.org/abs/1705.06560 |
| A3D | IROS 2019 | ダッシュカム事故 | 時間 | — | https://arxiv.org/abs/1903.00618 |
| DoTA | arXiv 2020 | ダッシュカム異常 | 時間・空間（異常物体 BBox）・カテゴリ | 4,677 本 | https://arxiv.org/abs/2004.03044 |
| ROL | T-IV（AM-Net） | ダッシュカム事故 | 危険物体 BBox | — | https://arxiv.org/abs/2209.07922 |
| MM-AU | CVPR 2024 | エゴ視点の事故 | 物体 BBox 223 万個＋事故理由テキスト | 11,727 本 | https://arxiv.org/abs/2403.00436 |
| CAP | arXiv 2022 | 事故（DADA 拡張） | 注視＋テキスト | — | https://arxiv.org/abs/2212.09381 |
| DRAMA | WACV 2023 | 車載・東京 | 危険物体 BBox＋キャプション＋QA | 17,785 シナリオ | https://arxiv.org/abs/2209.10767 |
| DRAMA-X | ICCVW 2025 | DRAMA 由来 | BBox＋意図＋リスク＋行動提案 | 5,686 フレーム | https://arxiv.org/abs/2506.17590 |
| Rank2Tell | WACV 2024 | 車載・交差点 | 重要度ランク＋理由 | — | https://arxiv.org/abs/2309.06597 |
| HOIST | RA-L 2024 | シミュレーション | 物体重要度（人手） | — | https://arxiv.org/abs/2312.02467 |
| TOI | NeurIPS 2024 | 車載 | 物体重要度 | — | https://arxiv.org/abs/2411.17152 |
| IDD-X | ICRA 2024 | インド・密で非構造的な交通、多視点 | 重要物体＋説明 | — | https://arxiv.org/abs/2404.08561 |
| RiskBench | ICRA 2024 | CARLA | 時空間リスク注釈 | 6,916 シナリオ | https://arxiv.org/abs/2312.01659 |
| RAID | IV 2026 | 実走行 | 意図・リスク状況・ドライバー反応・歩行者の注意 | 4,691 クリップ | https://arxiv.org/abs/2603.05926 |
| DHPR | T-IV 2024 | 静止画（ダッシュカム） | 危険仮説の文章＋視覚エンティティ | 15K 画像 | https://arxiv.org/abs/2310.04671 |
| COOOL / 2COOOL | WACVW 2025 / ICCVW 2025 | ダッシュカム、ラベル外の危険 | 危険物体＋キャプション（2COOOL はレポート） | 約 200 本（COOOL） | https://arxiv.org/abs/2508.21080 |
| NuRisk | ICRA 2026 | nuScenes / Waymo / CommonRoad | エージェント単位のリスク VQA | 2.9K シナリオ、110 万サンプル | https://arxiv.org/abs/2509.25944 |
| CODA / CODA-LM | ECCV 2022 / WACV 2025 | コーナーケース | BBox / LVLM 評価 | — | https://arxiv.org/abs/2203.07724 / https://arxiv.org/abs/2404.10595 |
| Lost and Found | IROS 2016 | ステレオ・小さな路上障害物 | 画素マスク | 2,000 フレーム超 | https://arxiv.org/abs/1609.04653 |
| SegmentMeIfYouCan | NeurIPS 2021 D&B | 路上の異常 / 障害物 | 画素マスク | — | https://arxiv.org/abs/2104.14812 |
| ACCIDENT | arXiv 2026 | CCTV（第三者視点） | 事故の時間・空間位置・種別 | 実 2,027＋合成 2,211 | https://arxiv.org/abs/2604.09819 |
| **歩行者 / BLV 系** | | | | | |
| SANPO | arXiv 2023（WACV 投稿） | 人の一人称・ステレオ | 密なパノプティックセグメンテーション＋深度 | 701 本（ステレオ） | https://arxiv.org/abs/2309.12172 |
| PEDESTRIAN | arXiv 2025 | スマホ一人称・歩道 | 障害物 29 クラス | 340 本 | https://arxiv.org/abs/2512.19190 |
| HEADS-UP | arXiv 2024 | 頭部装着・BLV 想定 | 歩行者軌道（衝突リスク） | — | https://arxiv.org/abs/2409.20324 |
| EgoBlind | NeurIPS 2025 D&B | 盲人が撮影した一人称映像 | VQA | 1,392 本 | https://arxiv.org/abs/2503.08221 |
| GuideDog | ACL 2026 | 歩行動画（46 か国） | 案内用の説明＋QA | 22K 画像（2K 人手検証） | https://arxiv.org/abs/2503.12844 |
| WalkVLM | arXiv 2024 | 歩行支援 | 映像と注釈の対 | 12,000 対 | https://arxiv.org/abs/2412.20903 |
| mmWalk | NeurIPS 2025 D&B | CARLA 歩行、多視点 | RGB / 深度 / セグメンテーション＋VQA | 62k フレーム | https://arxiv.org/abs/2510.11520 |
| GuideTWSI | arXiv 2026 | 点字ブロック | 画素マスク | — | https://arxiv.org/abs/2603.07060 |
| EgoSafe-Bench | arXiv 2026 | モバイル一人称 | 安全推論ベンチマーク | — | https://arxiv.org/abs/2607.26518 |
| EgoPathBench | arXiv 2026 | 一人称のウェイポイント選択 | 通行可能な候補点 | 34K 問 | https://arxiv.org/abs/2609.16610 |

（「—」は本調査で規模を確認していないもの）

---

## 4. 手法の系譜と比較

### 4.1 系譜（概略）

```
[2016-17] 事故予測 DSA/DAD (Chan, ACCV16) ── Zeng CVPR17: 事故予測＋危険領域定位（agent-centric）
              │
[2017-19] 注意予測: DR(eye)VE ─ BDD-A ─ DADA-2000 ─ SAGE-Net ─ MEDIRL
              │                         \
[2019-20] 重要物体: Gao ICRA19 ─ Zhang ICRA20(GCN)       \──> 注意を事故予測に利用（DRIVE, CAP, CAMERA）
              │
[2020-21] ドライバー中心 ROI: Li IROS20 (因果介入: 物体除去→Stop/Go) ─ DROID ─ RAIST
              │
[2022-24] 半教師あり重要物体 (Li J. ICRA22) ─ HOIST(反事実) ─ TOI(NeurIPS24) ─ IDD-X ─ Rank2Tell
          ベンチマーク化: RiskBench (ICRA24) ─ DoTA/STAUC ─ ROL/AM-Net ─ WWW/AOLA ─ MM-AU
              │
[2022-25] 言語＋定位: DRAMA (WACV23) ─ DHPR ─ DriveLM/DriveVLM ─ CODA-LM ─ COOOL ─ DRAMA-X ─ INSIGHT
              │
[2025-26] BEV / 場: PF-ROI (ICRA25) ─ MC-Risk ─ Unified Risk Map
          不確実性: Conformal Risk Tube (ICRA26)
          弱教師＋歩行者の注意: RAID (IV26)
          世界モデル: RiskWorld (2026)
          VLM 推論: NuRisk (ICRA26) ─ PotentialRiskQA ─ UniDrive ─ C-CoT ─ What's Hidden Matters (IROS26)
          説明可能な事故予測: CARA (MM26)
          歩行者 / BLV: SANPO ─ HEADS-UP ─ WalkVLM ─ EgoBlind ─ GuideDog ─ mmWalk ─ PEDESTRIAN ─ ForeSightGuide
```

### 4.2 手法ファミリーの比較

| ファミリー | 代表 | 「危険」の根拠 | 必要な教師 | 長所 | 短所 |
|---|---|---|---|---|---|
| 因果介入型 ROI（物体を除去して行動予測の変化を見る） | Li 2020、DROID、RAIST、RAID | 行動変化 | 行動ラベル（物体ラベル不要） | 弱教師で済み、解釈しやすい | 物体の数だけ推論が要る。画像空間でのマスクが不自然。行動変化を伴わない潜在的危険は捉えられない |
| BEV 場型 | PF-ROI、MC-Risk、Unified Risk Map | 斥力・引力のポテンシャル / 軌道予測 | BEV セグメンテーション、HD マップ | 空間的・時間的一貫性が高い | BEV 化や地図に依存し、歩行者視点への移植が難しい |
| 重要度分類・ランキング型 | Gao 2019、Zhang 2020、Li J. 2022、TOI、Ling 2026、Rank2Tell | 人手の重要度 | 物体単位の人手ラベル | 直接教師で精度が高い | ラベル付けが主観的で高コスト |
| 反事実シミュレーション型 | HOIST、Biswas 2024 | 物体の動きを変えたときの自車への影響 | シミュレータまたは軌道予測器 | 理論的に明快 | 実映像への適用が難しい |
| 不確実性・保証型 | Conformal Risk Tube | 被覆保証付きのリスクスコア | 校正用データ | 閾値設定や時間的安定性の問題に対処できる | 新しく、検証例が少ない |
| 世界モデル型 | RiskWorld | 自車と物体の関係を潜在空間でロールアウト | 物体履歴＋事故 / リスクラベル | 危険になる前に検知できる | 計算コスト、評価がまだ少ない |
| 注意予測型 | DR(eye)VE、BDD-A、DADA、MEDIRL、W³DA | 人の注視 | 視線計測 | 密な出力、人間の認知に近い | 注視は危険と一致しない（見ていない危険こそ重要）。トップダウン要因の欠落（Kotseruba IV 2024） |
| 空間注意付き事故予測 | Zeng 2017、DSTA、DRIVE、AM-Net、WWW、CARA、CAMERA | 衝突への関与 | 事故時刻＋（関与物体） | 「いつ」と「どこ」を同時に出せる | 注意＝説明とは限らない（Karim TRR の XAI 検証） |
| VLM / LLM 根拠付き推論 | DRAMA、DHPR、INSIGHT、Shriram 2025、NuRisk、UniDrive、C-CoT、Chahe 2026 | 言語的推論 | QA / キャプション（＋BBox） | ラベル外の危険（COOOL）に対応でき、説明が付く | 小さい・遠い対象の定位が弱い（UniDrive が指摘）、遅延が大きい、出力が不安定（Richards 2026） |
| 障害物 / 異常セグメンテーション | Lost and Found、SMIYC、Fishyscapes | 「既知クラス外」かつ走行面上 | 画素ラベル | 未知の障害物を拾える | 動的な危険や意図を扱えない |

---

## 5. 主要論文の要約

### 5.1 危険物体の特定・重要物体

- **Li, C. et al., "Who Make Drivers Stop? Towards Driver-centric Risk Assessment: Risk Object Identification via Causal Inference", IROS 2020.** 「ドライバーの行動に影響する物体が危険」とドライバー中心に定義し、Risk Object Identification というタスクを提案した。Stop/Go を予測する物体レベル操作可能な運転モデルを学習し、物体を 1 つずつ除去したとき Stop→Go に変われば、その物体を原因（危険物体）とする 2 段階の因果推論を行う。HDD で評価し、強いベースラインを 7.5% 上回る。
- **Li, C. et al., "DROID", arXiv 2021（TPAMI 投稿版）.** 上記を拡張したもの。主観的なリスク評価を「行動変化の予測＋その原因の特定」として定式化し、介入をデータ拡張にも使う。
- **Suman, V. et al., "RAIST", IROS 2023.** 時空間グラフ畳み込みで物体間の相互作用をモデル化し、同じ 2 段階の Stop/Go 推論を行う。
- **Kung, C.-H. et al., "RiskBench", ICRA 2024.** CARLA 上の 6,916 シナリオに時空間リスク注釈を付けた、シナリオベースのベンチマーク。ルールベース、軌道予測＋衝突判定、衝突予測、行動予測ベース（BCP）の 10 手法を「定位・予測・意思決定支援」で比較し、PIC 指標を導入した。既存手法にはまだ大きな改善余地があると結論づけている。
- **Pao, P.-Y. et al., "Potential Field as Scene Affordance for Behavior Change-Based Visual Risk Object Identification", ICRA 2025.** 画像空間での因果推論は空間精度と効率が低いと指摘する。BEV セマンティックセグメンテーションから、道路構造や交通参加者による斥力と目的地による引力のポテンシャル場を作り、BEV 上で介入する。RiskBench で空間的一貫性を 20.3%、時間的一貫性を 11.6% 改善した（アブストラクト記載）。
- **Fu, K.-Y. et al., "Uncertainty-Aware Vision-based Risk Object Identification via Conformal Risk Tube Prediction", ICRA 2026.** 決定論的な閾値では曖昧な場面で検出が早すぎたり遅すぎたりし、時間的にも不安定になると指摘する。時空間の「リスクチューブ」をコンフォーマル予測で出力し、真の危険を被覆することを保証する。複数のリスクが連鎖する場面のデータセットと指標も新たに作った。誤った制動警報が減ったと報告している。
- **Agarwal, N. et al., "Towards Driver Behavior Understanding: Weakly-Supervised Risk Perception in Driving Scenes" (RAID), IV 2026.** 4,691 クリップに意図した操作、道路形状、リスク状況、ドライバー反応、**歩行者の注意（自車を見ているか）** をラベル付けした。意図と反応の関係から危険源を弱教師で特定する。
- **Li, J. et al., "RiskWorld: Object-Centric Latent World Modeling for Autonomous Driving Risk Identification", arXiv 2026.** 既存手法はシーンレベルの事故予測か、自車の行動からの間接推定か、軌道予測後の幾何判定であり、予測した自車と物体の関係を危険源の定位に使っていないと指摘する。RSSM 型の潜在ダイナミクスで物体単位のリスクをロールアウトする。
- **Gao, M. et al., "Goal-oriented Object Importance Estimation in On-road Driving Videos", ICRA 2019.** 交差点データに重要物体の人手ラベルを付け、自車の目的（計画経路）を条件として与えた。左折・右折の場面で特に効果が大きい。
- **Zhang, Z. et al., "Interaction Graphs for Object Importance Estimation", ICRA 2020.** 物体間の相互作用（例: 自車の前に飛び出す歩行者がいると、他の物体の重要度が下がる）をグラフ畳み込みで扱う。
- **Li, J. et al., "Important Object Identification with Semi-Supervised Learning for Autonomous Driving", ICRA 2022.** 重要物体の判定を二値分類として直接教師あり学習する。関係推論、半教師あり学習、補助タスクとしての自車行動予測を組み合わせ、H3D で評価した。
- **Gupta, P. et al., "Object Importance Estimation using Counterfactual Reasoning for Intelligent Driving" (HOIST), RA-L 2024.** シミュレーション上の重要度データセット HOIST を作り、物体の動きを反事実的に変えたときの自車運転への影響を重要度とした。
- **Nan, Z. et al., "On-Road Object Importance Estimation: A New Dataset and A Model with Multi-Fold Top-Down Guidance" (TOI), NeurIPS 2024.** ボトムアップ特徴だけ、または単一のトップダウン手がかりだけを使う従来法に対し、ドライバー意図や意味的文脈など複数のトップダウン手がかりを統合した。
- **Parikh, C. et al., "IDD-X", ICRA 2024.** インドの密で非構造的な交通を多視点で撮影し、自車から見た重要物体の定位と説明を付けた。
- **Ling, S. et al., "A Two-Stage Framework for Ego-Centric Key Object Identification via Object State Prediction", ICRA 2026.** 仮想の自車表現と物体状態予測器で、自車に対する各物体の状態を推定してから時空間推論で重要物体を選ぶ。
- **Sachdeva, E. et al., "Rank2Tell", WACV 2024.** 重要度のランク付けと理由の文章生成を同時に扱う、マルチモーダルなエゴ視点データセット。
- **Biswas, A. et al., "Modeling Drivers' Risk Perception via Attention to Improve Driving Assistance", arXiv 2024.** 前方衝突警報の実配備映像を使い、ドライバーの物体への注視を反事実的に軌道予測へ反映させて、ドライバーが知覚しているリスクを推定する。不要な警報を減らすのが狙い。

### 5.2 注意予測を危険領域として使う

- **Palazzi, A. et al., "Predicting the Driver's Focus of Attention: the DR(eye)VE Project", TPAMI.** 実ドライバーの注視データと、マルチブランチの注視予測モデルを提案した。
- **Xia, Y. et al., "Predicting Driver Attention in Critical Situations" (BDD-A), ACCV 2018.** 危機的場面を実験室で観察者に見せて注視を収集した。危険領域の代理ラベルとして広く使われる。
- **Fang, J. et al., "DADA-2000"（ITSC 2019 投稿）／ "DADA: Driver Attention Prediction in Driving Accident Scenarios", IEEE T-ITS.** 事故映像の注視マップと事故注釈を同時に提供する。SCAFNet はセマンティック画像と RGB を融合する。
- **Baee, S. et al., "MEDIRL", ICCV 2021.** 最大エントロピー逆強化学習で、事故が起きやすい場面の注視を予測する。EyeCar データセットを公開した。
- **Pal, A. et al., "Looking at the right stuff — Guided semantic-gaze for autonomous driving" (SAGE-Net), CVPR 2020.** 深度、自車速度、歩行者の横断意図を考慮して注視予測を補正する。
- **Zhu, P. et al., "Unsupervised Self-Driving Attention Prediction via Uncertainty Mining and Knowledge Embedding", ICCV 2023.** 自然画像で学習した顕著性モデルから疑似ラベルを作り、運転知識で補正する教師なし手法。
- **Kotseruba, I. & Tsotsos, J., IV 2024 の 2 本（"Understanding and Modeling the Effects of Task and Context on Drivers' Gaze Allocation"、"Data Limitations for Modeling Top-Down Effects on Drivers' Attention"）.** 既存の注視データセットはタスク（意図）や文脈といったトップダウン要因を欠いており、「注視＝危険」とみなすことには限界があると指摘している。
- **Zhou, Y. et al., "Where, What, Why: Towards Explainable Driver Attention Prediction" (W³DA), ICCV 2025.** 注視の位置（where）に加え、注視対象の意味（what）と認知的理由（why）を予測する。3,548 シーン、69,980 キーフレーム。
- **Gopinath, D. et al., "MAAD"（ICCVW 2021）／ Shen, Y. et al., "CoCAtt"（arXiv 2021）.** 「見た」ことと「気づいている」ことの区別、注意散漫や意図の条件付けを扱う。

### 5.3 物体レベルの事故予測（空間的説明付き）

- **Zeng, K.-H. et al., "Agent-Centric Risk Assessment: Accident Anticipation and Risky Region Localization", CVPR 2017.** エージェント中心の RNN で、事故予測（時間）と危険領域の定位（空間）を同時に評価した最初期の研究。Epic Fail データセット（3,000 本）を作った。
- **Yao, Y. et al., "Unsupervised Traffic Accident Detection in First-Person Videos", IROS 2019.** 交通参加者の将来位置予測の誤差と一貫性から異常を検出する。A3D データセットを公開した。
- **Yao, Y. et al., "When, Where, and What? A New Dataset for Anomaly Detection in Driving Videos" (DoTA), arXiv 2020.** 4,677 本に時間・空間・カテゴリの注釈を付け、STAUC 指標を提案した。
- **Bao, W. et al., "UString", ACM MM 2020 ／ "DRIVE: Deep Reinforced Accident Anticipation with Visual Explanation", ICCV 2021.** DRIVE は注視（fixation）報酬を使う強化学習で、視覚的な説明を同時に出す。
- **Karim, M. M. et al., "DSTA", arXiv 2021 ／ "Towards explainable AI for early anticipation of traffic accidents", TRR ／ AM-Net "An Attention-guided Multistream Feature Fusion Network for Localization of Risky Objects in Driving Videos"（IEEE T-IV）.** DSTA は時空間注意を使い、TRR の論文は Grad-CAM の顕著性マップを人の注視と比較する（ローカル PDF で確認）。AM-Net は物体ごとの危険度スコアを出力し、ROL データセットを公開した。
- **Liao, H. et al., "When, Where, and What? A Novel Benchmark for Accident Anticipation and Localization with Large Language Models", ACM MM 2024（ローカル PDF で確認）.** LLM を組み込み、事故の時間（when）・場所（where）・種類（what）を同時に推論する。事故関与物体を BBox で示し、AOLA を導入した。
- **Fang, J. et al., "Abductive Ego-View Accident Video Understanding for Safe Driving Perception" (MM-AU), CVPR 2024.** 11,727 本、223 万 BBox、58,650 組の事故理由テキスト。物体中心の動画拡散で事故の因果を扱う。
- **Fang, J. et al., "Cognitive Accident Prediction in Driving Scenes: A Multimodality Benchmark" (CAP), arXiv 2022 ／ Li, L.-L. et al., "Cognitive Traffic Accident Anticipation", IEEE ITS Magazine 2024（DOI 10.1109/MITS.2024.3378460、ローカル PDF で確認）.** DADA-2000 に文章説明を付け、注視とテキストを事故予測に組み込む。
- **Zhang, J. et al., "Eyes on the Road, Mind Beyond Vision" (CAMERA), ACM MM 2025.** ダッシュカム映像、テキスト、ドライバー注意マップを融合し、動的な閾値を使う。
- **Tao, Z. et al., "CARA: Concept-Aware Risk Attention for Interpretable Collision Anticipation", ACM MM 2026.** 事故の記述からリスク概念を抽出し、フレームと対応付けて、内在的に解釈可能な時空間リスク注意を作る。

### 5.4 VLM / LLM による根拠付き危険推論

- **Malla, S. et al., "DRAMA: Joint Risk Localization and Captioning in Driving", WACV 2023.** 東京で収集した 17,785 シナリオで、危険物体の BBox とその説明文を同時に予測する。根拠付き危険推論の基準となるデータセット。
- **Godbole, M. et al., "DRAMA-X", ICCVW 2025.** DRAMA に交通弱者（VRU）の意図、リスク、行動提案を自動注釈で加えた。訓練不要の SGG-Intent（シーングラフ→意図→リスク→行動）も提案した。
- **Charoenpitaks, K. et al., "Exploring the Potential of Multi-Modal AI for Driving Hazard Prediction" (DHPR), IEEE T-IV 2024.** 1 枚の画像から、数秒後に起こりうる事故を視覚的アブダクションとして推論する。
- **Kim, J. et al., "Textual Explanations for Self-Driving Vehicles"（ECCV 2018）／ "Grounding Human-to-Vehicle Advice for Self-driving Vehicles"（CVPR 2019）.** 言語説明と視覚注意を対応付ける先駆的研究。
- **Sima, C. et al., "DriveLM"（ECCV 2024）／ Tian, X. et al., "DriveVLM"（arXiv 2024）／ Ding, X. et al., "HiLM-D"（IJCV）.** 運転全般の VLM で、重要物体の特定を推論の中間ステップに含む。HiLM-D は DRAMA 上で高解像度の定位を改善した。
- **Chen, K. et al., "Automated Evaluation of Large Vision-Language Models on Self-driving Corner Cases" (CODA-LM), WACV 2025.** コーナーケースでの LVLM 評価。
- **Shriram, S. et al., "Towards a Multi-Agent Vision-Language System for Zero-Shot Novel Hazardous Object Detection", arXiv 2025.** COOOL のノイズを除いたうえで言語記述を付け、VLM と LLM、OWL 系のゼロショット検出器、CLIP による照合を組み合わせて、ラベル外の危険物体を検出する。
- **AlShami, A. K. et al., "2COOOL", ICCVW 2025 ／ Yokoi, S. et al., "Hierarchical Reasoning with VLMs for Incident Reports from Dashcam Videos", arXiv 2025（2COOOL 2 位）.** ラベル外（out-of-label）の危険を扱う。COOOL は危険検出とキャプション、2COOOL はインシデントレポート生成。
- **Chen, D. et al., "INSIGHT", arXiv 2025.** VLM を教師ありで微調整し、注意機構と座標回帰で危険位置を直接出力する（BDD100K）。
- **Gao, Y. et al., "NuRisk", ICRA 2026.** 「既存の VLM は静止画で対象を根拠付けて定性的に判断するだけで、時空間推論が欠けている」と指摘し、エージェント単位のリスク VQA（110 万サンプル）を作った。
- **Liu, J. et al., "Predicting Risks Before They Happen: Vision-Language Reasoning for Autonomous Driving", arXiv 2025.** まだ見えていない危険の前兆を推論する PotentialRiskQA を提案した。
- **Gao, X. et al., "UniDrive", arXiv 2026.** MLLM には時間的推論と空間精度のトレードオフがあり、小さい・遠い・遮蔽された危険を見落とすと指摘する。多フレームの推論枝と高解像度の知覚枝を統合した。
- **Chahe, A. et al., "What's Hidden Matters: Identifying Planning-Critical Occluded Agents using VLMs", IROS 2026.** Planning KL-divergence で遮蔽されたエージェントを計画への影響度でランク付けし、VLM に推論させる（nuScenes）。
- **Tian, K. et al., "C-CoT", arXiv 2026.** シーン記述→重要物体の特定→リスク予測→反事実リスク推論→行動計画の順に推論させる。
- **Zeng, T. et al., "Are Vision LLMs Road-Ready?" (DVBench), arXiv 2025 ／ Richards, E. et al., "Task-Aligned Stability Analysis of VLMs for Autonomous Driving Hazard Detection", ICMLW 2026.** 安全上重要な場面での VLLM の理解力と出力の安定性を評価した。

### 5.5 危険領域のセグメンテーション（参考）

- **Pinggera, P. et al., "Lost and Found", IROS 2016**、**Blum, H. et al., "Fishyscapes", arXiv 2019**、**Chan, R. et al., "SegmentMeIfYouCan", NeurIPS 2021 D&B**、**Li, K. et al., "CODA", ECCV 2022.** 走行面上の未知の障害物を画素単位で検出する系統。歩道上の段差や落下物の検出に直接転用しやすい。

---

## 6. 歩行者 / エゴセントリック視点への適用と課題

### 6.1 既存の歩行支援系の研究

| 研究 | 空間的出力 | 時間的予測 | メモ |
|---|---|---|---|
| SANPO（Waghmare, arXiv 2023） | パノプティックセグメンテーション＋深度 | × | 歩行者視点の密なシーン理解の基盤 |
| PEDESTRIAN（Thoma, arXiv 2025） | 障害物検出（29 クラス） | × | スマホ一人称 |
| GuideTWSI（Hwang, arXiv 2026） | 点字ブロックのセグメンテーション | × | 安全上重要なランドマーク |
| HEADS-UP（Haghighi, arXiv 2024） | 歩行者軌道 → 衝突リスク | ○ | 頭部装着、半局所座標での予測 |
| Qiu et al.（RA-L 2022） | 装着者自身の軌道予測 | ○ | 視覚障害者のナビゲーション支援を想定 |
| Ego4D（Grauman, CVPR 2022） | 移動（locomotion）予測タスク | ○ | 将来の地面上軌道 |
| WalkVLM（Yuan, arXiv 2024） | 言語による注意喚起 | 一部 | 冗長な応答と推論遅延を課題視 |
| EgoBlind（Xiao, NeurIPS 2025 D&B） | VQA | × | MLLM の限界を分析 |
| GuideDog（Kim, ACL 2026） | 言語＋QA | × | BLV 向け案内基準に沿った注釈 |
| mmWalk（Ying, NeurIPS 2025 D&B） | 多視点・多モーダル＋VQA | × | シミュレーション |
| Zhang et al. "I Can See Forever!"（arXiv 2025） | 実時間 VideoLLM の評価 | ○ | 視覚障害者支援 |
| ForeSightGuide（Wang, arXiv 2026） | 予測的な案内 | ○ | 誤検出と冗長性の削減 |
| Valls et al.（arXiv 2026） | VQA ベースのイベントマップ | — | ロービジョン向け都市リスクナビゲーション |
| EgoSafe-Bench（Chen, arXiv 2026）／ EgoPathBench（Zhao, arXiv 2026） | 安全推論 / 通行可能な点の選択 | — | EgoPathBench では 9 つの VLM の最高スコアが 28.3 にとどまる |

### 6.2 車載の手法を歩行者視点へ移す際の論点

1. **「行動変化」による教師の置き換え**: 車載 ROI（Li 2020、DROID、RAID）は Stop/Go や制動を弱教師に使う。歩行者では**立ち止まる・減速する・進路を変える・頭を振る**（IMU や視覚オドメトリで取得可能）が同じ役割を果たしうる。物体除去による介入推論もそのまま使える可能性がある。
2. **自車経路条件付け**: Gao 2019 の「目的条件付き重要度」、PF-ROI の目的地引力、Chahe 2026 の計画 KL は、いずれも**自分の将来経路と衝突するかどうか**で危険を定義している。歩行者では装着者自身の軌道予測（Qiu、Ego4D locomotion、HEADS-UP）と組み合わせ、「経路上の物体・領域」をリスクとする設計が自然である（papers.md の「将来経路の衝突＋アフォーダンスの喪失＋TTC＋不確実性」という定式化に合う）。
3. **静的危険の比重**: 車載研究は動的エージェントが中心だが、歩行では段差、穴、ポール、自転車の駐輪、点字ブロック上の障害物といった**静的で、クラス外のことも多い障害物**が主役になる。障害物セグメンテーション（Lost and Found、SMIYC）、SANPO の深度とセグメンテーション、COOOL 型のラベル外危険検出を組み合わせる必要がある。
4. **BEV が使えない**: 車載の BEV 場型（PF-ROI、MC-Risk）は HD マップや多カメラを前提にする。歩行者の単眼一人称カメラでは、単眼深度から局所的な占有グリッドを作るなど、代わりの表現が要る。
5. **激しいエゴモーションと視野の揺れ**: 頭部や胸に装着したカメラでは揺れやローリングが大きく、車載ダッシュカムで学習したモデルはドメインギャップが大きい。SANPO や PEDESTRIAN で事前学習するのが有力である。
6. **注意＝危険の仮定はさらに弱い**: BLV ユーザーには視線がない、または有効でない。注視ベースの代理ラベル（BDD-A、DADA）は晴眼の歩行者データでしか作れず、それも「見落とされた危険」を拾えない（Kotseruba 2024 の指摘を参照）。
7. **言語出力の簡潔さと遅延**: WalkVLM や ForeSightGuide が指摘するように、VLM の冗長な出力と推論遅延は実時間の警告に向かない。UniDrive が指摘する小さく遠い対象の見落としも、歩道上の小さな障害物では深刻になる。「軽量な検出・セグメンテーションで候補領域を出し、VLM は確認と説明だけを担う」という分担が現実的である。
8. **誤警報コスト**: Biswas 2024 や Conformal Risk Tube が扱う「不要な警報」の抑制は、歩行支援ではユーザーの受容性に直結する。被覆保証付きのリスク出力や、ユーザーの知覚状態（気づいているか）のモデル化が必要になる。
9. **評価**: 歩行者向けの「空間＋時間」の統合指標はまだない。STAUC や AOLA の考え方を、経路上の危険物体を TTC 以前に当てられたかという形に置き換えた指標を設計する必要がある。

---

## 7. オープン問題と研究の方向

1. **統一的な危険定義と指標**: 衝突ベース、行動変化ベース、重要度ベース、注意ベースの定義が混在し、RiskBench（PIC）、DoTA（STAUC）、WWW（AOLA）、DRAMA（IoU）などベンチマークごとに指標も違う。「早く、正しい物体・領域を、安定して」当てる性能を測る共通指標が必要である。
2. **潜在的・遮蔽された危険**: 見えていない危険の前兆（PotentialRiskQA）や、計画にとって重要な遮蔽エージェント（Chahe 2026）、遮蔽を考慮したリスクマップ（Unified Risk Map）は始まったばかりである。
3. **ラベル外・長い裾の危険**: COOOL/2COOOL、CODA、SMIYC が示すように、閉じたクラス集合の検出器では足りない。オープン語彙検出と VLM の組み合わせ（Shriram 2025）が有望だが、定位精度に課題が残る。
4. **不確実性と校正**: Conformal Risk Tube 以外に、空間的リスク出力の校正を扱う研究はほとんどない。
5. **世界モデル・生成による反事実**: RiskWorld や MM-AU の物体中心動画拡散、Communications Engineering 2025 の世界モデルによるシーン生成のように、生成モデルで反事実を作り、因果的に危険源を特定する方向が伸びている。
6. **VLM の空間精度・遅延・安定性**: UniDrive、Richards 2026、DVBench が示す弱点を、高解像度の知覚枝、検出器との協調、蒸留で補う研究が続いている。
7. **歩行者・BLV 向けの空間的危険ベンチマークの欠如**: 既存の歩行支援データは VQA や説明文（EgoBlind、GuideDog、WalkVLM）か、静的な障害物検出（PEDESTRIAN）に偏っている。**時間的な危険イベントと空間的な危険物体を両方注釈した一人称歩行データ**はほぼない。ここは本プロジェクトにとって貢献の余地が大きい。
8. **人の状態との結合**: 他者が自分に注意を向けているか（RAID の歩行者の注意）、ユーザー自身が気づいているか（MAAD、Biswas 2024）を危険度に反映させる研究。

---

## 8. 参考文献

**papers.md の記載について**: papers.md にある "Ego-Alert: Egocentric Future Hazard Forecasting (ICCV 2021)"、"EgoVis"（危険検出ベンチマークとして）、"EgoPATH" は、arXiv 検索・Web 検索・CVF 関連の検索のいずれでも実在を確認できなかったため、引用していない（名前が近い実在論文は EgoPathBench 2609.16610 と EgoVIS ワークショップ）。"Learning to Anticipate Traffic Accidents from Egocentric Video (ICCV 2019)" も確認できず、Yao et al. IROS 2019 を誤記したものと考えられる。"Anticipating Accidents in Dashcam Videos" は ECCV ではなく ACCV 2016 として広く引用されている（本調査では原著ページを確認していないため参考文献には含めていない）。

### 危険物体の特定・重要物体
1. Li, C., Chan, S. H., Chen, Y.-T. "Who Make Drivers Stop? Towards Driver-centric Risk Assessment: Risk Object Identification via Causal Inference." IROS 2020. https://arxiv.org/abs/2003.02425
2. Li, C. et al. "DROID: Driver-centric Risk Object Identification." arXiv 2021 (TPAMI 投稿). https://arxiv.org/abs/2106.13201
3. Suman, V. et al. "RAIST: Learning Risk Aware Traffic Interactions via Spatio-Temporal Graph Convolutional Networks." IROS 2023. https://arxiv.org/abs/2011.08722
4. Kung, C.-H. et al. "RiskBench: A Scenario-based Benchmark for Risk Identification." ICRA 2024. https://arxiv.org/abs/2312.01659 (DOI 10.1109/ICRA57147.2024.10610270)
5. Pao, P.-Y. et al. "Potential Field as Scene Affordance for Behavior Change-Based Visual Risk Object Identification." ICRA 2025. https://arxiv.org/abs/2409.15846
6. Fu, K.-Y. et al. "Uncertainty-Aware Vision-based Risk Object Identification via Conformal Risk Tube Prediction." ICRA 2026. https://arxiv.org/abs/2603.23919
7. Agarwal, N. et al. "Towards Driver Behavior Understanding: Weakly-Supervised Risk Perception in Driving Scenes." IV 2026. https://arxiv.org/abs/2603.05926
8. Li, J. et al. "RiskWorld: Object-Centric Latent World Modeling for Autonomous Driving Risk Identification." arXiv 2026. https://arxiv.org/abs/2608.21414
9. Link, M. et al. "MC-Risk: Multi-Component Risk Fields for Risk Identification and Motion Planning." arXiv 2026. https://arxiv.org/abs/2605.21406
10. Gao, M. et al. "Goal-oriented Object Importance Estimation in On-road Driving Videos." ICRA 2019. https://arxiv.org/abs/1905.02848
11. Zhang, Z. et al. "Interaction Graphs for Object Importance Estimation in On-road Driving Videos." ICRA 2020. https://arxiv.org/abs/2003.06045
12. Li, J. et al. "Important Object Identification with Semi-Supervised Learning for Autonomous Driving." ICRA 2022. https://arxiv.org/abs/2203.02634
13. Gupta, P. et al. "Object Importance Estimation using Counterfactual Reasoning for Intelligent Driving." RA-L 9(4), 2024. https://arxiv.org/abs/2312.02467
14. Nan, Z. et al. "On-Road Object Importance Estimation: A New Dataset and A Model with Multi-Fold Top-Down Guidance." NeurIPS 2024. https://arxiv.org/abs/2411.17152
15. Parikh, C. et al. "IDD-X: A Multi-View Dataset for Ego-relative Important Object Localization and Explanation in Dense and Unstructured Traffic." ICRA 2024. https://arxiv.org/abs/2404.08561
16. Ling, S. et al. "A Two-Stage Framework for Ego-Centric Key Object Identification via Object State Prediction." ICRA 2026. https://arxiv.org/abs/2609.07125
17. Sachdeva, E. et al. "Rank2Tell: A Multimodal Driving Dataset for Joint Importance Ranking and Reasoning." WACV 2024. https://arxiv.org/abs/2309.06597
18. Biswas, A. et al. "Modeling Drivers' Risk Perception via Attention to Improve Driving Assistance." arXiv 2024. https://arxiv.org/abs/2409.04738
19. Ramanishka, V. et al. "Toward Driving Scene Understanding: A Dataset for Learning Driver Behavior and Causal Reasoning" (HDD). CVPR 2018. https://arxiv.org/abs/1811.02307

### 注意予測
21. Palazzi, A. et al. "Predicting the Driver's Focus of Attention: the DR(eye)VE Project." TPAMI. https://arxiv.org/abs/1705.03854
22. Xia, Y. et al. "Predicting Driver Attention in Critical Situations." ACCV 2018. https://arxiv.org/abs/1711.06406
23. Fang, J. et al. "DADA-2000: Can Driving Accident be Predicted by Driver Attention? Analyzed by A Benchmark." ITSC 2019 投稿. https://arxiv.org/abs/1904.12634
24. Fang, J. et al. "DADA: Driver Attention Prediction in Driving Accident Scenarios." IEEE T-ITS. https://arxiv.org/abs/1912.12148
25. Baee, S. et al. "MEDIRL: Predicting the Visual Attention of Drivers via Maximum Entropy Deep Inverse Reinforcement Learning." ICCV 2021. https://arxiv.org/abs/1912.07773
26. Pal, A. et al. "Looking at the right stuff — Guided semantic-gaze for autonomous driving." CVPR 2020. https://arxiv.org/abs/1911.10455
27. Zhu, P. et al. "Unsupervised Self-Driving Attention Prediction via Uncertainty Mining and Knowledge Embedding." ICCV 2023. https://arxiv.org/abs/2303.09706
28. Kotseruba, I. et al. "Understanding and Modeling the Effects of Task and Context on Drivers' Gaze Allocation." IV 2024. https://arxiv.org/abs/2310.09275
29. Kotseruba, I. et al. "Data Limitations for Modeling Top-Down Effects on Drivers' Attention." IV 2024. https://arxiv.org/abs/2404.08749
30. Zhou, Y. et al. "Where, What, Why: Towards Explainable Driver Attention Prediction." ICCV 2025. https://arxiv.org/abs/2506.23088
31. Gopinath, D. et al. "MAAD: A Model and Dataset for 'Attended Awareness' in Driving." ICCVW (EPIC) 2021. https://arxiv.org/abs/2110.08610
32. Shen, Y. et al. "CoCAtt: A Cognitive-Conditioned Driver Attention Dataset." arXiv 2021. https://arxiv.org/abs/2111.10014
33. Zhao, T. et al. "Gated Driver Attention Predictor." ITSC 2023. https://arxiv.org/abs/2308.02530

### 事故予測・異常（空間的出力あり／対比）
34. Zeng, K.-H. et al. "Agent-Centric Risk Assessment: Accident Anticipation and Risky Region Localization." CVPR 2017. https://arxiv.org/abs/1705.06560
35. Yao, Y. et al. "Unsupervised Traffic Accident Detection in First-Person Videos." IROS 2019. https://arxiv.org/abs/1903.00618
36. Yao, Y. et al. "When, Where, and What? A New Dataset for Anomaly Detection in Driving Videos" (DoTA). arXiv 2020. https://arxiv.org/abs/2004.03044
37. Bao, W. et al. "Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning." ACM MM 2020. https://arxiv.org/abs/2008.00334
38. Bao, W. et al. "DRIVE: Deep Reinforced Accident Anticipation with Visual Explanation." ICCV 2021. https://arxiv.org/abs/2107.10189
39. Karim, M. M. et al. "A Dynamic Spatial-temporal Attention Network for Early Anticipation of Traffic Accidents." arXiv 2021. https://arxiv.org/abs/2106.10197
40. Karim, M. M. et al. "Towards explainable artificial intelligence (XAI) for early anticipation of traffic accidents." TRR. https://arxiv.org/abs/2108.00273
41. Karim, M. M. et al. "An Attention-guided Multistream Feature Fusion Network for Localization of Risky Objects in Driving Videos." IEEE T-IV. https://arxiv.org/abs/2209.07922
42. Liao, H. et al. "When, Where, and What? A Novel Benchmark for Accident Anticipation and Localization with Large Language Models." ACM MM 2024. https://arxiv.org/abs/2407.16277
43. Fang, J. et al. "Abductive Ego-View Accident Video Understanding for Safe Driving Perception" (MM-AU). CVPR 2024. https://arxiv.org/abs/2403.00436
44. Fang, J. et al. "Cognitive Accident Prediction in Driving Scenes: A Multimodality Benchmark." arXiv 2022. https://arxiv.org/abs/2212.09381
45. Li, L.-L., Fang, J., Xue, J. "Cognitive Traffic Accident Anticipation." IEEE ITS Magazine, 2024. https://doi.org/10.1109/MITS.2024.3378460
46. Zhang, J. et al. "Eyes on the Road, Mind Beyond Vision: Context-Aware Multi-modal Enhanced Risk Anticipation" (CAMERA). ACM MM 2025. https://arxiv.org/abs/2507.06444
47. Tao, Z. et al. "CARA: Concept-Aware Risk Attention for Interpretable Collision Anticipation." ACM MM 2026. https://arxiv.org/abs/2607.22494
48. Fang, J. et al. "Vision-Based Traffic Accident Detection and Anticipation: A Survey." IEEE TCSVT. https://arxiv.org/abs/2308.15985
49. Picek, L. et al. "ACCIDENT: A Benchmark Dataset for Vehicle Accident Detection from Traffic Surveillance Videos." arXiv 2026. https://arxiv.org/abs/2604.09819
50. （対比：シーンレベル）Zhao, T. et al. "Accident Anticipation via Temporal Occurrence Prediction." NeurIPS 2025.（ローカル PDF、プロジェクト https://happytianhao.github.io/TOP/）
51. （対比）Zou, Y. et al. "RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation." CVPR 2026. https://arxiv.org/abs/2603.27165
52. （対比）Hicsonmez, S. et al. "Progressive Risk Estimation for Accident Anticipation." NeurIPS 2026. https://arxiv.org/abs/2609.32811
53. （対比）Liu, X. et al. "Predict and Resist: Long-Term Accident Anticipation under Sensor Noise." AAAI 2026. https://arxiv.org/abs/2511.08640
54. （対比）Goldshmidt, R. et al. "BADAS: Context Aware Collision Prediction Using Real-World Dashcam Data." arXiv 2025. https://arxiv.org/abs/2510.14876 ／ "Beyond the Beep … BADAS-2.0." arXiv 2026. https://arxiv.org/abs/2604.05767
55. （対比）Moura, D. C. et al. "Nexar Dashcam Collision Prediction Dataset and Challenge." CVPRW 2025. https://arxiv.org/abs/2503.03848
56. （対比）Guan, Y. et al. "Domain-Enhanced Dual-Branch Model for Efficient and Interpretable Accident Anticipation." arXiv 2025. https://arxiv.org/abs/2507.12755
57. （対比）Guan, Y. et al. "World model-based end-to-end scene generation for accident anticipation in autonomous driving." Communications Engineering 4:144, 2025. https://doi.org/10.1038/s44172-025-00474-7

### VLM / LLM・言語グラウンディング
58. Malla, S. et al. "DRAMA: Joint Risk Localization and Captioning in Driving." WACV 2023. https://arxiv.org/abs/2209.10767
59. Godbole, M. et al. "DRAMA-X: A Fine-grained Intent Prediction and Risk Reasoning Benchmark for Driving." ICCVW 2025. https://arxiv.org/abs/2506.17590
60. Charoenpitaks, K. et al. "Exploring the Potential of Multi-Modal AI for Driving Hazard Prediction." IEEE T-IV 2024. https://arxiv.org/abs/2310.04671
61. Kim, J. et al. "Textual Explanations for Self-Driving Vehicles." ECCV 2018. https://arxiv.org/abs/1807.11546
62. Kim, J. et al. "Grounding Human-to-Vehicle Advice for Self-driving Vehicles." CVPR 2019. https://arxiv.org/abs/1911.06978
63. Xu, Y. et al. "Explainable Object-induced Action Decision for Autonomous Vehicles." arXiv 2020. https://arxiv.org/abs/2003.09405
64. Sima, C. et al. "DriveLM: Driving with Graph Visual Question Answering." ECCV 2024. https://arxiv.org/abs/2312.14150
65. Tian, X. et al. "DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models." arXiv 2024. https://arxiv.org/abs/2402.12289
66. Ding, X. et al. "HiLM-D: Enhancing MLLMs with Multi-Scale High-Resolution Details for Autonomous Driving." IJCV. https://arxiv.org/abs/2309.05186
67. Chen, K. et al. "Automated Evaluation of Large Vision-Language Models on Self-driving Corner Cases." WACV 2025. https://arxiv.org/abs/2404.10595
68. Shriram, S. et al. "Towards a Multi-Agent Vision-Language System for Zero-Shot Novel Hazardous Object Detection for Autonomous Driving Safety." arXiv 2025. https://arxiv.org/abs/2504.13399
69. AlShami, A. K. et al. "2COOOL: 2nd Workshop on the Challenge Of Out-Of-Label Hazards in Autonomous Driving." ICCVW 2025. https://arxiv.org/abs/2508.21080 （COOOL @ WACV 2025: https://www.kaggle.com/competitions/coolwacv25）
70. Yokoi, S. et al. "Hierarchical Reasoning with Vision-Language Models for Incident Reports from Dashcam Videos." arXiv 2025. https://arxiv.org/abs/2510.12190
71. Chen, D. et al. "INSIGHT: Enhancing Autonomous Driving Safety through Vision-Language Models on Context-Aware Hazard Detection and Edge Case Evaluation." arXiv 2025. https://arxiv.org/abs/2502.00262
72. Gao, Y. et al. "NuRisk: A Visual Question Answering Dataset for Agent-Level Risk Assessment in Autonomous Driving." ICRA 2026. https://arxiv.org/abs/2509.25944
73. Liu, J. et al. "Predicting Risks Before They Happen: Vision-Language Reasoning for Autonomous Driving." arXiv 2025. https://arxiv.org/abs/2511.22928
74. Gao, X. et al. "UniDrive: A Unified Vision-Language and Grounding Framework for Interpretable Risk Understanding in Autonomous Driving." arXiv 2026. https://arxiv.org/abs/2606.24759
75. Chahe, A. et al. "What's Hidden Matters: Identifying Planning-Critical Occluded Agents using Vision-Language Models." IROS 2026. https://arxiv.org/abs/2607.00283
76. Tian, K. et al. "C-CoT: Counterfactual Chain-of-Thought with Vision-Language Models for Safe Autonomous Driving." arXiv 2026. https://arxiv.org/abs/2605.10744
77. Zeng, T. et al. "Are Vision LLMs Road-Ready? A Comprehensive Benchmark for Safety-Critical Driving Video Understanding." arXiv 2025. https://arxiv.org/abs/2504.14526
78. Richards, E. et al. "Task-Aligned Stability Analysis of Vision-Language Models for Autonomous Driving Hazard Detection." ICML 2026 Workshop (CTB). https://arxiv.org/abs/2606.11889

### リスク場・セグメンテーション
79. Jia, J. et al. "Learning A Unified Risk Map for Autonomous Driving in Partially Observable Environments." RA-L 2026. https://arxiv.org/abs/2605.22189
80. Pinggera, P. et al. "Lost and Found: Detecting Small Road Hazards for Self-Driving Vehicles." IROS 2016. https://arxiv.org/abs/1609.04653
81. Blum, H. et al. "The Fishyscapes Benchmark: Measuring Blind Spots in Semantic Segmentation." arXiv 2019. https://arxiv.org/abs/1904.03215
82. Chan, R. et al. "SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation." NeurIPS 2021 D&B. https://arxiv.org/abs/2104.14812
83. Li, K. et al. "CODA: A Real-World Road Corner Case Dataset for Object Detection in Autonomous Driving." ECCV 2022. https://arxiv.org/abs/2203.07724

### 歩行者・エゴセントリック・BLV
84. Waghmare, S. M. et al. "SANPO: A Scene Understanding, Accessibility and Human Navigation Dataset." arXiv 2023. https://arxiv.org/abs/2309.12172
85. Thoma, M. et al. "PEDESTRIAN: An Egocentric Vision Dataset for Obstacle Detection on Pavements." arXiv 2025. https://arxiv.org/abs/2512.19190
86. Haghighi, Y. et al. "HEADS-UP: Head-Mounted Egocentric Dataset for Trajectory Prediction in Blind Assistance Systems." arXiv 2024. https://arxiv.org/abs/2409.20324
87. Qiu, J. et al. "Egocentric Human Trajectory Forecasting with a Wearable Camera and Multi-Modal Fusion." RA-L 2022. https://arxiv.org/abs/2111.00993
88. Grauman, K. et al. "Ego4D: Around the World in 3,000 Hours of Egocentric Video." CVPR 2022. https://arxiv.org/abs/2110.07058
89. Rodin, I. et al. "Predicting the Future from First Person (Egocentric) Vision: A Survey." CVIU 2021. https://arxiv.org/abs/2107.13411
90. Yuan, Z. et al. "WalkVLM: Aid Visually Impaired People Walking by Vision Language Model." arXiv 2024. https://arxiv.org/abs/2412.20903
91. Xiao, J. et al. "EgoBlind: Towards Egocentric Visual Assistance for the Blind." NeurIPS 2025 D&B. https://arxiv.org/abs/2503.08221
92. Kim, J. et al. "GuideDog: A Real-World Egocentric Multimodal Dataset for Blind and Low-Vision Accessibility-Aware Guidance." ACL 2026. https://arxiv.org/abs/2503.12844
93. Ying, K. et al. "mmWalk: Towards Multi-modal Multi-view Walking Assistance." NeurIPS 2025 D&B. https://arxiv.org/abs/2510.11520
94. Hwang, H. et al. "GuideTWSI: A Diverse Tactile Walking Surface Indicator Dataset from Synthetic and Real-World Images for Blind and Low-Vision Navigation." arXiv 2026. https://arxiv.org/abs/2603.07060
95. Zhang, Z. et al. "'I Can See Forever!': Evaluating Real-time VideoLLMs for Assisting Individuals with Visual Impairments." arXiv 2025. https://arxiv.org/abs/2505.04488
96. Wang, Z. et al. "ForeSightGuide: An Anticipatory Framework toward Accurate and Low-Redundancy Guidance for the Visually Impaired." arXiv 2026. https://arxiv.org/abs/2608.18993
97. Valls, A. et al. "Urban Risk-Aware Navigation via VQA-Based Event Maps for People with Low Vision." arXiv 2026. https://arxiv.org/abs/2605.11782
98. Chen, Y. et al. "EgoSafe: A First-Person Mobile-Captured Benchmark for Visual Safety Understanding." arXiv 2026. https://arxiv.org/abs/2607.26518
99. Zhao, Y. et al. "EgoPathBench: Evaluating Zero-Shot Egocentric Waypoint Decision-Making in Vision-Language Models." arXiv 2026. https://arxiv.org/abs/2609.16610
