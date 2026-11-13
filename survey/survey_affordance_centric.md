# アフォーダンス中心の危険予測（Affordance-centric Hazard Prediction）サーベイ — 2026年までの動向

> 対象: 一人称（egocentric）映像からの歩行支援向け危険検知
> 作成日: 2026-11
> 方針: 引用した論文はすべて、タイトル・第一著者・発表先・年を arXiv / DOI / 公式ページで確認した。確認できない数値は載せていない。本文中の数値は各論文のアブストラクトに書かれている値である。

---

## 0. 要旨

アフォーダンス中心の危険予測では、「何が写っているか」(物体検出) や「このフレームの事故確率はどれくらいか」(シーン中心) を問わない。問うのは次の一点である。

> **「エージェント（歩行者・ロボット・車椅子・車両）は、意図した経路に沿って、これからも安全に行動・移動し続けられるか？」**

危険とは、移動できる空間 (affordance) が**失われること**、または**変化すること**だと定義する。障害物の出現、通路の閉塞、段差、他者の進路との競合などがこれに当たる。この考え方は複数の系譜にまたがっている。

- ロボティクスの traversability 推定 (BADGR, WVN, V-STRONG)
- 自動運転の freespace / occupancy forecasting (Hu 2021, Khurana 2022/2023, Cam4DOcc, OccWorld)
- 一人称映像の将来経路予測 (Park 2016, Qiu 2022, EgoNav, LookOut)
- 一人称映像のアフォーダンス学習 (Interaction Hotspots, Ego-Topo)
- TTC 推定 (Manglik 2019, Binary TTC)
- 視覚障害者支援 (BBeep, HEADS-UP, AI Guide Dog, WalkVLM, GuideDog, mmWalk, WalkGPT)

2023〜2026 年には VLM/LLM による navigability 推論 (CoNVOI, VLM-Social-Nav, ZeST, CATNAV, NaviTrace, EgoPathBench) が急速に増えた。ただし評価では、VLM の空間的接地 (spatial grounding) と身体性 (embodiment) を考慮した経路判断が、まだ大きく弱いことが繰り返し示されている。

---

## 1. 定義とスコープ

### 1.1 三つのパラダイム

| 観点 | シーン中心 (Scene-centric) | リスク領域 (Risk-region) | **アフォーダンス中心 (Affordance-centric)** |
|---|---|---|---|
| 問い | 「近いうちに事故が起きるか？」 | 「どの物体・領域が危険か？」 | 「意図した経路を安全に進み続けられるか？」 |
| 出力 | フレーム毎の事故確率 $p_t$、TTA | 危険物体の bbox/マスク、重要度 | 歩行可能・走行可能マップ、将来の占有、将来経路、経路衝突/TTC、不確かさ |
| 危険の単位 | 時間 (いつ) | 物体 (どれが) | **空間 × 時間 × 自己の意図 (どこへ行けなくなるか)** |
| 代表例 | DSA/UString/TOP 系 | DRAMA, ROL 系 | 本サーベイの対象 |
| 自己の行動 | 通常は考慮しない | 部分的に考慮 | **明示的に条件付ける** |

- シーン中心の事故予測 (例: NeurIPS 2025 の TOP) と、リスク領域の手法 (危険物体の局在化) は**本サーベイの対象外**とし、比較のためにだけ触れる。これらはそれぞれ `survey_scene_centric.md` と `survey_risk_region.md` で扱う。
- アフォーダンス中心の手法の強みは、**行動に直結した判断を出せる**ことにある。「止まる」「右へよける」「このまま進む」が決まる。また、検出器の語彙に載っていない障害物 (out-of-label) にも、「通れない領域」として自然に対応できる。

### 1.2 「アフォーダンス」の二つの意味

1. **移動アフォーダンス (locomotion / navigational affordance)**: 歩ける・走れる・通れる空間。traversability、walkability、drivable area、freespace がこれに当たる。Park et al. (CVPR 2016) は一人称のステレオ映像から「歩ける面・避けるべき物体」を学び、将来の自己経路を予測した。この流れの起点と言える。
2. **相互作用アフォーダンス (interaction affordance)**: 物体のどこを掴むか、どこで何ができるか。代表例は Interaction Hotspots (Nagarajan, ICCV 2019)、Ego-Topo (CVPR 2020)、Liu et al. (CVPR 2022)。歩行の危険にとっては副次的だが、「ドアを押す」「手すりを掴む」のような環境との接触の予測に使える。

本サーベイは主に 1. を扱い、2. は関連する系譜として整理する。

---

## 2. 問題設定の分類と評価指標

### 2.1 問題設定のタクソノミー

| ID | 問題設定 | 入力 | 出力 | 代表研究 |
|---|---|---|---|---|
| P1 | 現在の traversability / walkability 推定 | RGB(-D)、LiDAR、自己運動 | 画素毎または BEV のコスト・通行可能確率 | GONet, BADGR, WVN, V-STRONG, SANPO, WalkGPT |
| P2 | 将来の freespace / occupancy 予測 | 過去の点群・画像列 | $t+\tau$ の占有/freespace (BEV, 3D) | Hu 2021, Khurana 2022/2023, FIERY, Cam4DOcc, OccWorld, UniAD |
| P3 | 一人称の将来自己経路予測 | 一人称映像 (+ IMU、視線、深度) | 将来軌跡の分布 | Park 2016, Qiu 2022, EgoNav, EgoCogNav, AI Guide Dog, LookOut |
| P4 | 他者の将来位置 + 自己との衝突/競合 | 一人称映像 | 他者軌跡、衝突リスク | Yagi 2018, InCrowdFormer, HEADS-UP, BBeep |
| P5 | TTC (time-to-collision) 推定 | 単眼映像 | TTC (連続値 / 閾値付き二値) | Manglik 2019, Binary TTC |
| P6 | 一人称映像からのアフォーダンス学習 | 一人称映像 | hotspot、環境アフォーダンスグラフ | Interaction Hotspots, Ego-Topo, Liu 2022, VRB, EgoPAT3D |
| P7 | 視覚障害者向け safe-path / ガイダンス生成 | 一人称映像・画像 | 経路、警告、言語による指示 | BBeep, AI Guide Dog, WalkVLM, GuideDog, EgoBlind, mmWalk, WalkGPT |
| P8 | VLM/LLM による navigability 推論 | 画像 + 言語の目標 | 候補点の選択、経路トレース、コスト | CoNVOI, VLM-Social-Nav, ZeST, CATNAV, NaviTrace, EgoPathBench, SceneAware |

### 2.2 評価指標

| 問題 | 指標 | 備考 |
|---|---|---|
| P1 | IoU / F1 (通行可能領域)、AUROC、実機での成功率・介入回数 | 自己教師ありの手法では実機評価が中心 |
| P2 | 将来占有の IoU、Chamfer/L1 (点群予測)、ray-based 指標 | Khurana 2023 は点群予測を 4D 占有予測の代理タスクにする |
| P3/P4 | ADE / FDE、best-of-K、モード被覆、**衝突率** (予測経路と障害物の距離) | LookOut は静的/動的障害物それぞれへの距離で衝突を評価 |
| P5 | TTC の絶対誤差 [s]、閾値を超えたかの二値精度、推論レイテンシ | Binary TTC は低遅延を重視 |
| P6 | KLD / SIM / AUC-J (hotspot)、3D 目標位置の誤差 | |
| P7/P8 | VQA 正答率、言語の質、経路スコア (実行可能性・隣接エッジの合法性・到達)、冗長性 | EgoPathBench は「実行可能か、合法か、到達するか」で採点 |
| 危険予測として | **mTTA、イベント単位の Precision/Recall、誤警報率 (件/分)、警告の早さ** | 歩行支援では誤警報のコストが大きい (§6) |

---

## 3. データセット表

| データセット | 年/発表先 | 視点・主体 | 主な内容 | アフォーダンス中心での用途 |
|---|---|---|---|---|
| Egocentric Future Localization (Park et al.) | CVPR 2016 | 一人称ステレオ、歩行者 | 将来の自己経路、歩行回避マップ | P1/P3 の原型 |
| First-Person Locomotion (Yagi et al.) | CVPR 2018 | 一人称、歩行者 | 他者の将来位置 | P4 |
| Near-Collision (Manglik et al.) | IROS 2019 | 押して運ぶスーツケース型ロボット、屋内 | 一人称スニペット 13,658 本、TTC ラベル | P5 |
| EPIC-KITCHENS / Ego4D (Grauman et al.) | CVPR 2022 (Ego4D) | 一人称、日常生活 | 大規模映像、Forecasting ベンチマーク (手・物体・行動の予測) | P6、事前学習 |
| TISS (Qiu et al.) | RA-L 2022 | 一人称、混雑空間 | 装着者の軌跡 + 周囲の人 + シーン | P3/P4 |
| EgoPAT3D (Li et al.) | CVPR 2022 | 一人称 RGB-D + IMU | 100 万フレーム超、3D 行動目標 | P6 |
| SCAND (Karnan et al.) | RA-L 2022 | ロボット、社会的環境 | 社会的に適切なナビゲーションのデモ | 社会的ナビゲーション |
| SANPO (Waghmare et al.) | arXiv 2023 (WACV 2025 投稿版) | 一人称ステレオ、屋外歩行 | 実映像 701 本 (全フレーム深度、11.2 万フレームにパノプティック) + 合成映像 | P1、BLV 支援 |
| EgoNav データ (Wang et al.) | arXiv 2024 | 身体装着カメラ、歩行 | 一人称の歩行ナビゲーション | P3 |
| HEADS-UP (Haghighi et al.) | arXiv 2024 | 頭部装着、BLV 想定 | 歩行者との衝突シナリオ (難易度別) | P4 + 衝突リスク |
| WAD (WalkVLM) | arXiv 2024 | 一人称、歩行 | 映像-注釈ペア 12,000 | P7 |
| EgoBlind (Xiao et al.) | arXiv 2025 | 視覚障害者本人による一人称映像 | 映像 1,392 本、質問 5,311 | P7 (VideoQA) |
| GuideDog / GuideDogQA (Kim et al.) | ACL 2026 | 歩行者視点、46 か国 | 約 2.2 万の画像-記述ペア、818 問の QA | P7 |
| AI Guide Dog (Jadhav et al.) | AAAI 2025 Spring Symposium | スマートフォン、一人称 | 屋内・屋外の経路予測用映像 | P3/P7 |
| EgoWalk (Akhtyamov et al.) | arXiv 2025 | 人間の一人称歩行、ステレオ | 50 時間、屋内外・季節も多様 | P1/P3、ロボットへの転移 |
| LookOut / Aria データ (Pan et al.) | ICCV 2025 | Project Aria | 6D の頭部姿勢軌跡、静的/動的障害物 | P3 + 衝突評価 |
| mmWalk (Ying et al.) | NeurIPS 2025 D&B | 合成、多視点 (BLV) | 120 経路、6.2 万フレーム、mmWalkVQA 6.9 万問 | P7/P8 (リスク評価 VQA) |
| CEN (EgoCogNav, Qiu et al.) | arXiv 2025 | 一人称 + 視線 | 6 時間 | P3 + 経路の不確かさ |
| PEDESTRIAN (Thoma et al.) | arXiv 2025 | スマートフォン、歩道 | 映像 340 本、障害物 29 種 | 歩道の障害物 (P1 の補助) |
| PAVE (WalkGPT) | CVPR 2026 | 歩行者視点 (SANPO 由来) | 4.1 万の画像-質問-回答 + 分割 + 深度 | P1/P7 |
| NaviTrace (Windecker et al.) | ICRA 2026 | 多様な身体性 | 1,000 シナリオ、3,000 超の専門家トレース | P8 |
| EgoPathBench (Zhao et al.) | arXiv 2026 | 一人称 RGB | 学習 31,852 / 検証 1,345 / 評価 1,111 問 | P8 |
| nuScenes/Occ 系 (Cam4DOcc, OccWorld) | CVPR 2024 / ECCV 2024 | 車載 | 4D 占有予測 | P2 |

---

## 4. 手法の系譜と比較

### 4.1 系譜 (年表)

```
2016  Park: Egocentric Future Localization ──► 一人称の「歩行アフォーダンス」+ 将来経路
2017  Kahn: 不確かさを考慮した衝突回避 RL
2018  Yagi: 一人称での将来人物位置 / Hirose: GONet (半教師ありの traversability)
2019  Manglik: 単眼 TTC 予測 / Nagarajan: Interaction Hotspots / Kayukawa: BBeep (BLV 向け衝突回避)
2020  Nagarajan: Ego-Topo / Mangalam: Y-Net (シーンに従う軌跡)
2021  Kahn: BADGR (自己教師ありのイベント予測) / Hu: Freespace Forecasting / Badki: Binary TTC / Hu: FIERY
2022  Khurana: 微分可能レイキャストによる占有予測 / Qiu: 一人称の軌跡予測 / Liu: 手 + hotspot / EgoPAT3D / Ego4D / SCAND
2023  Frey: WVN (オンライン自己教師あり) / Khurana: 4D 占有予測 / UniAD / Bahl: VRB / ViNT / NoMaD / InCrowdFormer / SANPO
2024  V-STRONG / Cam4DOcc / OccWorld / EgoNav / HEADS-UP / CoNVOI / VLM-Social-Nav / WalkVLM / CityWalker / NWM
2025  EgoBlind / GuideDog / AI Guide Dog / EgoWalk / LookOut / mmWalk / SceneAware / ZeST / EgoCogNav / NaviTrace / PEDESTRIAN
2026  WalkGPT / CATNAV / EgoNav (ヒューマノイド) / EgoPathBench
```

大まかには次の三段階で進んできた。

- **(a) 幾何と自己教師あり** (〜2021): 実際にぶつかった、通れたという経験をラベルにする。
- **(b) 将来予測と世界モデル** (2021〜2024): freespace や occupancy を時間方向へ外挿する。
- **(c) 基盤モデルと言語** (2023〜2026): DINO 特徴や VLM で zero-shot に navigability を判断し、言語で説明する。

### 4.2 比較表

| 手法 | 年/発表先 | 問題 | 入力 | 教師信号 | 危険の表現 | 不確かさ | 一人称歩行への近さ |
|---|---|---|---|---|---|---|---|
| Park et al. | CVPR'16 | P1+P3 | 一人称ステレオ | 実際の歩行軌跡 | 歩行回避マップ、経路 | — | ◎ |
| Kahn et al. (UA-RL) | arXiv'17 | 衝突回避 | 単眼 | 衝突経験 | 衝突確率 | ◎ (bootstrap/dropout) | △ |
| GONet | arXiv/IROS'18 | P1 | 単眼 | 半教師あり (GAN) | 通行可能か | — | ○ |
| Manglik et al. | IROS'19 | P5 | 単眼 | TTC ラベル | TTC [s] | — | ◎ (人が押すロボット) |
| BBeep | CHI'19 | P4+P7 | ステレオ (スーツケース) | — (検出 + 予測) | 予測衝突 → 音で警告 | — | ◎ |
| BADGR | RA-L'21 | P1 | 単眼 + センサ | 事後のイベント (衝突・凹凸) | イベント確率 | — | ○ |
| Hu et al. Freespace | CVPR'21 | P2 | LiDAR 列 | 将来のレイキャスト (注釈不要) | 将来 freespace | — | △ (車両) |
| Binary TTC | CVPR'21 | P5 | 単眼・未較正 | TTC | 時間的ジオフェンス | — | ○ |
| FIERY | ICCV'21 | P2 | 多視点カメラ | BEV 注釈 | 将来インスタンス (BEV) | ○ (確率的) | △ |
| Khurana et al. | ECCV'22 / CVPR'23 | P2 | LiDAR | 自己教師あり | 4D 占有 | — | △ |
| Qiu et al. | RA-L'22 | P3/P4 | 一人称 + 深度/意味 | 軌跡 | 将来軌跡 | — | ◎ |
| WVN | RSS'23 | P1 | RGB (DINO) | オンライン自己教師あり (速度追従誤差) | traversability | ○ (再構成誤差) | ○ |
| UniAD | CVPR'23 | P2+計画 | 多視点 | 注釈 | 占有 + 計画 | — | △ |
| V-STRONG | ICRA'24 | P1 | RGB (SAM + 基盤モデル) | 人の運転データ + マスク | コストマップ | — | △ (オフロード) |
| Cam4DOcc / OccWorld | CVPR'24 / ECCV'24 | P2 | カメラ / 占有 | 注釈 | 4D 占有・世界モデル | — | △ |
| EgoNav | arXiv'24 | P3 | 身体装着カメラ | 軌跡 | 経路分布 (拡散) | ○ (分布) | ◎ |
| HEADS-UP | arXiv'24 | P4 | 頭部カメラ | 軌跡 | 衝突リスク | — | ◎ (BLV) |
| CoNVOI | arXiv'24 | P8 | RGB + 占有 | VLM (zero-shot) | 障害物のない候補点の選択 | — | ○ |
| VLM-Social-Nav | arXiv'24 | P8 + 社会的 | RGB | VLM | 社会的コスト | — | ○ |
| WalkVLM | arXiv'24 | P7 | 一人称映像 | WAD | 言語による警告 | — | ◎ |
| AI Guide Dog | AAAI-SS'25 | P3/P7 | スマートフォン映像 | 軌跡 | 1 秒先の進行方向 | ○ (多経路) | ◎ |
| LookOut | ICCV'25 | P3 | Aria 映像 | 6D 頭部軌跡 | 衝突しない将来経路 | — | ◎ |
| ZeST | arXiv'25/'26 | P8 | RGB | VLM (zero-shot) | traversability の事後分布、期待ショートフォール | ◎ (偶然的/認識的を分離) | ○ |
| EgoCogNav | arXiv'25 | P3 | 一人称 + 視線 | 軌跡 | 経路 + 本人が感じる不確かさ | ◎ | ◎ |
| WalkGPT | CVPR'26 | P1+P7 | 歩行者視点画像 | PAVE | 通行可能/有害領域のマスク + 距離 + 言語 | — | ◎ |
| CATNAV | arXiv'26 | P8 | RGB | MLLM (zero-shot) | 身体性を考慮したコストマップ | — | ○ |
| EgoPathBench | arXiv'26 | P8 (評価) | 一人称 RGB | — | ウェイポイントの実行可能性 | — | ◎ |

◎ = 一人称の歩行に直接使える、○ = 適応すれば使える、△ = 概念だけ使える

---

## 5. 主要論文の要約

### 5.1 歩行可能性・traversability の推定 (P1)

- **Park et al., "Egocentric Future Localization", CVPR 2016.** 一人称のステレオ映像から、障害物を避け、物の間を抜け、角を曲がるような将来の自己経路を予測する。画素単位の「歩行回避マップ」も出力する。walkability と将来経路を一つにまとめた先駆的な仕事で、本テーマの原点に当たる。
- **Hirose et al., "GONet", 2018.** GAN を使った半教師ありの traversability 推定。単眼画像だけで「進めるか」を判定する。
- **Kahn et al., "BADGR", RA-L 2021.** 実世界で集めたオフポリシーデータに、衝突や凹凸といった「イベント」を事後的・自己教師ありで付け、行動系列ごとにイベントの確率を予測する。危険を「行動条件付きの将来イベント」として扱う点で、アフォーダンス中心の発想に近い。
- **Frey et al., "Fast Traversability Estimation for Wild Visual Navigation (WVN)", RSS 2023.** 自己教師ありの ViT (DINO) 特徴を使い、短いデモの後はオンラインの自己教師ありで適応する、視覚だけの traversability 推定。拡張版は arXiv:2404.07110 にある。
- **Jung et al., "V-STRONG", ICRA 2024.** 基盤モデルの特徴とクラスに依存しない分割マスクを使い、対照学習で traversability を学ぶ。分布外のデータに強い。
- **Waghmare et al., "SANPO", arXiv 2023.** 屋外歩行の一人称ステレオ映像に密な深度とパノプティック注釈を付けた。BLV 支援を明示的に目的にしている。
- **Sultan et al., "WalkGPT", CVPR 2026.** 歩行者視点の画像と質問から、通行可能な領域と有害な領域の分割マスク、相対的な距離、会話形式の案内を同時に出す画素接地型 LVLM。PAVE データセットも提案した。「歩ける領域」と「言語による説明」を一つのモデルに統合した最新の例。
- 補助: **Sevastopoulos et al., "A Survey of Traversability Estimation for Mobile Robots", 2022** は traversability 研究全体を整理したサーベイ。

### 5.2 将来の freespace・占有の予測 (P2)

- **Hu et al., "Safe Local Motion Planning with Self-Supervised Freespace Forecasting", CVPR 2021.** 将来の LiDAR 観測をレイキャストすると、注釈なしで将来の freespace の正解が得られることを示した。「通れる空間が今後どうなるか」を直接予測し、計画に使う。
- **Khurana et al., "Differentiable Raycasting for Self-supervised Occupancy Forecasting", ECCV 2022** と **"Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting", CVPR 2023.** 微分可能なレイキャストで、自己教師ありのまま 4D 占有を予測する。
- **Hu (Anthony) et al., "FIERY", ICCV 2021.** 周囲の単眼カメラから BEV 上の将来インスタンスを確率的に予測する。
- **Hu (Yihan) et al., "UniAD: Planning-oriented Autonomous Driving", CVPR 2023.** 占有予測と計画を一体で学習する。「計画にとって意味のある将来予測」へ向かう流れの代表。
- **Ma et al., "Cam4DOcc", CVPR 2024** と **Zheng et al., "OccWorld", ECCV 2024.** カメラだけを使う 4D 占有予測のベンチマークと、3D 占有の世界モデル。
- **Bar et al., "Navigation World Models", CVPR 2025.** 行動を条件として将来の一人称観測を生成するナビゲーション世界モデル。経路候補を「想像して」評価できる。

### 5.3 一人称の将来経路予測と衝突推論 (P3/P4/P5)

- **Yagi et al., "Future Person Localization in First-Person Videos", CVPR 2018.** 一人称映像で、他者の将来位置を装着者の自己運動を考慮して予測する。
- **Manglik et al., "Forecasting Time-to-Collision from Monocular Video", IROS 2019.** 人が押すスーツケース型ロボットの単眼映像から、周囲の歩行者との TTC を、明示的な深度や速度を使わずに回帰する。一人称のスニペット 13,658 本で、平均誤差は約 0.75 秒と報告されている。歩行支援に最も近い TTC の研究。
- **Badki et al., "Binary TTC", CVPR 2021.** TTC を「時間 τ 以内に衝突するか」の二値分類の列として推定する。未較正の単眼カメラで低遅延な「時間的ジオフェンス」を実現した。
- **Qiu et al., "Egocentric Human Trajectory Forecasting with a Wearable Camera and Multi-Modal Fusion", RA-L 2022.** 混雑空間で装着者の将来軌跡を、周囲の人の軌跡・シーンの意味・深度を融合して予測する (TISS データ)。視覚障害者のナビゲーションへの応用も明記している。
- **Nishimura et al., "InCrowdFormer", arXiv 2023.** 一人称の映像だけから、周囲の歩行者の地面上の位置を予測する「歩行者世界モデル」。
- **Wang et al., "EgoNav", arXiv 2024.** 身体装着センサで集めた歩行データに対し、拡散モデルで静的環境を条件とした将来経路の分布を生成する。衝突回避とモード被覆の改善を報告している。同じグループの **"Learning Humanoid Navigation from Human Data" (arXiv 2026)** は、約 5 時間の人間の歩行データだけで学習し、ヒューマノイドを zero-shot で動かした。
- **Haghighi et al., "HEADS-UP", arXiv 2024.** BLV 支援を目的とした頭部装着の一人称データセット。自己と他者の軌跡を別々に予測せず、カメラ基準の回転座標系で衝突リスクを直接推定する。
- **Jadhav et al., "AI Guide Dog", AAAI 2025 Spring Symposium.** スマートフォン上で、カメラだけから進むべき方向を予測する軽量な BLV ナビゲーション。複数の経路があり得る状況も扱う。
- **Pan et al., "LookOut", ICCV 2025.** 過去の一人称映像から、静的・動的障害物がある状況での将来の 6D 頭部姿勢軌跡を予測する。静的と動的の障害物に分けて衝突を評価する点が、危険予測の評価設計として参考になる。
- **Qiu (Zhiwen) et al., "EgoCogNav", arXiv 2025.** 一人称映像・視線・過去の運動から、将来の経路に加えて歩行者本人が感じる経路の不確かさを推定する。迷い、走査、引き返しといった行動と対応することを示した。
- **Mangalam et al., "Y-Net", 2020 (ICCV 2021).** シーンの意味地図の上で、ゴール・ウェイポイント・経路を分けて長期の軌跡を予測する。「通れる場所に従う」軌跡予測の代表。
- **Bai & Shim, "SceneAware", arXiv 2025.** MLLM が生成した二値の歩行可能性マスクで、歩行者の軌跡予測に制約をかける。

### 5.4 一人称映像からのアフォーダンス学習 (P6)

- **Nagarajan et al., "Grounded Human-Object Interaction Hotspots from Video", ICCV 2019.** 人の行動映像を見るだけで、物体の「どこで相互作用が起きるか」を表す hotspot を弱教師ありで学習する。
- **Nagarajan et al., "Ego-Topo", CVPR 2020.** 一人称映像を、環境の「場所 (zone)」と、そこで可能な行動からなるトポロジカルグラフに変換する。環境アフォーダンスを場所の単位で表した例。
- **Liu et al., "Joint Hand Motion and Interaction Hotspots Prediction from Egocentric Videos", CVPR 2022.** 手の将来軌跡と相互作用の hotspot を同時に予測する。
- **Li et al., "EgoPAT3D", CVPR 2022.** 一人称の RGB-D と IMU から、操作の目標位置を 3D でできるだけ早く予測する。
- **Bahl et al., "VRB: Affordances from Human Videos as a Versatile Representation for Robotics", CVPR 2023.** 人間の映像から接触点と接触後の軌跡を学び、ロボットに転移する。
- **Grauman et al., "Ego4D", CVPR 2022.** Forecasting ベンチマーク (将来の手・物体・行動) を含む大規模な一人称映像。walkability のラベルはないが、事前学習源として重要。
- 背景: **Rodin et al., "Predicting the Future from First Person (Egocentric) Vision: A Survey", CVIU 2021.**

### 5.5 視覚障害者向けの safe-path と歩行支援 (P7)

- **Kayukawa et al., "BBeep", CHI 2019.** スーツケース型のシステムで歩行者の将来位置を予測し、衝突が予測されたときに音で周囲へ知らせる。「予測した衝突で警告する」という、アフォーダンス中心の危険検知を実機で示した例。
- **Yuan et al., "WalkVLM", arXiv 2024** と **"WalkVLM-LR", arXiv 2025.** 歩行支援用の WAD データセット (12,000 ペア) と、CoT による階層的計画、時間的な冗長を抑える警告生成。後続の LR 版は人間の好みに基づく報酬で冗長さを減らす。
- **Xiao et al., "EgoBlind", arXiv 2025.** 視覚障害者自身が撮影した一人称映像の VideoQA。最良の MLLM でも約 60% で、人間 (87.4%) を大きく下回る。
- **Kim et al., "GuideDog", ACL 2026.** 46 か国の歩行者視点の画像に、BLV ガイダンスの基準に沿った記述を付けた。現行の MLLM は奥行きの知覚とガイダンス基準への準拠が苦手だと報告している。
- **Ying et al., "mmWalk", NeurIPS 2025 D&B.** 合成の多視点・多モーダルな歩行支援データ。主要な VLM はリスク評価とナビゲーションの質問で成績が悪かった。
- **Thoma et al., "PEDESTRIAN", arXiv 2025.** 歩道の障害物 29 種を含むスマートフォン映像。物体中心のデータだが、walkability の阻害要因を学習するのに使える。

### 5.6 社会的ナビゲーションのアフォーダンス

- **Karnan et al., "SCAND", RA-L 2022.** 社会的に適切なナビゲーションの大規模なデモデータ。
- **Song et al., "VLM-Social-Nav", arXiv 2024.** VLM の判断を社会的なコスト項に変換し、計画の行動を採点する。
- **Lindemann et al., "Safe Planning in Dynamic Environments using Conformal Prediction", arXiv 2022.** 他者の軌跡予測に conformal prediction で誤差の上限を付け、安全な計画を保証する。危険予測を校正する方法として重要。

### 5.7 VLM/LLM による navigability 推論 (P8, 2023–2026)

- **Shah et al., "ViNT", CoRL 2023**、**Sridhar et al., "NoMaD", 2023 (ICRA 2024)**、**Liu et al., "CityWalker", CVPR 2025.** 大規模なナビゲーション基盤モデル。CityWalker は Web 上の都市歩行映像から学習する。
- **Akhtyamov et al., "EgoWalk", arXiv 2025.** 人間の一人称歩行 50 時間。ViNT・NoMaD の fine-tuning や traversability の学習に使い、身体性の違いがあっても人間の歩行データが有用だと結論した。
- **Sathyamoorthy et al., "CoNVOI", arXiv 2024.** VLM が画像上の番号付き候補点 (障害物のない場所) を選び、それを局所占有マップに対応させて経路にする。
- **Gummadi et al., "ZeST", arXiv 2025 (v4: 2026).** VLM の出力を確率的な観測として扱い、Normal-Inverse-Gamma の事後分布で偶然的/認識的不確かさを分ける。リスクは期待ショートフォールで定量化する。**不確かさを危険の項として明示的に入れた** VLM-traversability 研究。
- **Potnis et al., "CATNAV", arXiv 2026.** MLLM で身体性を考慮したコストマップを zero-shot で生成し、シーンの新規性に応じてキャッシュする (VLM への問い合わせを 85.7% 減らしたと報告)。
- **Windecker et al., "NaviTrace", ICRA 2026.** 身体性 (人・脚ロボット・車輪・自転車) ごとに、画像上の 2D 経路トレースで VLM を評価する。VLM は空間的接地とゴールの局在化が弱く、人間に及ばない。
- **Zhao et al., "EgoPathBench", arXiv 2026.** 一人称画像上のウェイポイント選択で VLM を評価する。最良モデルの EgoPath Score は 28.3 で、身体性を考慮した経路 (Embodied Path) ではほぼ解けていない。fine-tuning で大きく改善した。

**まとめ**: VLM は「何が危ないか」の意味的な判断や言語での説明には役立つ。一方で、「自分の身体で、どこを、いつまで通れるか」という幾何・時間の判断はまだ弱い。現実的なのは、**幾何モジュール (深度・占有・TTC) と VLM を組み合わせるハイブリッド**である。

---

## 6. 歩行支援への適用と定式化

### 6.1 二つの問いへの対応

| 問い | アフォーダンス中心の解釈 | 必要なモジュール |
|---|---|---|
| Q1: 近いうちに危険なことが起きるか？ | 予測した**意図経路**上で、許容時間内に affordance が失われるか | 将来経路予測 (P3) + 将来 walkability/占有 (P1/P2) + TTC (P5) |
| Q2: 意図した経路を安全に進み続けられるか？ | 現在から少し先までの意図経路の通行可能性と、代替経路があるか | walkability (P1) + 経路分布 (P3) + 不確かさ |

### 6.2 定式化の案

時刻 $t$ の一人称観測を $o_{\le t}$ とする。

1. **意図経路の分布**: $\pi \sim p(\pi \mid o_{\le t})$。将来の地面上の経路 $\pi=\{x_{t+k}\}_{k=1}^{K}$ を EgoNav / AI Guide Dog / LookOut のように多峰の分布として出す。GPS やルート情報がある場合は、それで条件付ける。
2. **将来の歩行可能性**: $\hat A_{t+\tau}(x) = P(x \text{ が時刻 } t+\tau \text{ で歩行可能} \mid o_{\le t}) \in [0,1]$。
   静的な walkability (段差・穴・工事・水たまり。SANPO / WalkGPT 型の分割) と、動的な占有 (歩行者・自転車・車。InCrowdFormer / HEADS-UP 型の予測) の積で作る。
3. **経路の衝突 (affordance loss)**:
   $C(\pi) = \max_{k} \big(1 - \hat A_{t+k}(x_{t+k})\big)$。経路上で最も通りにくい点を見る。
   現在からの変化量 $\Delta A = A_t - \hat A_{t+\tau}$ は「急に通れなくなる」ことを表す。
4. **TTC**: $\mathrm{TTC}(\pi) = \min \{ \tau : 1-\hat A_{t+\tau}(x_{t+\tau}) > \theta \}$。
   Binary TTC のように「$\tau_{\text{react}}$ 以内に衝突するか」の二値分類にすると、低遅延で頑健になる。
5. **不確かさ**: $U$ = 経路分布のエントロピー + 歩行可能性の認識的不確かさ (アンサンブル、ZeST の NIG 事後分布) + 本人の迷い (EgoCogNav)。
6. **危険スコアと警告の判断**:
   $$H_t = \mathbb{E}_{\pi}\big[ C(\pi)\cdot g(\mathrm{TTC}(\pi)) \big] + \lambda\,U,\qquad g(\tau)=\exp(-\tau/\tau_0)$$
   警告条件は $P\big(\mathrm{TTC} < \tau_{\text{react}}\big) > \delta$ とする。$\delta$ は Lindemann et al. のような conformal prediction で校正し、誤警報率に上限を付ける。
   さらに代替経路 $\pi'$ の通行可能性 $\max_{\pi'}(1-C(\pi'))$ を出力すれば、「止まれ」と「右へ」を区別できる。

これはプロジェクトのメモにある $\text{Hazard} = f(\text{将来経路の衝突}, \text{affordance loss}, \text{TTC}, \text{不確かさ})$ を、測定できる量に分解したものになる。

### 6.3 実装の方針

- **教師信号**: 正解ラベルは歩行者本人の「その後の実際の行動」から自己教師ありで作れる。BADGR や Hu et al. の freespace 予測と同じ考え方である。例えば、急停止・回避・引き返しを危険イベントとし、実際に歩いた軌跡を通行可能領域の正例とする。EgoWalk / EgoNav / CEN 型の歩行データなら、手作業の注釈を大きく減らせる。
- **骨格の候補**: (i) 単眼深度 + walkability の分割で BEV 歩行可能マップを作る → (ii) 他の歩行者の将来占有 (InCrowdFormer 型) → (iii) 拡散モデルで意図経路の分布 (EgoNav 型) → (iv) 経路の衝突、TTC、不確かさを集約 → (v) 必要に応じて VLM が意味を補い、言語で警告する (WalkVLM / WalkGPT 型)。
- **評価**: 経路の衝突率 (LookOut 型の静的/動的別)、警告の早さ (mTTA 相当)、イベント単位の P/R、**1 分あたりの誤警報数**、警告の冗長さ (WalkVLM-LR の問題意識) で測る。
- **シーン中心・リスク領域の手法との併用**: シーン中心のスコアは「何かが起きそう」という事前分布に、リスク領域の手法は動的障害物の候補生成に使える。最終的な判断はアフォーダンス (通れるか) で決めるのが筋がよい。

---

## 7. オープン問題と研究の方向

1. **歩行者視点の「affordance loss」のベンチマークがない**: 一人称映像で「意図経路がふさがれる瞬間」と TTC を注釈した公開データは確認できなかった。HEADS-UP、LookOut、EgoBlind、GuideDog が部分的に近い。
2. **身体性への依存**: 杖、車椅子、ベビーカー、高齢者の歩行では通れる空間が変わる。NaviTrace、CATNAV、EgoPathBench が身体性の条件付けを扱い始めたが、歩行支援の文脈ではまだ少ない。
3. **VLM の幾何と時間の弱さ**: NaviTrace、EgoPathBench、GuideDog、mmWalk、EgoBlind が一貫して、空間的接地・奥行き・ナビゲーション判断の弱さを報告している。幾何モジュールとの統合と、不確かさの校正 (ZeST) が鍵になる。
4. **不確かさの校正と誤警報**: 歩行支援では、誤警報が多いとユーザーが警告を信用しなくなる。conformal prediction などで校正した警告閾値と、言語出力の冗長さの抑制 (WalkVLM-LR) を合わせて設計する必要がある。
5. **自己の意図の推定**: 意図経路が不明だと、経路の衝突を定義できない。視線 (EgoCogNav)、GPS やルート (AI Guide Dog)、頭部の向き (LookOut) を統合する研究が必要。
6. **一人称の将来占有予測**: 車載の 4D 占有予測 (Cam4DOcc, OccWorld) の手法を、手ぶれの大きい頭部や胸部の装着カメラに移すことは未開拓。
7. **自己教師ありの危険ラベル**: 実際の回避・停止をラベルにする方法 (BADGR 型) を歩行者に当てはめるには、意図的な停止 (信号待ちなど) と危険による停止を区別する必要がある。
8. **リアルタイム性と計算量**: スマートフォンやウェアラブルでの実行 (AI Guide Dog、HEADS-UP の Jetson 実装、CATNAV のキャッシュ) が求められる。

---

## 8. 参考文献

### 歩行可能性・traversability
1. Park, H. S., Hwang, J.-J., Niu, Y., Shi, J. "Egocentric Future Localization." CVPR 2016. https://openaccess.thecvf.com/content_cvpr_2016/papers/Park_Egocentric_Future_Localization_CVPR_2016_paper.pdf
2. Hirose, N., et al. "GONet: A Semi-Supervised Deep Learning Approach For Traversability Estimation." 2018. https://arxiv.org/abs/1803.03254
3. Kahn, G., Abbeel, P., Levine, S. "BADGR: An Autonomous Self-Supervised Learning-Based Navigation System." IEEE RA-L 6(2), 2021. https://arxiv.org/abs/2002.05700
4. Frey, J., Mattamala, M., et al. "Fast Traversability Estimation for Wild Visual Navigation." RSS 2023. https://arxiv.org/abs/2305.08510 (拡張版: https://arxiv.org/abs/2404.07110)
5. Jung, S., et al. "V-STRONG: Visual Self-Supervised Traversability Learning for Off-road Navigation." ICRA 2024. https://arxiv.org/abs/2312.16016
6. Sevastopoulos, C., et al. "A Survey of Traversability Estimation for Mobile Robots." arXiv 2022. https://arxiv.org/abs/2204.10883
7. Waghmare, S. M., et al. "SANPO: A Scene Understanding, Accessibility and Human Navigation Dataset." arXiv 2023. https://arxiv.org/abs/2309.12172
8. Sultan, R. I., et al. "WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation." CVPR 2026. https://arxiv.org/abs/2603.10703

### 将来の freespace・占有の予測
9. Hu, P., Huang, A., Dolan, J., Held, D., Ramanan, D. "Safe Local Motion Planning with Self-Supervised Freespace Forecasting." CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Hu_Safe_Local_Motion_Planning_With_Self-Supervised_Freespace_Forecasting_CVPR_2021_paper.html
10. Khurana, T., et al. "Differentiable Raycasting for Self-supervised Occupancy Forecasting." ECCV 2022. https://arxiv.org/abs/2210.01917
11. Khurana, T., et al. "Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting." CVPR 2023. https://arxiv.org/abs/2302.13130
12. Hu, A., et al. "FIERY: Future Instance Prediction in Bird's-Eye View from Surround Monocular Cameras." ICCV 2021. https://arxiv.org/abs/2104.10490
13. Hu, Y., et al. "Planning-oriented Autonomous Driving (UniAD)." CVPR 2023. https://arxiv.org/abs/2212.10156
14. Ma, J., et al. "Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications." CVPR 2024. https://arxiv.org/abs/2311.17663
15. Zheng, W., et al. "OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving." ECCV 2024. https://arxiv.org/abs/2311.16038
16. Bar, A., et al. "Navigation World Models." CVPR 2025. https://arxiv.org/abs/2412.03572

### 一人称の経路予測・衝突・TTC
17. Yagi, T., et al. "Future Person Localization in First-Person Videos." CVPR 2018. https://arxiv.org/abs/1711.11217
18. Manglik, A., Weng, X., Ohn-Bar, E., Kitani, K. M. "Forecasting Time-to-Collision from Monocular Video: Feasibility, Dataset, and Challenges." IROS 2019. https://arxiv.org/abs/1903.09102
19. Badki, A., Gallo, O., Kautz, J., Sen, P. "Binary TTC: A Temporal Geofence for Autonomous Navigation." CVPR 2021. https://arxiv.org/abs/2101.04777
20. Kahn, G., et al. "Uncertainty-Aware Reinforcement Learning for Collision Avoidance." arXiv 2017. https://arxiv.org/abs/1702.01182
21. Qiu, J., et al. "Egocentric Human Trajectory Forecasting With a Wearable Camera and Multi-Modal Fusion." IEEE RA-L 7(4), 2022. https://arxiv.org/abs/2111.00993
22. Nishimura, M., et al. "InCrowdFormer: On-Ground Pedestrian World Model From Egocentric Views." arXiv 2023. https://arxiv.org/abs/2303.09534
23. Wang, W., Liu, C. K., Kennedy III, M. "EgoNav: Egocentric Scene-aware Human Trajectory Prediction." arXiv 2024. https://arxiv.org/abs/2403.19026
24. Wang, W., Ze, Y., Liu, C. K., Kennedy III, M. "Learning Humanoid Navigation from Human Data." arXiv 2026. https://arxiv.org/abs/2604.00416
25. Haghighi, Y., et al. "HEADS-UP: Head-Mounted Egocentric Dataset for Trajectory Prediction in Blind Assistance Systems." arXiv 2024. https://arxiv.org/abs/2409.20324
26. Jadhav, A., et al. "AI Guide Dog: Egocentric Path Prediction on Smartphone." AAAI 2025 Spring Symposium. https://arxiv.org/abs/2501.07957
27. Pan, B., et al. "LookOut: Real-World Humanoid Egocentric Navigation." ICCV 2025. https://arxiv.org/abs/2508.14466
28. Qiu, Z., et al. "EgoCogNav: Cognition-aware Human Egocentric Navigation." arXiv 2025. https://arxiv.org/abs/2511.17581
29. Mangalam, K., et al. "From Goals, Waypoints & Paths To Long Term Human Trajectory Forecasting (Y-Net)." ICCV 2021. https://arxiv.org/abs/2012.01526
30. Bai, J., Shim, I. "SceneAware: Scene-Constrained Pedestrian Trajectory Prediction with LLM-Guided Walkability." arXiv 2025. https://arxiv.org/abs/2506.14144

### 一人称映像からのアフォーダンス学習
31. Nagarajan, T., Feichtenhofer, C., Grauman, K. "Grounded Human-Object Interaction Hotspots from Video." ICCV 2019. https://arxiv.org/abs/1812.04558
32. Nagarajan, T., et al. "EGO-TOPO: Environment Affordances from Egocentric Video." CVPR 2020. https://arxiv.org/abs/2001.04583
33. Liu, S., Tripathi, S., Majumdar, S., Wang, X. "Joint Hand Motion and Interaction Hotspots Prediction from Egocentric Videos." CVPR 2022. https://arxiv.org/abs/2204.01696
34. Li, Y., et al. "Egocentric Prediction of Action Target in 3D (EgoPAT3D)." CVPR 2022. https://arxiv.org/abs/2203.13116
35. Bahl, S., et al. "Affordances from Human Videos as a Versatile Representation for Robotics (VRB)." CVPR 2023. https://arxiv.org/abs/2304.08488
36. Grauman, K., et al. "Ego4D: Around the World in 3,000 Hours of Egocentric Video." CVPR 2022. https://arxiv.org/abs/2110.07058
37. Rodin, I., et al. "Predicting the Future from First Person (Egocentric) Vision: A Survey." CVIU 2021. https://arxiv.org/abs/2107.13411

### 視覚障害者支援
38. Kayukawa, S., et al. "BBeep: A Sonic Collision Avoidance System for Blind Travellers and Nearby Pedestrians." CHI 2019. https://doi.org/10.1145/3290605.3300282
39. Yuan, Z., et al. "WalkVLM: Aid Visually Impaired People Walking by Vision Language Model." arXiv 2024. https://arxiv.org/abs/2412.20903
40. "Less Redundancy: Boosting Practicality of Vision Language Model in Walking Assistants (WalkVLM-LR)." arXiv 2025. https://arxiv.org/abs/2508.16070
41. Xiao, J., et al. "EgoBlind: Towards Egocentric Visual Assistance for the Blind." arXiv 2025. https://arxiv.org/abs/2503.08221
42. Kim, J., et al. "GuideDog: A Real-World Egocentric Multimodal Dataset for Blind and Low-Vision Accessibility-Aware Guidance." ACL 2026. https://arxiv.org/abs/2503.12844
43. Ying, K., et al. "mmWalk: Towards Multi-modal Multi-view Walking Assistance." NeurIPS 2025 Datasets & Benchmarks. https://arxiv.org/abs/2510.11520
44. Thoma, M., et al. "PEDESTRIAN: An Egocentric Vision Dataset for Obstacle Detection on Pavements." arXiv 2025. https://arxiv.org/abs/2512.19190

### 社会的ナビゲーション・安全な計画
45. Karnan, H., et al. "Socially Compliant Navigation Dataset (SCAND)." IEEE RA-L 2022. https://arxiv.org/abs/2203.15041
46. Song, D., et al. "VLM-Social-Nav: Socially Aware Robot Navigation through Scoring using Vision-Language Models." arXiv 2024. https://arxiv.org/abs/2404.00210
47. Lindemann, L., et al. "Safe Planning in Dynamic Environments using Conformal Prediction." arXiv 2022. https://arxiv.org/abs/2210.10254

### ナビゲーション基盤モデル・VLM による navigability 推論
48. Shah, D., et al. "ViNT: A Foundation Model for Visual Navigation." CoRL 2023. https://arxiv.org/abs/2306.14846
49. Sridhar, A., et al. "NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration." arXiv 2023 (ICRA 2024). https://arxiv.org/abs/2310.07896
50. Liu, X., et al. "CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos." CVPR 2025. https://arxiv.org/abs/2411.17820
51. Akhtyamov, T., et al. "EgoWalk: A Multimodal Dataset for Robot Navigation in the Wild." arXiv 2025. https://arxiv.org/abs/2505.21282
52. Sathyamoorthy, A. J., et al. "CoNVOI: Context-aware Navigation using Vision Language Models in Outdoor and Indoor Environments." arXiv 2024. https://arxiv.org/abs/2403.15637
53. Gummadi, S., et al. "ZeST: A VLM-based Zero-Shot Traversability Navigation for Unknown Environments." arXiv 2025 (v4 2026). https://arxiv.org/abs/2508.19131
54. Potnis, et al. "CATNAV: Cached Vision-Language Traversability for Efficient Zero-Shot Robot Navigation." arXiv 2026. https://arxiv.org/abs/2603.22800
55. Windecker, T., et al. "NaviTrace: Evaluating Embodied Navigation of Vision-Language Models." ICRA 2026. https://arxiv.org/abs/2510.26909
56. Zhao, Y., Chen, Z., Yang, X. "EgoPathBench: Evaluating Zero-Shot Egocentric Waypoint Decision-Making in Vision-Language Models." arXiv 2026. https://arxiv.org/abs/2609.16610

### 対比のためだけに参照 (スコープ外)
57. Zhao, T., et al. "Accident Anticipation via Temporal Occurrence Prediction (TOP)." NeurIPS 2025. https://arxiv.org/abs/2510.22260 (シーン中心の事故予測の例。https://happytianhao.github.io/TOP/)

---

### 付記: papers.md の記載のうち存在を確認できなかったもの

- **"Ego-Alert: Egocentric Future Hazard Forecasting" (ICCV 2021)**: 該当する論文は見つからなかった。
- **"EgoVis: A Benchmark for Visual Hazard Detection in Egocentric Video"**: 該当するベンチマーク論文は見つからなかった。"EgoVis" は CVPR の一人称視覚ワークショップの名称 (https://egovis.github.io/cvpr25/) である。
- **"EgoPATH: Forecasting the Egocentric Path of the Camera Wearer"**: 該当する論文は見つからなかった。内容が近い実在の研究は Qiu et al. (RA-L 2022)、EgoNav (2024)、AI Guide Dog (2025)、LookOut (ICCV 2025)、EgoPathBench (2026; ただしこれは VLM 評価ベンチマーク)。
- **"Learning to Anticipate Traffic Accidents from Egocentric Video" (ICCV 2019)**: この題名・発表先では確認できなかった。近い実在の研究は Yao et al., "Unsupervised Traffic Accident Detection in First-Person Videos", IROS 2019 (https://arxiv.org/abs/1903.00618)。シーン中心の研究なので本サーベイでは扱わない。
- 参考: "Anticipating Accidents in Dashcam Videos" は ECCV 2016 ではなく ACCV 2016 (Chan et al.)。"Forecasting Human-Object Interaction ... in Egocentric Video" と "Ego4D Forecasting Benchmarks" は、それぞれ Liu et al. CVPR 2022 [33] と Ego4D [36] として扱った。
