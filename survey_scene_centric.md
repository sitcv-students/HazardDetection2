# シーン中心の危険予測（Scene-centric Hazard / Accident Anticipation）サーベイ ― 2016〜2026

作成日: 2026-11-11 ／ 対象: 映像全体（フレーム・クリップ・動画）単位で「近い将来に危険事象が起きるか／いつ起きるか」を推定する研究
背景: エゴセントリック映像を用いた歩行支援システムのハザード検出

> **検証方針**: 本稿で引用する文献は、すべて arXiv API・arXiv 抄録ページ・Crossref・CVF Open Access・手元 PDF のいずれかで、タイトル・第一著者・会議/論文誌・年を確認した。性能値は、原著 PDF または原著抄録で確認できたものだけを載せている。会議採録は arXiv の comment 欄や出版社 DOI で確認できたものだけを書き、確認できないものは「arXiv」と書いた。

---

## 1. 定義とスコープ

### 1.1 定義
**シーン中心の危険予測**は、観測済みの映像 $x_{1:t}$ から、**シーン全体についての危険度スカラー（またはその時系列）** を出力するタスクである。

- 代表的な出力: フレームごとの事故発生確率 $p_t \in [0,1]$（risk score / accident probability curve）
- 拡張された出力:
  - 将来の複数時刻での発生確率 $p(\text{accident at } t+\Delta)$（例: TOP [27]）
  - 衝突までの時間（Time-to-Collision / Time-to-Accident）の回帰（例: Manglik ら [40]）
  - 警報行動（SILENT/OBSERVE/ALERT）の方策（例: VLAlert [36]）

いずれも「**危険がいつ、どの程度起きそうか**」に答える。「**どこが危険か**」には答えない。

### 1.2 スコープ外（対比のみ）
危険の**場所**を推定する研究は別系統として扱う（別稿 `survey_risk_region.md` の対象）。具体的には、危険物体の同定（risky object localization）、リスクヒートマップ、危険領域のセグメンテーションである。
- 例: Agent-Centric Risk Assessment [4] は事故予測と危険領域の位置推定を同時に行う。DRIVE [7] は注視（fixation）による視覚的説明を出す。これらは位置推定の要素を持つが、**シーン単位の確率曲線も出す**ので、その部分だけを本稿で扱う。
- 事故**検出**（事故が起きた後の時間的局在化。例: A3D/DoTA のビデオ異常検出 [12][13]、ACCIDENT ベンチマーク [33]）も厳密には対象外である。ただし、データセットを共有しているので必要な範囲で触れる。

### 1.3 なぜシーン中心か
- 警報（「止まれ」「注意」）は本質的に**スカラーの意思決定**であり、その時間的な早さと誤報率（False Alarm Rate）が実用性を左右する。
- 物体検出に依存しない大域特徴型（VAGNet [30]、BADAS [31]、FLaRA [35]、VideoMAE-v2 型 [34]）が近年主流化している。これらは検出器の失敗（見切れ、遮蔽、未知クラス）に頑健であり、歩行者視点への移植にも向く。

---

## 2. 問題設定・評価指標

### 2.1 標準的な問題設定（DSA [1] 以来）
- 入力: ドライブレコーダー映像。事故動画（positive）と非事故動画（negative）の短いクリップで、DAD なら 100 フレーム・5 秒 [1][5]。
- 出力: 各フレーム $t$ の事故確率 $a_t$。閾値 $\tau$ を初めて超えた時刻 $t_\tau$ を警報時刻とする。
- 学習:
  - DSA [1] は positive の各フレームの損失に、事故時刻 $y$ に近いほど大きくなる**指数重み**を掛けた交差エントロピーを導入した（anticipation loss）。
  - AdaLEA [2] はこの重みを学習の進行に応じて適応的に変える。
  - DSTA [6]・UString [5] など多くの後続研究がこの枠組みを継承している。
- 問題点: 近年 [27][28][29] は、「事故動画の全フレームを positive とする**二値ラベル**」が曖昧な教師になり、誤報の原因になると指摘している（§4.2 参照）。

### 2.2 評価指標

| 指標 | 定義 | 由来・主な使用 |
|---|---|---|
| **AP**（Average Precision） | 閾値 $\tau$ を動かして得られる（動画単位の）Precision–Recall 曲線の面積 | DSA [1] 以降の標準 |
| **TTA**（Time-to-Accident） | positive 動画で、事故時刻 $y$ と初回警報時刻の差 $y - t_\tau$ | DSA [1] |
| **mTTA** | 全閾値 $\tau$ にわたる TTA の平均 | DSA [1], UString [5] |
| **TTA@R80**（TTA@0.8） | Recall が 80% となる閾値での TTA | UString [5] で使用（TTA@0.8）。DSTA [6], DRIVE [7] も報告 |
| ATTC | 平均 time-to-collision。AdaLEA [2] で用いられた | AdaLEA [2] |
| **AUC**（ROC） | フレーム単位またはクリップ単位の ROC-AUC | CAP [9], TOP [27], BADAS [31] |
| **FAR 制約付き Recall / TTA / mAUC** | FAR ≤ λ（例: 10%）の範囲内でだけ Recall と TTA を集計し、事故の 0.5 / 1.0 / 1.5 秒前の区間別にも評価 | TOP [27]（NeurIPS 2025） |
| mAP @ Δ（time-to-event 別） | 事故/ニアミスの 500 / 1000 / 1500 ms 前での mAP | Nexar Challenge [26] |
| **Separation Score** | 予測リスク曲線全体の大域挙動を評価。異常開始前の早すぎる陽性を罰し、開始後の高リスクを報いる | PRE-ACT [29] |

**指標上の注意**
1. AP と mTTA はトレードオフであり、論文によって「AP 最大化点」「mTTA 最大化点」など報告の作法が異なる。数値を論文間で直接比べるのは危険である。
2. mTTA は $\tau$ をすべて平均するため、実用不能なほど誤報の多い閾値まで含んでしまう。TOP [27] はこれを批判し、FAR 制約付きの評価を提案した。
3. 自車が関与しない事故（他車同士の事故）も positive に含むデータセット（DAD/CCD など）では、「自車への脅威」と評価が一致しない。BADAS [31] は主要データセットを ego-involvement で再アノテーションしてこれを是正した。

---

## 3. データセット

| データセット | 年・出典 | 視点 | 規模（原著で確認した値） | 主なアノテーション | 主な用途 |
|---|---|---|---|---|---|
| **DAD**（Dashcam Accident Dataset / Street Accident） | 2016, Chan ら ACCV [1] | 車載 | 事故 620 本・通常 1,130 本。各 100 フレーム（5 秒）、事故は最後の 10 フレーム内。学習 1,284 / テスト 466 本 [5] | 事故時刻、物体 bbox | 予測の標準ベンチマーク |
| **NIDB**（Near-miss Incident DB） | 2018, Suzuki ら CVPR [2] | 車載 | 大規模ニアミス（原著参照） | cyclist/pedestrian/vehicle/background の 4 クラス | risk / risk-factor anticipation |
| **A3D**（AnAn Accident Detection） | 2019, Yao ら IROS [12] | 車載 | 1,500 本（すべて事故を含む）[5] | 事故区間 | 本来は検出用。予測にも流用 |
| **CCD**（Car Crash Dataset） | 2020, Bao ら ACM MM [5] | 車載 | 事故 1,500 本 + BDD100K 由来の通常 3,000 本。各 5 秒。事故開始時刻は最後の 2 秒内にランダム配置 | 環境属性（昼夜・天候）、参加者 bbox、事故原因 | 予測の標準ベンチマーク |
| **DADA-2000** | 2019, Fang ら ITSC [10] / T-ITS 2022 [11] | 車載 | 2,000 本（658,476 フレーム）[27] | ドライバー注視、事故カテゴリ、異常出現・事故時刻 | 注視誘導型の予測 |
| **DoTA**（Detection of Traffic Anomaly） | 2020, Yao ら arXiv [13] | 車載 | 4,677 本 | 時間・空間・カテゴリ | 異常検出（BADAS [31] は予測評価にも使用） |
| **CAP**（Cognitive Accident Prediction） | 2022, Fang ら [9] | 車載 | 11,727 本（2,195,613 フレーム）[27] | テキスト記述、注視、異常出現・事故時刻 | MM-AU の予測サブセット |
| **MM-AU** | 2024, Fang ら CVPR [14] | 車載 | 実環境のエゴ視点事故動画 11,727 本、58 カテゴリ、物体 bbox 223 万、事故理由 QA 58.6K 組 | 事故理由・防止策・カテゴリのテキスト | 予測・理由推定・動画生成。CAP と DADA を内包 |
| **Nexar Collision Prediction** | 2025, Moura ら CVPRW [26] | 車載（実ユーザーのドラレコ） | 公開 1,500 本（各約 40 秒）。テスト 1,344 本（各約 10 秒） | 衝突/ニアミス、事故・警報時刻、天候・照明 | Kaggle 競技。mAP@Δ で評価 |
| **BADAS 再アノテーション** | 2025, Goldshmidt ら [31] | 車載 | DAD・DADA-2000・DoTA・Nexar を再注釈 | ego-involvement、合意警報時刻 | 「自車に対する」衝突予測評価 |
| **Near-Collision** | 2019, Manglik ら IROS [40] | **人が押すスーツケース型ロボット**（屋内・人ごみ） | 13,658 セグメント | LiDAR 由来の time-to-near-collision（〜6 秒） | 単眼 TTC 回帰。歩行者視点に最も近い |
| DHPR | 2023/2024, Charoenpitaks ら T-IV [21] | 車載・**静止画** | 15K 画像 | 速度、仮説ハザード記述、関与エンティティ | VLM によるハザード推論 |
| PotentialRiskQA | 2025, Liu ら arXiv [39] | 車載 | ― | 前兆（precursor）・リスク出現時刻、対処行動 | 前兆に基づく VLM 推論 |

補足:
- ACCIDENT（CVPR 2026 W）[33] は**監視カメラ**の事故検出（時間・位置・種類）であり、予測ではない。
- 同じ CVPR 2026 AUTOPILOT ワークショップでは、別に**ドラレコのゼロショット事故予測**競技も開かれた [34]。

---

## 4. 手法の系譜

### 4.1 系譜の概観

```
[第1世代: RNN + 物体注意 + 早期損失] 2016–2019
  DSA (ACCV'16) ─ 物体単位の動的空間注意 + LSTM + 指数重み付き anticipation loss
   ├ Agent-Centric Risk Assessment (CVPR'17) ─ 予測 + 危険領域（位置推定、スコープ外の要素）
   └ AdaLEA (CVPR'18) ─ 学習進行に応じた適応的早期損失、NIDB
[第2世代: 関係・グラフ・不確実性・強化学習] 2020–2022
  UString (MM'20) ─ GCN + RNN + ベイズNN、不確実性ランキング損失、CCD
  GFA / Fatima (ICPR'20) ─ 大域特徴集約
  DSTA (T-ITS'22) ─ 動的空間・時間注意 + GRU
  DRIVE (ICCV'21) ─ 強化学習、bottom-up/top-down 注意、説明可能性
  XAI-GRU (TRR) ─ Grad-CAM による事後説明
[第3世代: グラフ高度化・3D/深度・周波数・Transformer] 2023–2025
  GSC (T-IV) ─ 欠落エージェントを考慮した時空間連続グラフ
  DAA-GN (PR'24) / Graph(Graph) (WACV'24) / STAGNet (Access'25)
  AccNet (AAP'24) / Physical depth-aware (arXiv'25) ─ 単眼深度による 3D 化
  CRASH (MM'24) ─ 物体注意 + FFT による文脈 + 多層融合
  AAT-DA (WACVW'25) ─ ドライバー注意付き Transformer
  LATTE (Inf. Fusion) / MsFIN / Song & Lee (ICIP'25) ─ 軽量化・多スケール
[第4世代: 認知・言語・VLM/LLM] 2022–2026
  CAP / Cog-TAA (ITS Mag.'24) ─ テキスト記述 + ドライバー注視
  When-Where-What (MM'24) ─ LLM による言語警報
  CAMERA (MM'25), Dual-Branch, Mind-the-Gap (AAAI'26), VLAlert (CoRL'26)
[第5世代: 時間的定式化の見直し・基盤モデル・データ拡張] 2025–2026
  TOP (NeurIPS'25) ─ 将来複数時刻の発生確率 + FAR 制約評価
  RiskProp (CVPR'26) ─ 衝突時刻のみ使う自己教師リスク伝播 + 単調性制約
  PRE-ACT (NeurIPS'26) ─ 時間順序・事故距離を考慮した漸進リスク推定 + Separation Score
  BADAS / BADAS-2.0 ─ V-JEPA2 の端到端学習、自車関与、大規模実データ
  FLaRA (ITSC'26) ─ V-JEPA2 で将来潜在表現を予測
  VAGNet (OJVT'26), VideoMAE-v2 ゼロショット (CVPR'26 競技)
  頑健性: Predict & Resist (AAAI'26), ROAR (AAP), SECURE
  生成によるデータ拡張: EQ-TAA (TMM), World-model scene generation (Commun. Eng.'25)
```

### 4.2 主要な設計軸
1. **表現**: 物体ベース（検出器 + 物体特徴、DSA〜CRASH）か、大域ベース（フレーム/クリップ全体、VAGNet・BADAS・FLaRA）か。後者は検出器の計算コストと失敗モードを回避できる [30]。
2. **関係モデリング**: 注意機構（DSA, DSTA）→ GCN（UString, GSC, Graph(Graph), STAGNet）→ Transformer（AAT-DA, MsFIN, TOP のデコーダ）。
3. **時間的教師の設計**（2025〜2026 年の中心論点）:
   - 従来: 事故動画の全フレームを 1 とし、指数重みを掛ける [1][2][6]。
   - TOP [27]: 予測対象を「現在フレームのリスク」から「**0.1〜2.0 秒先の各時刻に事故が起きるか**」に変え、正確な事故時刻だけを教師にする。
   - RiskProp [28]: 主観的な「異常開始」アノテーションを廃止し、衝突フレームだけを錨（アンカー）にする。次フレームの予測をソフトターゲットにしてリスクを後方へ伝播させ、さらに適応的な単調性制約をかける。
   - PRE-ACT [29]: リスクを、事故に近づくほど増加する連続信号とみなす。時間順序の選好と事故までの距離を明示的に学習する。
4. **不確実性・意思決定**: ベイズ NN（UString）、強化学習（DRIVE、Predict & Resist の actor–critic [42]）、三値行動方策（VLAlert [36] の OBSERVE による証拠収集）。
5. **補助モダリティ**: ドライバー注視（DADA、CAP、Cog-TAA、AAT-DA、CAMERA、Mind-the-Gap）、テキスト（CAP、Dual-Branch、CAMERA）、深度（AccNet、Physical depth-aware）。
6. **データ**: 合成・生成による拡張（EQ-TAA の Attentive Video Diffusion、World-model scene generation、MM-AU の OAVD）と、大規模実データ（BADAS-2.0 は 178,500 本のラベル付き動画）。

### 4.3 比較表

| 手法 | 年・会議/誌 | 入力表現 | 時間モデル | 教師・損失の工夫 | 評価データ（原著記載） | 出力 |
|---|---|---|---|---|---|---|
| DSA [1] | 2016 ACCV | 物体 + フレーム | LSTM + 空間注意 | 指数重み anticipation loss | DAD | $a_t$ |
| Agent-Centric [4] | 2017 CVPR | 物体（エージェント） | RNN + soft attention | ― | Epic Fail, Street Accident | $a_t$ + 危険領域 |
| AdaLEA [2] | 2018 CVPR | フレーム | RNN 系（原著参照） | 適応的早期損失 | NIDB, DAD | $a_t$ |
| UString [5] | 2020 ACM MM | 物体グラフ | GCN + RNN（ベイズ） | 不確実性ランキング損失 | DAD, A3D, CCD | $a_t$ + 不確実性 |
| GFA [8] | 2020 ICPR | 大域 + 物体 | 特徴集約 + RNN | ― | DAD 等 | $a_t$ |
| DRIVE [7] | 2021 ICCV | フレーム + 注視 | 強化学習エージェント | 密な予測報酬 + 疎な注視報酬 | DAD, A3D, DADA-2000 | $a_t$ + 注視 |
| DSTA [6] | 2022 T-ITS | 物体 + フレーム | GRU + 動的空間/時間注意 | 指数重み | DAD, CCD, A3D | $a_t$ |
| XAI-GRU [19] | TRR（arXiv 2021） | 物体関係 | GRU | Grad-CAM（事後説明） | DAD 等 | $a_t$ + saliency |
| CAP [9] | 2022 arXiv | フレーム + テキスト + 注視 | 注意融合 | テキスト→視覚シフト融合 | CAP/MM-AU | $a_t$ |
| GSC [15] | 2024 T-IV（2023 早期公開） | 物体グラフ | GCN（時空間連続性） | 新しい隣接行列 | DAD 等 | $a_t$ |
| DAA-GN [16] | 2024 Pattern Recognit. | 物体グラフ | 動的注意拡張グラフ | ― | （原著参照） | $a_t$ |
| Graph(Graph) [17] | 2024 WACV | 入れ子グラフ（フレーム内・フレーム間） | GNN | ― | DAD 等 | $a_t$ |
| Cog-TAA [18] | 2024 IEEE ITS Mag. | フレーム + テキスト + 注視 | 注意 | ― | DADA-2000 拡張 | $a_t$ |
| CRASH [22] | 2024 ACM MM | 物体 + 大域（FFT） | 多層融合 | ― | DAD, CCD, A3D | $a_t$ |
| When-Where-What [23] | 2024 ACM MM | 物体 + LLM | chain-based attention | ― | DAD, CCD, A3D | $a_t$ + 言語警報 |
| AccNet [24] | 2024 AAP | 物体 + 単眼深度 | ― | BA-LEA | DAD, CCD, A3D, DADA-2000 | $a_t$ |
| AAT-DA [20] | 2025 WACVW | フレーム + ドライバー注意 | Transformer | ― | DAD, CCD | $a_t$ |
| LATTE [43] | 2025 Inf. Fusion | 多スケール空間 | Memory Attention | VLM 警報 | DAD, CCD, A3D | $a_t$ + 言語警報 |
| CAMERA [25] | 2025 ACM MM | 映像 + テキスト + 注視 | Bi-GRU | 適応閾値（シーン複雑度・注視エントロピー） | DADA-2000 等 | $a_t$ + 言語警報 |
| **TOP** [27] | **2025 NeurIPS** | スニペット（大域） | Transformer デコーダ（時刻クエリ） | **将来複数時刻（0.1〜2.0 秒）の発生確率** | CAP, DADA（MM-AU） | $p(t+\Delta)$ |
| BADAS [31] | 2025 arXiv | 大域（V-JEPA2） | 端到端 | 自車関与ラベル | DAD, DADA-2000, DoTA, Nexar | $a_t$ |
| Predict & Resist [42] | 2026 AAAI | 物体 + 画像（拡散デノイズ） | actor–critic | 時間重み付き報酬 | DAD, CCD, A3D | 警報時刻 |
| Mind-the-Gap [44] | 2026 AAAI | フレーム + 人間注意整合 | ― | 人間–AI 注意の整合 | （原著参照） | $a_t$ |
| **RiskProp** [28] | **2026 CVPR** | 大域 | ― | 衝突アンカー型自己教師伝播 + 単調性制約 | CAP, Nexar | $a_t$ |
| VAGNet [30] | 2026 IEEE OJVT | 大域（VideoMAE-V2） | Transformer + グラフ | 物体特徴不要 | （原著参照） | $a_t$ |
| FLaRA [35] | 2026 ITSC | 大域（V-JEPA2） | 将来潜在予測 | 特徴再構成 + CE | Nexar ほか | $a_t$ |
| VLAlert [36] | 2026 CoRL | VLM（Qwen3-VL-4B） | 逐次決定 | SILENT/OBSERVE/ALERT 方策 | VLAlert-Bench | 警報行動 |
| **PRE-ACT** [29] | **2026 NeurIPS**（arXiv 記載） | 大域（VideoMAE 系） | ― | 時間順序・事故距離の選好学習 | CAP, DADA, Nexar | $a_t$（連続リスク） |

---

## 5. 主要論文の要約

### 5.1 基礎（2016–2018）
- **DSA — Chan et al., "Anticipating Accidents in Dashcam Videos", ACCV 2016 [1]**
  - 事故予測というタスクと DAD データセットを確立した。
  - 検出した物体に動的な空間注意をかけ、LSTM で逐次的に事故確率を出す。
  - 事故時刻に近いほど重い指数重み付き損失を導入した。AP と mTTA の評価も本論文が起点である。
  - なお papers.md はこの論文を「ECCV 2016」と記しているが誤りで、正しくは ACCV 2016（LNCS 10114, DOI 10.1007/978-3-319-54190-7_9）である。
- **Agent-Centric Risk Assessment — Zeng et al., CVPR 2017 [4]**
  - soft attention RNN で事故予測と危険領域の位置推定を同時に行う。Epic Fail データセットを導入した。
  - 位置推定の部分はスコープ外である。
- **AdaLEA — Suzuki et al., CVPR 2018 [2]**
  - エポックごとに「どれだけ早く予測できているか」を見て損失の重みを調整し、学習を通じて徐々に早期予測を学ばせる。
  - ニアミス DB（NIDB）も公開した。

### 5.2 関係・不確実性・強化学習（2020–2022）
- **UString — Bao et al., ACM MM 2020 [5]**
  - GCN と RNN で物体間の時空間関係を学習し、ベイズ NN で潜在関係表現の不確実性を扱う。
  - 不確実性ランキング損失を導入し、CCD データセットを公開した。
  - TTA@0.8（TTA@R80）を AP・mTTA とともに報告し、以後の標準指標（AP / mTTA / TTA@R80）を定めた。
- **DRIVE — Bao et al., ICCV 2021 [7]**
  - 事故予測を、注視しながら判断する確率的マルチタスク強化学習エージェントとして定式化した。
  - 密な予測報酬と疎な注視報酬で学習する。視覚的説明（注視マップ）も出力する。
- **DSTA — Karim et al., IEEE T-ITS 2022 [6]**
  - 動的空間注意（物体の選択）と動的時間注意（過去フレームの選択）を GRU と組み合わせた。
  - DAD、CCD、A3D で評価している。
- **GFA — Fatima et al., ICPR 2020 [8]** は大域特徴の集約、**XAI-GRU — Karim et al., TRR [19]** は Grad-CAM による事後説明を提案した。

### 5.3 グラフ・深度・Transformer（2023–2025）
- **GSC — Wang et al., IEEE T-IV (vol.9 no.1, 2024; 2023 早期公開) [15]**
  - 検出の途切れ（欠落エージェント）があってもエージェントの時空間連続性を保つ GCN を提案した。
  - 後続研究の主要ベースラインになっている。
- **Graph(Graph) — Thakur et al., WACV 2024 [17]**: フレーム内とフレーム間の物体相互作用に、大域特徴を加えた入れ子グラフで表現する。
- **DAA-GN — Song et al., Pattern Recognition 2024 [16]**: 動的注意で拡張したグラフネットワーク。
- **STAGNet — Vipulananthan et al., IEEE Access 2025 [45]**: 時空間グラフと LSTM を組み合わせる。
- **AccNet — Liao et al., Accident Analysis & Prevention 2024 [24]**: 単眼深度による 3D シーンモデリングと BA-LEA 損失。**Physical depth-aware — Huang et al., arXiv 2025 [46]** は Depth-Anything の深度特徴を取り込む。
- **CRASH — Liao et al., ACM MM 2024 [22]**
  - 物体注意モジュールと、FFT で周波数領域に拡張した文脈モジュールを、多層融合で統合する。
  - 欠損データ（-missing 設定）での頑健性も評価している。
- **AAT-DA — Kumamoto et al., WACV Workshops 2025 [20]**: ドライバー注意を取り込んだ Transformer。Video Swin や UniformerV2 と比較している。
- 軽量化: **LATTE [43]**（Information Fusion 採録と arXiv に記載）、**Song & Lee, ICIP 2025 [47]**（特徴再利用によるリアルタイム化）、**MsFIN [48]**（多スケール + Transformer、arXiv 2025）。

### 5.4 認知・言語・VLM/LLM 系（2022–2026）
- **CAP — Fang et al., arXiv 2022 [9]**
  - テキスト記述（シーンの意味）とドライバー注視（重要領域）で学習を誘導するマルチモーダル予測。
  - MM-AU の予測サブセット「CAP」の起点でもある。
- **Cog-TAA — Li, Fang, Xue, IEEE ITS Magazine 2024 [18]**: CAP 系の手法。DADA-2000 を拡張して学習する。
- **MM-AU / AdVersa-SD — Fang et al., CVPR 2024 [14]**
  - 11,727 本の事故動画と事故理由・防止策テキストからなるデータセット。
  - abductive CLIP と物体中心動画拡散（OAVD）で、事故の因果連鎖を学習する。
- **When, Where, and What? — Liao et al., ACM MM 2024 [23]**: LLM で「いつ・どこで・何が」を推定し、言語警報を生成する。
- **CAMERA — Zhang et al., ACM MM 2025 [25]**: シーン複雑度と注視エントロピーで警報閾値を適応的に変え、誤報を減らす。
- **Dual-Branch — Guan et al., arXiv 2025 [37]**: GPT-4o と Long-CLIP を用い、事故報告書テキストと映像を融合する。
- **Mind the Gap — Ryu & Wallraven, AAAI 2026 [44]**: 人間と AI の視覚注意の差を定量化し、整合させる。
- **VLAlert（Observe Before You Alert）— Wang et al., CoRL 2026 [36]**
  - 警報生成を SILENT / OBSERVE / ALERT の三値方策として扱う。OBSERVE は、不確かなときに警報を遅らせて追加の証拠を集める内部行動である。
  - Qwen3-VL-4B を安全エビデンス生成器として使う。
  - 「確率曲線を閾値で切る」という従来の枠組みを、逐次意思決定に置き換える試みである。
- 対比（静止画・推論系）: **DHPR [21]**、**PotentialRiskQA / PR-Reasoner [39]**、**ScVLM [38]** は VLM によるハザード推論である。いずれもフレーム単位の確率曲線は出さない。

### 5.5 時間的定式化の見直し（2025–2026、最重要）
- **TOP — Zhao et al., "Accident Anticipation via Temporal Occurrence Prediction", NeurIPS 2025 [27]**
  - 予測対象を、現在フレームのリスクから**将来の複数時刻（0.1〜2.0 秒先）での事故スコア**に変えた。
  - 構成: スニペットエンコーダと、時刻ごとのクエリを持つ Transformer 時間デコーダ。
  - 評価: **FAR が許容範囲（λ）内のときだけ** Recall と TTA を集計する新しいプロトコルを提案し、事故の 0.5 / 1.0 / 1.5 秒前の区間別にも評価する。
  - データ: MM-AU の CAP と DADA を 10 FPS で使用。
  - 原著値（λ=0.1、FAR ≤ 10%）: mAUC は CAP 0.4290、DADA 0.3315。0.5 秒前の AUC は CAP 0.6747、DADA 0.5669。
- **RiskProp — Zou et al., CVPR 2026 [28]**
  - 主観的で揺れの大きい「異常開始」ラベルを使わず、確実な衝突フレームだけを教師にする。
  - 損失: 未来フレームの予測をソフトターゲットとする正則化と、適応的単調性制約。
  - CAP と Nexar で SOTA を報告し、より滑らかなリスク曲線が得られるとしている。
- **PRE-ACT — Hicsonmez et al., arXiv 2026（NeurIPS 2026 採録と記載）[29]**
  - リスクを、事故に近づくにつれて単調に高まる連続信号としてモデル化する。時間順序の選好と事故までの距離を学習する。
  - 原著主張: CAP で mean AUC +10% 超、DADA で約 +30%、警報がそれぞれ 0.25 秒・0.18 秒早い、Nexar Challenge の優勝解を +1 mAP 上回る。
  - mAUC・mTTA・mAP は局所的な評価窓しか見ないとして、**Separation Score** を提案した。

### 5.6 基盤モデル・実データ・頑健性（2025–2026）
- **Nexar Dataset & Challenge — Moura et al., CVPR Workshops 2025 [26]**: 実ユーザーのドラレコから集めた衝突・ニアミス 1,500 本と、Kaggle 競技。
- **BADAS — Goldshmidt et al., arXiv 2025 [31]**
  - 「自車への脅威」と「自車と無関係な事故」を区別し、主要ベンチマークを再アノテーションした。
  - V-JEPA2 を端到端で学習する。BADAS-Open は公開 1.5k 本、BADAS1.0 は非公開 40k 本で学習した。
- **BADAS-2.0 [32]**
  - 10 グループのロングテール評価セットを構築し、学習データを 178,500 本に拡大した。
  - エッジ向けに蒸留した Flash（86M）と Flash-Lite（22M）も提供する。
- **FLaRA — Caselli et al., ITSC 2026 [35]**: V-JEPA2 で**将来の潜在表現を予測**し、その予測表現の上で事故を分類する。
- **VAGNet — Vipulananthan et al., IEEE OJVT 2026 [30]**: VideoMAE-V2 の大域特徴に Transformer とグラフを組み合わせ、物体特徴なしで予測する。
- **VideoMAE-v2 ゼロショット予測 — Li et al., arXiv 2026 [34]**: CVPR 2026 AUTOPILOT のゼロショット事故予測競技で 2 位。
- 頑健性:
  - **Predict and Resist — Liu et al., AAAI 2026 [42]**: 拡散によるデノイズと時間を考慮した actor–critic。
  - **ROAR — Liu et al., AAP [49]**: 離散ウェーブレット変換（DWT）と dynamic focal loss。
  - **SECURE — Wang et al., arXiv 2026 [50]**: CRASH などが微小な摂動で不安定になることを示し、予測空間と潜在空間の安定性を学習で強制する。
- データ生成:
  - **EQ-TAA — Fang et al., IEEE TMM [51]**: 正常クリップから事故クリップを生成する拡散モデルと、等変三重項損失。
  - **World-model scene generation — Guan et al., Communications Engineering 2025 [52]**: 世界モデルで多様なリスクシナリオを生成し、新しいベンチマークも公開した。

---

## 6. 歩行者／エゴセントリック視点への適用と課題

### 6.1 現状: 歩行者視点の「シーン単位危険予測」は未開拓
調べた範囲では、**歩行者が装着したカメラの映像から、DAD/CCD 型のフレーム単位の危険確率曲線を予測するベンチマークや手法は確立していない**。近い研究は次の 4 系統に分かれる。

| 系統 | 代表研究 | シーン単位危険予測との関係 |
|---|---|---|
| (a) 単眼 TTC / ニアミス予測 | Manglik ら IROS 2019 [40]（Near-Collision、スーツケース型ロボット視点）。Urban ら J. Imaging 2021 [41]（スマートグラス想定の軽量 TTC 予測、視覚障害者ナビ） | **最も直接的**。シーン全体から衝突までの時間を回帰する |
| (b) 歩行支援 VLM・データセット | WalkVLM [53]（12,000 組の歩行支援動画注釈と時間適応的リマインダ）、mmWalk [54]（NeurIPS 2025 D&B、BLV 向け多視点歩行、リスク評価 VQA）、GuideDog [55]（ACL 2026）、VISTA [56]（Project Aria、ハザード検出タスクを含む）、PEDESTRIAN [57]（歩道障害物の検出） | 主に**現在の**危険の認識・説明。将来予測と時間指標はない |
| (c) エゴセントリック安全ベンチマーク | EgoSafe-Bench [58]（スマホ一人称、因果推論 QA）、EgoSafetyBench [59]（ロボット視点 1,200 シナリオ、0.5 秒粒度。VLM は危険を含む動画は分かっても「危険な瞬間」を見逃しやすいと報告） | 時間粒度のある安全判定。0.5 秒粒度の注釈は TTA 型評価へ転用できる |
| (d) 自己経路・行動予測 | Ego4D の Forecasting（locomotion）[60]、Qiu ら RA-L 2022 [61]（装着者の軌跡予測）、AI Guide Dog [62]、EgoPathBench [63]、エゴセントリック未来予測サーベイ [64] | 「進路と衝突するか」を判定するための前段 |

### 6.2 車載からの移植で何が変わるか
1. **自己運動の性質**: 歩行には上下動、頭部の回転、急な向き変更があり、車載とは大きく違う。DSA 型の物体追跡と大域特徴は手ぶれに弱い。EgoSafe-Bench [58] も手ぶれと低照度への頑健性を課題に挙げている。V-JEPA2 / VideoMAE 系の大域表現 [30][31][35] は検出器に依存しないので、移植の第一候補になる。
2. **危険の種類**: 車載は「衝突」がほぼ唯一の正例である。歩行では衝突（車・自転車・人・ポール）に加えて、段差や転倒、穴、階段、路面状態など「通行可能性の喪失」が加わる。TTC 回帰 [40][41] は前者にしか対応しない。
3. **自己関与（ego-involvement）**: BADAS [31] の指摘はそのまま当てはまる。歩行者視点では、遠くの事故より「**自分の進路上**」の脅威だけが警報に値する。(d) の自己経路予測との統合が要る。
4. **警報の時間スケール**: 歩行速度は 1〜1.5 m/s 程度と遅い一方、反応に必要な時間は人によって大きく異なる（視覚障害者・高齢者）。TOP [27] の「将来複数時刻での確率」と FAR 制約評価は、警報設計に直接使える。
5. **誤報の負担**: 歩行支援では誤報が利用者の信頼と移動効率を損なう。FAR 制約付き評価 [27]、Separation Score [29]、OBSERVE 行動 [36]、時間適応的リマインダ [53] は、いずれも誤報の抑制を狙っている。
6. **教師ラベル**:
   - 歩行者の事故・ニアミス映像はドラレコより桁違いに少ない。
   - RiskProp [28] のように「衝突（接触）時刻だけ」を錨にする自己教師方式は、アノテーションの負担を減らせるので歩行者データに向く。
   - Near-Collision [40] のように、LiDAR などの距離センサから TTC ラベルを自動生成する方式も有効である。
7. **データ生成**: EQ-TAA [51] や世界モデル [52] による事故映像の生成は、歩行者視点のニアミス生成に応用できる可能性がある（未検証）。

### 6.3 推奨される移植パイプライン（研究提案）
- **入力**: エゴセントリック RGB クリップ（＋可能なら IMU・深度）
- **バックボーン**: V-JEPA2 / VideoMAE-V2 系の凍結表現、または軽量蒸留版（BADAS-2.0-Flash [32] の考え方）
- **ヘッド**: TOP 型の多時刻発生確率 $p(\text{hazard at } t+\Delta)$（Δ = 0.5 / 1.0 / 1.5 / 2.0 秒）と、TTC 回帰ヘッドの併用
- **教師**: 接触時刻アンカー + 単調性制約（RiskProp）、または距離センサ由来の TTC（Near-Collision）
- **評価**: FAR 制約付きの Recall / TTA と Separation Score。ego-involvement で層別化する
- **説明**: VLM による短い言語警報（WalkVLM、LATTE の警報モジュール）

---

## 7. オープン問題と研究の方向

包括的な既存サーベイとしては Fang ら [3]（検出と予測の両方を扱う）があり、文献一覧としては [66] がある。

1. **時間的教師の正しさ**: 二値ラベル＋指数重み → 多時刻発生確率（TOP）→ 衝突アンカー自己教師（RiskProp）→ 漸進リスク（PRE-ACT）と移ってきたが、「真のリスク」の定義は未解決である。人間の判断（Mind the Gap [44]）や物理量（TTC）との整合も課題である。
2. **評価指標の再設計**: AP と mTTA の限界は明確になった [27][29]。FAR 制約、区間別 Recall、Separation Score、自車関与による層別化 [31] を組み合わせた統一プロトコルが必要である。
3. **ドメイン汎化とゼロショット**: 学習データにない地域・カメラへの汎化。CVPR 2026 のゼロショット競技 [34] がこの方向を示している。車載から歩行者への移植は、その極端なケースにあたる。
4. **ロングテールと実データの規模**: BADAS-2.0 [32] は、データの規模とロングテール対策の効果を示した。ただし学習データは非公開で、学術的な再現性に課題が残る。
5. **頑健性**: センサ劣化 [42]、欠損データ [22][49]、敵対的摂動 [50]。ウェアラブル機器では、手ぶれ・露出変化・低フレームレートが加わる。
6. **VLM/LLM の位置づけ**: 現状の VLM は、事故予測の確率曲線よりも説明と警報文の生成に使われることが多い。VLAlert [36] のように VLM を逐次意思決定に組み込む方向が出てきている。一方 EgoSafetyBench [59] は、VLM が「危険な瞬間」の時間的局在に弱いことを示している。
7. **予測的表現学習**: FLaRA [35] のように将来の潜在表現を予測する世界モデル型の手法は、予測時間を伸ばすうえで有望である。
8. **リアルタイム・省電力**: 車載でも LATTE [43] や Song & Lee [47] が軽量化を進めている。スマートフォンやスマートグラスではさらに厳しい制約がかかる。
9. **歩行者視点のベンチマーク不在**: 予測型の時間注釈（ハザード出現時刻・接触時刻）を持つエゴセントリック歩行データセットの構築が、最優先の空白である。

---

## 8. 参考文献

### 交通事故予測（ドラレコ）
1. F.-H. Chan, Y.-T. Chen, Y. Xiang, M. Sun. "Anticipating Accidents in Dashcam Videos." ACCV 2016 (LNCS). DOI: https://doi.org/10.1007/978-3-319-54190-7_9 ／ Project: http://aliensunmin.github.io/project/dashcam/
2. T. Suzuki, H. Kataoka, Y. Aoki, Y. Satoh. "Anticipating Traffic Accidents with Adaptive Loss and Large-scale Incident DB." CVPR 2018. https://arxiv.org/abs/1804.02675 ／ https://openaccess.thecvf.com/content_cvpr_2018/html/Suzuki_Anticipating_Traffic_Accidents_CVPR_2018_paper.html
3. J. Fang ほか. "Vision-Based Traffic Accident Detection and Anticipation: A Survey." IEEE TCSVT. https://arxiv.org/abs/2308.15985
4. K.-H. Zeng, S.-H. Chou, F.-H. Chan, J. C. Niebles, M. Sun. "Agent-Centric Risk Assessment: Accident Anticipation and Risky Region Localization." CVPR 2017. https://arxiv.org/abs/1705.06560
5. W. Bao, Q. Yu, Y. Kong. "Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning." ACM MM 2020. https://arxiv.org/abs/2008.00334 ／ DOI: https://doi.org/10.1145/3394171.3413827
6. M. M. Karim, Y. Li, R. Qin, Z. Yin. "A Dynamic Spatial-temporal Attention Network for Early Anticipation of Traffic Accidents." IEEE T-ITS 23(7), 2022. https://arxiv.org/abs/2106.10197
7. W. Bao, Q. Yu, Y. Kong. "DRIVE: Deep Reinforced Accident Anticipation with Visual Explanation." ICCV 2021. https://arxiv.org/abs/2107.10189
8. M. Fatima, M. U. K. Khan, C.-M. Kyung. "Global Feature Aggregation for Accident Anticipation." ICPR 2020. https://arxiv.org/abs/2006.08942
9. J. Fang, L.-L. Li, K. Yang, Z. Zheng, J. Xue, T.-S. Chua. "Cognitive Accident Prediction in Driving Scenes: A Multimodality Benchmark." arXiv 2022. https://arxiv.org/abs/2212.09381
10. J. Fang ほか. "DADA-2000: Can Driving Accident be Predicted by Driver Attention? Analyzed by A Benchmark." IEEE ITSC 2019. DOI: https://doi.org/10.1109/ITSC.2019.8917218
11. J. Fang, D. Yan, J. Qiao, J. Xue, H. Yu. "DADA: Driver Attention Prediction in Driving Accident Scenarios." IEEE T-ITS 23(6), 2022. https://arxiv.org/abs/1912.12148
12. Y. Yao ほか. "Unsupervised Traffic Accident Detection in First-Person Videos." IROS 2019（A3D）. https://arxiv.org/abs/1903.00618
13. Y. Yao ほか. "When, Where, and What? A New Dataset for Anomaly Detection in Driving Videos."（DoTA）arXiv 2020. https://arxiv.org/abs/2004.03044
14. J. Fang, L.-L. Li, J. Zhou ほか. "Abductive Ego-View Accident Video Understanding for Safe Driving Perception."（MM-AU）CVPR 2024. https://arxiv.org/abs/2403.00436 ／ http://www.lotvsmmau.net
15. T. Wang, K. Chen, G. Chen ほか. "GSC: A Graph and Spatio-Temporal Continuity Based Framework for Accident Anticipation." IEEE T-IV 9(1), 2024. DOI: https://doi.org/10.1109/TIV.2023.3257169
16. W. Song, S. Li, T. Chang ほか. "Dynamic attention augmented graph network for video accident anticipation." Pattern Recognition 147, 2024. DOI: https://doi.org/10.1016/j.patcog.2023.110071
17. N. Thakur, P. Gouripeddi, B. Li. "Graph(Graph): A Nested Graph-Based Framework for Early Accident Anticipation." WACV 2024. https://openaccess.thecvf.com/content/WACV2024/html/Thakur_GraphGraph_A_Nested_Graph-Based_Framework_for_Early_Accident_Anticipation_WACV_2024_paper.html
18. L.-L. Li, J. Fang, J. Xue. "Cognitive Traffic Accident Anticipation." IEEE Intelligent Transportation Systems Magazine, 2024. DOI: https://doi.org/10.1109/MITS.2024.3378460
19. M. M. Karim, Y. Li, R. Qin. "Towards explainable artificial intelligence (XAI) for early anticipation of traffic accidents." Transportation Research Record. https://arxiv.org/abs/2108.00273
20. Kumamoto ほか. "AAT-DA: Accident Anticipation Transformer with Driver Attention." WACV Workshops 2025. DOI: https://doi.org/10.1109/WACVW65960.2025.00127 ／ https://openaccess.thecvf.com/content/WACV2025W/HAVI/papers/Kumamoto_AAT-DA_Accident_Anticipation_Transformer_with_Driver_Attention_WACVW_2025_paper.pdf
21. K. Charoenpitaks ほか. "Exploring the Potential of Multi-Modal AI for Driving Hazard Prediction."（DHPR）IEEE T-IV 2024. https://arxiv.org/abs/2310.04671
22. H. Liao ほか. "CRASH: Crash Recognition and Anticipation System Harnessing with Context-Aware and Temporal Focus Attentions." ACM MM 2024. https://arxiv.org/abs/2407.17757
23. H. Liao, Y. Li ほか. "When, Where, and What? A Novel Benchmark for Accident Anticipation and Localization with Large Language Models." ACM MM 2024. https://arxiv.org/abs/2407.16277 ／ DOI: https://doi.org/10.1145/3664647.3681326
24. H. Liao, Y. Li ほか. "Real-time Accident Anticipation for Autonomous Driving Through Monocular Depth-Enhanced 3D Modeling."（AccNet）Accident Analysis & Prevention 207, 2024. https://arxiv.org/abs/2409.01256
25. J. Zhang, H. Liao ほか. "Eyes on the Road, Mind Beyond Vision: Context-Aware Multi-modal Enhanced Risk Anticipation."（CAMERA）ACM MM 2025. https://arxiv.org/abs/2507.06444
26. D. C. Moura, S. Zhu, O. Zvitia. "Nexar Dashcam Collision Prediction Dataset and Challenge." CVPR Workshops 2025. https://arxiv.org/abs/2503.03848 ／ https://openaccess.thecvf.com/content/CVPR2025W/WAD/html/Moura_Nexar_Dashcam_Collision_Prediction_Dataset_and_Challenge_CVPRW_2025_paper.html
27. T. Zhao, Y. Zou ほか. "Accident Anticipation via Temporal Occurrence Prediction."（TOP）NeurIPS 2025. https://arxiv.org/abs/2510.22260 ／ https://happytianhao.github.io/TOP/
28. Y. Zou, T. Zhao ほか. "RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation." CVPR 2026. https://arxiv.org/abs/2603.27165 ／ https://github.com/xingyueye5/RiskProp/
29. S. Hicsonmez, E. Çakar, N. Samet, F. Güney. "Progressive Risk Estimation for Accident Anticipation."（PRE-ACT）arXiv 2026（NeurIPS 2026 採録と記載）. https://arxiv.org/abs/2609.32811 ／ https://github.com/giddyyupp/PRE-ACT
30. V. Vipulananthan ほか. "VAGNet: Vision-based Accident Anticipation with Global Features." IEEE Open Journal of Vehicular Technology 7, 2026. https://arxiv.org/abs/2604.09305
31. R. Goldshmidt ほか. "BADAS: Context Aware Collision Prediction Using Real-World Dashcam Data." arXiv 2025. https://arxiv.org/abs/2510.14876
32. R. Goldshmidt ほか. "Beyond the Beep: Scalable Collision Anticipation and Real-Time Explainability with BADAS-2.0." arXiv 2026. https://arxiv.org/abs/2604.05767
33. L. Picek ほか. "ACCIDENT: A Benchmark Dataset for Vehicle Accident Detection from Traffic Surveillance Videos." CVPR Workshops 2026. https://arxiv.org/abs/2604.09819
34. S. Li ほか. "A VideoMAE-v2 Approach to Zero-Shot Traffic Accident Anticipation." arXiv 2026. https://arxiv.org/abs/2606.09542
35. L. Caselli ほか. "FLaRA: Predicting Future Latent Representations for Accident Anticipation." ITSC 2026. https://arxiv.org/abs/2606.14380
36. Y. Wang ほか. "Observe Before You Alert: Adaptive Driver Alerting with Vision-Language Models."（VLAlert）CoRL 2026. https://arxiv.org/abs/2609.08130
37. Y. Guan ほか. "Domain-Enhanced Dual-Branch Model for Efficient and Interpretable Accident Anticipation." arXiv 2025. https://arxiv.org/abs/2507.12755
38. L. Shi ほか. "ScVLM: Enhancing Vision-Language Model for Safety-Critical Event Understanding." WACV Workshops 2025. https://arxiv.org/abs/2410.00982
39. J. Liu ほか. "Predicting Risks Before They Happen: Vision-Language Reasoning for Autonomous Driving."（PotentialRiskQA）arXiv 2025. https://arxiv.org/abs/2511.22928

### 歩行者・エゴセントリック
40. A. Manglik ほか. "Forecasting Time-to-Collision from Monocular Video: Feasibility, Dataset, and Challenges." IROS 2019. https://arxiv.org/abs/1903.09102
41. D. Urban ほか. "Time- and Resource-Efficient Time-to-Collision Forecasting for Indoor Pedestrian Obstacles Avoidance." Journal of Imaging 7(4), 2021. DOI: https://doi.org/10.3390/jimaging7040061

### 交通事故予測（続き）
42. X. Liu, B. Rao ほか. "Predict and Resist: Long-Term Accident Anticipation under Sensor Noise." AAAI 2026. https://arxiv.org/abs/2511.08640
43. J. Zhang ほか. "LATTE: Lightweight Attention-based Traffic Accident Anticipation Engine." Information Fusion（arXiv 記載）. https://arxiv.org/abs/2504.04103
44. H. S. Ryu, C. Wallraven. "Mind the Gap: Quantifying and Aligning Human-AI Visual Attention for Accident Anticipation." AAAI 2026. DOI: https://doi.org/10.1609/aaai.v40i44.41125
45. V. Vipulananthan ほか. "STAGNet: A Spatio-Temporal Graph and LSTM Framework for Accident Anticipation." IEEE Access 2025. DOI: https://doi.org/10.1109/ACCESS.2025.3645127
46. H. Huang ほか. "Physical Depth-aware Early Accident Anticipation: A Multi-dimensional Visual Feature Fusion Framework." arXiv 2025. https://arxiv.org/abs/2502.18496
47. I. Song, J. Lee. "Real-time Traffic Accident Anticipation with Feature Reuse." ICIP 2025. https://arxiv.org/abs/2505.17449
48. T. Wu ほか. "MsFIN: Multi-scale Feature Interaction Network for Traffic Accident Anticipation." arXiv 2025. https://arxiv.org/abs/2509.19227
49. X. Liu ほか. "ROAR: Robust Accident Recognition and Anticipation for Autonomous Driving." Accident Analysis & Prevention（arXiv 記載）. https://arxiv.org/abs/2511.06226
50. W. Wang ほか. "SECURE: Stable Early Collision Understanding via Robust Embeddings in Autonomous Driving." arXiv 2026. https://arxiv.org/abs/2604.01337
51. J. Fang ほか. "EQ-TAA: Equivariant Traffic Accident Anticipation via Diffusion-Based Accident Video Synthesis." IEEE TMM（arXiv 記載）. https://arxiv.org/abs/2506.10002
52. Y. Guan, H. Liao ほか. "World model-based end-to-end scene generation for accident anticipation in autonomous driving." Communications Engineering, 2025. DOI: https://doi.org/10.1038/s44172-025-00474-7 ／ https://arxiv.org/abs/2507.12762

### 歩行者・エゴセントリック（続き）
53. Z. Yuan ほか. "WalkVLM: Aid Visually Impaired People Walking by Vision Language Model." arXiv 2024. https://arxiv.org/abs/2412.20903
54. K. Ying ほか. "mmWalk: Towards Multi-modal Multi-view Walking Assistance." NeurIPS 2025 Datasets & Benchmarks. https://arxiv.org/abs/2510.11520
55. J. Kim ほか. "GuideDog: A Real-World Egocentric Multimodal Dataset for Blind and Low-Vision Accessibility-Aware Guidance." ACL 2026. https://arxiv.org/abs/2503.12844
56. N. Mosca ほか. "VISTA: A Multimodal Egocentric Benchmark for Goal-Oriented Perception and Assistance to Blind and Low-Vision Users." arXiv 2026. https://arxiv.org/abs/2610.22822
57. M. Thoma ほか. "PEDESTRIAN: An Egocentric Vision Dataset for Obstacle Detection on Pavements." arXiv 2025. https://arxiv.org/abs/2512.19190
58. Y. Chen ほか. "EgoSafe: A First-Person Mobile-Captured Benchmark for Visual Safety Understanding." arXiv 2026. https://arxiv.org/abs/2607.26518
59. S. Panpatil ほか. "EgoSafetyBench: A Diagnostic Egocentric Video Benchmark for Evaluating Embodied VLMs as Runtime Safety Guards." arXiv 2026. https://arxiv.org/abs/2607.00218
60. K. Grauman ほか. "Ego4D: Around the World in 3,000 Hours of Egocentric Video." CVPR 2022. https://arxiv.org/abs/2110.07058
61. J. Qiu ほか. "Egocentric Human Trajectory Forecasting with a Wearable Camera and Multi-Modal Fusion." IEEE RA-L 2022. https://arxiv.org/abs/2111.00993
62. A. Jadhav ほか. "AI Guide Dog: Egocentric Path Prediction on Smartphone." AAAI 2025 Spring Symposium. https://arxiv.org/abs/2501.07957
63. Y. Zhao, Z. Chen, X. Yang. "EgoPathBench: Evaluating Zero-Shot Egocentric Waypoint Decision-Making in Vision-Language Models." arXiv 2026. https://arxiv.org/abs/2609.16610
64. I. Rodin ほか. "Predicting the Future from First Person (Egocentric) Vision: A Survey." CVIU 2021. https://arxiv.org/abs/2107.13411

### 関連
65. M. Liu ほか. "Forecasting Human-Object Interaction: Joint Prediction of Motor Attention and Actions in First Person Video." ECCV 2020. https://arxiv.org/abs/1911.10967
66. 事故予測の文献リスト: https://github.com/LIUTIGHE/Accident-Anticipation-List

---

### 付録: `papers.md` 記載文献の検証結果
| papers.md の記載 | 検証結果 |
|---|---|
| Ego-Alert: Egocentric Future Hazard Forecasting (ICCV 2021) | **存在を確認できず**（Web 検索で該当なし）。引用していない |
| Learning to Anticipate Traffic Accidents from Egocentric Video (ICCV 2019) | **存在を確認できず**。近いのは Yao ら IROS 2019 [12] と Manglik ら IROS 2019 [40] |
| Anticipating Accidents in Dashcam Videos (ECCV 2016) | 実在するが **ACCV 2016** の誤記 [1] |
| EgoVis: A Benchmark for Visual Hazard Detection in Egocentric Video | **存在を確認できず**。EgoVis は CVPR の Joint Egocentric Vision ワークショップの名称。近いのは EgoSafe [58]、EgoSafetyBench [59]、VISTA [56] |
| EgoPATH: Forecasting the Egocentric Path of the Camera Wearer | **存在を確認できず**。近いのは EgoPathBench [63]（VLM 評価）、Qiu ら [61]、Ego4D locomotion [60] |
| Forecasting Human-Object Interaction and Future Activities in Egocentric Video | 題名が完全には一致しない。実在するのは Liu ら ECCV 2020 "Forecasting Human-Object Interaction: Joint Prediction of Motor Attention and Actions in First Person Video" [65] |
| Ego4D Forecasting Benchmarks | 実在する [60] |
