# [日本語](#dmrサブタスクでのdockerイメージの作成提出方法) | [English](#how-to-create-and-submit-a-docker-image-for-the-dmr-subtask)

# DMRサブタスクでのDockerイメージの作成・提出方法
本ページは，AI王〜クイズAI日本一決定戦〜の[リーダボードへの投稿](https://sites.google.com/view/project-aio/competition2/how-to-use-leaderboard)を参考に作成しています．

## Dockerイメージの要件
投稿されるDockerイメージは，以下の要件を全て満たすものとします．

### 要件１:推論スクリプト`~/submission.sh`を含むこと
Dockerイメージは，与えられるトピックデータに対して検索結果の出力を行うスクリプト`~/submission.sh`が含まれており，以下のコマンドで実行可能であることとします．
```
$ bash ./submission.sh <input_file> <outout_file>
```

#### 入力データのフォーマット
ここで，`<input_file>`はシステムに入力されるトピックデータで，以下のようなJSON形式であるとします．
```json
[{"topic_id":"img2sen_001", "dmr_minute_id":"00001", "input_modal":"image", "output_modal":"sensor", "image":..., "heart_rate(bpm)":87.0, ...},
 {"topic_id":"img2sen_001", "dmr_minute_id":"00002", "input_modal":"image", "output_modal":"sensor", "image":..., "heart_rate(bpm)":91.0, ...},
 ...
]
```
- `topic_id`: 検索トピックごとに付与されるIDです
- `dmr_minute_id`: データごとに付与されるIDです．Lifelog-6データに付与されている`minute_id`とは異なります．
- `input_modal`: 検索クエリとして与えられるモダリティです．画像`image`かセンサ`sensor`が記述されます．
- `output_modal`: 検索対象となるモダリティです．画像`image`かセンサ`sensor`が記述されます．
- `image`: 画像データの数値が記述されています．
- 以降の要素はLifelog-6で提供されるメタデータや位置情報から構成されています．

**詳しくは，オーガナイザが提供する`create_topics.ipynb`をご覧ください．**

#### 出力データのフォーマット
出力する`<output_file>`は，以下のようなCSV形式であるとします．参加者が提出するシステム内で出力する必要があります．

```csv
group_id,run_id,topic_id,dmr_minute_id,score
ORG,baseline,img2sen_001,536,-0.26862192
ORG,baseline,img2sen_001,7,0.73308593
ORG,baseline,img2sen_001,413,-0.12440574
```

- `group_id`: 参加GROUP IDです．参加者が独自に半角5文字以内で設定します．
- `run_id`: RUN IDです．参加者が独自に半角10文字以内で設定します
- `topic_id`: `<input_file>`で指定されているトピックIDです
- `dmr_minute_id`: `<input_file>`で指定されいてるデータIDです
- `score`: システムが`topic_id`内の`dmr_minute_id`に対して予測したスコアです

### 要件２:モデルの動作に必要な全てのファイルを含むこと
Dockerイメージには，モデルの動作に必要なファイルがすべて含まれており，それ単体で動作可能であることとします．  
Dockerイメージ内から，イメージの外側にあるファイルをダウンロードすることなどは不可とします．

### 要件３:gzip圧縮されたDockerイメージのサイズが10GiB以内であること
Docker イメージの保存・圧縮は，以下のコマンドで行います．  
圧縮後のファイル（以下の例では`image.tar.gz`）がサイズ制限の対象となります．
```bash
$ docker save <image_name> | gzip > image.tar.gz
```

## Dockerイメージの構築例
オーガナイザ公開のベースラインシステムを用いて，Dockerイメージの構築例を以下に示します．

1. モデルの訓練
訓練用ノートブック`train.ipynb`を用いてモデルを訓練します．  
手順について詳しくは[`README.md`](./README.md)と[`train.ipynb`](./train.ipynb)のノートブック内をご覧ください．

2. Dockerイメージのビルド
[`Dockerfile`](./Dockerfile)を用いて，Dockerイメージをビルドします．
```bash
$ docker build -t org-baseline-run .
```

3. Dockerイメージの動作確認
`create_topics.ipynb`を用いて，Dockerイメージの動作確認をします．
```bash
$ docker run --rm -v "<DMR dir absolute path>:/app/" org-baseline-run bash ./submission.sh input/input_test.json output/org-baseline_scores.csv
```

4. Dockerイメージの保存と圧縮
Dockerイメージが正しく動作することを確認できたら，イメージをファイルに保存・圧縮します．
```bash
$ docker save org-baseline-run | gzip > org-baseline-run.tar.gz
```

## 評価方法

### 評価尺度
後日公開

### 評価用マシンのスペック
後日公開

## Dockerイメージ提出のルール
- 参加チームは，保存・圧縮したDockerイメージのファイルをアップロードすることによりシステムの投稿を行います．  
- 各参加チームで提出可能なDockerイメージ（RUN）は，最大3件までとします．  
- Dockerイメージのファイル名は`<GROUP ID>-<RUN ID>-run.tar.gz`としてください．  
例：オーガナイザ(`GROUP ID: org`)のベースラインシステム(`RUN ID: baseline`)のファイル名は，`org-baseline-run.tar.gz`
- アップロード方法は後日公開します．

---
&nbsp;
なお，以上のルールは参加者の参加状況に応じて今後変更される可能性があります．

&nbsp;
&nbsp;

# How to Create and Submit a Docker Image for the DMR Subtask
To be updated.