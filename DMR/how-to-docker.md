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
Mean reciprocal rank

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
This page is based on the guidelines provided for [submissions to the leaderboard](https://sites.google.com/view/project-aio/competition2/how-to-use-leaderboard) of the AI King—Quiz AI Japan Championship.

## Requirements for Docker Images
Submitted Docker images must meet the following requirements:

### Requirement 1: Include an inference script `~/submission.sh`
The Docker image must include a script `~/submission.sh` that outputs search results based on the given topic data. It should be executable using the following command:
```bash
$ bash ./submission.sh <input_file> <output_file>
```

#### Input Data Format
Here, `<input_file>` is the topic data provided to the system in the following JSON format:
```json
[{"topic_id":"img2sen_001", "dmr_minute_id":"00001", "input_modal":"image", "output_modal":"sensor", "image":..., "heart_rate(bpm)":87.0, ...},
 {"topic_id":"img2sen_001", "dmr_minute_id":"00002", "input_modal":"image", "output_modal":"sensor", "image":..., "heart_rate(bpm)":91.0, ...},
 ...
]
```

- `topic_id`: The ID assigned to each search topic.
- `dmr_minute_id`: The ID assigned to each data point, which differs from the `minute_id` provided in Lifelog-6 data.
- `input_modal`: The modality provided as the search query, either `image` or `sensor`.
- `output_modal`: The modality being searched, either `image` or `sensor`.
- `image`: The numerical data representing the image.
- Additional elements consist of metadata and location information provided in Lifelog-6.

For more details, please refer to the create_topics.ipynb notebook provided by the organizers.

#### Output Data Format
The `<output_file>` should be output in the following CSV format. Participants' systems must generate this output:

```
group_id,run_id,topic_id,dmr_minute_id,score
ORG,baseline,img2sen_001,536,-0.26862192
ORG,baseline,img2sen_001,7,0.73308593
ORG,baseline,img2sen_001,413,-0.12440574
```

- `group_id`: The participant's GROUP ID, set independently with up to 5 alphanumeric characters.
- `run_id`: The RUN ID, set independently with up to 10 alphanumeric characters.
- `topic_id`: The topic ID specified in `<input_file>`.
- `dmr_minute_id`: The data ID specified in `<input_file>`.
- `score`: The score predicted by the system for `dmr_minute_id` within `topic_id`.

### Requirement 2: Include all necessary files for the model
The Docker image must contain all the files necessary for the model to function and be able to operate independently.
Downloading files from outside the Docker image is not allowed.

### Requirement 3: The compressed Docker image must be within 10 GiB
The Docker image should be saved and compressed using the following command:
The compressed file (e.g., `image.tar.gz`) will be subject to the size limitation.

```bash
$ docker save <image_name> | gzip > image.tar.gz
```

## Example of Building a Docker Image
Below is an example of building a Docker image using the baseline system published by the organizers:

1. Train the model using the train.ipynb notebook.
For detailed steps, refer to [`README.md`](./README.md) and the notebook [`train.ipynb`](./train.ipynb).

2. Build the Docker image using the [`Dockerfile`](./Dockerfile).

```bash
$ docker build -t org-baseline-run .
```

3. Verify the Docker image using the `create_topics.ipynb` notebook.

```bash
$ docker run --rm -v "<DMR dir absolute path>:/app/" org-baseline-run bash ./submission.sh input/input_test.json output/org-baseline_scores.csv
```

4. After confirming that the Docker image works correctly, save and compress the image.

```bash
$ docker save org-baseline-run | gzip > org-baseline-run.tar.gz
```

## Evaluation Method
### Evaluation Metrics
To be announced.

### Specifications of the Evaluation Machine
To be announced.

## Rules for Submitting Docker Images
- Teams submit their systems by uploading the saved and compressed Docker image file.
- Each team can submit up to three Docker images (RUNs).
- File names for Docker images should follow the format `<GROUP ID>-<RUN ID>-run.tar.gz`.
- For example, the baseline system of the organizers (`GROUP ID: org`, RUN ID: `baseline`) would be named `org-baseline-run.tar.gz`.
- Submission methods will be announced later.
