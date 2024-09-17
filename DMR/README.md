# [日本語](#dmrサブタスクの始め方) | [English](#how-to-start-dmr-subtask)

# DMRサブタスクの始め方
## 前提
- NTCIR-18 Transfer-2タスクの参加申し込みを完了している．

## データセットの入手
- DMRサブタスクは，NTCIR-18 Lifelog-6タスクのデータセットを使用してモデルの訓練・評価を行います．
- データセットの入手は，[NTCIR-18 Lifelog-6](http://lifelogsearch.org/ntcir-lifelog/dataset/)のタスクページを参照してください．
- データセット入手後，想定するディレクトリ構成は以下のとおりです．

```
DMR/
 ├ Lifelog-6/    NTCIR-18 Lifelog-6データセットの保存場所
 ├ utils/        テスト用トピックファイルの作成スクリプトや評価スクリプトの保存場所
 ├ models/       モデルの保存場所（画像認識モデルも含む）
 ├ input/        入力となるトピックファイルの保存場所
 ├ output/       出力となる検索結果の保存場所
 ├ logs/         モデルの学習結果等ログファイルの保存場所
 ├ submission.sh モデルの実行について記述されたシェルスクリプトファイル
 ├ predict.py    検索結果出力用のPythonスクリプト
 ├ train.ipynb   モデル学習用のJupyter notebook
 ├ train.py      `train.ipynb`のPythonスクリプト
 ├ dpr_models.py モデルの定義
 └ Dockerfile    Dockerイメージを作成するための手順を記したファイル
```

- `Lifelog-6/`以下のディレクトリ構成は以下を想定しています．

```
Lifelog-6/
 ├ images/
 ├ lsc22_metadata.csv
 ├ lsc22_visual_concepts.csv
 └ vaisl_gps.csv
```

## 大切なルール
- Transfer-2 DMRサブタスクでは，Lifelog-6の外部データセットを学習と評価で使用する関係上，トピックファイルの配布は行いません．
- このため参加者が構築したシステムの評価は，参加者が検索システムのDockerイメージをアップロードすることにより行います．
- Dockerイメージの要件は[こちらのページ](./how-to-docker.md)を確認してください．

## ノートブック
GitHubのWebページに表示されるノートブックのコードは不完全な場合がありますので，コードをコピーせずに必ずファイルをダウンロードして実行してください．
- `dpr_models.py`: モデルの定義
- `train.ipynb`: モデルの訓練例
- `predict.py`: 検索結果の出力例
- `utils/create_topics.ipynb`: トピックファイルの作成例（参加者のローカル環境でモデルの評価確認用）
- `utils/evaluate.ipynb`: 検索結果の評価例

## ベースラインモデルの動作方法
以下はベースラインモデルの実行手順です．各ノートブックのコメントとコードを読みながら進めてください．
1. MIT CSAILが公開している[Google Drive](https://drive.google.com/drive/folders/1k2nggK3LqyBE5huGpL3E-JXoEv7o6qRq)からPlaces365データセットで事前学習したResNet18モデル（`resnet18_places365.pth.tar`）をダウンロードし，`models/`以下に配置します．  
1. `create_topics.ipynb`を実行し，トピックファイル（`input/input_test.json`，`input/qrels_test.csv`）を作成する．
1. `train.ipynb`を実行し，訓練済みモデル（`models/baseline.pth`）とモデル設定ファイル（`models/baseline.pkl`）を作成する．
1. `predict.py`を実行し，検索結果（`output/baseline_scores.csv`）を出力する．
1. `evaluate.ipynb`を実行し，検索結果を評価する．

## 検索結果の提出方法
- Dockerイメージを提出していただきます．詳細は，[Dockerイメージの提出方法](./how-to-docker.md)を参照してください．

&nbsp;
&nbsp;

# How to start DMR subtask

## Prerequisite
- The registration for participation in the NTCIR-18 Transfer-2 task has been completed.

## Rquirement: Lifelog-6 Dataset
- The DMR subtask uses the NTCIR-18 Lifelog-6 dataset.
- To use this dataset, please refer to the [NTCIR-18 Lifelog-6](http://lifelogsearch.org/ntcir-lifelog/dataset/) task page.
- After obtaining the dataset, the assumed directory structure is as follows:

```
DMR/
 ├ Lifelog-6/        Directory for the NTCIR-18 Lifelog-6 dataset  
 ├ utils/            Directory for test topic file creation scripts and evaluation scripts  
 ├ models/           Directory for saving models (including image recognition models)  
 ├ input/            Directory for storing input topic files  
 ├ output/           Directory for storing the output search results  
 ├ logs/             Directory for storing log files, such as model training results  
 ├ submission.sh     Shell script file describing model execution  
 ├ predict.py        Python script for outputting search results  
 ├ train.ipynb       Jupyter notebook for model training  
 ├ train.py          Python script equivalent to `train.ipynb`  
 ├ dpr_models.py     Model definition file  
 └ Dockerfile    A file with instructions for creating a Docker image
```

- The directory structure under `Lifelog-6/` is assumed to be as follows:

```
Lifelog-6/
 ├ images/
 ├ lsc22_metadata.csv
 ├ lsc22_visual_concepts.csv
 └ vaisl_gps.csv
```

## Important Rules
- In the Transfer-2 DMR subtask, no topic files will be distributed due to the use of the external Lifelog-6 dataset for both training and evaluation.
- Therefore, the evaluation of the systems built by participants will be conducted by having them upload the Docker image of their search system.
- Please check the Docker image requirements on this [page](./how-to-docker.md).

## Notebook
The code displayed in the notebook on the GitHub webpage may be incomplete. Please make sure to download the files and execute them instead of copying the code.
- `create_topics.ipynb`: Example of creating topic files (for participants to verify model evaluation in their local environment)
- `dpr_models.py`: Model definition
- `train.ipynb`: Example of model training
- `predict.py`: Example of search results output
- `evaluate.ipynb`: Example of search results evaluation

## How to Run the Baseline Model
The following are the steps to execute the baseline model. Please proceed while reading the comments and code in each notebook.
1. Download the ResNet18 model pre-trained on the Places365 dataset (`resnet18_places365.pth.tar`) from the [Google Drive](https://drive.google.com/drive/folders/1k2nggK3LqyBE5huGpL3E-JXoEv7o6qRq) published by MIT CSAIL, and place it in the `models/` directory.
1. Run `create_topics.ipynb` to generate the topic files (`input/topics_test.pkl`, `input/qrels_test.csv`).
1. Run `train.ipynb` to create the trained model (`models/baseline.pth`) and model configuration file (`models/baseline.pkl`).
1. Run `predict.py` to output the search results (`output/baseline_scores.csv`).
1. Run `evaluate.ipynb` to evaluate the search results.

## How to Submit Search Results
You will be required to submit a Docker image.  
For more details, please refer to this [page](./how-to-docker.md).