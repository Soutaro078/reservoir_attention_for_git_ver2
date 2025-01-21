# ベースイメージとして Python 3.8 を使用
FROM python:3.8-slim

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 依存関係をインストール
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードをコピー
COPY . /app

# メインスクリプトを実行するためのコマンド
CMD ["python", "main.py"]

# ベースイメージとして Python 3.8 を使用
# FROM python:3.8-slim

# # 作業ディレクトリを設定
# WORKDIR /app

# # 必要なパッケージをインストール
# RUN apt-get update && apt-get install -y \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # 依存関係をインストール
# COPY requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # ソースコードをコピー
# COPY . /app

# # メインスクリプトを実行するためのコマンド
# CMD ["python", "main.py"]