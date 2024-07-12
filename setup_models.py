"""
このスクリプトは、Voskの日本語モデルと英語モデルをダウンロードし、適切なディレクトリに配置します。

以下のbashコマンドと同等の処理を行います：

日本語モデル:
mkdir -p models/ja
wget https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip
unzip vosk-model-small-ja-0.22.zip -d models/ja
rm vosk-model-small-ja-0.22.zip

英語モデル:
mkdir -p models/en
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip -d models/en
rm vosk-model-small-en-us-0.15.zip

このスクリプトは上記の操作を自動化し、複数の言語モデルに対して実行します。
"""

import os
import urllib.request
import zipfile
import shutil

def download_and_extract(url, target_dir):
    # ターゲットディレクトリが存在しない場合は作成
    os.makedirs(target_dir, exist_ok=True)

    # ファイル名を URL から抽出
    filename = url.split('/')[-1]
    filepath = os.path.join(target_dir, filename)

    # ファイルをダウンロード
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filepath)

    # ファイルを解凍
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    # ダウンロードした zip ファイルを削除
    os.remove(filepath)

    # 解凍されたディレクトリ名を取得
    extracted_dir = next(os.walk(target_dir))[1][0]

    # ファイルを一つ上の階層に移動
    src_dir = os.path.join(target_dir, extracted_dir)
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            shutil.move(s, d)
        else:
            shutil.move(s, target_dir)

    # 空になった解凍ディレクトリを削除
    os.rmdir(src_dir)

    print(f"Model setup completed for {target_dir}")

def main():
    models = {
        "ja": "https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip",
        "en": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    }

    for lang, url in models.items():
        target_dir = os.path.join("models", lang)
        download_and_extract(url, target_dir)

if __name__ == "__main__":
    main()
    print("All models have been successfully downloaded and set up.")
