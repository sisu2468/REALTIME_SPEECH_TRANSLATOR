# 多言語対応リアルタイム音声翻訳・文字起こしアプリ

このアプリケーションは、リアルタイムで日本語と英語の音声を認識し、文字起こしと翻訳を行います。Vosk、Google Cloud Translation API、および Gradio を使用しています。

## セットアップ手順

### 1. Poetry のインストール（まだの場合）

```bash
pip install poetry
```

### 2. プロジェクトの依存関係のインストール

```bash
poetry install
```

### 3. Vosk モデルのダウンロードとセットアップ

プロジェクトに含まれるセットアップスクリプトを使用して、必要な Vosk モデル（日本語と英語）を自動的にダウンロードし、セットアップします。

```bash
poetry run python setup_models.py
```

このスクリプトは、日本語モデルと英語モデルを自動的にダウンロードし、`models/ja` と `models/en` ディレクトリにそれぞれ配置します。

注意: ダウンロードには時間がかかる場合があります。インターネット接続が必要です。


### 4. Google Cloud credentials の設定

1. Google Cloud Console で新しいプロジェクトを作成し、Cloud Translation API を有効にします。
2. サービスアカウントキーを作成し、JSON ファイルをダウンロードします。
3. 環境変数を設定します：

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

## アプリケーションの実行

セットアップが完了したら、以下のコマンドでアプリケーションを起動できます：

```bash
poetry run python main.py
```

ブラウザで表示される URL にアクセスし、アプリケーションを使用します。

## 注意事項

- このアプリケーションは、インターネット接続が必要です（Google Cloud Translation API を使用するため）。
- 音声認識の精度は、使用する Vosk モデルとマイクの品質に依存します。
- Google Cloud Translation API の使用には課金が発生する可能性があります。使用量に注意してください。

## トラブルシューティング

- モデルのダウンロードに失敗する場合は、Vosk の公式ウェブサイトで最新のモデル URL を確認してください。
- `GOOGLE_APPLICATION_CREDENTIALS` が正しく設定されていることを確認してください。
- 音声認識が機能しない場合は、マイクの設定とブラウザの権限を確認してください。
