# 🚀 Kaggle セットアップガイド

このガイドでは、Kaggle Notebookで**Apertus-8B AI モデル**を起動し、recapisureアプリから利用できるようにする手順を説明します。

## 📋 目次

1. [前提条件](#前提条件)
2. [Kaggle Notebookの作成](#kaggle-notebookの作成)
3. [APIサーバーのセットアップ](#apiサーバーのセットアップ)
4. [ngrokでトンネル作成](#ngrokでトンネル作成)
5. [ローカルアプリの設定](#ローカルアプリの設定)
6. [トラブルシューティング](#トラブルシューティング)

---

## 🎯 前提条件

### 必要なアカウント

1. **Kaggle アカウント**: https://www.kaggle.com/
   - GPU利用: 週30時間まで無料
   - T4 GPU推奨（Apertus-8Bに最適）

2. **HuggingFace アカウント**: https://huggingface.co/
   - Apertus-8Bモデルへのアクセスに必要
   - アクセストークンを取得

3. **ngrok アカウント**: https://ngrok.com/
   - 無料プランで十分
   - 認証トークンを取得

---

## 📝 Step 1: HuggingFaceトークンの取得

### 1.1 HuggingFaceにログイン
https://huggingface.co/ にアクセスしてログイン

### 1.2 アクセストークンを作成
1. Settings → Access Tokens に移動
2. "New token" をクリック
3. Token名: `kaggle-apertus` (任意)
4. Type: **Read** を選択
5. "Generate a token" をクリック
6. トークンをコピーして安全な場所に保存
   - 形式: `hf_xxxxxxxxxxxxxxxxxxxx`

### 1.3 Apertus-8Bモデルへのアクセス許可
1. https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509 にアクセス
2. "Agree and access repository" をクリック（初回のみ）

---

## 📝 Step 2: Kaggle Notebookの作成

### 2.1 新しいNotebookを作成
1. https://www.kaggle.com/ にログイン
2. "Code" → "New Notebook" をクリック
3. Notebook名: `recapisure-apertus-server` （任意）

### 2.2 GPU設定
1. 右側の "Settings" パネルを開く
2. "Accelerator" → **GPU T4 x2** を選択
3. "Internet" → **ON** にする（必須）
4. "Save" をクリック

### 2.3 必要なパッケージをインストール
新しいセルに以下を貼り付けて実行:

```python
!pip install -q flask pyngrok transformers torch accelerate
```

---

## 📝 Step 3: APIサーバーのセットアップ

### 3.1 APIキーの生成
別のセルで以下を実行してランダムなAPIキーを生成:

```python
import secrets
api_key = secrets.token_urlsafe(32)
print(f"Generated API Key: {api_key}")
```

**重要**: このAPIキーを安全にコピーしてください（後で使います）

### 3.2 サーバーコードを配置
`kaggle_server_template.py` の内容をKaggle Notebookの新しいセルにコピー

### 3.3 認証情報を設定
コピーしたコード内の以下の部分を置き換え:

```python
# 🔒 この2つを必ず置き換えてください
API_KEY = "YOUR_RANDOM_API_KEY_HERE_REPLACE_ME"  # → Step 3.1で生成したキー
hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"         # → Step 1.2で取得したトークン
```

### 3.4 サーバーを起動
セルを実行してサーバーを起動（3-5分かかります）

✅ 成功すると以下のように表示されます:
```
✅ モデルロード完了!
📊 dtype: torch.bfloat16
🎮 GPU: Tesla T4
💾 VRAM使用量: 14.2 GB
🚀 Flaskサーバーを起動します...
```

---

## 📝 Step 4: ngrokでトンネル作成

### 4.1 ngrokトークンの取得
1. https://dashboard.ngrok.com/ にログイン
2. "Your Authtoken" をコピー

### 4.2 ngrokトンネルを作成
**新しいセル**に以下を貼り付けて実行:

```python
from pyngrok import ngrok
import os

# ngrok認証トークンを設定
ngrok_token = "YOUR_NGROK_AUTH_TOKEN"  # ← ここを置き換え
ngrok.set_auth_token(ngrok_token)

# トンネル作成
public_url = ngrok.connect(5000)
print(f"\n🌐 Public URL: {public_url}")
print(f"\n📋 この URL を .env ファイルの KAGGLE_API_URL に設定してください\n")
```

✅ 以下のように表示されます:
```
🌐 Public URL: https://xxxx-xx-xx-xxx-xx.ngrok-free.app
📋 この URL を .env ファイルの KAGGLE_API_URL に設定してください
```

**重要**: このURLをコピーして保存してください

### 4.3 動作確認
別のセルで以下を実行:

```python
import requests

health_url = f"{public_url}/health"
response = requests.get(health_url)
print(response.json())
```

✅ 成功すると以下が表示されます:
```json
{
  "success": true,
  "status": "running",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_name": "Tesla T4",
  "model": "swiss-ai/Apertus-8B-Instruct"
}
```

---

## 📝 Step 5: ローカルアプリの設定

### 5.1 .envファイルを作成
プロジェクトフォルダで `.env.template` を `.env` にコピー:

```bash
# Windows PowerShell
Copy-Item .env.template .env

# Mac/Linux
cp .env.template .env
```

### 5.2 環境変数を設定
`.env` ファイルを開いて以下を設定:

```env
# Kaggle API設定
KAGGLE_API_URL=https://xxxx-xx-xx-xxx-xx.ngrok-free.app  # Step 4.2のURL
KAGGLE_API_KEY=xxxxxxxxxxxxxxxxxxxxxx                      # Step 3.1のAPIキー
```

### 5.3 アプリを起動
```bash
# Windows PowerShell
$env:PYTHONIOENCODING="utf-8"; python app.py

# Mac/Linux
export PYTHONIOENCODING=utf-8
python app.py
```

### 5.4 動作確認
1. ブラウザで http://localhost:5000 を開く
2. 設定ページ（/settings）で "Apertus (Kaggle)" が表示されることを確認
3. テキストを入力して要約を試す

---

## 🔧 トラブルシューティング

### ❌ "Model not loaded" エラー
**原因**: モデルのロードに失敗

**対処法**:
1. HuggingFaceトークンが正しいか確認
2. Apertus-8Bモデルへのアクセス許可を確認
3. Kaggle NotebookでGPUが有効か確認
4. サーバーセルを再実行

### ❌ "Unauthorized" エラー
**原因**: APIキーが一致していない

**対処法**:
1. Kaggle側の `API_KEY` と .env の `KAGGLE_API_KEY` が同じか確認
2. .envファイルを編集後、アプリを再起動

### ❌ "Connection refused" エラー
**原因**: ngrok URLが無効または期限切れ

**対処法**:
1. Kaggle Notebookでngrokセルを再実行
2. 新しいURLを.envファイルに設定
3. アプリを再起動

### ❌ VRAM不足エラー
**原因**: GPUメモリ不足

**対処法**:
1. Notebookを再起動
2. T4 x2 GPUを選択
3. 他のセルを実行していないか確認

### ⏱️ タイムアウトエラー
**原因**: 初回実行時のモデルダウンロードに時間がかかる

**対処法**:
- 3-5分待つ（初回のみ）
- 2回目以降は高速化

---

## 📌 重要な注意事項

### セキュリティ
- ⚠️ **API_KEY** と **HuggingFaceトークン** は絶対に公開しない
- ⚠️ `.env` ファイルはGitにコミットしない
- ⚠️ ngrok URLは定期的に変わるため、毎回更新が必要

### Kaggle制限
- GPU利用: 週30時間まで無料
- セッション: 12時間で自動終了
- 再起動時はngrok URLが変わる

### コスト
- すべて無料（Kaggle GPU + ngrok無料プラン）
- 追加料金なし

---

## 🎉 完了！

これでrecapisureアプリからKaggle上のApertus-8Bモデルを利用できるようになりました。

質問や問題がある場合は、GitHubのIssuesで報告してください。
