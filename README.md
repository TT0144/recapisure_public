# 🚀 recapisure - 多言語AI要約Webアプリケーション (Apertus統合版)

Hugging Face Transformersを活用した**完全無料・APIキー不要**の高性能多言語テキスト処理Webアプリケーションです。
長文要約、多言語翻訳、PDF処理の主要機能に加え、**🇨🇭 Apertus-8B**（スイス政府製AI）を統合しました。

## ✨ 主要機能

### 🇨🇭 **Apertus-8B 統合**
- **1,811言語対応**: 世界最高クラスの多言語AIモデル
- **Kaggle GPU**: 完全無料でGPU利用（週30時間）
- **高品質要約**: 学術論文から技術文書まで対応
- **学習システム**: ユーザーフィードバックで精度向上

### 📝 多言語要約・翻訳
- **70以上の言語対応**: 日本語、英語、中国語、韓国語、スペイン語など
- **自動言語検出**: 入力言語を自動判別
- **2つの要約モード**:
  - 📄 **通常要約** (200-400字): 素早く要点を把握
  - 📚 **詳細要約** (800-1000字): 包括的な内容理解
- **3つの要約スタイル**:
  - ⚖️ **balanced**: バランスの取れた標準的な要約
  - 📖 **detailed**: より詳細で包括的な要約
  - 🎯 **concise**: 簡潔で要点を絞った要約

### 📄 ファイル対応
- **PDF処理**: 複数ページを自動抽出（pdfplumber + PyPDF2）
- **画像OCR**: スキャンPDF・画像から文字認識（Tesseract）
- **テキストファイル**: TXT, MD対応
- **文字化け対策**: 自動エンコーディング検出と修正

### 🎯 高品質処理
- **キーワード抽出**: テキストから重要キーワードを自動抽出
- **品質スコア**: 要約品質を0-100点で評価
- **履歴管理**: 最大200件の処理履歴を保存
- **詳細統計**: 圧縮率、実行時間、トークン使用量

## 🛠️ 技術スタック

- **Backend**: Python 3.11+, Flask 2.3
- **Frontend**: Bootstrap 5, jQuery, Font Awesome
- **AI**: 
  - 要約/翻訳: **Apertus-8B-Instruct** (Kaggle GPU経由)
  - フォールバック: `facebook/nllb-200-distilled-600M`
- **PDF処理**: pdfplumber, PyPDF2, Tesseract OCR
- **データベース**: SQLite3
- **完全無料**: API課金なし

## 📦 クイックスタート

### 1. リポジトリクローン
```bash
git clone https://github.com/YOUR_USERNAME/recapisure.git
cd recapisure
```

### 2. 依存関係インストール
```bash
pip install -r requirements.txt
```

### 3. 環境変数設定（オプション）
基本機能のみ使う場合はスキップ可能。Apertus-8Bを使う場合は設定が必要です。

```bash
# テンプレートをコピー
cp .env.template .env

# .env を編集（後述）
```

### 4. アプリケーション起動

**Windows PowerShell:**
```powershell
$env:PYTHONIOENCODING="utf-8"; python app.py
```

**Mac/Linux:**
```bash
export PYTHONIOENCODING=utf-8
python app.py
```

### 5. ブラウザでアクセス
```
http://localhost:5000
```

---

## 🔐 セキュリティとセットアップ

### ⚠️ 重要: セキュリティの原則

このプロジェクトでは、**APIキーやトークンなどの機密情報を絶対にGitHubにコミットしない**ことが重要です。

#### 保護される情報
- 🔑 **API キー**: Kaggle API_KEY、HuggingFaceトークン
- 🔒 **SECRET_KEY**: Flaskのセッション暗号化キー
- 🗃️ **データベース**: 個人の履歴データ
- 📂 **アップロードファイル**: ユーザーがアップロードしたPDF等

#### .gitignoreで自動的に除外されるファイル
```
.env                    # 環境変数（API キー等）
.env.local             # ローカル環境設定
.env.production        # 本番環境設定
*.key, *.pem           # 証明書ファイル
data/                  # データベースと履歴
uploads/               # アップロードファイル
test_articles/         # テストファイル
__pycache__/          # Pythonキャッシュ
*.log                  # ログファイル
```

### 🚀 Apertus-8B (Kaggle) のセットアップ

Apertus-8Bは高性能なAIモデルですが、セットアップが必要です。詳細な手順は以下を参照してください：

📖 **[KAGGLE_SETUP.md](KAGGLE_SETUP.md) - 完全セットアップガイド**

#### 必要な手順（概要）
1. **HuggingFaceトークン取得**: https://huggingface.co/settings/tokens
2. **Kaggle Notebook作成**: GPU T4 x2 を有効化
3. **APIサーバー起動**: `kaggle_server_template.py` を使用
4. **ngrokトンネル**: 公開URLを取得
5. **ローカル設定**: `.env` にURLとAPIキーを設定

#### .env ファイルの設定例
```env
# Kaggle API設定（Apertus-8B使用時）
KAGGLE_API_URL=https://your-ngrok-url.ngrok-free.app
KAGGLE_API_KEY=your-generated-api-key-here

# アプリケーション設定
SECRET_KEY=your-secret-key-change-this
```

**重要**: 
- ⚠️ `.env` ファイルは絶対にGitにコミットしない
- ⚠️ `kaggle_server_template.py` を使用（実際のトークンは含まれていません）
- ⚠️ APIキーは `python -c "import secrets; print(secrets.token_urlsafe(32))"` で生成

---

## 🚀 使用方法

1. **処理モード選択**: メインページで要約・展開・URL要約を選択
2. **テキスト入力**: 直接入力またはファイルアップロード
3. **処理実行**: AIによる高速処理
4. **結果確認**: コピー・保存・履歴管理

## ⚙️ AI設定

- **モデル**: Apertus-8B (Kaggle GPU経由)
- **パフォーマンス調整**: タイムアウト・リトライ設定
- **使用統計**: トークン使用量・処理数の確認

## 📁 プロジェクト構造

```
recapisure/
├── app.py                 # メインアプリケーション
├── config.py             # 設定管理
├── setup_apertus.py      # セットアップスクリプト
├── .env.example          # 環境変数テンプレート
├── .gitignore           # Git除外設定
├── services/            # AIサービス
│   ├── apertus_client.py
│   └── ai_service.py
├── models/              # データモデル
│   └── processing.py
└── templates/           # HTMLテンプレート
    ├── index.html
    ├── history.html
    ├── about.html
    └── settings.html
```

## 🔧 開発情報

### 環境要件
- Python 3.8+
- Apertus API アカウント
- インターネット接続

### 開発モード
```bash
# デバッグモードで起動
FLASK_DEBUG=true python app.py
```

### 依存関係更新
```bash
pip install -r requirements.txt
```

## 📝 ライセンス

MIT License

## 👨‍💻 作者

**TT0144**
- GitHub: [@TT0144](https://github.com/TT0144)

---

**注意**: このプロジェクトは就職活動用の個人制作として開発されています。
API キーやセキュリティ情報の管理には十分注意してください。
