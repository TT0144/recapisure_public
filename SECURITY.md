# 🔒 セキュリティガイドライン

このドキュメントでは、recapisureプロジェクトにおけるセキュリティのベストプラクティスを説明します。

## 📋 目次

1. [機密情報の管理](#機密情報の管理)
2. [環境変数の取り扱い](#環境変数の取り扱い)
3. [Gitへのコミット前チェックリスト](#gitへのコミット前チェックリスト)
4. [トークン・APIキーの生成方法](#トークンapiキーの生成方法)
5. [セキュリティ脆弱性の報告](#セキュリティ脆弱性の報告)

---

## 🔐 機密情報の管理

### 絶対に公開してはいけない情報

以下の情報は**絶対にGitHubや公開リポジトリにコミットしないでください**：

1. **APIキー・トークン**
   - HuggingFace アクセストークン (`hf_xxxxx`)
   - Kaggle API キー
   - ngrok認証トークン
   - その他のサービスAPIキー

2. **セキュリティキー**
   - Flask SECRET_KEY
   - 暗号化キー
   - SSL/TLS証明書

3. **個人データ**
   - データベースファイル (`.db`, `.sqlite`)
   - ユーザーアップロードファイル
   - ログファイル

4. **環境設定**
   - `.env` ファイル
   - `.env.local`, `.env.production`
   - 設定ファイルの実際の値

### 安全な管理方法

#### ✅ 良い例
```python
# config.py - 環境変数から読み込む
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('KAGGLE_API_KEY')  # ✅ 環境変数から取得
SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))  # ✅ デフォルト値も安全
```

#### ❌ 悪い例
```python
# config.py - ハードコーディング（絶対NG）
API_KEY = "your_actual_api_key_here_DO_NOT_DO_THIS"  # ❌ 直接書いている
HF_TOKEN = "hf_YourActualTokenHereDO_NOT_DO_THIS"    # ❌ トークンが見える
```

---

## 📁 環境変数の取り扱い

### .envファイルの作成手順

1. **テンプレートをコピー**
   ```bash
   cp .env.template .env
   ```

2. **.envファイルを編集**
   ```env
   # 実際の値を設定（この値は例です）
   SECRET_KEY=your-actual-secret-key-here
   KAGGLE_API_KEY=your-actual-api-key-here
   KAGGLE_API_URL=https://your-actual-ngrok-url.ngrok-free.app
   ```

3. **.gitignoreで除外されていることを確認**
   ```bash
   # .gitignoreに以下が含まれていることを確認
   .env
   .env.local
   .env.production
   ```

### 環境変数の読み込み

```python
# Python での読み込み例
import os
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# 環境変数を取得
api_key = os.getenv('KAGGLE_API_KEY')
if not api_key:
    raise ValueError("KAGGLE_API_KEY is not set")
```

---

## ✅ Gitへのコミット前チェックリスト

コミット前に必ず以下を確認してください：

### 1. 機密情報のチェック
```bash
# ファイル内にトークンパターンがないか検索
grep -r "hf_" . --exclude-dir=.git
grep -r "sk-" . --exclude-dir=.git
grep -r "Bearer " . --exclude-dir=.git
```

### 2. .envファイルのチェック
```bash
# .envがstaged（追加予定）になっていないか確認
git status

# もし追加されていたら削除
git reset .env
```

### 3. .gitignoreの確認
```bash
# .gitignoreに必要な項目が含まれているか確認
cat .gitignore | grep -E "\.env|uploads|data|__pycache__"
```

### 4. git-secretsの使用（推奨）
```bash
# git-secretsをインストール (Mac)
brew install git-secrets

# git-secretsをインストール (Linux)
git clone https://github.com/awslabs/git-secrets.git
cd git-secrets
sudo make install

# セットアップ
git secrets --install
git secrets --register-aws
```

---

## 🔑 トークン・APIキーの生成方法

### SECRET_KEY の生成

**方法1: Pythonで生成**
```python
import secrets
print(secrets.token_hex(32))
# 出力例: 8f3a9b2c7d1e6f4a5b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0
```

**方法2: コマンドラインで生成 (Linux/Mac)**
```bash
openssl rand -hex 32
```

**方法3: オンラインツール**
- https://randomkeygen.com/ (Fort Knox Passwordsを使用)

### Kaggle API_KEY の生成

```python
import secrets
print(secrets.token_urlsafe(32))
# 出力例: xK8mN2pQ5rT9vW1yZ4bC7dF0gH3jL6nP8qS1tV4xY7zA
```

### HuggingFace トークンの取得

1. https://huggingface.co/settings/tokens にアクセス
2. "New token" をクリック
3. Token名を入力（例: `kaggle-apertus`）
4. Type: **Read** を選択
5. "Generate a token" をクリック
6. トークンをコピーして `.env` に保存

---

## 🚨 セキュリティ脆弱性の報告

### もしトークンを誤ってコミットしてしまった場合

#### 1. 直ちにトークンを無効化
- HuggingFace: https://huggingface.co/settings/tokens でトークンを削除
- Kaggle: 新しいAPIキーを生成

#### 2. Gitの履歴から削除
```bash
# BFG Repo-Cleanerをインストール (推奨)
brew install bfg

# トークンを含むコミットを削除
bfg --replace-text passwords.txt  # passwords.txtに削除したいトークンを記載

# または、git filter-branchを使用
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all
```

#### 3. 強制プッシュ
```bash
git push origin --force --all
git push origin --force --tags
```

#### 4. GitHubサポートに連絡
キャッシュをクリアしてもらうため、GitHubサポートに連絡してください。

### 脆弱性を発見した場合

セキュリティ上の問題を発見した場合は、**公開のIssueではなく**、以下の方法で報告してください：

- **Email**: [メールアドレスを記載]
- **件名**: `[SECURITY] recapisure セキュリティ報告`

報告には以下を含めてください：
- 脆弱性の説明
- 再現手順
- 影響範囲
- 可能であれば修正案

---

## 📚 参考資料

- [GitHub のセキュリティベストプラクティス](https://docs.github.com/ja/code-security/getting-started/best-practices-for-securing-your-github-account)
- [OWASP トップ10](https://owasp.org/www-project-top-ten/)
- [12 Factor App - Config](https://12factor.net/ja/config)

---

## ✅ セキュリティチェックリスト（要約）

- [ ] `.env` ファイルを `.gitignore` に追加済み
- [ ] テンプレートファイル（`.env.template`）のみコミット
- [ ] 実際のトークン・APIキーはテンプレートに含めない
- [ ] コード内にハードコードされたトークンがない
- [ ] `data/`, `uploads/` フォルダが `.gitignore` に含まれている
- [ ] コミット前に `git status` で確認
- [ ] 定期的にトークンをローテーション
- [ ] 本番環境では強力な `SECRET_KEY` を使用

---

**最終更新日**: 2026年2月6日
