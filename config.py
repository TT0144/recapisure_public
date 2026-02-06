#!/usr/bin/env python3
"""
Configuration Management for recapisure
設定管理モジュール
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from services.apertus_client import ModelType

# 環境変数を安全に読み込み
try:
    from dotenv import load_dotenv
    load_dotenv()  # .envファイルを読み込み
except ImportError:
    # python-dotenvがインストールされていない場合は警告
    print("⚠️  python-dotenv not installed. Environment variables from .env file will not be loaded.")
    print("   Install with: pip install python-dotenv")

@dataclass
class AIConfig:
    """AI関連設定"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: ModelType = ModelType.GPT4_TURBO
    request_timeout: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        # 環境変数から設定を読み込み
        self.api_key = self.api_key or os.getenv('APERTUS_API_KEY')
        self.base_url = self.base_url or os.getenv('APERTUS_BASE_URL', 'https://api.apertus.ai')
        
        # モデル設定
        model_name = os.getenv('APERTUS_DEFAULT_MODEL', self.default_model.value)
        for model in ModelType:
            if model.value == model_name:
                self.default_model = model
                break

@dataclass
class AppConfig:
    """アプリケーション設定"""
    # Flask設定
    secret_key: str = os.environ.get('SECRET_KEY', 'recapisure-secret-key-2024')
    debug: bool = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host: str = os.environ.get('FLASK_HOST', '127.0.0.1')
    port: int = int(os.environ.get('FLASK_PORT', '5000'))
    
    # ファイル設定
    max_content_length: int = 16 * 1024 * 1024  # 16MB
    upload_folder: Path = Path(__file__).parent / 'uploads'
    allowed_extensions: set = None
    
    # テキスト処理設定
    max_text_length: int = int(os.environ.get('MAX_TEXT_LENGTH', '10000'))
    max_url_content_length: int = int(os.environ.get('MAX_URL_CONTENT_LENGTH', '50000'))
    request_timeout: int = int(os.environ.get('REQUEST_TIMEOUT', '30'))
    
    # AI設定
    ai_config: AIConfig = None
    
    # Kaggle API設定 (外部AI処理用)
    KAGGLE_API_URL: Optional[str] = os.environ.get('KAGGLE_API_URL', None)
    KAGGLE_API_KEY: Optional[str] = os.environ.get('KAGGLE_API_KEY', None)  # 🔒 認証キー
    KAGGLE_API_TIMEOUT: int = int(os.environ.get('KAGGLE_API_TIMEOUT', '60'))
    USE_KAGGLE_API: bool = os.environ.get('USE_KAGGLE_API', 'False').lower() == 'true'
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            # ⭐ 画像ファイル対応追加 (PNG, JPG, JPEG, GIF, BMP, WEBP)
            self.allowed_extensions = {'.txt', '.md', '.rtf', '.doc', '.docx', '.pdf', 
                                       '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        
        if self.ai_config is None:
            self.ai_config = AIConfig()
        
        # アップロードフォルダ作成
        self.upload_folder.mkdir(exist_ok=True)

# デフォルト設定インスタンス
config = AppConfig()

# ========================================
# 言語マッピング (Apertusの言語コード → 人間可読な言語名)
# ========================================
LANGUAGE_MAP = {
    'jpn_Jpan': 'Japanese',
    'eng_Latn': 'English', 
    'deu_Latn': 'German',
    'fra_Latn': 'French',
    'zho_Hans': 'Chinese (Simplified)',
    'zho_Hant': 'Chinese (Traditional)',
    'kor_Hang': 'Korean',
    'spa_Latn': 'Spanish',
    'por_Latn': 'Portuguese',
    'ita_Latn': 'Italian',
    'rus_Cyrl': 'Russian',
    'ara_Arab': 'Arabic',
    'hin_Deva': 'Hindi',
    'vie_Latn': 'Vietnamese',
    'tha_Thai': 'Thai',
    'auto': 'auto-detect'
}

def get_language_name(lang_code: str, default: str = None) -> str:
    """言語コードを人間可読な言語名に変換する
    
    Args:
        lang_code: Apertusの言語コード (例: 'jpn_Jpan')
        default: マッチしない場合のデフォルト値。Noneの場合はlang_code自体を返す
    
    Returns:
        人間可読な言語名 (例: 'Japanese')
    """
    if default is None:
        default = lang_code
    return LANGUAGE_MAP.get(lang_code, default)