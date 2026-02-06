"""
Kaggle AI Service Client
=========================
ローカルFlaskアプリからKaggle APIにリクエストを送信

使用方法:
    from services.kaggle_ai_client import KaggleAIClient
    
    client = KaggleAIClient("https://your-ngrok-url.ngrok.io")
    result = client.translate("Hello", "English", "Japanese")
"""

import os
import requests
import time
from typing import Dict, Optional
import urllib3

# SSL警告を抑制（ngrok使用時）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class KaggleAIClient:
    """<Kaggle APIクライアント
    
    Apertus-8BモデルをKaggle Notebookで実行し、
    ngrok経由でローカルアプリと通信するクライアント
    """
    
    def __init__(self, base_url: str, api_key: str = None, timeout: int = None):
        """
        Args:
            base_url: KaggleサーバーのURL (ngrok URL)
            api_key: 認証用APIキー
            timeout: リクエストタイムアウト(秒) - Noneの場合は環境変数から取得
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.environ.get('KAGGLE_API_KEY')
        # ⭐ タイムアウト: 引数 > 環境変数 > デフォルト300秒
        self.timeout = timeout or int(os.environ.get('KAGGLE_API_TIMEOUT', '300'))
        self._is_available = None
        self._last_check = 0
        
        if not self.api_key:
            print("⚠️ KAGGLE_API_KEYが設定されていません")
            print("   .envに追加してください: KAGGLE_API_KEY=your-api-key")
    
    def _get_headers(self) -> dict:
        """認証ヘッダーを生成"""
        headers = {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true',  # ngrok警告ページをスキップ
            'User-Agent': 'RecapisureApp/1.0'  # カスタムUser-Agent
        }
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers
    
    def is_available(self, force_check: bool = False) -> bool:
        """Kaggle APIが利用可能かチェック
        
        Args:
            force_check: キャッシュを無視して再チェック
            
        Returns:
            利用可能ならTrue
        """
        # 5分間キャッシュ
        if not force_check and self._is_available is not None:
            if time.time() - self._last_check < 300:
                return self._is_available
        
        try:
            # ⭐ ngrok警告ページをスキップするヘッダーを追加
            response = requests.get(
                f"{self.base_url}/health",
                headers=self._get_headers(),  # ⭐ 認証ヘッダー追加
                timeout=10,  # ヘルスチェックは10秒で十分
                verify=False  # ngrokのSSL証明書検証を無効化
            )
            
            if response.status_code == 200:
                data = response.json()
                self._is_available = data.get("status") == "ok" and data.get("model_loaded", False)
            else:
                self._is_available = False
                
        except Exception as e:
            print(f"⚠️ Kaggle APIヘルスチェック失敗: {e}")
            self._is_available = False
        
        self._last_check = time.time()
        return self._is_available
    
    def translate(
        self,
        text: str,
        source_lang: str = "English",
        target_lang: str = "Japanese"
    ) -> Optional[Dict]:
        """テキスト翻訳
        
        Args:
            text: 翻訳するテキスト
            source_lang: 元言語 (English, German, French, Italian, Japanese)
            target_lang: 翻訳先言語
            
        Returns:
            成功時: {"success": True, "translation": "翻訳結果", "time": 8.5}
            失敗時: {"success": False, "error": "エラーメッセージ"}
        """
        try:
            response = requests.post(
                f"{self.base_url}/translate",
                json={
                    "text": text,
                    "source_lang": source_lang,
                    "target_lang": target_lang
                },
                headers=self._get_headers(),  # 🔒 認証ヘッダー追加
                timeout=self.timeout,
                verify=False  # ngrokのSSL証明書検証を無効化
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"タイムアウト ({self.timeout}秒)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def summarize(self, text: str, max_length: int = 400, source_lang: str = "auto-detect", target_lang: str = "Japanese", style: str = "balanced", summary_mode: str = "short") -> Optional[Dict]:
        """テキスト要約 - Apertus-8Bの多言語要約機能を使用
        
        Args:
            text: 要約するテキスト
            max_length: 目標文字数
            source_lang: 入力言語 (auto-detect, English, Japanese, etc.)
            target_lang: 出力言語 (Japanese, English, etc.)
            style: 要約スタイル (bullet, narrative, balanced)
            summary_mode: 要約モード (short, long)
            
        Returns:
            成功時: {"success": True, "summary": "要約結果", "time": 5.2, ...}
            失敗時: {"success": False, "error": "エラーメッセージ"}
        """
        try:
            response = requests.post(
                f"{self.base_url}/summarize",
                json={
                    "text": text,
                    "max_length": max_length,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "style": style,
                    "summary_mode": summary_mode
                },
                headers=self._get_headers(),
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"タイムアウト ({self.timeout}秒)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def expand(self, text: str, target_length: int = 300, target_lang: str = "Japanese") -> Optional[Dict]:
        """短文展開 - Apertus-8Bで短いテキストを詳細な文章に展開
        
        Args:
            text: 展開する短文（300文字以下）
            target_length: 目標文字数（最大500）
            target_lang: 出力言語
            
        Returns:
            成功時: {"success": True, "result": "展開結果", "time": 5.2, ...}
            失敗時: {"success": False, "error": "エラーメッセージ"}
        """
        try:
            response = requests.post(
                f"{self.base_url}/expand",
                json={
                    "text": text,
                    "target_length": min(target_length, 500),
                    "target_lang": target_lang
                },
                headers=self._get_headers(),
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"タイムアウト ({self.timeout}秒)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def explain_code(self, code: str, language: str = "auto", target_lang: str = "Japanese") -> Optional[Dict]:
        """コード解説 - Apertus-8Bでプログラミングコードを解説
        
        Args:
            code: 解説するコード
            language: プログラミング言語（auto, Python, JavaScript等）
            target_lang: 解説の出力言語
            
        Returns:
            成功時: {"success": True, "explanation": "解説", "detected_language": "Python", ...}
            失敗時: {"success": False, "error": "エラーメッセージ"}
        """
        try:
            response = requests.post(
                f"{self.base_url}/explain-code",
                json={
                    "code": code,
                    "language": language,
                    "target_lang": target_lang
                },
                headers=self._get_headers(),
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"タイムアウト ({self.timeout}秒)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# デフォルトクライアント (config.pyから設定読み込み)
_default_client = None

def get_kaggle_client() -> Optional[KaggleAIClient]:
    """デフォルトKaggle AIクライアントを取得"""
    global _default_client
    
    if _default_client is None:
        from config import config  # ⭐ Configクラスではなくconfigインスタンスを使用
        
        kaggle_url = config.KAGGLE_API_URL
        kaggle_key = config.KAGGLE_API_KEY
        kaggle_timeout = config.KAGGLE_API_TIMEOUT
        
        if kaggle_url:
            _default_client = KaggleAIClient(
                base_url=kaggle_url,
                api_key=kaggle_key,
                timeout=kaggle_timeout
            )
            print(f"✅ Kaggle APIクライアント初期化: {kaggle_url}")
            print(f"   タイムアウト: {kaggle_timeout}秒")
        else:
            print("⚠️ KAGGLE_API_URL が設定されていません")
    
    return _default_client
