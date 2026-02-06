#!/usr/bin/env python3
"""
Apertus LLM Service
スイス政府製オープンソースLLM (swiss-ai/Apertus-8B-Instruct-2509)
または軽量代替モデル (rinna/japanese-gpt2-medium)
"""

import os
import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Apertus利用可能チェック
APERTUS_AVAILABLE = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    APERTUS_AVAILABLE = True
    logger.info("✅ Apertus LLM利用可能")
except ImportError as e:
    logger.warning(f"⚠️ Apertus LLM利用不可: {e}")


# 🇨🇭 利用可能なモデル定義
AVAILABLE_MODELS = {
    "apertus-8b": {
        "model_id": "swiss-ai/Apertus-8B-Instruct-2509",
        "name": "Apertus 8B (Swiss AI)",
        "size": "8B parameters",
        "requires_gpu": True,
        "memory_gb": 16,
        "description": "スイス政府製オープンソースLLM (高性能)"
    },
    "gpt2-small": {
        "model_id": "gpt2",  # OpenAI GPT-2 (動作確認済み)
        "name": "GPT-2 Small",
        "size": "124M parameters",
        "requires_gpu": False,
        "memory_gb": 1,
        "description": "軽量汎用モデル (CPU動作可、動作確認済み)"
    },
    "rinna-bilingual": {
        "model_id": "rinna/bilingual-gpt-neox-4b",
        "name": "Rinna Bilingual 4B",
        "size": "4B parameters",
        "requires_gpu": True,
        "memory_gb": 8,
        "description": "日英バイリンガルモデル (中性能)"
    }
}


def get_recommended_model() -> str:
    """
    システム環境に応じた推奨モデルを返す
    
    Returns:
        推奨モデルのID
    """
    try:
        import torch
        if torch.cuda.is_available():
            # GPU利用可能 → Apertus 8Bを推奨
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory >= 16:
                logger.info("🎯 推奨: Apertus 8B (GPU 16GB+)")
                return "apertus-8b"
            elif gpu_memory >= 8:
                logger.info("🎯 推奨: Rinna Bilingual 4B (GPU 8GB+)")
                return "rinna-bilingual"
        
        # CPU環境 → 軽量モデル
        logger.info("🎯 推奨: GPT-2 Small (CPU)")
        return "gpt2-small"
    except:
        return "gpt2-small"


@dataclass
class ApertusResponse:
    """Apertusレスポンス"""
    success: bool
    result: str
    model_used: str
    execution_time: float = 0.0
    confidence: float = 0.90
    token_usage: Dict[str, int] = None
    error: Optional[str] = None


class ApertusService:
    """Apertus AI Service (スイス政府製8B LLM)"""
    
    def __init__(self, model_name: str = None):
        """
        初期化
        
        Args:
            model_name: モデル指定 (None=自動選択, "apertus-8b", "rinna-medium", etc.)
        """
        # モデル自動選択
        if model_name is None:
            model_key = get_recommended_model()
            model_name = AVAILABLE_MODELS[model_key]["model_id"]
            logger.info(f"🤖 自動選択: {AVAILABLE_MODELS[model_key]['name']}")
        elif model_name in AVAILABLE_MODELS:
            # ショートカット名からモデルIDを取得
            model_name = AVAILABLE_MODELS[model_name]["model_id"]
        
        self.model_name = model_name
        self.available = APERTUS_AVAILABLE
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
        # 4bit量子化設定
        self.use_4bit = True  # メモリ節約のため4bit量子化を使用
        
        # Apertus互換モード
        self.is_apertus_compatible = "apertus" in model_name.lower() or "swiss" in model_name.lower()
        
        logger.info(f"🇨🇭 Apertus Service初期化: {model_name}")
        logger.info(f"   4bit量子化: {'有効' if self.use_4bit else '無効'}")
        logger.info(f"   Apertus互換: {'はい' if self.is_apertus_compatible else 'いいえ (代替モデル)'}")
    
    def load_model(self) -> bool:
        """
        モデルをロード (初回のみ)
        
        Returns:
            成功したらTrue
        """
        if self.loaded:
            return True
        
        if not self.available:
            logger.error("❌ Transformersライブラリが利用できません")
            return False
        
        try:
            logger.info(f"📥 Apertusモデルをロード中... ({self.model_name})")
            start_time = time.time()
            
            # トークナイザーをロード
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.use_4bit:
                # 4bit量子化でロード (メモリ効率化)
                logger.info("   🔧 4bit量子化モードでロード中...")
                
                try:
                    from transformers import BitsAndBytesConfig
                    
                    # 4bit量子化設定
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=quantization_config,
                        device_map="auto"
                    )
                    logger.info("   ✅ 4bit量子化ロード成功")
                    
                except ImportError:
                    logger.warning("   ⚠️ bitsandbytesが未インストール。通常ロードにフォールバック")
                    self.use_4bit = False
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
            else:
                # 通常ロード (FP16)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            load_time = time.time() - start_time
            self.loaded = True
            
            logger.info(f"✅ Apertusモデルロード完了 ({load_time:.1f}秒)")
            logger.info(f"   モデル: {self.model_name}")
            logger.info(f"   量子化: {'4bit' if self.use_4bit else 'FP16'}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Apertusモデルロード失敗: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> ApertusResponse:
        """
        テキスト生成
        
        Args:
            prompt: 入力プロンプト
            max_new_tokens: 最大生成トークン数
            temperature: 温度パラメータ (0.0-1.0)
            top_p: Top-p sampling (0.0-1.0)
            do_sample: サンプリングを使用するか
        
        Returns:
            ApertusResponse
        """
        if not self.available:
            return ApertusResponse(
                success=False,
                result="",
                model_used="unavailable",
                error="Transformersライブラリが利用できません"
            )
        
        # モデルをロード (初回のみ)
        if not self.loaded:
            if not self.load_model():
                return ApertusResponse(
                    success=False,
                    result="",
                    model_used=self.model_name,
                    error="モデルのロードに失敗しました"
                )
        
        try:
            start_time = time.time()
            
            # トークナイズ
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # デバイスに移動
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # デコード
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # プロンプト部分を除去
            if generated_text.startswith(prompt):
                result = generated_text[len(prompt):].strip()
            else:
                result = generated_text
            
            execution_time = time.time() - start_time
            
            # トークン使用量
            input_tokens = inputs['input_ids'].shape[1]
            output_tokens = outputs.shape[1]
            
            logger.info(f"✅ Apertus生成完了 ({execution_time:.1f}秒)")
            logger.info(f"   入力: {input_tokens} tokens, 出力: {output_tokens} tokens")
            
            return ApertusResponse(
                success=True,
                result=result,
                model_used=self.model_name,
                execution_time=execution_time,
                confidence=0.92,
                token_usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens - input_tokens,
                    "total_tokens": output_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Apertus生成エラー: {e}")
            return ApertusResponse(
                success=False,
                result="",
                model_used=self.model_name,
                error=str(e)
            )
    
    def summarize(self, text: str, max_length: int = 200) -> ApertusResponse:
        """
        テキスト要約
        
        Args:
            text: 要約するテキスト
            max_length: 最大文字数
        
        Returns:
            ApertusResponse
        """
        prompt = f"""以下のテキストを{max_length}文字程度で簡潔に要約してください。

テキスト:
{text}

要約:"""
        
        return self.generate(
            prompt=prompt,
            max_new_tokens=max_length * 2,  # 日本語は1文字≈2トークン
            temperature=0.3,  # 要約は決定的に
            do_sample=True
        )
    
    def expand(self, text: str, target_length: int = 500) -> ApertusResponse:
        """
        短文展開
        
        Args:
            text: 展開する短文
            target_length: 目標文字数
        
        Returns:
            ApertusResponse
        """
        prompt = f"""以下の短い文章を、約{target_length}文字程度の詳細な文章に展開してください。

元の文章:
{text}

詳細な文章:"""
        
        return self.generate(
            prompt=prompt,
            max_new_tokens=target_length * 2,
            temperature=0.7,
            do_sample=True
        )
    
    def get_status(self) -> ApertusResponse:
        """サービスステータス"""
        # モデル情報を取得
        model_info = None
        for key, info in AVAILABLE_MODELS.items():
            if info["model_id"] == self.model_name:
                model_info = info
                break
        
        status_info = {
            'service': 'Apertus LLM',
            'model_name': self.model_name,
            'model_display_name': model_info['name'] if model_info else self.model_name,
            'model_size': model_info['size'] if model_info else 'Unknown',
            'available': self.available,
            'loaded': self.loaded,
            'quantization': '4bit' if self.use_4bit else 'FP16',
            'device': 'auto',
            'api_key_required': False,
            'completely_free': True,
            'developer': 'Swiss AI (スイス政府)' if self.is_apertus_compatible else 'Rinna Co., Ltd.',
            'is_apertus_official': self.is_apertus_compatible,
            'memory_requirement_gb': model_info['memory_gb'] if model_info else 'Unknown'
        }
        
        return ApertusResponse(
            success=True,
            result=status_info,
            model_used=self.model_name,
            execution_time=0.0,
            confidence=1.0
        )
    
    def list_available_models(self) -> Dict[str, Any]:
        """利用可能なモデル一覧を取得"""
        return AVAILABLE_MODELS


# グローバルインスタンス
_apertus_service = None

def get_apertus_service() -> ApertusService:
    """ApertusServiceのシングルトン取得"""
    global _apertus_service
    if _apertus_service is None:
        _apertus_service = ApertusService()
    return _apertus_service
