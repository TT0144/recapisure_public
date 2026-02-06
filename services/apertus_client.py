#!/usr/bin/env python3
"""
Apertus API Client for recapisure
Apertus AIプラットフォームとの連携クライアント
"""

import os
import json
import time
import logging
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """使用可能なモデルタイプ"""
    GPT4_TURBO = "gpt-4-turbo"
    GPT4 = "gpt-4"
    GPT35_TURBO = "gpt-3.5-turbo"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    GEMINI_PRO = "gemini-pro"

class TaskType(Enum):
    """タスクタイプ"""
    SUMMARIZE = "summarize"
    EXPAND = "expand"
    ANALYZE = "analyze"
    TRANSLATE = "translate"

@dataclass
class ApertusRequest:
    """Apertus APIリクエストデータクラス"""
    text: str
    task_type: TaskType
    model: ModelType
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    temperature: float = 0.7
    language: str = "ja"
    custom_prompt: Optional[str] = None

@dataclass
class ApertusResponse:
    """Apertus APIレスポンスデータクラス"""
    success: bool
    result: str
    model_used: str
    execution_time: float
    token_usage: Dict[str, int]
    confidence: float
    metadata: Dict[str, Any]
    error: Optional[str] = None

class ApertusClient:
    """Apertus LLM クライアント (swiss-ai/Apertus-8B-Instruct-2509)"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Apertus クライアント初期化
        
        Note:
            api_key, base_urlは後方互換性のため残していますが、
            実際にはローカルのApertus LLMを使用します。
        """
        # ローカルApertusサービスを使用
        from .apertus_service import get_apertus_service
        self.apertus = get_apertus_service()
        self.mock_mode = not self.apertus.available
        
        if self.mock_mode:
            logger.warning("⚠️ Apertus LLM利用不可。モックモードで動作します。")
        else:
            logger.info("✅ Apertus LLM (swiss-ai/Apertus-8B-Instruct-2509) を使用")
        
        # デフォルト設定
        self.default_model = ModelType.GPT4_TURBO  # 互換性のため残す
        self.request_timeout = 30
        self.max_retries = 3
    
    # 旧API呼び出しメソッド削除（ローカルLLM使用のため不要）
    
    def _create_prompt(self, request: ApertusRequest) -> str:
        """タスクタイプに応じたプロンプト生成（多言語対応）"""
        if request.custom_prompt:
            return request.custom_prompt
        
        # 言語判定
        japanese_chars = sum(1 for c in request.text[:1000] if ord(c) > 0x3000)
        is_japanese = japanese_chars > 50
        
        if request.task_type == TaskType.SUMMARIZE:
            if is_japanese:
                # 日本語テキストの要約
                return f"""
以下のテキストを{request.min_length or 50}文字以上{request.max_length or 200}文字以下で要約してください。
重要なポイントを漏らさず、簡潔で分かりやすい日本語で要約してください。

テキスト:
{request.text}

要約:
"""
            else:
                # 英語テキスト → 日本語翻訳要約
                max_len = request.max_length or 400
                return f"""
以下の英語テキストを日本語に翻訳した上で、約{max_len}文字程度で要約してください。

処理手順:
1. まず英語の内容を正確に理解する
2. テキスト全体を日本語に翻訳する感覚で読み解く
3. 翻訳された内容から重要なポイントを抽出して要約する

要求事項:
- 論文の主要なポイントを日本語で明確に記述
- 専門用語は適切な日本語訳を使用（必要に応じて英語を併記）
- 自然で読みやすい日本語の要約にする
- 論文の論理構成を保つ

【英語テキスト】
{request.text}

【日本語翻訳要約】
"""
        
        base_prompts = {
            TaskType.EXPAND: f"""
以下の短いテキストを詳細で具体的な文章に展開してください。
目標文字数: {request.max_length or 500}文字程度
元の意味を保ちながら、背景情報や詳細説明を追加してください。

元のテキスト:
{request.text}

展開された文章:
""",
            TaskType.ANALYZE: f"""
以下のテキストを分析し、主要なポイント、論点、含意を抽出してください。

テキスト:
{request.text}

分析結果:
"""
        }
        
        return base_prompts.get(request.task_type, request.text)
    
    def _mock_response(self, request: ApertusRequest) -> ApertusResponse:
        """モックレスポンス生成（API キーが無い場合）"""
        time.sleep(1)  # 実際のAPI遅延をシミュレート
        
        if request.task_type == TaskType.SUMMARIZE:
            # 言語判定（日本語文字が50文字以上あれば日本語）
            japanese_chars = sum(1 for c in request.text[:1000] if ord(c) > 0x3000)
            is_japanese = japanese_chars > 50
            
            if is_japanese:
                # 日本語要約
                sentences = request.text.split('。')
                summary_sentences = []
                current_length = 0
                max_len = request.max_length or 200
                
                for sentence in sentences:
                    if sentence.strip():
                        sentence = sentence.strip() + '。'
                        if current_length + len(sentence) <= max_len:
                            summary_sentences.append(sentence)
                            current_length += len(sentence)
                        else:
                            break
                
                result = ''.join(summary_sentences)
                if not result:
                    result = request.text[:max_len] + "..." if len(request.text) > max_len else request.text
            else:
                # 英語テキスト → 日本語翻訳要約
                total_chars = len(request.text)
                mock_summary = f"""【Apertusモック翻訳要約（英語→日本語）】
📊 元テキスト: {total_chars:,}文字（英語）

本論文では、重要な研究テーマについて包括的に論じています。研究者らは特定の手法を用いて実験を行い、興味深い知見を得ました。得られた結果は、当該分野において重要な意義を持つと考えられます。

━━━━━━━━━━━━━━━━━━━━━━━━
💡 実際のApertus API使用時:
━━━━━━━━━━━━━━━━━━━━━━━━

Apertusは複数の高性能AIモデル（GPT-4, Claude 3, Gemini Pro）を使用して、英語論文を正確に理解し、自然な日本語で翻訳要約します。

【使用可能なモデル】
- GPT-4 Turbo: 最高品質の翻訳・要約
- Claude 3 Sonnet: 長文理解に優れる
- Gemini Pro: バランスの取れた高速処理

※ APERTUS_API_KEYを設定すると、これらのモデルを利用できます。
━━━━━━━━━━━━━━━━━━━━━━━━"""
                result = mock_summary
                
        elif request.task_type == TaskType.EXPAND:
            # 簡単な展開ロジック
            base_expansion = f"{request.text}について詳しく説明すると、これは現代社会において重要な要素の一つです。"
            base_expansion += "この概念は多角的な視点から理解することができ、様々な分野に応用されています。"
            base_expansion += "さらに詳細な分析を行うと、その背景には複数の要因が関係していることが分かります。"
            
            target_len = request.max_length or 500
            result = base_expansion[:target_len] if len(base_expansion) > target_len else base_expansion
            
        else:
            result = f"[Mock] {request.task_type.value} result for: {request.text[:100]}..."
        
        return ApertusResponse(
            success=True,
            result=result,
            model_used=f"mock-{request.model.value}",
            execution_time=1.0,
            token_usage={"input": len(request.text), "output": len(result)},
            confidence=0.85,
            metadata={"mock": True, "timestamp": time.time()}
        )
    
    def process_text(self, request: ApertusRequest) -> ApertusResponse:
        """テキスト処理メイン関数（Apertus LLM使用）"""
        start_time = time.time()
        
        try:
            if self.mock_mode:
                return self._mock_response(request)
            
            # プロンプト生成
            prompt = self._create_prompt(request)
            
            # Apertus LLMで処理
            if request.task_type == TaskType.SUMMARIZE:
                # 要約タスク
                max_len = request.max_length or 400
                response = self.apertus.summarize(
                    text=request.text,
                    max_length=max_len
                )
            elif request.task_type == TaskType.EXPAND:
                # 展開タスク
                target_len = request.max_length or 500
                response = self.apertus.expand(
                    text=request.text,
                    target_length=target_len,
                    prompt_template=prompt if request.custom_prompt else None
                )
            else:
                # その他のタスク（汎用生成）
                response = self.apertus.generate(
                    prompt=prompt,
                    max_new_tokens=request.max_length or 512,
                    temperature=request.temperature
                )
            
            execution_time = time.time() - start_time
            
            # ApertusResponseに変換
            if response.success:
                return ApertusResponse(
                    success=True,
                    result=response.result,
                    model_used=response.model_used,
                    execution_time=execution_time,
                    token_usage=response.token_usage,
                    confidence=response.confidence,
                    metadata={
                        "timestamp": time.time(),
                        "task_type": request.task_type.value,
                        "apertus_version": "8B-Instruct-2509"
                    }
                )
            else:
                # Apertus LLMからのエラー
                return ApertusResponse(
                    success=False,
                    result="",
                    model_used=response.model_used or "unknown",
                    execution_time=execution_time,
                    token_usage={},
                    confidence=0.0,
                    metadata={"error_timestamp": time.time()},
                    error=response.error or "Unknown error from Apertus LLM"
                )
            
        except Exception as e:
            logger.error(f"Apertus処理エラー: {e}")
            execution_time = time.time() - start_time
            
            return ApertusResponse(
                success=False,
                result="",
                model_used=request.model.value,
                execution_time=execution_time,
                token_usage={},
                confidence=0.0,
                metadata={"error_timestamp": time.time()},
                error=str(e)
            )
    
    def summarize(self, text: str, max_length: int = 200, min_length: int = 50, 
                  model: ModelType = None) -> ApertusResponse:
        """テキスト要約"""
        request = ApertusRequest(
            text=text,
            task_type=TaskType.SUMMARIZE,
            model=model or self.default_model,
            max_length=max_length,
            min_length=min_length
        )
        return self.process_text(request)
    
    def expand(self, text: str, target_length: int = 500, 
               model: ModelType = None) -> ApertusResponse:
        """テキスト展開"""
        request = ApertusRequest(
            text=text,
            task_type=TaskType.EXPAND,
            model=model or self.default_model,
            max_length=target_length
        )
        return self.process_text(request)
    
    def analyze(self, text: str, model: ModelType = None) -> ApertusResponse:
        """テキスト分析"""
        request = ApertusRequest(
            text=text,
            task_type=TaskType.ANALYZE,
            model=model or self.default_model
        )
        return self.process_text(request)
    
    def health_check(self) -> bool:
        """Apertus LLMの状態確認"""
        if self.mock_mode:
            return True
        
        try:
            status = self.apertus.get_status()
            return status.success and self.apertus.available
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """使用可能なモデル一覧取得"""
        if self.mock_mode:
            return [model.value for model in ModelType]
        
        # Apertus LLM使用時
        if self.apertus.available:
            return ["swiss-ai/Apertus-8B-Instruct-2509"]
        else:
            return []