#!/usr/bin/env python3
"""
AI Summarization Service using Apertus
Apertusを使用した高性能要約サービス
"""

import logging
from typing import Dict, Any, Optional
from .apertus_client import ApertusClient, ModelType, TaskType, ApertusRequest

logger = logging.getLogger(__name__)

class AISummarizationService:
    """Apertus AI を使用した要約サービス"""
    
    def __init__(self, api_key: Optional[str] = None, default_model: ModelType = ModelType.GPT4_TURBO):
        """
        AIサービス初期化
        
        Args:
            api_key: Apertus API キー
            default_model: デフォルトで使用するモデル
        """
        self.client = ApertusClient(api_key=api_key)
        self.default_model = default_model
        self.is_available = self.client.health_check()
        
        if self.client.mock_mode:
            logger.info("Running in mock mode - Set APERTUS_API_KEY for full functionality")
        else:
            logger.info(f"Connected to Apertus API - Default model: {default_model.value}")
    
    def summarize_text(self, text: str, max_length: int = 200, min_length: int = 50) -> Dict[str, Any]:
        """
        テキスト要約
        
        Args:
            text: 要約対象テキスト
            max_length: 要約の最大文字数
            min_length: 要約の最小文字数
            
        Returns:
            要約結果辞書
        """
        try:
            response = self.client.summarize(
                text=text,
                max_length=max_length,
                min_length=min_length,
                model=self.default_model
            )
            
            if response.success:
                return {
                    'success': True,
                    'summary': response.result,
                    'original_length': len(text),
                    'summary_length': len(response.result),
                    'compression_ratio': 1 - (len(response.result) / len(text)) if len(text) > 0 else 0,
                    'model_used': response.model_used,
                    'execution_time': response.execution_time,
                    'confidence': response.confidence,
                    'token_usage': response.token_usage
                }
            else:
                return {
                    'success': False,
                    'error': response.error or 'Unknown error occurred',
                    'summary': '',
                    'original_length': len(text),
                    'summary_length': 0,
                    'compression_ratio': 0
                }
                
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return {
                'success': False,
                'error': f'要約処理中にエラーが発生しました: {str(e)}',
                'summary': '',
                'original_length': len(text),
                'summary_length': 0,
                'compression_ratio': 0
            }
    
    def expand_text(self, text: str, target_length: int = 500) -> Dict[str, Any]:
        """
        テキスト展開
        
        Args:
            text: 展開対象テキスト
            target_length: 目標文字数
            
        Returns:
            展開結果辞書
        """
        try:
            response = self.client.expand(
                text=text,
                target_length=target_length,
                model=self.default_model
            )
            
            if response.success:
                return {
                    'success': True,
                    'expanded_text': response.result,
                    'original_length': len(text),
                    'expanded_length': len(response.result),
                    'expansion_ratio': len(response.result) / len(text) if len(text) > 0 else 0,
                    'model_used': response.model_used,
                    'execution_time': response.execution_time,
                    'confidence': response.confidence,
                    'token_usage': response.token_usage
                }
            else:
                return {
                    'success': False,
                    'error': response.error or 'Unknown error occurred',
                    'expanded_text': '',
                    'original_length': len(text),
                    'expanded_length': 0,
                    'expansion_ratio': 0
                }
                
        except Exception as e:
            logger.error(f"Expansion error: {e}")
            return {
                'success': False,
                'error': f'展開処理中にエラーが発生しました: {str(e)}',
                'expanded_text': '',
                'original_length': len(text),
                'expanded_length': 0,
                'expansion_ratio': 0
            }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        テキスト分析
        
        Args:
            text: 分析対象テキスト
            
        Returns:
            分析結果辞書
        """
        try:
            response = self.client.analyze(
                text=text,
                model=self.default_model
            )
            
            if response.success:
                return {
                    'success': True,
                    'analysis': response.result,
                    'original_length': len(text),
                    'model_used': response.model_used,
                    'execution_time': response.execution_time,
                    'confidence': response.confidence,
                    'token_usage': response.token_usage
                }
            else:
                return {
                    'success': False,
                    'error': response.error or 'Unknown error occurred',
                    'analysis': ''
                }
                
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                'success': False,
                'error': f'分析処理中にエラーが発生しました: {str(e)}',
                'analysis': ''
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        サービス状態取得
        
        Returns:
            サービス状態辞書
        """
        return {
            'available': self.is_available,
            'mock_mode': self.client.mock_mode,
            'default_model': self.default_model.value,
            'available_models': self.client.get_available_models()
        }
    
    def change_model(self, model_name: str) -> bool:
        """
        使用モデル変更
        
        Args:
            model_name: 新しいモデル名
            
        Returns:
            変更成功可否
        """
        try:
            # ModelTypeからモデルを検索
            for model in ModelType:
                if model.value == model_name:
                    self.default_model = model
                    logger.info(f"Model changed to: {model_name}")
                    return True
            
            logger.warning(f"Unknown model: {model_name}")
            return False
            
        except Exception as e:
            logger.error(f"Model change error: {e}")
            return False