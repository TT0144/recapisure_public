#!/usr/bin/env python3
"""
Text Processing Models for recapisure
テキスト処理用のデータモデル
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

class ProcessingType(Enum):
    """処理タイプ"""
    SUMMARIZE = "summarize"
    EXPAND = "expand"
    URL_SUMMARIZE = "url_summarize"
    ANALYZE = "analyze"

class ProcessingStatus(Enum):
    """処理ステータス"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProcessingResult:
    """処理結果データクラス"""
    id: str
    timestamp: datetime
    type: ProcessingType
    status: ProcessingStatus
    
    # 入力データ
    original_text: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    
    # 出力データ
    result: Optional[str] = None
    
    # メタデータ
    execution_time: float = 0.0
    model_used: Optional[str] = None
    confidence: float = 0.0
    token_usage: Dict[str, int] = None
    
    # 統計情報
    original_length: int = 0
    result_length: int = 0
    compression_ratio: float = 0.0
    expansion_ratio: float = 0.0
    
    # エラー情報
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.token_usage is None:
            self.token_usage = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'type': self.type.value,
            'status': self.status.value,
            'original_text': self.original_text,
            'url': self.url,
            'title': self.title,
            'result': self.result,
            'execution_time': self.execution_time,
            'model_used': self.model_used,
            'confidence': self.confidence,
            'token_usage': self.token_usage,
            'original_length': self.original_length,
            'result_length': self.result_length,
            'compression_ratio': self.compression_ratio,
            'expansion_ratio': self.expansion_ratio,
            'error': self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingResult':
        """辞書から作成"""
        return cls(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            type=ProcessingType(data['type']),
            status=ProcessingStatus(data['status']),
            original_text=data.get('original_text'),
            url=data.get('url'),
            title=data.get('title'),
            result=data.get('result'),
            execution_time=data.get('execution_time', 0.0),
            model_used=data.get('model_used'),
            confidence=data.get('confidence', 0.0),
            token_usage=data.get('token_usage', {}),
            original_length=data.get('original_length', 0),
            result_length=data.get('result_length', 0),
            compression_ratio=data.get('compression_ratio', 0.0),
            expansion_ratio=data.get('expansion_ratio', 0.0),
            error=data.get('error')
        )

@dataclass
class ProcessingRequest:
    """処理リクエストデータクラス"""
    type: ProcessingType
    text: Optional[str] = None
    url: Optional[str] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    target_length: Optional[int] = None
    model: Optional[str] = None
    custom_prompt: Optional[str] = None
    
    def validate(self) -> bool:
        """リクエストの妥当性検証"""
        if self.type == ProcessingType.URL_SUMMARIZE:
            return bool(self.url)
        else:
            return bool(self.text)

@dataclass
class UserSession:
    """ユーザーセッションデータ"""
    session_id: str
    history: List[ProcessingResult] = None
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
        if self.preferences is None:
            self.preferences = {}
    
    def add_result(self, result: ProcessingResult):
        """結果を履歴に追加"""
        self.history.append(result)
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        if not self.history:
            return {
                'total_operations': 0,
                'average_execution_time': 0,
                'total_text_processed': 0,
                'success_rate': 0
            }
        
        completed = [r for r in self.history if r.status == ProcessingStatus.COMPLETED]
        total_time = sum(r.execution_time for r in completed)
        total_chars = sum(r.original_length for r in completed)
        
        return {
            'total_operations': len(self.history),
            'completed_operations': len(completed),
            'average_execution_time': total_time / len(completed) if completed else 0,
            'total_text_processed': total_chars,
            'success_rate': len(completed) / len(self.history) if self.history else 0
        }