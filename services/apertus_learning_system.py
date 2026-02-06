#!/usr/bin/env python3
"""
Apertus Learning System
要約・翻訳スコアを学習して精度を向上させるシステム
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .apertus_client import ApertusClient, ModelType, TaskType, ApertusRequest

logger = logging.getLogger(__name__)


@dataclass
class FeedbackScore:
    """フィードバックスコア"""
    task_id: str
    original_text: str
    result_text: str
    user_score: float  # 1-5の評価
    accuracy_score: float  # 正確性
    fluency_score: float  # 流暢性
    completeness_score: float  # 完全性
    timestamp: str
    task_type: str  # summarize/translate/expand
    model_used: str
    user_feedback: Optional[str] = None


@dataclass
class LearningMetrics:
    """学習メトリクス"""
    total_tasks: int
    average_score: float
    accuracy_trend: List[float]  # 時系列の正確性
    fluency_trend: List[float]  # 時系列の流暢性
    best_score: float
    worst_score: float
    improvement_rate: float  # 改善率


class ApertusLearningSystem:
    """
    Apertusを使った学習型要約・翻訳システム
    
    機能:
    - ユーザーフィードバックを収集
    - スコアを学習・分析
    - プロンプトを動的に最適化
    - 精度を継続的に向上
    """
    
    def __init__(self, apertus_client: Optional[ApertusClient] = None):
        """
        初期化
        
        Args:
            apertus_client: Apertusクライアント
        """
        self.client = apertus_client or ApertusClient()
        
        # データ保存ディレクトリ
        self.data_dir = Path("data/learning")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # フィードバックデータファイル
        self.feedback_file = self.data_dir / "feedback_scores.jsonl"
        self.metrics_file = self.data_dir / "learning_metrics.json"
        
        # メモリ内キャッシュ
        self.feedback_history: List[FeedbackScore] = []
        self.metrics: Optional[LearningMetrics] = None
        
        # 学習済みパラメータ
        self.learned_params = self._load_learned_params()
        
        # データロード
        self._load_feedback_history()
        self._calculate_metrics()
        
        logger.info(f"✅ 学習システム初期化完了: {len(self.feedback_history)}件のフィードバック")
    
    def _load_learned_params(self) -> Dict[str, Any]:
        """学習済みパラメータをロード"""
        param_file = self.data_dir / "learned_params.json"
        
        if param_file.exists():
            with open(param_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # デフォルトパラメータ
        return {
            "temperature": 0.7,
            "max_length_ratio": 0.3,  # 要約時の元テキストに対する長さ比率
            "min_length_ratio": 0.1,
            "prompt_style": "default",  # default/detailed/concise
            "translation_formality": "neutral",  # formal/neutral/casual
        }
    
    def _save_learned_params(self):
        """学習済みパラメータを保存"""
        param_file = self.data_dir / "learned_params.json"
        with open(param_file, 'w', encoding='utf-8') as f:
            json.dump(self.learned_params, f, ensure_ascii=False, indent=2)
    
    def _load_feedback_history(self):
        """フィードバック履歴をロード"""
        if not self.feedback_file.exists():
            return
        
        with open(self.feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.feedback_history.append(FeedbackScore(**data))
    
    def _save_feedback(self, feedback: FeedbackScore):
        """フィードバックを保存"""
        with open(self.feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(feedback), ensure_ascii=False) + '\n')
    
    def _calculate_metrics(self):
        """メトリクスを計算"""
        if not self.feedback_history:
            self.metrics = LearningMetrics(
                total_tasks=0,
                average_score=0.0,
                accuracy_trend=[],
                fluency_trend=[],
                best_score=0.0,
                worst_score=0.0,
                improvement_rate=0.0
            )
            return
        
        scores = [fb.user_score for fb in self.feedback_history]
        accuracy_scores = [fb.accuracy_score for fb in self.feedback_history]
        fluency_scores = [fb.fluency_score for fb in self.feedback_history]
        
        # 改善率計算（最初の10件と最新10件を比較）
        improvement_rate = 0.0
        if len(scores) >= 20:
            early_avg = sum(scores[:10]) / 10
            recent_avg = sum(scores[-10:]) / 10
            improvement_rate = ((recent_avg - early_avg) / early_avg) * 100
        
        self.metrics = LearningMetrics(
            total_tasks=len(self.feedback_history),
            average_score=sum(scores) / len(scores),
            accuracy_trend=accuracy_scores[-50:],  # 直近50件
            fluency_trend=fluency_scores[-50:],
            best_score=max(scores),
            worst_score=min(scores),
            improvement_rate=improvement_rate
        )
        
        # メトリクスを保存
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.metrics), f, ensure_ascii=False, indent=2)
    
    def add_feedback(
        self,
        task_id: str,
        original_text: str,
        result_text: str,
        user_score: float,
        accuracy_score: float,
        fluency_score: float,
        completeness_score: float,
        task_type: str,
        model_used: str,
        user_feedback: Optional[str] = None
    ) -> bool:
        """
        フィードバックを追加
        
        Args:
            task_id: タスクID
            original_text: 元のテキスト
            result_text: 生成結果
            user_score: ユーザー評価 (1-5)
            accuracy_score: 正確性スコア (0-100)
            fluency_score: 流暢性スコア (0-100)
            completeness_score: 完全性スコア (0-100)
            task_type: タスクタイプ (summarize/translate/expand)
            model_used: 使用モデル
            user_feedback: ユーザーコメント
        
        Returns:
            成功したらTrue
        """
        try:
            feedback = FeedbackScore(
                task_id=task_id,
                original_text=original_text[:500],  # 保存容量削減のため最大500文字
                result_text=result_text[:500],
                user_score=user_score,
                accuracy_score=accuracy_score,
                fluency_score=fluency_score,
                completeness_score=completeness_score,
                timestamp=datetime.now().isoformat(),
                task_type=task_type,
                model_used=model_used,
                user_feedback=user_feedback
            )
            
            # メモリに追加
            self.feedback_history.append(feedback)
            
            # ファイルに保存
            self._save_feedback(feedback)
            
            # メトリクス再計算
            self._calculate_metrics()
            
            # パラメータ最適化 (10件ごと)
            if len(self.feedback_history) % 10 == 0:
                self._optimize_params_from_feedback()
            
            logger.info(f"✅ フィードバック追加: task_id={task_id}, score={user_score}")
            return True
            
        except Exception as e:
            logger.error(f"❌ フィードバック追加エラー: {e}")
            return False
    
    def get_metrics(self) -> LearningMetrics:
        """現在のメトリクスを取得"""
        if self.metrics is None:
            self._calculate_metrics()
        return self.metrics
    
    def _optimize_params_from_feedback(self):
        """フィードバックからパラメータを最適化"""
        if len(self.feedback_history) < 10:
            return  # データが少ない場合は最適化しない
        
        recent_feedback = self.feedback_history[-50:]  # 直近50件
        
        # 高評価のタスクの特徴を分析
        high_score_tasks = [fb for fb in recent_feedback if fb.user_score >= 4.0]
        
        if high_score_tasks:
            # 高評価タスクの平均長さ比率を計算
            # （ここでは簡易的な実装。実際はより詳細な分析が可能）
            avg_accuracy = sum(fb.accuracy_score for fb in high_score_tasks) / len(high_score_tasks)
            avg_fluency = sum(fb.fluency_score for fb in high_score_tasks) / len(high_score_tasks)
            
            # パラメータ調整
            if avg_accuracy > 4.0 and avg_fluency > 4.0:
                # 高品質なので現在のパラメータを維持
                pass
            elif avg_accuracy < 3.5:
                # 正確性が低い → より詳細なプロンプトに
                self.learned_params["prompt_style"] = "detailed"
                self.learned_params["max_length_ratio"] = min(0.4, self.learned_params["max_length_ratio"] + 0.05)
            elif avg_fluency < 3.5:
                # 流暢性が低い → より簡潔なプロンプトに
                self.learned_params["prompt_style"] = "concise"
                self.learned_params["temperature"] = min(0.9, self.learned_params["temperature"] + 0.1)
            
            self._save_learned_params()
            logger.info(f"📊 パラメータ最適化完了: {self.learned_params}")
    
    def _get_optimized_prompt(self, text: str, task_type: TaskType) -> str:
        """学習結果を反映した最適化プロンプト生成"""
        style = self.learned_params["prompt_style"]
        
        if task_type == TaskType.SUMMARIZE:
            max_len = int(len(text) * self.learned_params["max_length_ratio"])
            min_len = int(len(text) * self.learned_params["min_length_ratio"])
            
            if style == "detailed":
                return f"""
以下のテキストを{min_len}文字以上{max_len}文字以下で詳細に要約してください。

【要求事項】
- 重要なポイントを全て含める
- 具体的な数値や固有名詞を保持
- 論理的な流れを維持
- 読みやすく自然な日本語で

【テキスト】
{text}

【要約】
"""
            elif style == "concise":
                return f"""
以下のテキストを{max_len}文字以下で簡潔に要約してください。
最も重要なポイントのみを抽出し、明確で読みやすい日本語で記述してください。

テキスト: {text}

要約:
"""
            else:  # default
                return f"""
以下のテキストを{min_len}文字以上{max_len}文字以下で要約してください。
重要なポイントを漏らさず、簡潔で分かりやすい日本語で要約してください。

テキスト: {text}

要約:
"""
        
        elif task_type == TaskType.TRANSLATE:
            formality = self.learned_params["translation_formality"]
            
            formality_instruction = {
                "formal": "丁寧で格式高い日本語に翻訳してください。",
                "neutral": "自然で読みやすい日本語に翻訳してください。",
                "casual": "親しみやすく分かりやすい日本語に翻訳してください。"
            }
            
            return f"""
以下の英語テキストを日本語に翻訳してください。

【翻訳方針】
{formality_instruction[formality]}
専門用語は適切な日本語訳を使用し、必要に応じて英語を併記してください。
原文の意図とニュアンスを正確に伝えてください。

【英語テキスト】
{text}

【日本語翻訳】
"""
        
        return text
    
    def process_with_learning(
        self, 
        text: str, 
        task_type: TaskType, 
        model: ModelType = ModelType.GPT4_TURBO
    ) -> Tuple[Dict[str, Any], str]:
        """
        学習機能付きでテキスト処理
        
        Args:
            text: 処理対象テキスト
            task_type: タスクタイプ
            model: 使用モデル
            
        Returns:
            (結果辞書, タスクID)
        """
        # タスクID生成
        task_id = f"{task_type.value}_{int(time.time() * 1000)}"
        
        # 最適化されたプロンプト生成
        prompt = self._get_optimized_prompt(text, task_type)
        
        # Apertus APIリクエスト
        request = ApertusRequest(
            text=text,
            task_type=task_type,
            model=model,
            temperature=self.learned_params["temperature"],
            custom_prompt=prompt
        )
        
        start_time = time.time()
        response = self.client.process_text(request)
        execution_time = time.time() - start_time
        
        result = {
            'success': response.success,
            'result': response.result,
            'task_id': task_id,
            'model_used': response.model_used,
            'execution_time': execution_time,
            'confidence': response.confidence,
            'learned_params_used': self.learned_params.copy(),
            'total_learning_samples': len(self.feedback_history),
            'average_score': self.metrics.average_score if self.metrics else 0.0,
        }
        
        if not response.success:
            result['error'] = response.error
        
        return result, task_id
    
    def submit_feedback(
        self,
        task_id: str,
        original_text: str,
        result_text: str,
        user_score: float,
        accuracy_score: float,
        fluency_score: float,
        completeness_score: float,
        task_type: str,
        model_used: str,
        user_feedback: Optional[str] = None
    ):
        """
        ユーザーフィードバックを送信
        
        Args:
            task_id: タスクID
            original_text: 元のテキスト
            result_text: 結果テキスト
            user_score: 総合評価 (1-5)
            accuracy_score: 正確性 (1-5)
            fluency_score: 流暢性 (1-5)
            completeness_score: 完全性 (1-5)
            task_type: タスクタイプ
            model_used: 使用モデル
            user_feedback: テキストフィードバック
        """
        feedback = FeedbackScore(
            task_id=task_id,
            original_text=original_text[:500],  # 長すぎる場合は切り詰め
            result_text=result_text[:1000],
            user_score=user_score,
            accuracy_score=accuracy_score,
            fluency_score=fluency_score,
            completeness_score=completeness_score,
            timestamp=datetime.now().isoformat(),
            task_type=task_type,
            model_used=model_used,
            user_feedback=user_feedback
        )
        
        # フィードバック保存
        self.feedback_history.append(feedback)
        self._save_feedback(feedback)
        
        # メトリクス再計算
        self._calculate_metrics()
        
        # パラメータ最適化（10件ごと）
        if len(self.feedback_history) % 10 == 0:
            self._optimize_params_from_feedback()
            logger.info(f"🎓 学習完了: {len(self.feedback_history)}件のフィードバックから最適化")
        
        logger.info(f"✅ フィードバック受信: タスク{task_id}, スコア{user_score}/5")
        
        return feedback  # 追加: フィードバックオブジェクトを返す
    
    def get_metrics(self) -> LearningMetrics:
        """学習メトリクスを取得"""
        if not self.metrics:
            self._calculate_metrics()
        return self.metrics
    
    def get_learning_status(self) -> Dict[str, Any]:
        """学習状態を取得"""
        if not self.metrics:
            return {"status": "no_data"}
        
        return {
            "total_feedback": self.metrics.total_tasks,
            "average_score": round(self.metrics.average_score, 2),
            "best_score": round(self.metrics.best_score, 2),
            "worst_score": round(self.metrics.worst_score, 2),
            "improvement_rate": round(self.metrics.improvement_rate, 2),
            "current_params": self.learned_params,
            "recent_accuracy_avg": round(sum(self.metrics.accuracy_trend[-10:]) / 10, 2) if self.metrics.accuracy_trend else 0,
            "recent_fluency_avg": round(sum(self.metrics.fluency_trend[-10:]) / 10, 2) if self.metrics.fluency_trend else 0,
        }
    
    def get_performance_chart_data(self) -> Dict[str, List]:
        """パフォーマンスチャート用データ取得"""
        if not self.feedback_history:
            return {"labels": [], "scores": [], "accuracy": [], "fluency": []}
        
        recent = self.feedback_history[-50:]  # 直近50件
        
        return {
            "labels": [f"#{i+1}" for i in range(len(recent))],
            "scores": [fb.user_score for fb in recent],
            "accuracy": [fb.accuracy_score for fb in recent],
            "fluency": [fb.fluency_score for fb in recent],
        }


# グローバルインスタンス
_learning_system = None

def get_learning_system(apertus_client: Optional[ApertusClient] = None) -> ApertusLearningSystem:
    """学習システムのシングルトン取得"""
    global _learning_system
    if _learning_system is None:
        _learning_system = ApertusLearningSystem(apertus_client)
    return _learning_system
