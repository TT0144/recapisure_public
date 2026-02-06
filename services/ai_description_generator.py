#!/usr/bin/env python3
"""
AI Description Generator
Apertusを使った説明文の自動生成サービス
"""

import logging
from typing import Optional
from .apertus_client import ApertusClient, ModelType, TaskType, ApertusRequest

logger = logging.getLogger(__name__)


class AIDescriptionGenerator:
    """AI説明文自動生成クラス"""
    
    def __init__(self, apertus_client: Optional[ApertusClient] = None):
        """
        初期化
        
        Args:
            apertus_client: Apertusクライアント（Noneの場合は新規作成）
        """
        self.client = apertus_client or ApertusClient()
        self.use_ai = not self.client.mock_mode
        
        if self.use_ai:
            logger.info("✅ AI自動生成モード: 有効")
        else:
            logger.info("⚠️ AI自動生成モード: 無効（テンプレート使用）")
    
    def generate_food_description(self, food: str) -> str:
        """
        食べ物の説明を自動生成
        
        Args:
            food: 食べ物の名前
            
        Returns:
            説明文
        """
        if not self.use_ai:
            return self._fallback_food_description(food)
        
        try:
            prompt = f"""
以下の食べ物について、簡潔で魅力的な説明文を50-80文字程度で作成してください。

食べ物: {food}

含めるべき要素:
- 外見や特徴（色、形、食感など）
- 栄養価や健康面のメリット
- 一般的な調理法や食べ方の特徴

出力形式: 説明文のみ（前置きや補足は不要）
口調: 自然な日本語、丁寧体
"""
            
            response = self.client.process_text(ApertusRequest(
                text=prompt,
                task_type=TaskType.ANALYZE,
                model=ModelType.GPT4_TURBO,
                temperature=0.7,
                max_length=200
            ))
            
            if response.success and response.result:
                return response.result.strip()
            else:
                logger.warning(f"AI生成失敗: {response.error}")
                return self._fallback_food_description(food)
                
        except Exception as e:
            logger.error(f"説明文生成エラー: {e}")
            return self._fallback_food_description(food)
    
    def generate_hobby_description(self, hobby: str) -> str:
        """
        趣味の説明を自動生成
        
        Args:
            hobby: 趣味の名前
            
        Returns:
            説明文
        """
        if not self.use_ai:
            return self._fallback_hobby_description(hobby)
        
        try:
            prompt = f"""
以下の趣味について、その魅力や効果を30-50文字程度で説明してください。

趣味: {hobby}

含めるべき要素:
- その趣味がもたらす効果（心理的・身体的）
- 魅力や楽しさ
- 得られるもの

出力形式: 説明文のみ
口調: 簡潔で前向き
"""
            
            response = self.client.process_text(ApertusRequest(
                text=prompt,
                task_type=TaskType.ANALYZE,
                model=ModelType.GPT4_TURBO,
                temperature=0.7,
                max_length=150
            ))
            
            if response.success and response.result:
                return response.result.strip()
            else:
                return self._fallback_hobby_description(hobby)
                
        except Exception as e:
            logger.error(f"趣味説明生成エラー: {e}")
            return self._fallback_hobby_description(hobby)
    
    def generate_vehicle_description(self, vehicle: str, category: str = "") -> str:
        """
        乗り物の説明を自動生成
        
        Args:
            vehicle: 乗り物の名前
            category: カテゴリ（車、バイクなど）
            
        Returns:
            説明文
        """
        if not self.use_ai:
            return self._fallback_vehicle_description(vehicle, category)
        
        try:
            prompt = f"""
{vehicle}（{category}）について、特徴を30-50文字程度で説明してください。

含めるべき要素:
- 主な用途や特徴
- デザインや性能の特色
- 人気の理由

出力形式: 説明文のみ
"""
            
            response = self.client.process_text(ApertusRequest(
                text=prompt,
                task_type=TaskType.ANALYZE,
                model=ModelType.GPT4_TURBO,
                temperature=0.7,
                max_length=150
            ))
            
            if response.success and response.result:
                return response.result.strip()
            else:
                return self._fallback_vehicle_description(vehicle, category)
                
        except Exception as e:
            logger.error(f"乗り物説明生成エラー: {e}")
            return self._fallback_vehicle_description(vehicle, category)
    
    def generate_color_description(self, color: str) -> str:
        """色の説明を自動生成"""
        if not self.use_ai:
            return self._fallback_color_description(color)
        
        try:
            prompt = f"""
{color}という色について、イメージや特徴を20-40文字で説明してください。

含めるべき要素:
- その色が持つイメージや印象
- よく使われる場面
- 心理的効果

出力形式: 説明文のみ
"""
            
            response = self.client.process_text(ApertusRequest(
                text=prompt,
                task_type=TaskType.ANALYZE,
                model=ModelType.GPT4_TURBO,
                temperature=0.7,
                max_length=120
            ))
            
            if response.success:
                return response.result.strip()
            else:
                return self._fallback_color_description(color)
                
        except Exception as e:
            return self._fallback_color_description(color)
    
    def generate_place_description(self, place: str) -> str:
        """場所の説明を自動生成"""
        if not self.use_ai:
            return "独自の文化や雰囲気を持つ場所"
        
        try:
            prompt = f"{place}について、その特徴や魅力を30-50文字で簡潔に説明してください。"
            
            response = self.client.process_text(ApertusRequest(
                text=prompt,
                task_type=TaskType.ANALYZE,
                model=ModelType.GPT4_TURBO,
                temperature=0.7,
                max_length=150
            ))
            
            if response.success:
                return response.result.strip()
            else:
                return "独自の文化や雰囲気を持つ場所"
                
        except:
            return "独自の文化や雰囲気を持つ場所"
    
    # === フォールバック用テンプレート ===
    
    def _fallback_food_description(self, food: str) -> str:
        """フォールバック: 食べ物の説明テンプレート"""
        templates = {
            'カリフラワー': '白い花蕾が特徴的な野菜で、ビタミンCや食物繊維が豊富',
            'ブロッコリー': '緑色の花蕾が特徴で、栄養価が高く万能な野菜',
            'トマト': '赤く熟した果実で、リコピンやビタミンが豊富',
        }
        return templates.get(food, '独特の風味と食感を持つ食材')
    
    def _fallback_hobby_description(self, hobby: str) -> str:
        """フォールバック: 趣味の説明テンプレート"""
        templates = {
            '読書': '知識を深め、想像力を養い、ストレス解消にもなる',
            '映画': '様々なストーリーや世界観を楽しみ、感動を得られる',
            '音楽': '心を癒し、感情を表現し、創造性を刺激する',
        }
        return templates.get(hobby, 'リラックスでき、自己表現に繋がる')
    
    def _fallback_vehicle_description(self, vehicle: str, category: str) -> str:
        """フォールバック: 乗り物の説明テンプレート"""
        return "快適で実用的な移動手段として人気"
    
    def _fallback_color_description(self, color: str) -> str:
        """フォールバック: 色の説明テンプレート"""
        templates = {
            '赤': '情熱的でエネルギッシュな印象を与える',
            '青': '落ち着きと信頼感を感じさせる',
            '緑': '自然や癒しをイメージさせる',
        }
        return templates.get(color, '独特の印象を与える')


# シングルトンインスタンス
_ai_generator = None

def get_ai_description_generator(apertus_client: Optional[ApertusClient] = None) -> AIDescriptionGenerator:
    """AIDescriptionGeneratorのシングルトン取得"""
    global _ai_generator
    if _ai_generator is None:
        _ai_generator = AIDescriptionGenerator(apertus_client)
    return _ai_generator
