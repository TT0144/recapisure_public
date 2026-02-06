#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recapisure - Professional Japanese Text Summarization Web Application

高性能日本語要約Webアプリケーション - Apertus AI統合版
機能: 長文要約、短文展開、URL記事要約、マルチユーザー対応
"""

import os
import sys
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

# OpenMPライブラリの重複初期化警告を回避
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# UTF-8エンコーディング強制設定（文字化け対策）
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'
import logging
import json
import time
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, Any, List
import requests
import re
import hashlib
from functools import wraps

# PDF処理
import PyPDF2
import pdfplumber
import sqlite3

# Flask関連
from flask import Flask, render_template, request, jsonify, session, flash, redirect, url_for
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

# 独自モジュール
from config import config
from services.huggingface_service import get_hf_service
from services.kaggle_ai_client import KaggleAIClient  # ⭐ Kaggle統合
from models.processing import ProcessingResult, ProcessingType, ProcessingStatus, ProcessingRequest
from database import get_db

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCR処理用（loggerの後に配置）
try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
    logger.info("✅ OCR機能が利用可能です（pytesseract + pdf2image）")
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("⚠️ OCR機能が利用できません。pip install pytesseract pdf2image Pillow")

# 設定(旧Config クラスは config.py に移行)
class Config:
    """下位互換性のための設定クラス"""
    SECRET_KEY = config.secret_key
    MAX_CONTENT_LENGTH = config.max_content_length
    UPLOAD_FOLDER = config.upload_folder
    MAX_TEXT_LENGTH = config.max_text_length
    MAX_URL_CONTENT_LENGTH = config.max_url_content_length
    REQUEST_TIMEOUT = config.request_timeout
    ALLOWED_EXTENSIONS = config.allowed_extensions

# ユーティリティ関数
def allowed_file(filename):
    """ファイル形式チェック"""
    return '.' in filename and Path(filename).suffix.lower() in Config.ALLOWED_EXTENSIONS

def _fix_ocr_garbled_text(text: str) -> str:
    """
    OCRで文字化けしたテキストを修正
    
    Args:
        text: OCRで抽出されたテキスト
        
    Returns:
        修正後のテキスト
    """
    import re
    
    # よくある文字化けパターンを修正
    replacements = {
        # Unicode文字化け（よく見られるパターン）
        '༊': '区',
        'ໃ': '西',
        '໭': '北',
        'ᆶ': '垂',
        'Ỉ': '水',
        'ᮾ': '東',
        'ℿ': '灘',
        '㡲': '須',
        '☻': '磨',
        '୰': '中',
        'ኸ': '央',
        'රᗜ': '兵庫',
        'රᗜ༊': '兵庫区',
        '㛗⏣': '長田',
        '▲': '',  # 記号除去
        '〈': '',
        '〉': '',
        
        # 数字・記号の文字化け
        '㸫': '−',
        'Ϩ': 'Ⅰ',
        'ϩ': 'Ⅱ',
        'Ϫ': 'Ⅲ',
        
        # よくある誤認識
        'すす༊': '須磨区',
        'ໃ༊': '西区',
        '໭༊': '北区',
        'ᆶỈ༊': '垂水区',
        'ᮾℿ༊': '東灘区',
        '㡲☻༊': '須磨区',
        'ℿ༊': '灘区',
        '୰ኸ༊': '中央区',
        'රᗜ༊': '兵庫区',
        '㛗⏣༊': '長田区',
    }
    
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    # 余分な空白を削除
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    
    return text

def extract_text_with_ocr(file_path):
    """
    OCRを使用してPDFからテキストを抽出（画像PDF対応）
    
    Args:
        file_path: PDFファイルのパス
    
    Returns:
        str: 抽出されたテキスト
    """
    if not OCR_AVAILABLE:
        raise ValueError("OCR機能が利用できません。pip install pytesseract pdf2image Pillow でインストールしてください。")
    
    try:
        logger.info("🔍 OCR処理を開始します...")
        
        # PDFを画像に変換
        logger.info("📄 PDFを画像に変換中...")
        images = convert_from_path(file_path, dpi=300)  # 300dpiで高品質
        total_pages = len(images)
        logger.info(f"📄 {total_pages}ページの画像を生成しました")
        
        extracted_pages = []
        
        for page_num, image in enumerate(images, 1):
            try:
                logger.info(f"🔍 ページ {page_num}/{total_pages} をOCR処理中...")
                
                # Tesseract OCRでテキスト抽出
                # ⭐ 日本語優先 + 英語のフォールバック
                try:
                    # まず日本語+英語で試行
                    text = pytesseract.image_to_string(
                        image,
                        lang='jpn+eng',  # 日本語と英語
                        config='--psm 1 --oem 1'  # 自動ページセグメンテーション + LSTMエンジン
                    )
                except Exception as lang_error:
                    logger.warning(f"⚠️ 日本語OCRが失敗、英語のみで再試行: {lang_error}")
                    # 日本語が使えない場合は英語のみ
                    text = pytesseract.image_to_string(
                        image,
                        lang='eng',
                        config='--psm 1 --oem 1'
                    )
                
                if text and text.strip():
                    # ⭐ OCRの文字化けを後処理で修正
                    text = _fix_ocr_garbled_text(text)
                    extracted_pages.append(f"━━━━━ ページ {page_num}/{total_pages} (OCR) ━━━━━\n{text}")
                    logger.info(f"✅ ページ {page_num}/{total_pages} OCR完了 ({len(text)} 文字)")
                else:
                    logger.warning(f"⚠️ ページ {page_num}/{total_pages} からテキストを抽出できませんでした")
                    
            except Exception as e:
                logger.error(f"❌ ページ {page_num}/{total_pages} のOCR処理に失敗: {e}")
                continue
        
        if extracted_pages:
            result = "\n\n".join(extracted_pages)
            logger.info(f"🎉 OCR処理完了: {len(extracted_pages)}/{total_pages} ページ抽出成功")
            return result
        else:
            raise ValueError("OCR処理でテキストを抽出できませんでした")
            
    except Exception as e:
        logger.error(f"❌ OCR処理エラー: {e}")
        raise ValueError(f"OCR処理に失敗しました: {str(e)}")


def extract_text_from_image(file_path):
    """
    ⭐ 画像ファイル（PNG, JPG等）からOCRでテキストを抽出
    
    Args:
        file_path: 画像ファイルのパス（Path or str）
    
    Returns:
        str: 抽出されたテキスト
    """
    if not OCR_AVAILABLE:
        raise ValueError("OCR機能が利用できません。pip install pytesseract Pillow でインストールしてください。")
    
    try:
        logger.info(f"🖼️ 画像OCR処理を開始: {file_path}")
        
        # 画像を読み込み
        image = Image.open(file_path)
        
        # 画像のサイズを確認
        width, height = image.size
        logger.info(f"📐 画像サイズ: {width}x{height} pixels")
        
        # 画像が小さすぎる場合は拡大
        if width < 800 or height < 600:
            scale = max(800 / width, 600 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"📏 画像を拡大: {new_width}x{new_height}")
        
        # グレースケールに変換（OCR精度向上）
        if image.mode != 'L':
            image = image.convert('L')
        
        # Tesseract OCRでテキスト抽出
        try:
            # まず日本語+英語で試行
            text = pytesseract.image_to_string(
                image,
                lang='jpn+eng',
                config='--psm 3 --oem 1'  # 完全自動ページセグメンテーション + LSTMエンジン
            )
        except Exception as lang_error:
            logger.warning(f"⚠️ 日本語OCRが失敗、英語のみで再試行: {lang_error}")
            text = pytesseract.image_to_string(
                image,
                lang='eng',
                config='--psm 3 --oem 1'
            )
        
        if text and text.strip():
            # 文字化け修正
            text = _fix_ocr_garbled_text(text)
            # テキストクリーニング
            text = clean_text(text)
            logger.info(f"✅ 画像OCR完了: {len(text)} 文字を抽出")
            return text
        else:
            raise ValueError("画像からテキストを抽出できませんでした。文字が含まれているか確認してください。")
            
    except Exception as e:
        logger.error(f"❌ 画像OCRエラー: {e}")
        raise ValueError(f"画像OCR処理に失敗しました: {str(e)}")


def clean_pdf_text(text):
    """
    PDFから抽出したテキストのクリーニング
    - ページ番号、過剰な改行、記号の羅列などを除去
    """
    if not text:
        return ""
    
    import re
    
    # デバッグ: 元のテキストをログ出力
    sample = text[:300] if len(text) > 300 else text
    logger.info(f"🔍 PDF抽出直後 ({len(text)}文字): {repr(sample)}")
    
    # 1. CID文字パターン（例: (cid:12255)）を除去
    text = re.sub(r'\(cid:\d+\)', '', text)
    
    # 2. ページ番号を除去（例: -62-, -64-, -65- など）
    text = re.sub(r'-\d+-', ' ', text)
    
    # 3. 記号の羅列を除去（◎●▼▲■□◆◇が3個以上連続）
    text = re.sub(r'[◎●▼▲■□◆◇○]{3,}', ' ', text)
    
    # 4. 区切り線を除去（ハイフン、イコール、アンダースコアが5個以上連続）
    text = re.sub(r'[-=_]{5,}', ' ', text)
    
    # 5. パーセンテージの羅列を除去（数字%が連続）
    text = re.sub(r'(?:\d+%\s*){5,}', ' ', text)
    
    # 6. 単独の記号を除去（前後に空白がある記号1文字）
    text = re.sub(r'\s[●▼▲■□◆◇○◎]\s', ' ', text)
    
    # 7. 行頭の記号を除去
    text = re.sub(r'^[●▼▲■□◆◇○◎・]\s*', '', text, flags=re.MULTILINE)
    
    # 8. 図表記号を除去（(図1)、(表2)、[図3]、※など）
    text = re.sub(r'[(\[（](?:図|表|写真|資料|グラフ)\d*[)\]）]', '', text)
    text = re.sub(r'※+', '', text)
    
    # 9. 章番号パターンを除去（I-II、II-I、I.、II.など）
    text = re.sub(r'\b[IVX]+-[IVX]+\b', '', text)
    text = re.sub(r'\b[IVX]+\.\b', '', text)
    text = re.sub(r'\b[IVX]+−[IVX]+\b', '', text)  # 全角ダッシュ
    
    # 10. 括弧内の補足情報を除去（※〜）など
    text = re.sub(r'[（(][※＊][^)）]*[)）]', '', text)
    
    # 11. 波線や矢印記号を除去
    text = re.sub(r'[〜～→⇒⇨➡▶►]', ' ', text)
    
    # 12. 括弧だけが残った場合を除去
    text = re.sub(r'[（()\[\]）]+', '', text)
    
    # 13. PDFレイアウトによる不自然な改行を除去
    # 日本語の文中での改行（句読点以外で終わる行の改行）を除去
    # これによりPDFの左揃えで発生する改行を自然なテキストに変換
    text = re.sub(r'(?<=[^\n。．.!！?？、，,\n])\n(?=[^\n\s])', '', text)
    
    # 14. 半角スペースやタブの連続を1つのスペースに統一
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 15. 3つ以上連続する改行を2つに統一
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 16. 行頭・行末の空白を削除
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # 17. 空行が2つ以上連続する場合は1つに統一
    text = re.sub(r'\n\n+', '\n\n', text)

    # 18. 空行だけの行を削除
    lines = [line for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)    # デバッグ: クリーニング後のテキスト
    sample_after = text[:300] if len(text) > 300 else text
    logger.info(f"✅ クリーニング後 ({len(text)}文字): {repr(sample_after)}")
    
    return text.strip()


def is_garbled_text(text):
    """
    テキストが文字化けしているかを検出
    
    Args:
        text: チェックするテキスト
    
    Returns:
        bool: 文字化けしている場合True
    """
    if not text or len(text) < 10:
        return False
    
    # サンプル文字を取得（最初の500文字）
    sample = text[:500]
    
    # 文字化けパターン1: チベット文字・ラオス文字など（PDFフォント問題）
    garbled_chars = 0
    for char in sample:
        code = ord(char)
        # チベット文字 (U+0F00-0FFF)
        if 0x0F00 <= code <= 0x0FFF:
            garbled_chars += 1
        # ラオス文字 (U+0E80-0EFF)
        elif 0x0E80 <= code <= 0x0EFF:
            garbled_chars += 1
        # デーヴァナーガリー文字 (U+0900-097F)
        elif 0x0900 <= code <= 0x097F:
            garbled_chars += 1
        # ベンガル文字 (U+0980-09FF)
        elif 0x0980 <= code <= 0x09FF:
            garbled_chars += 1
        # ミャンマー文字 (U+1000-109F)
        elif 0x1000 <= code <= 0x109F:
            garbled_chars += 1
        # ハングル互換Jamo (U+3130-318F)
        elif 0x3130 <= code <= 0x318F:
            garbled_chars += 1
    
    # 文字化け文字が10%以上含まれている場合
    garbled_ratio = garbled_chars / len(sample)
    if garbled_ratio > 0.1:
        logger.warning(f"⚠️ 文字化け検出: {garbled_ratio*100:.1f}% ({garbled_chars}/{len(sample)}文字)")
        return True
    
    # 文字化けパターン2: 日本語PDFなのに日本語文字が少なすぎる
    japanese_chars = 0
    for char in sample:
        code = ord(char)
        # ひらがな、カタカナ、漢字
        if (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF) or (0x4E00 <= code <= 0x9FFF):
            japanese_chars += 1
    
    japanese_ratio = japanese_chars / len(sample)
    
    # 日本語文字が5%未満の場合（ファイル名などから日本語PDFと推定される場合）
    if japanese_ratio < 0.05:
        logger.warning(f"⚠️ 日本語文字が少なすぎます: {japanese_ratio*100:.1f}% - 画像PDFの可能性")
        # このケースは文字化けではなく画像PDFの可能性があるため、
        # 他の条件も確認
        if garbled_chars > 0:
            return True
    
    return False


def extract_text_from_pdf(file_path):
    """
    PDFからテキストを抽出（全ページ対応・文字化け対策・OCR対応版）
    
    全ページから確実にテキストを抽出し、ページ情報も含めて返す
    通常のテキスト抽出で失敗した場合、自動的にOCRを試行する
    """
    
    text = ""
    errors = []
    total_pages = 0
    
    # 方法1: pdfminer.sixを使用（フォントエンコーディング問題に最強）
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        from pdfminer.layout import LAParams
        
        logger.info("🔧 pdfminer.sixでPDF抽出を試行...")
        
        # LAParamsでレイアウト解析を調整
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            detect_vertical=True  # 縦書きも検出
        )
        
        text = pdfminer_extract(str(file_path), laparams=laparams)
        
        if text and len(text.strip()) > 10:
            # デバッグ: 抽出直後のテキストを確認
            sample = text[:300] if len(text) > 300 else text
            logger.info(f"📄 pdfminer.six 抽出直後: {repr(sample)}")
            
            # クリーニング
            cleaned = clean_pdf_text(text)
            
            # ⭐ 文字化けチェック
            if is_garbled_text(cleaned):
                logger.warning("🚨 pdfminer.sixで抽出したテキストに文字化けを検出 - OCRにフォールバック")
                # 文字化けが検出された場合、後でOCRを試行
                text = ""
            elif len(cleaned) >= 10:
                logger.info(f"✅ pdfminer.sixで抽出成功: {len(cleaned)} 文字")
                return cleaned
        
        logger.warning("pdfminer.sixでは十分なテキストを抽出できませんでした")
        
    except ImportError:
        logger.warning("⚠️ pdfminer.sixがインストールされていません")
        errors.append("pdfminer.six: not installed")
    except Exception as e:
        errors.append(f"pdfminer.six: {str(e)}")
        logger.warning(f"pdfminer.six抽出エラー: {e}")
    
    # 方法2: pdfplumberを使用（pdfminer.sixで失敗した場合のフォールバック）
    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            extracted_pages = []
            
            logger.info(f"PDFの総ページ数: {total_pages}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    
                    # デバッグ: 抽出直後のテキストを確認
                    if page_num == 1 and page_text:
                        sample = page_text[:200] if len(page_text) > 200 else page_text
                        logger.info(f"📄 pdfplumber 1ページ目抽出直後: {repr(sample)}")
                    
                    if page_text and page_text.strip():
                        # PDFテキストをクリーニング（CID文字除去など）
                        cleaned_page_text = clean_pdf_text(page_text)
                        
                        # クリーニング後にテキストが残っているか確認
                        if cleaned_page_text and len(cleaned_page_text) > 10:
                            extracted_pages.append(f"━━━━━ ページ {page_num}/{total_pages} ━━━━━\n{cleaned_page_text}")
                            logger.info(f"ページ {page_num}/{total_pages} を抽出成功 ({len(cleaned_page_text)} 文字)")
                        else:
                            logger.warning(f"ページ {page_num}/{total_pages} はクリーニング後にテキストが残りませんでした")
                    else:
                        logger.warning(f"ページ {page_num}/{total_pages} にテキストがありません")
                except Exception as e:
                    logger.warning(f"ページ {page_num}/{total_pages} の抽出に失敗: {e}")
                    continue
            
            if extracted_pages:
                text = "\n\n".join(extracted_pages)
                
                # ⭐ 文字化けチェック
                if is_garbled_text(text):
                    logger.warning("🚨 pdfplumberで抽出したテキストに文字化けを検出 - OCRにフォールバック")
                    text = ""  # テキストをクリアしてOCRに進む
                else:
                    logger.info(f"✅ pdfplumberで {len(extracted_pages)}/{total_pages} ページ抽出成功")
    except Exception as e:
        errors.append(f"pdfplumber: {str(e)}")
        logger.warning(f"pdfplumber抽出エラー: {e}")
    
    # 方法2: pdfplumberで失敗した場合、PyPDF2を試す
    if not text.strip():
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                extracted_pages = []
                
                logger.info(f"PyPDF2で再試行 - 総ページ数: {total_pages}")
                
                for page_num in range(total_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        
                        if page_text and page_text.strip():
                            # PDFテキストをクリーニング
                            cleaned_page_text = clean_pdf_text(page_text)
                            
                            if cleaned_page_text and len(cleaned_page_text) > 10:
                                extracted_pages.append(f"━━━━━ ページ {page_num + 1}/{total_pages} ━━━━━\n{cleaned_page_text}")
                                logger.info(f"ページ {page_num + 1}/{total_pages} を抽出成功")
                        else:
                            logger.warning(f"ページ {page_num + 1}/{total_pages} にテキストがありません")
                    except Exception as e:
                        logger.warning(f"ページ {page_num + 1}/{total_pages} の抽出に失敗: {e}")
                        continue
                
                if extracted_pages:
                    text = "\n\n".join(extracted_pages)
                    
                    # ⭐ 文字化けチェック
                    if is_garbled_text(text):
                        logger.warning("🚨 PyPDF2で抽出したテキストに文字化けを検出 - OCRにフォールバック")
                        text = ""  # テキストをクリアしてOCRに進む
                    else:
                        logger.info(f"✅ PyPDF2で {len(extracted_pages)}/{total_pages} ページ抽出成功")
        except Exception as e:
            errors.append(f"PyPDF2: {str(e)}")
            logger.warning(f"PyPDF2抽出エラー: {e}")
    
    # 抽出できたテキストをクリーニング
    if text.strip():
        cleaned = clean_text(text)
        
        # ⭐ 文字化けチェック（最終確認）
        if is_garbled_text(cleaned):
            logger.warning("🚨 クリーニング後も文字化けを検出 - OCRにフォールバック")
            # 文字化けが検出された場合、OCRを強制実行
            if OCR_AVAILABLE:
                try:
                    logger.info("🔄 文字化けが検出されたため、OCRで再抽出します...")
                    ocr_text = extract_text_with_ocr(file_path)
                    cleaned = clean_text(ocr_text)
                    if len(cleaned) >= 10:
                        logger.info(f"【PDF抽出完了（OCR使用・文字化け対策)】抽出文字数: {len(cleaned)} 文字")
                        return cleaned
                except Exception as ocr_error:
                    logger.error(f"❌ OCR処理も失敗: {ocr_error}")
                    raise ValueError(f"PDFから十分なテキストを抽出できませんでした。OCR処理も失敗しました: {str(ocr_error)}")
            else:
                raise ValueError("PDFに文字化けが検出されましたが、OCR機能が利用できません。\n\nOCR機能を使用するには: pip install pytesseract pdf2image Pillow")
        
        # 最終チェック: 意味のあるテキストが含まれているか
        if len(cleaned) < 10:
            logger.warning("⚠️ 通常の方法では十分なテキストを抽出できませんでした。OCRを試行します...")
            # OCRフォールバック
            if OCR_AVAILABLE:
                try:
                    ocr_text = extract_text_with_ocr(file_path)
                    cleaned = clean_text(ocr_text)
                    if len(cleaned) >= 10:
                        # サマリーメッセージはログのみに出力(テキストには含めない)
                        logger.info(f"【PDF抽出完了（OCR使用)】総ページ数: {total_pages}、抽出文字数: {len(cleaned)} 文字")
                        return cleaned
                except Exception as ocr_error:
                    logger.error(f"❌ OCR処理も失敗: {ocr_error}")
                    raise ValueError(f"PDFから十分なテキストを抽出できませんでした。OCR処理も失敗しました: {str(ocr_error)}")
            else:
                raise ValueError("PDFから十分なテキストを抽出できませんでした。画像ベースのPDFの可能性があります。\n\nOCR機能を使用するには: pip install pytesseract pdf2image Pillow")
        
        # 抽出結果のサマリーをログに出力(テキストには含めない)
        logger.info(f"【PDF抽出完了】総ページ数: {total_pages}、抽出文字数: {len(cleaned)} 文字")
        return cleaned
    
    # すべての方法で失敗 → OCRを試行
    logger.warning("⚠️ 通常の方法ではテキストを抽出できませんでした。OCRを試行します...")
    if OCR_AVAILABLE:
        try:
            ocr_text = extract_text_with_ocr(file_path)
            cleaned = clean_text(ocr_text)
            if len(cleaned) >= 10:
                # サマリーメッセージはログのみに出力
                logger.info(f"【PDF抽出完了（OCR使用）】総ページ数: {total_pages}、抽出文字数: {len(cleaned)} 文字")
                return cleaned
        except Exception as ocr_error:
            logger.error(f"❌ OCR処理も失敗: {ocr_error}")
    
    # OCRも失敗した場合のエラーメッセージ
    error_msg = f"PDFからテキストを抽出できませんでした（総ページ数: {total_pages}）。"
    if errors:
        error_msg += f"\n\nエラー詳細: {'; '.join(errors)}"
    error_msg += "\n\n考えられる原因:\n- 画像ベースのPDF（OCR処理が必要）\n- 暗号化されたPDF\n- 破損したPDFファイル"
    
    if not OCR_AVAILABLE:
        error_msg += "\n\n💡 OCR機能をインストールすると画像PDFも処理できます:\n   pip install pytesseract pdf2image Pillow"
    
    raise ValueError(error_msg)

def clean_text(text):
    """
    テキストクリーニング（文字化け対策強化版 + 日本語空白除去）
    """
    if not text:
        return ""
    
    # 制御文字を除去（改行、タブ、キャリッジリターンは保持）
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # 改行の正規化
    text = re.sub(r'\r\n|\r', '\n', text)
    
    # 【追加】日本語文字間の不要な空白を除去
    # 例: "岩 の 中 で" → "岩の中で"
    # ひらがな、カタカナ、漢字の間の単一空白を削除
    text = re.sub(r'([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF])\s+([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF])', r'\1\2', text)
    
    # 上記パターンを複数回適用（連続する文字間の空白すべてに対応）
    for _ in range(5):  # 最大5回繰り返して完全に除去
        text = re.sub(r'([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF])\s+([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF])', r'\1\2', text)
    
    # 連続する空白の削除（英語などの通常の空白）
    text = re.sub(r' +', ' ', text)
    
    # 連続する改行を最大2つまでに制限
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 各行の前後の空白を削除
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # 先頭と末尾の空白・改行を削除
    return text.strip()


def clean_summary_result(text: str) -> str:
    """
    ⭐ 要約結果のクリーニング
    AIからの出力に含まれる不要な記号やフォーマットを除去
    
    Args:
        text: AIからの生の要約結果
        
    Returns:
        クリーニング後のテキスト
    """
    if not text:
        return ""
    
    import re
    
    # 先頭・末尾の空白を除去
    text = text.strip()
    
    # ⭐ コードブロックを除去（AIが出力してしまった場合）
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    
    # ⭐ コマンドライン出力を除去
    text = re.sub(r'^[\$\#]\s+.+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^python\s+\S+\.py.*$', '', text, flags=re.MULTILINE)
    
    # ⭐ import文やコード行を除去
    text = re.sub(r'^from\s+\S+\s+import.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^import\s+\S+.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*path\s*\(.*\).*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*urlpatterns\s*=.*$', '', text, flags=re.MULTILINE)
    
    # ⭐ 既知の技術用語リスト（連結防止＆正規化）
    known_words = [
        'Django', 'Flask', 'Python', 'JavaScript', 'TypeScript',
        'API', 'APIs', 'REST', 'GraphQL',
        'HTTP', 'HTTPS', 'HTML', 'CSS', 'JSON', 'XML', 'SQL',
        'ORM', 'WSGI', 'ASGI', 'URL', 'URI',
        'URLconf', 'URLconfs',
        'Web', 'App', 'Framework', 'Database',
        'Model', 'View', 'Template', 'Controller',
        'GET', 'POST', 'PUT', 'DELETE', 'PATCH',
        'JOIN', 'SELECT', 'INSERT', 'UPDATE',
        'Jinja', 'Werkzeug', 'Click', 'CLI',
    ]
    
    # ⭐ 連続する既知の単語を分離（複数回適用）
    for _ in range(3):  # 3回繰り返して連続した単語を確実に分離
        for word in known_words:
            # 単語の直後に大文字が続く場合にスペースを挿入
            pattern = f'({word})([A-Z])'
            text = re.sub(pattern, r'\1 \2', text)
            # 単語が重複している場合も分離（APIAPI → API API）
            pattern = f'({word})({word})'
            text = re.sub(pattern, r'\1 \2', text, flags=re.IGNORECASE)
    
    # ⭐ 保護する単語（分離しない）
    protected_words = ['URLconf', 'URLconfs', 'JavaScript', 'TypeScript', 'GraphQL']
    for word in protected_words:
        placeholder = f'__PROTECT_{word}__'
        text = text.replace(word, placeholder)
    
    # 小文字の後に大文字が来るパターン（キャメルケース）を検出
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # 大文字の連続の後に小文字が来るパターン（WSGIFlask → WSGI Flask）
    text = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', text)
    
    # 保護した単語を復元
    for word in protected_words:
        placeholder = f'__PROTECT_{word}__'
        text = text.replace(placeholder, word)
    
    # ⭐ 日本語と英語の間にスペースを追加（読みやすさ向上）
    # 英語→日本語
    text = re.sub(r'([a-zA-Z0-9])([ぁ-んァ-ン一-龥])', r'\1 \2', text)
    # 日本語→英語
    text = re.sub(r'([ぁ-んァ-ン一-龥])([a-zA-Z])', r'\1 \2', text)
    
    # ⭐ 重複した単語を1つに（Web Web → Web）
    for word in known_words:
        pattern = f'\\b({word})\\s+\\1\\b'
        text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
    
    # ⭐ 番号付きリストの改行を修正（1. 2. 3. などの前に改行を追加）
    text = re.sub(r'(?<=[^\n\d])(\d{1,2}\.\s)', r'\n\1', text)
    
    # ⭐ 箇条書き記号の前にも改行を追加
    text = re.sub(r'(?<=[^\n])([・•\-]\s)', r'\n\1', text)
    
    # 末尾の不要な記号を除去 (>, <, |, \, / など)
    text = re.sub(r'[><|\\\/]+\s*$', '', text)
    
    # 先頭の不要な記号を除去 (>, <, |, \, / など)  
    text = re.sub(r'^[><|\\\/]+\s*', '', text)
    
    # 「要約:」「Summary:」などのプレフィックスを除去
    text = re.sub(r'^(要約|Summary|翻訳|Translation|結果|Result)\s*[:：]\s*', '', text, flags=re.IGNORECASE)
    
    # 連続するスペースを1つに
    text = re.sub(r' +', ' ', text)
    
    # 連続する改行を最大2つに
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 先頭の改行を除去
    text = text.lstrip('\n')
    
    return text.strip()


def extract_keywords(text: str, max_keywords: int = 8) -> list:
    """
    ⭐ テキストからキーワードを抽出
    シンプルな頻度ベースのキーワード抽出
    
    Args:
        text: 入力テキスト
        max_keywords: 最大キーワード数
        
    Returns:
        キーワードのリスト
    """
    import re
    from collections import Counter
    
    if not text:
        return []
    
    # ストップワード（日本語と英語の一般的な単語）
    stop_words = set([
        # 日本語
        'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ',
        'ある', 'いる', 'も', 'する', 'から', 'な', 'こと', 'として', 'いく',
        'い', 'これ', 'それ', 'あれ', 'この', 'その', 'など', 'もの', 'ため',
        'より', 'よう', 'また', 'および', 'なる', 'へ', 'か', 'でき', 'とき',
        'れる', 'られる', 'ます', 'です', 'だ', 'である', 'という', 'ない',
        # 英語
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'just', 'and', 'but', 'if', 'or', 'because', 'while', 'although',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'we', 'they', 'what', 'which', 'who', 'whom', 'his', 'her', 'your'
    ])
    
    # 日本語の名詞・固有名詞を抽出（カタカナ、漢字を含む2文字以上の単語）
    japanese_words = re.findall(r'[一-龯ァ-ヴー]{2,}|[ァ-ヴー]{3,}', text)
    
    # 英単語を抽出（3文字以上）
    english_words = re.findall(r'[A-Za-z]{3,}', text)
    
    # 単語をカウント
    word_counts = Counter()
    
    for word in japanese_words:
        word_lower = word.lower()
        if word_lower not in stop_words and len(word) >= 2:
            word_counts[word] += 1
    
    for word in english_words:
        word_lower = word.lower()
        if word_lower not in stop_words and len(word) >= 3:
            # 英単語は元のケースを保持（技術用語のため）
            word_counts[word] += 1
    
    # 重要度でソート（頻度が高いものを優先）
    # ただし、一度しか出現しない単語は除外
    keywords = [
        word for word, count in word_counts.most_common(max_keywords * 2)
        if count >= 1  # 1回以上出現
    ][:max_keywords]
    
    return keywords


def preprocess_text_for_summarization(text, max_chars=5000):
    """
    要約用にテキストを前処理（技術記事対策強化版）
    - コードブロック全体を除去
    - コードっぽい行を除去
    - 意味のない短い行を除去
    - 適切な長さに制限
    """
    if not text:
        return text
    
    # ⭐ コードブロック全体を除去（```...```や、インデントされたブロック）
    # Markdownコードブロック
    text = re.sub(r'```[\s\S]*?```', '', text)
    # インラインコード
    text = re.sub(r'`[^`]+`', '', text)
    # $記号で始まるシェルコマンド行
    text = re.sub(r'^\$\s+.+$', '', text, flags=re.MULTILINE)
    # >>>で始まるPythonプロンプト
    text = re.sub(r'^>>>.*$', '', text, flags=re.MULTILINE)
    
    lines = text.split('\n')
    cleaned_lines = []
    in_code_block = False
    indent_code_count = 0
    
    for line in lines:
        stripped = line.strip()
        
        # 空行はスキップ
        if not stripped:
            indent_code_count = 0  # インデントコードブロックのリセット
            continue
        
        # インデントされた行が連続する場合（コードブロックの可能性）
        if line.startswith('    ') or line.startswith('\t'):
            indent_code_count += 1
            if indent_code_count >= 2:  # 2行以上インデントが続いたらコードとみなす
                continue
        else:
            indent_code_count = 0
        
        # コードっぽい行をスキップ（プログラミング記号が多い）
        code_chars = sum(1 for c in stripped if c in '{}[]();=<>|&$@#`')
        if len(stripped) > 10 and code_chars / len(stripped) > 0.12:
            continue
        
        # ⭐ コードパターンを強化
        code_patterns = [
            r'^(import|from)\s+[a-zA-Z]',  # Python import文
            r'^(def|class)\s+[a-zA-Z_]',  # Python関数/クラス定義
            r'^(function|const|let|var|export|async)\s+',  # JavaScript定義
            r'^(return|if|else|elif|for|while|try|except|with)\s*[\(\{:]',  # 制御構文
            r'^\s*(#!|//|/\*|\*/|\*\s)',  # コメント
            r'^[a-zA-Z_][a-zA-Z0-9_\.]*\s*[\(\[=]',  # 関数呼び出し/代入
            r'^\s*@\w+',  # デコレータ
            r'^[A-Z][a-z]+[A-Z]',  # キャメルケースのクラス名（単独）
            r'^\s*self\.',  # Pythonのself参照
            r'^\s*models\.',  # Django models
            r'^.*\(.*\)\s*:?\s*$',  # 関数定義っぽい行
        ]
        is_code = False
        for pattern in code_patterns:
            if re.match(pattern, stripped):
                is_code = True
                break
        if is_code:
            continue
        
        # コマンドライン出力っぽい行
        if stripped.startswith('$') or stripped.startswith('%') or stripped.startswith('>>>'):
            continue
        
        # 短すぎる行（5文字以下）はスキップ
        if len(stripped) <= 5:
            continue
        
        # 数字だけの行はスキップ
        if re.match(r'^[\d\s\.\-:]+$', stripped):
            continue
        
        # ファイルパスっぽい行はスキップ
        if re.match(r'^[\w\-]+/[\w\-/\.]+$', stripped):
            continue
        
        cleaned_lines.append(stripped)
    
    result = '\n'.join(cleaned_lines)
    
    # 長さ制限（文の途中で切れないように調整）
    if len(result) > max_chars:
        result = result[:max_chars]
        # 最後の文の終わりで切る
        last_period = max(result.rfind('。'), result.rfind('．'), result.rfind('.'))
        if last_period > max_chars * 0.7:
            result = result[:last_period + 1]
        logger.info(f"📝 テキスト制限適用: {max_chars}文字 → {len(result)}文字")
    
    return result


def extract_text_from_url(url):
    """
    URLからテキストを抽出（trafilatura版 - 高精度な本文抽出）
    
    対応サイト:
    - 一般的なニュースサイト、ブログ
    - note.com, Qiita, Zenn などの技術ブログ
    - 技術ドキュメント（Django, Flask, Python docs等）
    - Wikipedia
    - PDFファイル
    
    非対応（技術的制限）:
    - JavaScript必須のSPA（React/Vue単体）
    - ログイン必須のページ
    - ペイウォールのあるページ
    """
    try:
        import trafilatura
        
        # ⭐ PDFファイルのチェック（先にチェック）
        if url.lower().endswith('.pdf'):
            logger.info(f"📄 PDFファイルを検出: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(response.content)
                tmp_path = Path(tmp_file.name)
            
            try:
                text = extract_text_from_pdf(tmp_path)
                return {
                    'success': True,
                    'title': f'PDF: {url.split("/")[-1]}',
                    'content': text,
                    'url': url
                }
            finally:
                tmp_path.unlink(missing_ok=True)
        
        # ⭐ trafilaturaでコンテンツを取得
        logger.info(f"🔍 trafilaturaでURL取得開始: {url}")
        
        # HTMLを取得
        downloaded = trafilatura.fetch_url(url)
        
        if not downloaded:
            logger.warning(f"⚠️ trafilatura: URLの取得に失敗: {url}")
            return {
                'success': False,
                'error': 'URLの取得に失敗しました。サイトがアクセスをブロックしている可能性があります。'
            }
        
        # ⭐ メインコンテンツを抽出（コードブロックを除外するオプションなし）
        # trafilatura.extract() はコードブロックも含めて抽出する
        content_text = trafilatura.extract(
            downloaded,
            include_comments=False,     # コメント除外
            include_tables=False,       # テーブル除外（コードっぽいものが多い）
            include_images=False,       # 画像除外
            include_links=False,        # リンク除外
            output_format='txt',        # プレーンテキスト
            favor_precision=True,       # 精度優先
        )
        
        # ⭐ メタデータも取得（タイトル用）
        metadata = trafilatura.extract_metadata(downloaded)
        title_text = metadata.title if metadata and metadata.title else ''
        
        if not content_text or len(content_text) < 50:
            logger.warning(f"⚠️ trafilatura: コンテンツが短すぎます: {len(content_text) if content_text else 0}文字")
            # フォールバック: BeautifulSoupで試す
            return extract_text_from_url_fallback(url, downloaded)
        
        # ⭐ 抽出後のコード除去処理
        content_text = remove_code_from_text(content_text)
        
        # ⭐ テキストクリーニング
        content_text = content_text.replace('¶', '')
        content_text = re.sub(r'[\u00b6\u00a7\u2020\u2021]', '', content_text)
        content_text = re.sub(r' {2,}', ' ', content_text)
        content_text = re.sub(r'\n{3,}', '\n\n', content_text)
        content_text = '\n'.join(line.strip() for line in content_text.split('\n') if line.strip())
        
        full_text = f"{title_text}\n\n{content_text}" if title_text else content_text
        full_text = clean_text(full_text)
        
        # 長さ制限
        if len(full_text) > Config.MAX_URL_CONTENT_LENGTH:
            full_text = full_text[:Config.MAX_URL_CONTENT_LENGTH] + "..."
            logger.info(f"📝 長さ制限適用: {Config.MAX_URL_CONTENT_LENGTH}文字に切り詰め")
        
        logger.info(f"✅ trafilatura取得成功: {title_text[:50] if title_text else 'No title'}... ({len(full_text)}文字)")
        
        return {
            'success': True,
            'title': title_text,
            'content': full_text,
            'url': url
        }
        
    except ImportError:
        logger.warning("⚠️ trafilaturaがインストールされていません。BeautifulSoupにフォールバック")
        return extract_text_from_url_fallback(url, None)
    except Exception as e:
        logger.error(f"❌ URL抽出エラー: {str(e)}")
        return {
            'success': False,
            'error': f'URLからのテキスト抽出に失敗しました: {str(e)}'
        }


def remove_code_from_text(text):
    """
    テキストからコードっぽい行を除去
    """
    if not text:
        return text
    
    lines = text.split('\n')
    cleaned_lines = []
    
    in_code_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # コードブロックの開始/終了
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        
        if in_code_block:
            continue
        
        # コード行のパターン
        code_patterns = [
            r'^(import|from)\s+\w+',           # import文
            r'^(def|class|async def)\s+\w+',   # 関数/クラス定義
            r'^(if|elif|else|for|while|try|except|finally|with)\s*[:\(]',  # 制御構文
            r'^(return|yield|raise|pass|break|continue)\s',  # キーワード
            r'^\s*(self\.|cls\.)',              # self/cls
            r'^[a-z_]+\s*=\s*[\'"\[\{\(]',     # 変数代入
            r'^[A-Z][a-zA-Z]+\s*=\s*',         # 定数代入
            r'^@\w+',                           # デコレータ
            r'^\s*>>>\s',                       # Python REPL
            r'^\s*\$\s',                        # シェルコマンド
            r'^\s*#\s*(coding|-\*-)',          # コーディング宣言
            r'^\s*"""',                         # docstring
            r"^\s*'''",                         # docstring
        ]
        
        is_code = False
        for pattern in code_patterns:
            if re.match(pattern, stripped):
                is_code = True
                break
        
        # 括弧だらけの行（関数呼び出し等）
        if not is_code and len(stripped) > 0:
            bracket_count = stripped.count('(') + stripped.count(')') + stripped.count('[') + stripped.count(']')
            if bracket_count > 4 and bracket_count / len(stripped) > 0.1:
                is_code = True
        
        if not is_code and stripped:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def extract_text_from_url_fallback(url, html_content=None):
    """
    BeautifulSoupを使用したフォールバック抽出
    """
    try:
        if html_content is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
            }
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            html_content = response.text
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 不要な要素を削除
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 
                            'noscript', 'iframe', 'form', 'button', 'input',
                            'pre', 'code']):
            if element:
                element.decompose()
        
        # タイトル取得
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ''
        
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            title_text = og_title.get('content').strip()
        
        # コンテンツ取得
        content_text = ''
        content_selectors = ['article', '[role="main"]', 'main', '.body', '.content', '#content']
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                text = content.get_text(separator=' ', strip=True)
                if len(text) > len(content_text):
                    content_text = text
                    if len(content_text) > 500:
                        break
        
        if not content_text or len(content_text) < 100:
            body = soup.find('body')
            if body:
                paragraphs = body.find_all('p')
                if paragraphs:
                    content_text = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
        
        # クリーニング
        content_text = content_text.replace('¶', '')
        content_text = re.sub(r' {2,}', ' ', content_text)
        
        full_text = f"{title_text}\n\n{content_text}" if title_text else content_text
        full_text = clean_text(full_text)
        
        if len(full_text) < 50:
            return {
                'success': False,
                'error': f'コンテンツを十分に取得できませんでした（{len(full_text)}文字）'
            }
        
        if len(full_text) > Config.MAX_URL_CONTENT_LENGTH:
            full_text = full_text[:Config.MAX_URL_CONTENT_LENGTH] + "..."
        
        logger.info(f"✅ フォールバック取得成功: {len(full_text)}文字")
        
        return {
            'success': True,
            'title': title_text,
            'content': full_text,
            'url': url
        }
        
    except Exception as e:
        logger.error(f"❌ フォールバック抽出エラー: {str(e)}")
        return {
            'success': False,
            'error': f'URLからのテキスト抽出に失敗しました: {str(e)}'
        }


# ====== 以下は削除された旧コード（BeautifulSoup版）の残り部分を置き換え ======
def _legacy_pdf_extraction(url, response):
    """PDF抽出のレガシーコード（互換性のために残す）"""
    logger.info(f"📄 PDFファイルを検出: {url}")
    
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(response.content)
        tmp_path = Path(tmp_file.name)
    
    try:
        text = extract_text_from_pdf(tmp_path)
        return {
            'success': True,
            'title': f'PDF: {url.split("/")[-1]}',
            'content': text,
        'url': url
    }
    finally:
        tmp_path.unlink(missing_ok=True)


def process_uploaded_file(file_path):
    """アップロードファイル処理"""
    ext = file_path.suffix.lower()
    
    try:
        if ext == '.txt':
            return file_path.read_text(encoding='utf-8')
        elif ext == '.md':
            return file_path.read_text(encoding='utf-8')
        elif ext == '.pdf':
            return extract_text_from_pdf(file_path)
        # ⭐ 画像ファイル対応 (PNG, JPG, JPEG, GIF, BMP, WEBP)
        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            return extract_text_from_image(file_path)
        else:
            return f"ファイル形式 {ext} は現在サポートされていません。TXT、MD、PDF、または画像ファイル（PNG, JPG等）をお試しください。"
    except Exception as e:
        logger.error(f"ファイル処理エラー: {str(e)}")
        return f"ファイル処理エラー: {str(e)}"

def calculate_translation_quality(original: str, translated: str, source_lang: str, target_lang: str) -> Dict[str, float]:
    """
    ⭐ 翻訳品質スコアを詳細に計算
    
    Args:
        original: 元のテキスト
        translated: 翻訳されたテキスト
        source_lang: 元言語
        target_lang: 翻訳先言語
        
    Returns:
        品質スコアの詳細 {'total': 総合点, 'length': 長さ点, 'completeness': 完全性点, ...}
    """
    scores = {}
    
    # 1. 長さの適切さ (25点)
    orig_len = len(original)
    trans_len = len(translated)
    length_ratio = trans_len / orig_len if orig_len > 0 else 0
    
    # 翻訳先言語による理想的な長さ比率
    ideal_ratios = {
        'jpn_Jpan': (0.6, 1.2),  # 日本語は英語より短くなりがち
        'eng_Latn': (0.8, 1.3),
        'default': (0.7, 1.3)
    }
    min_ratio, max_ratio = ideal_ratios.get(target_lang, ideal_ratios['default'])
    
    if min_ratio <= length_ratio <= max_ratio:
        scores['length'] = 25.0
    else:
        # 範囲外の場合、距離に応じて減点
        distance = min(abs(length_ratio - min_ratio), abs(length_ratio - max_ratio))
        scores['length'] = max(0, 25 - distance * 50)
    
    # 2. 完全性 (25点) - 翻訳が極端に短すぎないか
    if trans_len < orig_len * 0.3:
        scores['completeness'] = 0  # 元の30%未満は不完全
    elif trans_len < orig_len * 0.5:
        scores['completeness'] = 15  # 50%未満は中程度
    else:
        scores['completeness'] = 25  # 50%以上は完全
    
    # 3. 文字種の適切さ (20点)
    if target_lang == 'jpn_Jpan':
        # 日本語: ひらがな・カタカナ・漢字のバランス
        hiragana = len(re.findall(r'[ぁ-ん]', translated))
        katakana = len(re.findall(r'[ァ-ヴー]', translated))
        kanji = len(re.findall(r'[一-龯]', translated))
        
        total_jp = hiragana + katakana + kanji
        if total_jp > 0:
            # 理想的な比率: ひらがな40-60%, カタカナ10-30%, 漢字20-40%
            h_ratio = hiragana / total_jp
            k_ratio = katakana / total_jp
            j_ratio = kanji / total_jp
            
            balance_score = 0
            if 0.35 <= h_ratio <= 0.65:
                balance_score += 8
            if 0.05 <= k_ratio <= 0.35:
                balance_score += 6
            if 0.15 <= j_ratio <= 0.45:
                balance_score += 6
            
            scores['character_balance'] = balance_score
        else:
            scores['character_balance'] = 0  # 日本語文字がない
    else:
        # 英語など: 基本的に満点
        scores['character_balance'] = 20
    
    # 4. 句読点の適切さ (15点)
    if target_lang == 'jpn_Jpan':
        # 日本語句読点(、。)の数
        jp_punctuation = len(re.findall(r'[、。]', translated))
        # 英語句読点(, .)が残っていないか
        en_punctuation = len(re.findall(r'[,.]', translated))
        
        if jp_punctuation > 0 and en_punctuation == 0:
            scores['punctuation'] = 15
        elif jp_punctuation > 0:
            scores['punctuation'] = 10  # 英語句読点が混在
        else:
            scores['punctuation'] = 5  # 句読点なし
    else:
        scores['punctuation'] = 15
    
    # 5. 繰り返しの少なさ (15点)
    # 同じ単語が3回以上連続していないか
    repetitions = len(re.findall(r'(\w+)\1{2,}', translated))
    if repetitions == 0:
        scores['no_repetition'] = 15
    elif repetitions <= 2:
        scores['no_repetition'] = 10
    else:
        scores['no_repetition'] = max(0, 15 - repetitions * 3)
    
    # 総合スコア計算
    scores['total'] = sum(scores.values())
    
    logger.info(f"⭐ 品質詳細: 総合{scores['total']:.1f}% "
                f"(長さ{scores['length']:.0f} 完全性{scores['completeness']:.0f} "
                f"文字種{scores['character_balance']:.0f} 句読点{scores['punctuation']:.0f} "
                f"繰返{scores['no_repetition']:.0f})")
    
    return scores

# Flaskアプリケーション
app = Flask(__name__)
app.config.from_object(Config)

# SocketIO初期化 (リアルタイム進捗通知用)
socketio = SocketIO(app, cors_allowed_origins="*")

# アップロードフォルダ作成
Config.UPLOAD_FOLDER.mkdir(exist_ok=True)

# Hugging Face AIサービス初期化
hf_service = get_hf_service()

logger.info(f"Hugging Face Service initialized. Available: {hf_service.available}")

# ⭐ Kaggle AIクライアント初期化（Apertus-8B専用）
kaggle_client = None
if os.getenv('USE_KAGGLE_API', '').lower() == 'true':
    try:
        kaggle_api_url = os.getenv('KAGGLE_API_URL')
        kaggle_api_key = os.getenv('KAGGLE_API_KEY')
        kaggle_timeout = int(os.getenv('KAGGLE_API_TIMEOUT', '60'))
        
        if kaggle_api_url:
            kaggle_client = KaggleAIClient(
                base_url=kaggle_api_url,
                api_key=kaggle_api_key,
                timeout=kaggle_timeout
            )
            if kaggle_client.is_available():
                logger.info(f"🚀 Kaggle Apertus-8B 初期化完了. URL: {kaggle_api_url}")
            else:
                logger.warning(f"⚠️ Kaggle APIは設定されていますが、現在利用できません")
                logger.warning(f"   Kaggle Notebookが起動しているか確認してください")
        else:
            logger.error("❌ KAGGLE_API_URLが設定されていません")
    except Exception as e:
        logger.error(f"❌ Kaggle AI Client初期化失敗: {e}")
        kaggle_client = None
else:
    logger.error("❌ USE_KAGGLE_API=true に設定してください (.envファイル)")

if not kaggle_client or not kaggle_client.is_available():
    logger.error("=" * 60)
    logger.error("⚠️ Kaggle Apertus-8Bが利用できません")
    logger.error("   このアプリはKaggle専用です。以下を確認してください:")
    logger.error("   1. .envファイルでUSE_KAGGLE_API=trueに設定")
    logger.error("   2. KAGGLE_API_URLとKAGGLE_API_KEYを設定")
    logger.error("   3. Kaggle Notebookが起動中であること")
    logger.error("=" * 60)

# ⭐ ログイン必須デコレータ
def login_required(f):
    """ログインが必要なエンドポイント用デコレータ"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'ログインが必要です'}), 401
        return f(*args, **kwargs)
    return decorated_function

def hash_password(password):
    """パスワードをハッシュ化"""
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')

@app.route('/history')
def history():
    """履歴ページ"""
    return render_template('history.html')

@app.route('/about')
def about():
    """アバウトページ"""
    return render_template('about.html')

@app.route('/learning-dashboard')
def learning_dashboard():
    """Apertus学習ダッシュボード - AIゼミ用"""
    return render_template('learning_dashboard.html')

@app.route('/api/upload-pdf', methods=['POST'])
def api_upload_pdf():
    """PDFアップロード専用API - pdfminer.sixでテキスト抽出"""
    try:
        # ファイルが送信されているか確認
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'ファイルが送信されていません'
            }), 400
        
        file = request.files['file']
        
        # ファイル名が空でないか確認
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'ファイルが選択されていません'
            }), 400
        
        # PDFファイルか確認
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({
                'success': False,
                'error': 'PDFファイルのみアップロード可能です'
            }), 400
        
        # 一時ファイルとして保存
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = Path(tmp_file.name)
        
        try:
            # extract_text_from_pdf()を使用してテキスト抽出
            logger.info(f"📄 PDFファイルをサーバー側で処理: {file.filename}")
            extracted_text = extract_text_from_pdf(tmp_path)
            
            return jsonify({
                'success': True,
                'text': extracted_text,
                'filename': file.filename
            })
        
        finally:
            # 一時ファイルを削除
            if tmp_path.exists():
                tmp_path.unlink()
    
    except Exception as e:
        logger.error(f"❌ PDFアップロードエラー: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'PDFの処理中にエラーが発生しました: {str(e)}'
        }), 500


@app.route('/api/upload-image', methods=['POST'])
def api_upload_image():
    """⭐ 画像アップロードAPI - OCRでテキスト抽出（説明書の写真対応）"""
    try:
        # ファイルが送信されているか確認
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'ファイルが送信されていません'
            }), 400
        
        file = request.files['file']
        
        # ファイル名が空でないか確認
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'ファイルが選択されていません'
            }), 400
        
        # 画像ファイルか確認
        allowed_image_ext = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_image_ext:
            return jsonify({
                'success': False,
                'error': f'対応している画像形式: {", ".join(allowed_image_ext)}'
            }), 400
        
        # OCR機能チェック
        if not OCR_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'OCR機能が利用できません。サーバー管理者にお問い合わせください。'
            }), 503
        
        # 一時ファイルとして保存
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            file.save(tmp_file.name)
            tmp_path = Path(tmp_file.name)
        
        try:
            # 画像からテキスト抽出
            logger.info(f"🖼️ 画像ファイルをOCR処理: {file.filename}")
            extracted_text = extract_text_from_image(tmp_path)
            
            return jsonify({
                'success': True,
                'text': extracted_text,
                'filename': file.filename,
                'message': '📷 画像からテキストを抽出しました'
            })
        
        finally:
            # 一時ファイルを削除
            if tmp_path.exists():
                tmp_path.unlink()
    
    except Exception as e:
        logger.error(f"❌ 画像アップロードエラー: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'画像の処理中にエラーが発生しました: {str(e)}'
        }), 500


@app.route('/api/summarize', methods=['POST'])

def api_summarize():
    """要約API - 多言語翻訳対応版 + 動的調整 + 履歴保存 + スタイル選択 + Apertus LLM対応 + リアルタイム進捗"""
    try:
        # ⭐ タスクID生成（進捗追跡用）
        task_id = str(uuid.uuid4())
        
        data = request.get_json() if request.is_json else request.form
        text = data.get('text', '').strip()
        max_length = int(data.get('max_length', 200))
        min_length = int(data.get('min_length', 50))
        summary_mode = data.get('summary_mode', 'short')  # 'short' or 'long'
        source_lang = data.get('source_lang', 'auto')  # 入力言語
        target_lang = data.get('target_lang', 'jpn_Jpan')  # 出力言語
        style = data.get('style', 'balanced')  # ⭐ 要約スタイル
        model_type = 'kaggle'  # ⭐ Kaggle Apertus-8B固定
        
        # ⭐ 初期進捗送信
        send_progress(task_id, 'validate', 5, 'テキストを検証中...')
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'テキストが入力されていません'
            }), 400
        
        # ⭐ PDF等の長文対応: 制限を30,000文字に緩和
        max_allowed_length = 30000
        if len(text) > max_allowed_length:
            return jsonify({
                'success': False,
                'error': f'テキストが長すぎます（最大{max_allowed_length:,}文字）。現在: {len(text):,}文字'
            }), 400
        
        # 🔥 要約モードに応じた固定文字数
        text_length = len(text)
        if summary_mode == 'long':
            # 詳細要約: 800-1000文字
            max_length = 1000
            min_length = 800
        else:
            # 通常要約: 200-400文字
            max_length = 400
            min_length = 200
        
        logger.info(f"📏 動的調整: 入力{text_length}文字 → 要約{min_length}-{max_length}文字")
        logger.info(f"🎨 スタイル: {style}")
        logger.info(f"🚀 モデル: Kaggle Apertus-8B (固定)")
        
        # ⭐ 進捗送信: 準備完了
        send_progress(task_id, 'prepare', 15, '要約処理の準備中...')
        
        # Kaggle Apertus-8Bで要約・翻訳実行
        start_time = time.time()
        
        if not kaggle_client:
            logger.error("❌ Kaggle APIクライアントが初期化されていません")
            return jsonify({
                'success': False,
                'error': '⚠️ サーバー設定エラー: KAGGLE_API_URLが設定されていません。管理者に連絡してください。',
                'task_id': task_id
            }), 500
        
        if not kaggle_client.is_available():
            logger.error("❌ Kaggle APIが利用できません")
            return jsonify({
                'success': False,
                'error': '🔌 AI サーバーに接続できません。\n\n考えられる原因:\n• Kaggle Notebookが停止している\n• ngrok URLが変更された\n• ネットワーク接続の問題\n\nしばらく待ってから再試行してください。',
                'task_id': task_id
            }), 503
        
        # 🚀 Kaggle Apertus-8Bで要約実行
        logger.info("🚀 Kaggle Apertus-8B で要約実行...")
        logger.info(f"📝 設定: 入力言語={source_lang}, 出力言語={target_lang}, モード={summary_mode}, スタイル={style}")
        
        # ⭐ 進捗送信: 要約中
        send_progress(task_id, 'summarize', 30, 'AI が要約を生成中... (これには数秒かかります)')
        
        # ⭐ 言語コードを人間可読な言語名に変換 (config.pyから共通定義を使用)
        from config import get_language_name
        source_lang_name = get_language_name(source_lang)
        target_lang_name = get_language_name(target_lang, 'Japanese')
        
        # ⭐ 新しい/summarizeエンドポイントを使用（Apertusの多言語要約機能）
        kaggle_result = kaggle_client.summarize(
            text=text,
            max_length=max_length,
            source_lang=source_lang_name,
            target_lang=target_lang_name,
            style=style,
            summary_mode=summary_mode
        )
        
        # ⭐ 進捗送信: 要約完了、後処理中
        send_progress(task_id, 'process', 70, '要約結果を整形中...')
        
        if not kaggle_result or not kaggle_result.get('success'):
            error_msg = kaggle_result.get('error', '不明なエラー') if kaggle_result else '応答なし'
            logger.error(f"❌ Kaggle要約失敗: {error_msg}")
            
            # ⭐ ユーザーフレンドリーなエラーメッセージに変換
            if 'タイムアウト' in error_msg or 'Timeout' in error_msg:
                user_error = '⏱️ 処理に時間がかかりすぎました。\n\nテキストが長すぎる可能性があります。短いテキストで再試行してください。'
            elif 'HTTP 5' in error_msg:
                user_error = '🔧 AIサーバーで内部エラーが発生しました。\n\nしばらく待ってから再試行してください。'
            elif 'HTTP 4' in error_msg:
                user_error = '🔐 認証エラーが発生しました。\n\n管理者に連絡してください。'
            else:
                user_error = f'❌ 要約処理に失敗しました。\n\n詳細: {error_msg}'
            
            return jsonify({
                'success': False,
                'error': user_error
            }), 500
        
        # /summarizeエンドポイントは'summary'キーで結果を返す
        summary_text = kaggle_result.get('summary', '')
        
        # ⭐ 要約結果をクリーニング（不要な記号を除去）
        summary_text = clean_summary_result(summary_text)
        
        # デバッグログ（クリーニング後）
        logger.info(f"🔍 要約結果（クリーニング後）: {summary_text[:200]}...")
        
        response = type('obj', (object,), {
            'success': True,
            'result': summary_text,
            'model_used': 'Kaggle Apertus-8B (1,811言語対応)',
            'token_usage': None,
            'error': None
        })()
        
        execution_time = time.time() - start_time
        
        # ⭐ 進捗送信: 品質評価中
        send_progress(task_id, 'evaluate', 85, '品質を評価中...')
        
        # ⭐ 品質スコア計算（詳細版）
        quality_score = 0.0
        quality_details = {}
        if response.success and response.result:
            quality_details = calculate_translation_quality(text, response.result, source_lang, target_lang)
            quality_score = quality_details.get('total', 0)
        
        # ⭐ 進捗送信: データベース保存中
        send_progress(task_id, 'save', 95, '履歴を保存中...')
        
        # 処理結果をモデルとして記録
        processing_result = ProcessingResult(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            type=ProcessingType.SUMMARIZE,
            status=ProcessingStatus.COMPLETED if response.success else ProcessingStatus.FAILED,
            original_text=text[:200] + "..." if len(text) > 200 else text,
            result=response.result,
            execution_time=execution_time,
            model_used=response.model_used,
            confidence=quality_score / 100.0,
            token_usage=response.token_usage or {},
            original_length=len(text),
            result_length=len(response.result),
            compression_ratio=len(response.result) / len(text) if text else 0.0,
            error=response.error if not response.success else None
        )
        
        # セッションには保存しない（Cookieサイズ制限対策）
        # 代わりにデータベースから取得する
        
        # 🔥 データベースに履歴保存（⭐ 文字数制限を増加 + キーワード追加）
        try:
            db = get_db()
            token_count = response.token_usage.get('total_tokens', 0) if response.token_usage else 0
            
            # ⭐ キーワードを抽出
            keywords = extract_keywords(text, max_keywords=8)
            keywords_json = json.dumps(keywords, ensure_ascii=False) if keywords else None
            
            db.save_translation(
                source_lang=source_lang,
                target_lang=target_lang,
                original_text=text[:5000],  # ⭐ 最初の5000文字まで保存（増加）
                translated_text=response.result[:3000] if response.success else "",  # ⭐ 最大3000文字
                summary_mode=summary_mode,
                quality_score=quality_score,
                file_name=None,
                processing_time=execution_time,
                token_count=token_count,
                keywords=keywords_json  # ⭐ キーワードを保存
            )
            logger.info("💾 翻訳履歴をデータベースに保存しました")
        except Exception as db_error:
            logger.warning(f"⚠️ データベース保存エラー（処理は継続）: {db_error}")
        
        # ⭐ 進捗送信: 完了
        send_progress(task_id, 'complete', 100, '完了！')
        
        if response.success:
            return jsonify({
                'success': True,
                'summary': response.result,
                'original_length': len(text),
                'summary_length': len(response.result),
                'compression_ratio': len(response.result) / len(text) if text else 0.0,
                'execution_time': execution_time,
                'model_used': response.model_used,
                'confidence': quality_score / 100.0,
                'quality_score': round(quality_score, 1),
                'quality_details': {
                    'length': round(quality_details.get('length', 0), 1),
                    'completeness': round(quality_details.get('completeness', 0), 1),
                    'character_balance': round(quality_details.get('character_balance', 0), 1),
                    'punctuation': round(quality_details.get('punctuation', 0), 1),
                    'no_repetition': round(quality_details.get('no_repetition', 0), 1)
                },
                'token_usage': response.token_usage or {},
                'task_id': task_id
                # quality_metrics属性は存在しないため削除
            })
        else:
            return jsonify({
                'success': False,
                'error': response.error or '要約処理中にエラーが発生しました'
            }), 500
        
    except Exception as e:
        logger.error(f"要約エラー: {e}")
        return jsonify({
            'success': False,
            'error': f'要約処理中にエラーが発生しました: {str(e)}'
        }), 500

@app.route('/api/expand', methods=['POST'])
def api_expand():
    """文章展開API - Kaggle Apertus-8B統合版"""
    try:
        data = request.get_json() if request.is_json else request.form
        text = data.get('text', '').strip()
        target_length = int(data.get('target_length', 300))
        target_lang_code = data.get('target_lang', 'jpn_Jpan')
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'テキストが入力されていません'
            }), 400
        
        if len(text) > 300:
            return jsonify({
                'success': False,
                'error': '展開元テキストは300文字以下にしてください'
            }), 400
        
        # ⭐ Kaggle APIで展開
        start_time = time.time()
        
        if not kaggle_client:
            return jsonify({
                'success': False,
                'error': 'Kaggle APIが設定されていません。.envファイルを確認してください。'
            }), 503
        
        if not kaggle_client.is_available():
            return jsonify({
                'success': False,
                'error': '🔌 AIサーバーに接続できません。Kaggle Notebookが起動しているか確認してください。'
            }), 503
        
        # ⭐ 言語コードを言語名に変換
        from config import get_language_name
        target_lang = get_language_name(target_lang_code, 'Japanese')
        
        # Kaggle /expand エンドポイントを呼び出し
        kaggle_result = kaggle_client.expand(
            text=text,
            target_length=target_length,
            target_lang=target_lang
        )
        
        execution_time = time.time() - start_time
        
        if kaggle_result and kaggle_result.get('success'):
            expanded_text = kaggle_result.get('result', '')
            
            # ⭐ 展開結果をクリーニング（不要な記号を除去）
            expanded_text = clean_summary_result(expanded_text)
            
            # ⭐ データベースに展開履歴を保存
            try:
                db = get_db()
                
                # ⭐ キーワードを抽出（展開後のテキストから）
                keywords = extract_keywords(expanded_text, max_keywords=8)
                keywords_json = json.dumps(keywords, ensure_ascii=False) if keywords else None
                
                db.save_translation(
                    source_lang='auto',
                    target_lang=target_lang_code,
                    original_text=text[:5000],
                    translated_text=expanded_text[:3000],
                    summary_mode='expand',  # ⭐ 展開として記録
                    quality_score=90.0,
                    file_name=None,
                    processing_time=execution_time,
                    token_count=0,
                    keywords=keywords_json  # ⭐ キーワードを保存
                )
                logger.info("💾 展開履歴をデータベースに保存しました")
            except Exception as db_error:
                logger.warning(f"⚠️ 展開履歴保存エラー（処理は継続）: {db_error}")
            
            return jsonify({
                'success': True,
                'expanded_text': expanded_text,
                'original_length': len(text),
                'expanded_length': len(expanded_text),
                'expansion_ratio': len(expanded_text) / len(text) if text else 0.0,
                'execution_time': execution_time,
                'model_used': 'Apertus-8B-Instruct',
                'confidence': 0.9
            })
        else:
            error_msg = kaggle_result.get('error', '展開処理に失敗しました') if kaggle_result else '展開処理に失敗しました'
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
        
    except Exception as e:
        logger.error(f"展開エラー: {e}")
        return jsonify({
            'success': False,
            'error': f'文章展開中にエラーが発生しました: {str(e)}'
        }), 500


@app.route('/api/explain-code', methods=['POST'])
def api_explain_code():
    """コード解説API - Kaggle Apertus-8B統合版"""
    try:
        data = request.get_json() if request.is_json else request.form
        code = data.get('code', '').strip()
        language = data.get('language', 'auto')  # プログラミング言語
        target_lang_code = data.get('target_lang', 'jpn_Jpan')  # 解説の出力言語
        
        if not code:
            return jsonify({
                'success': False,
                'error': 'コードが入力されていません'
            }), 400
        
        if len(code) > 5000:
            return jsonify({
                'success': False,
                'error': 'コードは5000文字以下にしてください'
            }), 400
        
        # ⭐ Kaggle APIで解説
        start_time = time.time()
        
        if not kaggle_client:
            return jsonify({
                'success': False,
                'error': 'Kaggle APIが設定されていません。.envファイルを確認してください。'
            }), 503
        
        if not kaggle_client.is_available():
            return jsonify({
                'success': False,
                'error': '🔌 AIサーバーに接続できません。Kaggle Notebookが起動しているか確認してください。'
            }), 503
        
        # ⭐ 言語コードを言語名に変換
        from config import get_language_name
        target_lang = get_language_name(target_lang_code, 'Japanese')
        
        # Kaggle /explain-code エンドポイントを呼び出し
        kaggle_result = kaggle_client.explain_code(
            code=code,
            language=language,
            target_lang=target_lang
        )
        
        execution_time = time.time() - start_time
        
        if kaggle_result and kaggle_result.get('success'):
            explanation = kaggle_result.get('explanation', '')
            detected_lang = kaggle_result.get('detected_language', language)
            
            # ⭐ 結果をクリーニング
            explanation = clean_summary_result(explanation)
            
            return jsonify({
                'success': True,
                'explanation': explanation,
                'code_length': len(code),
                'explanation_length': len(explanation),
                'detected_language': detected_lang,
                'execution_time': execution_time,
                'model_used': 'Apertus-8B-Instruct'
            })
        else:
            error_msg = kaggle_result.get('error', 'コード解説に失敗しました') if kaggle_result else 'コード解説に失敗しました'
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
        
    except Exception as e:
        logger.error(f"コード解説エラー: {e}")
        return jsonify({
            'success': False,
            'error': f'コード解説中にエラーが発生しました: {str(e)}'
        }), 500


@app.route('/api/url-summarize', methods=['POST'])
def api_url_summarize():
    """URL要約API - Kaggle Apertus-8B統合版"""
    try:
        data = request.get_json() if request.is_json else request.form
        url = data.get('url', '').strip()
        max_length = int(data.get('max_length', 200))
        summary_mode = data.get('summary_mode', 'short')  # 'short' or 'long'
        target_lang = data.get('target_lang', 'Japanese')
        style = data.get('style', 'balanced')
        
        # ⭐ 技術ドキュメントを検出して解説スタイルに変更
        tech_doc_patterns = [
            'docs.', 'documentation', '/docs/', '/doc/',
            'readthedocs', 'palletsprojects', 'djangoproject',
            'reactjs.org', 'react.dev', 'vuejs.org', 'angular.io',
            'numpy.org', 'pandas.pydata', 'pytorch.org', 'tensorflow.org',
            'developer.mozilla', 'devdocs.io',
            '/api/', '/reference/', '/guide/', '/tutorial/',
            'github.com', 'gitlab.com',
        ]
        is_tech_doc = any(pattern in url.lower() for pattern in tech_doc_patterns)
        
        if is_tech_doc:
            style = 'tech_doc'
            logger.info(f"📚 技術ドキュメント検出: {url[:50]}... → tech_docスタイル適用")
        
        if not url:
            return jsonify({
                'success': False,
                'error': 'URLが入力されていません'
            }), 400
        
        # URL形式チェック
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return jsonify({
                'success': False,
                'error': '有効なURLを入力してください'
            }), 400
        
        # URL記事取得（ローカルで実行）
        extract_result = extract_text_from_url(url)
        if not extract_result['success']:
            return jsonify(extract_result), 400
        
        # ⭐ テキスト前処理（コードや不要な行を除去、長さ制限）
        original_length = len(extract_result['content'])
        processed_text = preprocess_text_for_summarization(extract_result['content'], max_chars=5000)
        logger.info(f"📝 前処理: {original_length}文字 → {len(processed_text)}文字")
        
        if len(processed_text) < 50:
            return jsonify({
                'success': False,
                'error': 'コンテンツを十分に取得できませんでした。このサイトはコードが中心の技術記事、またはJavaScriptで動的に生成されている可能性があります。'
            }), 400
        
        # ⭐ Kaggle APIで要約
        start_time = time.time()
        
        if not kaggle_client:
            return jsonify({
                'success': False,
                'error': 'Kaggle APIが設定されていません。.envファイルを確認してください。'
            }), 503
        
        if not kaggle_client.is_available():
            return jsonify({
                'success': False,
                'error': '🔌 AIサーバーに接続できません。Kaggle Notebookが起動しているか確認してください。'
            }), 503
        
        # ⭐ 言語コードを人間可読な言語名に変換 (config.pyから共通定義を使用)
        from config import get_language_name
        target_lang_name = get_language_name(target_lang, 'Japanese')
        logger.info(f"📝 URL要約: 出力言語={target_lang} → {target_lang_name}")
        
        # Kaggle /summarize エンドポイントを呼び出し
        kaggle_result = kaggle_client.summarize(
            text=processed_text,  # ⭐ 前処理済みテキストを使用
            max_length=max_length,
            source_lang='auto-detect',
            target_lang=target_lang_name,  # ⭐ 変換後の言語名を使用
            style=style,
            summary_mode=summary_mode
        )
        
        execution_time = time.time() - start_time
        
        if kaggle_result and kaggle_result.get('success'):
            summary = kaggle_result.get('summary', '')
            
            # ⭐ 要約結果をクリーニング（不要な記号を除去）
            summary = clean_summary_result(summary)
            
            return jsonify({
                'success': True,
                'title': extract_result['title'],
                'url': url,
                'summary': summary,
                'original_length': len(extract_result['content']),
                'summary_length': len(summary),
                'compression_ratio': len(summary) / len(extract_result['content']) if extract_result['content'] else 0.0,
                'execution_time': execution_time,
                'model_used': 'Apertus-8B-Instruct',
                'confidence': 0.9
            })
        else:
            error_msg = kaggle_result.get('error', 'URL要約に失敗しました') if kaggle_result else 'URL要約に失敗しました'
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
        
    except Exception as e:
        logger.error(f"URL要約エラー: {e}")
        return jsonify({
            'success': False,
            'error': f'URL要約中にエラーが発生しました: {str(e)}'
        }), 500

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """ファイルアップロードAPI"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'ファイルが選択されていません'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'ファイルが選択されていません'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'サポートされていないファイル形式です'
            }), 400
        
        # ファイル保存
        filename = secure_filename(file.filename)
        file_path = Config.UPLOAD_FOLDER / filename
        file.save(file_path)
        
        # ファイル処理
        text = process_uploaded_file(file_path)
        
        # ファイル削除
        file_path.unlink(missing_ok=True)
        
        return jsonify({
            'success': True,
            'text': text,
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"アップロードエラー: {e}")
        return jsonify({
            'success': False,
            'error': f'ファイル処理中にエラーが発生しました: {str(e)}'
        }), 500

@app.route('/api/history')
def api_history():
    """履歴API（データベースから取得）- キーワード・品質スコア対応"""
    try:
        db = get_db()
        # ⭐ 最新200件を取得（直接SQL接続を使用）
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            history_rows = cursor.execute('''
                SELECT id, timestamp as created_at, source_lang, target_lang, 
                       original_text, translated_text as result_text, 
                       summary_mode as operation_type, processing_time as execution_time,
                       quality_score, keywords
                FROM translation_history
                ORDER BY timestamp DESC
                LIMIT 200
            ''').fetchall()
        
        # 辞書形式に変換
        history = []
        for row in history_rows:
            # キーワードをパース
            keywords = []
            if row['keywords']:
                try:
                    keywords = json.loads(row['keywords'])
                except:
                    keywords = []
            
            history.append({
                'id': row['id'],
                'operation_type': row['operation_type'] or 'summarize',
                'original_text': row['original_text'],
                'result_text': row['result_text'],
                'source_lang': row['source_lang'],
                'target_lang': row['target_lang'],
                'execution_time': row['execution_time'],
                'created_at': row['created_at'],
                'quality_score': row['quality_score'],  # ⭐ 品質スコア追加
                'keywords': keywords  # ⭐ キーワード追加
            })
        
        return jsonify({
            'success': True,
            'history': history,
            'total': len(history)
        })
    except Exception as e:
        logger.error(f"履歴取得エラー: {e}")
        return jsonify({
            'success': False,
            'history': [],
            'total': 0
        })

@app.route('/api/stats')
def api_stats():
    """統計API（データベースから取得）"""
    try:
        db = get_db()
        
        # データベースから統計情報を取得（直接SQL接続を使用）
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            stats_query = cursor.execute('''
                SELECT 
                    COUNT(*) as total_operations,
                    AVG(processing_time) as avg_execution_time,
                    SUM(LENGTH(original_text)) as total_chars_processed
                FROM translation_history
            ''').fetchone()
        
        if not stats_query or stats_query['total_operations'] == 0:
            return jsonify({
                'total_operations': 0,
                'average_execution_time': 0,
                'total_text_processed': 0
            })
        
        return jsonify({
            'total_operations': stats_query['total_operations'],
            'average_execution_time': stats_query['avg_execution_time'] or 0,
            'total_text_processed': stats_query['total_chars_processed'] or 0
        })
    except Exception as e:
        logger.error(f"統計取得エラー: {e}")
        return jsonify({
            'total_operations': 0,
            'average_execution_time': 0,
            'total_text_processed': 0
        })

# 🎓 Apertus学習システム - フィードバックAPI

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """
    フィードバック収集API - AIゼミ用（UTF-8対応強化版）
    Apertus学習システムと連携
    """
    try:
        from services.apertus_learning_system import get_learning_system
        
        # UTF-8デコードを明示的に実行
        data = request.get_json(force=True, silent=False)
        
        if data is None:
            return jsonify({
                'success': False,
                'error': 'リクエストボディが空か、JSONフォーマットが不正です'
            }), 400
        
        # 必須フィールドの検証
        required_fields = ['task_id', 'original_text', 'result_text', 'task_type',
                          'user_score', 'accuracy_score', 'fluency_score', 'completeness_score']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'必須フィールド "{field}" がありません'
                }), 400
        
        # 学習システムを取得
        learning_system = get_learning_system()
        
        # テキストフィールドのUTF-8クリーニング
        def clean_utf8_text(text):
            if not text:
                return ""
            # Unicode制御文字を除去
            import re
            cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', str(text))
            return cleaned
        
        original_text = clean_utf8_text(data.get('original_text', ''))
        result_text = clean_utf8_text(data.get('result_text', ''))
        comment = clean_utf8_text(data.get('comment', ''))
        
        # フィードバックを送信
        feedback = learning_system.submit_feedback(
            task_id=str(data['task_id']),
            original_text=original_text,
            result_text=result_text,
            user_score=float(data['user_score']),
            accuracy_score=float(data['accuracy_score']),
            fluency_score=float(data['fluency_score']),
            completeness_score=float(data['completeness_score']),
            task_type=str(data['task_type']),
            user_feedback=comment if comment else None
        )
        
        # メトリクスを取得
        metrics = learning_system.get_metrics()
        
        logger.info(f"✅ フィードバック受信: task_id={data['task_id']}, score={data['user_score']}")
        
        return jsonify({
            'success': True,
            'feedback_id': feedback.task_id,
            'total_feedbacks': metrics.total_tasks,
            'average_score': round(metrics.average_score, 2),
            'improvement_rate': round(metrics.improvement_rate, 2)
        })
        
    except KeyError as e:
        logger.error(f"フィールド不足エラー: {e}")
        return jsonify({
            'success': False,
            'error': f'必須フィールドが不足しています: {str(e)}'
        }), 400
    except ValueError as e:
        logger.error(f"データ型エラー: {e}")
        return jsonify({
            'success': False,
            'error': f'データ型が不正です: {str(e)}'
        }), 400
    except Exception as e:
        import traceback
        logger.error(f"フィードバックエラー: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'フィードバック処理中にエラーが発生しました: {str(e)}'
        }), 500

@app.route('/api/learning-metrics', methods=['GET'])
def api_learning_metrics():
    """
    学習メトリクス取得API - AIゼミ用
    ダッシュボードで使用
    """
    try:
        from services.apertus_learning_system import get_learning_system
        
        learning_system = get_learning_system()
        metrics = learning_system.get_metrics()
        
        return jsonify({
            'success': True,
            'metrics': {
                'total_tasks': metrics.total_tasks,
                'average_score': round(metrics.average_score, 2),
                'improvement_rate': round(metrics.improvement_rate, 2),
                'best_score': round(metrics.best_score, 2),
                'worst_score': round(metrics.worst_score, 2),
                'accuracy_trend': [round(s, 2) for s in metrics.accuracy_trend[-20:]],  # 直近20件
                'fluency_trend': [round(s, 2) for s in metrics.fluency_trend[-20:]]
            }
        })
        
    except Exception as e:
        logger.error(f"メトリクス取得エラー: {e}")
        return jsonify({
            'success': False,
            'error': f'メトリクス取得中にエラーが発生しました: {str(e)}'
        }), 500

@app.route('/settings')
def settings():
    """設定ページ"""
    return render_template('settings.html')

@app.route('/api/service-status')
def api_service_status():
    """AIサービス状態API"""
    try:
        status = hf_service.get_status()
        return jsonify({
            'success': True,
            'service': status['service'],
            'model': status['model'],
            'available': status['available'],
            'device': status['device'],
            'api_key_required': status['api_key_required'],
            'completely_free': status['completely_free']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health-check')
def api_health_check():
    """ヘルスチェックAPI"""
    try:
        is_healthy = hf_service.available
        
        return jsonify({
            'success': is_healthy,
            'status': 'healthy' if is_healthy else 'unhealthy',
            'completely_free': True,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ===== 学習機能エンドポイント =====

# 🔥 新機能: バッチ処理API
@app.route('/api/batch-process', methods=['POST'])
def api_batch_process():
    """バッチ処理API - 複数ファイルの一括処理"""
    try:
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'ファイルが選択されていません'
            }), 400
        
        files = request.files.getlist('files')
        summary_mode = request.form.get('summary_mode', 'short')
        source_lang = request.form.get('source_lang', 'auto')
        target_lang = request.form.get('target_lang', 'jpn_Jpan')
        
        batch_id = str(uuid.uuid4())
        results = []
        total_files = len(files)
        completed = 0
        failed = 0
        
        start_time = time.time()
        
        logger.info(f"📦 バッチ処理開始: {total_files}ファイル (ID: {batch_id})")
        
        for i, file in enumerate(files):
            try:
                if file.filename == '':
                    failed += 1
                    continue
                
                if not allowed_file(file.filename):
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': '対応していないファイル形式です'
                    })
                    failed += 1
                    continue
                
                # ファイル保存
                filename = secure_filename(file.filename)
                file_path = Config.UPLOAD_FOLDER / f"{uuid.uuid4()}_{filename}"
                file.save(file_path)
                
                # テキスト抽出
                text = process_uploaded_file(file_path)
                
                # ⭐ Kaggle APIで要約処理
                if not kaggle_client or not kaggle_client.is_available():
                    results.append({
                        'filename': filename,
                        'success': False,
                        'error': 'AIサーバーに接続できません'
                    })
                    failed += 1
                    file_path.unlink(missing_ok=True)
                    continue
                
                from config import get_language_name
                source_lang_name = get_language_name(source_lang)
                target_lang_name = get_language_name(target_lang, 'Japanese')
                
                kaggle_result = kaggle_client.summarize(
                    text=text,
                    max_length=400,
                    source_lang=source_lang_name,
                    target_lang=target_lang_name,
                    style='balanced',
                    summary_mode=summary_mode
                )
                
                # ファイル削除
                file_path.unlink(missing_ok=True)
                
                if kaggle_result and kaggle_result.get('success'):
                    summary = kaggle_result.get('summary', '')
                    results.append({
                        'filename': filename,
                        'success': True,
                        'summary': summary,
                        'original_length': len(text),
                        'summary_length': len(summary)
                    })
                    completed += 1
                else:
                    error_msg = kaggle_result.get('error', '要約に失敗しました') if kaggle_result else '要約に失敗しました'
                    results.append({
                        'filename': filename,
                        'success': False,
                        'error': error_msg
                    })
                    failed += 1
                
                logger.info(f"📄 処理完了: {i+1}/{total_files} - {filename}")
                
            except Exception as file_error:
                logger.error(f"❌ ファイル処理エラー ({file.filename}): {file_error}")
                results.append({
                    'filename': file.filename if file else 'unknown',
                    'success': False,
                    'error': str(file_error)
                })
                failed += 1
        
        total_time = time.time() - start_time
        
        # バッチ履歴を保存
        try:
            db = get_db()
            db.save_batch_history(
                batch_id=batch_id,
                total_files=total_files,
                completed_files=completed,
                failed_files=failed,
                total_time=total_time,
                status='completed' if failed == 0 else 'partial' if completed > 0 else 'failed'
            )
        except Exception as db_error:
            logger.warning(f"⚠️ バッチ履歴保存エラー: {db_error}")
        
        logger.info(f"✅ バッチ処理完了: {completed}/{total_files}成功, {failed}失敗 ({total_time:.1f}秒)")
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'total_files': total_files,
            'completed': completed,
            'failed': failed,
            'results': results,
            'total_time': total_time
        })
        
    except Exception as e:
        logger.error(f"バッチ処理エラー: {e}")
        return jsonify({
            'success': False,
            'error': f'バッチ処理中にエラーが発生しました: {str(e)}'
        }), 500


# 🔥 新機能: 全ページ翻訳API
@app.route('/api/full-translate', methods=['POST'])
def api_full_translate():
    """PDFの全ページを翻訳"""
    try:
        data = request.get_json() if request.is_json else request.form
        text = data.get('text', '').strip()
        source_lang = data.get('source_lang', 'auto')
        target_lang = data.get('target_lang', 'jpn_Jpan')
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'テキストが入力されていません'
            }), 400
        
        # テキストをページごとに分割（ページマーカーで判定）
        pages = []
        current_page = []
        
        for line in text.split('\n'):
            if line.startswith('━━━━━ ページ'):
                if current_page:
                    pages.append('\n'.join(current_page))
                current_page = []
            else:
                current_page.append(line)
        
        if current_page:
            pages.append('\n'.join(current_page))
        
        logger.info(f"📚 全ページ翻訳開始: {len(pages)}ページ")
        
        translated_pages = []
        start_time = time.time()
        
        # ⭐ Kaggle APIで翻訳
        if not kaggle_client or not kaggle_client.is_available():
            return jsonify({
                'success': False,
                'error': 'AIサーバーに接続できません'
            }), 503
        
        from config import get_language_name
        source_lang_name = get_language_name(source_lang)
        target_lang_name = get_language_name(target_lang, 'Japanese')
        
        for i, page_text in enumerate(pages):
            if not page_text.strip():
                continue
            
            # ページごとにKaggle APIで翻訳
            kaggle_result = kaggle_client.summarize(
                text=page_text,
                max_length=2000,  # 全文翻訳なので長め
                source_lang=source_lang_name,
                target_lang=target_lang_name,
                style='narrative',
                summary_mode='long'
            )
            
            if kaggle_result and kaggle_result.get('success'):
                translated_pages.append(f"━━━━━ ページ {i+1} ━━━━━\n{kaggle_result.get('summary', '')}")
                logger.info(f"✅ ページ {i+1}/{len(pages)} 翻訳完了")
            else:
                translated_pages.append(f"━━━━━ ページ {i+1} (翻訳失敗) ━━━━━\n{page_text}")
                logger.warning(f"⚠️ ページ {i+1} 翻訳失敗: {kaggle_result.get('error') if kaggle_result else '不明'}")
        
        total_time = time.time() - start_time
        full_translated_text = '\n\n'.join(translated_pages)
        
        logger.info(f"🎉 全ページ翻訳完了: {len(pages)}ページ ({total_time:.1f}秒)")
        
        return jsonify({
            'success': True,
            'translated_text': full_translated_text,
            'total_pages': len(pages),
            'total_time': total_time,
            'original_length': len(text),
            'translated_length': len(full_translated_text)
        })
        
    except Exception as e:
        logger.error(f"全ページ翻訳エラー: {e}")
        return jsonify({
            'success': False,
            'error': f'全ページ翻訳中にエラーが発生しました: {str(e)}'
        }), 500


# ⭐ Kaggle API翻訳エンドポイント
@app.route('/api/kaggle/translate', methods=['POST'])
def api_kaggle_translate():
    """Kaggle APIを使用した翻訳"""
    try:
        if not kaggle_client:
            return jsonify({
                'success': False,
                'error': 'Kaggle APIが利用できません'
            }), 503
        
        if not kaggle_client.is_available():
            return jsonify({
                'success': False,
                'error': 'Kaggle APIサーバーに接続できません'
            }), 503
        
        data = request.get_json() if request.is_json else request.form
        text = data.get('text', '').strip()
        source_lang = data.get('source_lang', 'English')
        target_lang = data.get('target_lang', 'Japanese')
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'テキストが入力されていません'
            }), 400
        
        logger.info(f"🚀 Kaggle翻訳開始: {source_lang} → {target_lang} ({len(text)}文字)")
        
        result = kaggle_client.translate(text, source_lang, target_lang)
        
        if result and result.get('success'):
            logger.info(f"✅ Kaggle翻訳完了: {result.get('time', 0):.1f}秒")
            return jsonify({
                'success': True,
                'translated_text': result.get('translation', ''),
                'processing_time': result.get('time', 0),
                'service': 'Kaggle Apertus-8B',
                'source_lang': source_lang,
                'target_lang': target_lang,
                'original_length': len(text),
                'translated_length': len(result.get('translation', ''))
            })
        else:
            error_msg = result.get('error', '不明なエラー') if result else '応答なし'
            logger.error(f"❌ Kaggle翻訳失敗: {error_msg}")
            return jsonify({
                'success': False,
                'error': f'Kaggle翻訳失敗: {error_msg}'
            }), 500
            
    except Exception as e:
        logger.error(f"Kaggle翻訳エラー: {e}")
        return jsonify({
            'success': False,
            'error': f'Kaggle翻訳エラー: {str(e)}'
        }), 500


@app.route('/api/kaggle/status', methods=['GET'])
def api_kaggle_status():
    """Kaggle APIのステータス確認"""
    try:
        if not kaggle_client:
            return jsonify({
                'available': False,
                'message': 'Kaggle APIが設定されていません'
            })
        
        is_available = kaggle_client.is_available(force_check=True)
        
        return jsonify({
            'available': is_available,
            'message': 'Kaggle APIは正常に動作しています' if is_available else 'Kaggle APIに接続できません',
            'url': kaggle_client.base_url if kaggle_client else None
        })
    except Exception as e:
        logger.error(f"Kaggleステータスチェックエラー: {e}")
        return jsonify({
            'available': False,
            'message': f'エラー: {str(e)}'
        })


# 🔥 新機能: ユーザー辞書API
@app.route('/api/dictionary', methods=['GET'])
def api_get_dictionary():
    """ユーザー辞書を取得"""
    try:
        db = get_db()
        source_lang = request.args.get('source_lang')
        target_lang = request.args.get('target_lang')
        
        dictionary = db.get_user_dictionary(source_lang, target_lang)
        
        return jsonify({
            'success': True,
            'dictionary': dictionary,
            'total': len(dictionary)
        })
    except Exception as e:
        logger.error(f"辞書取得エラー: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/dictionary', methods=['POST'])
def api_add_dictionary():
    """ユーザー辞書に用語を追加"""
    try:
        data = request.get_json()
        source_term = data.get('source_term', '').strip()
        target_term = data.get('target_term', '').strip()
        source_lang = data.get('source_lang', 'eng_Latn')
        target_lang = data.get('target_lang', 'jpn_Jpan')
        category = data.get('category', '')
        
        if not source_term or not target_term:
            return jsonify({
                'success': False,
                'error': '用語が入力されていません'
            }), 400
        
        db = get_db()
        term_id = db.add_user_term(source_term, target_term, source_lang, target_lang, category)

        # キャッシュを無効化して最新辞書を反映
        hf_service.invalidate_dictionary_cache(source_lang=source_lang, target_lang=target_lang)
        
        return jsonify({
            'success': True,
            'message': 'ユーザー辞書に追加しました',
            'term_id': term_id
        })
    except Exception as e:
        logger.error(f"辞書追加エラー: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/dictionary/<int:term_id>', methods=['DELETE'])
def api_delete_dictionary(term_id):
    """ユーザー辞書から用語を削除"""
    try:
        db = get_db()
        success = db.delete_user_term(term_id)
        
        if success:
            hf_service.invalidate_dictionary_cache()
            return jsonify({
                'success': True,
                'message': '用語を削除しました'
            })
        else:
            return jsonify({
                'success': False,
                'error': '用語が見つかりません'
            }), 404
    except Exception as e:
        logger.error(f"辞書削除エラー: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# 🔥 新機能: データベース履歴API
@app.route('/api/db-history', methods=['GET'])
def api_db_history():
    """データベースから翻訳履歴を取得"""
    try:
        db = get_db()
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        search = request.args.get('search', '')
        
        if search:
            history = db.search_translation_history(search)
        else:
            history = db.get_translation_history(limit, offset)
        
        return jsonify({
            'success': True,
            'history': history,
            'total': len(history)
        })
    except Exception as e:
        logger.error(f"履歴取得エラー: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/db-stats', methods=['GET'])
def api_db_stats():
    """データベース統計情報を取得"""
    try:
        db = get_db()
        stats = db.get_statistics()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"統計取得エラー: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch-history', methods=['GET'])
def api_batch_history():
    """バッチ処理履歴を取得"""
    try:
        db = get_db()
        limit = int(request.args.get('limit', 20))
        
        history = db.get_batch_history(limit)
        
        return jsonify({
            'success': True,
            'history': history,
            'total': len(history)
        })
    except Exception as e:
        logger.error(f"バッチ履歴取得エラー: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ⭐ ユーザー認証API
@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """ユーザー登録"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        email = data.get('email', '').strip()
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'ユーザー名とパスワードは必須です'}), 400
        
        if len(password) < 6:
            return jsonify({'success': False, 'error': 'パスワードは6文字以上で入力してください'}), 400
        
        db = get_db()
        password_hash = hash_password(password)
        user_id = db.create_user(username, password_hash, email)
        
        if user_id is None:
            return jsonify({'success': False, 'error': 'ユーザー名が既に使用されています'}), 409
        
        # 自動ログイン
        session['user_id'] = user_id
        session['username'] = username
        
        logger.info(f"✅ ユーザー登録成功: {username}")
        return jsonify({
            'success': True,
            'message': '登録が完了しました',
            'user': {'id': user_id, 'username': username}
        })
    except Exception as e:
        logger.error(f"❌ 登録エラー: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """ログイン"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'ユーザー名とパスワードを入力してください'}), 400
        
        db = get_db()
        user = db.get_user_by_username(username)
        
        if not user:
            return jsonify({'success': False, 'error': 'ユーザーが見つかりません'}), 404
        
        password_hash = hash_password(password)
        if password_hash != user['password_hash']:
            return jsonify({'success': False, 'error': 'パスワードが正しくありません'}), 401
        
        # ログイン成功
        session['user_id'] = user['id']
        session['username'] = user['username']
        db.update_user_login(user['id'])
        
        logger.info(f"✅ ログイン成功: {username}")
        return jsonify({
            'success': True,
            'message': 'ログインしました',
            'user': {'id': user['id'], 'username': user['username']}
        })
    except Exception as e:
        logger.error(f"❌ ログインエラー: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """ログアウト"""
    username = session.get('username', 'Unknown')
    session.clear()
    logger.info(f"👋 ログアウト: {username}")
    return jsonify({'success': True, 'message': 'ログアウトしました'})


@app.route('/api/auth/status', methods=['GET'])
def api_auth_status():
    """ログイン状態確認"""
    if 'user_id' in session:
        return jsonify({
            'logged_in': True,
            'user': {
                'id': session['user_id'],
                'username': session['username']
            }
        })
    else:
        return jsonify({'logged_in': False})


# ========================================
# 🇨🇭 Apertus学習システムAPI
# ========================================

# ⭐ リアルタイム進捗通知関数
def send_progress(task_id: str, stage: str, progress: int, message: str):
    """
    WebSocketで進捗状況をクライアントに送信
    
    Args:
        task_id: タスクID
        stage: 処理ステージ (extract, translate, summarize, etc.)
        progress: 進捗率 (0-100)
        message: 進捗メッセージ
    """
    try:
        socketio.emit('progress_update', {
            'task_id': task_id,
            'stage': stage,
            'progress': progress,
            'message': message,
            'timestamp': time.time()
        })
        logger.info(f"📡 進捗送信 [{task_id}] {stage}: {progress}% - {message}")
    except Exception as e:
        logger.warning(f"⚠️ 進捗送信失敗: {e}")

# ⭐ WebSocketイベントハンドラ
@socketio.on('connect')
def handle_connect():
    """クライアント接続時"""
    logger.info(f"🔌 クライアント接続: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """クライアント切断時"""
    logger.info(f"🔌 クライアント切断: {request.sid}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 recapisure - 高性能多言語要約サービス (Apertus-8B版)")
    print("="*60)
    print("✨ 機能:")
    print("   📝 長文要約 - 大量テキストの効率的要約 (箇条書き整形対応)")
    print("   📈 短文展開 - 短文を詳細な長文に変換")
    print("   🌐 URL要約 - Web記事の自動取得＋要約")
    print("   📄 ファイル対応 - TXT, MD, PDF, 画像 (OCR)")
    print("   📡 リアルタイム進捗表示")
    print("="*60)
    
    # Kaggle Apertus-8Bの状態確認
    if kaggle_client and kaggle_client.is_available():
        print(f"✅ Kaggle Apertus-8B 接続成功")
        print(f"🎯 使用モデル: swiss-ai/Apertus-8B-Instruct-2509")
        print(f"🌍 対応言語: 1,811言語")
        print(f"💰 完全無料・APIキー不要!")
    else:
        print("⚠️  Kaggle Apertus-8B に接続できません")
        print("💡 Kaggle Notebookを起動してください")
        print("   詳細: KAGGLE_NGROK_SETUP.md を参照")
    
    print("="*60)
    print("🌐 http://localhost:5000 でアクセス可能")
    print("⏹️  停止するには Ctrl+C を押してください")
    print("="*60 + "\n")
    
    # セッション履歴のクリーニング（データベース移行のため）
    @app.before_request
    def clear_old_session_history():
        """古いセッション履歴をクリア（Cookieサイズ制限対策）"""
        if 'history' in session:
            session.pop('history', None)
            session.modified = True
    
    # SocketIO対応の実行 (use_reloader=False: ファイル変更時の自動リロードを無効化)
    socketio.run(app, host='127.0.0.1', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
