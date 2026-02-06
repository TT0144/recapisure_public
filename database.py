#!/usr/bin/env python3
"""
データベース管理モジュール
翻訳履歴・要約履歴の保存と管理
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Database:
    """SQLiteデータベース管理クラス"""
    
    def __init__(self, db_path="data/history.db"):
        """
        データベース初期化
        
        Args:
            db_path: データベースファイルのパス
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """データベーステーブルを初期化"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # ⭐ ユーザーテーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT UNIQUE,
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    is_active INTEGER DEFAULT 1,
                    settings TEXT DEFAULT '{}'
                )
            ''')
            
            # 翻訳履歴テーブル（ユーザーID追加）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS translation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    timestamp TEXT NOT NULL,
                    source_lang TEXT NOT NULL,
                    target_lang TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    summary_mode TEXT,
                    quality_score REAL,
                    file_name TEXT,
                    processing_time REAL,
                    token_count INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # ユーザー辞書テーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_dictionary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_term TEXT NOT NULL,
                    target_term TEXT NOT NULL,
                    source_lang TEXT NOT NULL,
                    target_lang TEXT NOT NULL,
                    category TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(source_term, source_lang, target_lang)
                )
            ''')
            
            # バッチ処理履歴テーブル
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS batch_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    total_files INTEGER NOT NULL,
                    completed_files INTEGER NOT NULL,
                    failed_files INTEGER NOT NULL,
                    total_time REAL,
                    status TEXT NOT NULL
                )
            ''')
            
            # ⭐ キーワード列を追加（既存のテーブルの場合）
            try:
                cursor.execute('ALTER TABLE translation_history ADD COLUMN keywords TEXT')
                logger.info("✅ キーワード列を追加しました")
            except sqlite3.OperationalError:
                pass  # 列が既に存在する場合は無視
            
            conn.commit()
            logger.info("✅ データベース初期化完了")
    
    def save_translation(self, source_lang, target_lang, original_text, 
                        translated_text, summary_mode=None, quality_score=None,
                        file_name=None, processing_time=None, token_count=None, keywords=None):
        """
        翻訳履歴を保存
        
        Args:
            source_lang: 元言語
            target_lang: 翻訳先言語
            original_text: 元のテキスト
            translated_text: 翻訳されたテキスト
            summary_mode: 要約モード
            quality_score: 品質スコア (0-100)
            file_name: ファイル名
            processing_time: 処理時間（秒）
            token_count: トークン数
            keywords: キーワード（JSON文字列）
            
        Returns:
            保存されたレコードのID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO translation_history 
                (timestamp, source_lang, target_lang, original_text, translated_text,
                 summary_mode, quality_score, file_name, processing_time, token_count, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, source_lang, target_lang, original_text, translated_text,
                  summary_mode, quality_score, file_name, processing_time, token_count, keywords))
            
            conn.commit()
            record_id = cursor.lastrowid
            logger.info(f"💾 翻訳履歴保存: ID={record_id}")
            return record_id
    
    def get_translation_history(self, limit=200, offset=0):
        """
        翻訳履歴を取得
        
        Args:
            limit: 取得件数（デフォルト200件）
            offset: オフセット
            
        Returns:
            翻訳履歴のリスト
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM translation_history 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def search_translation_history(self, search_term, source_lang=None, target_lang=None):
        """
        翻訳履歴を検索
        
        Args:
            search_term: 検索キーワード
            source_lang: 元言語でフィルタ
            target_lang: 翻訳先言語でフィルタ
            
        Returns:
            検索結果のリスト
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = '''
                SELECT * FROM translation_history 
                WHERE (original_text LIKE ? OR translated_text LIKE ?)
            '''
            params = [f'%{search_term}%', f'%{search_term}%']
            
            if source_lang:
                query += ' AND source_lang = ?'
                params.append(source_lang)
            
            if target_lang:
                query += ' AND target_lang = ?'
                params.append(target_lang)
            
            query += ' ORDER BY timestamp DESC LIMIT 200'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def add_user_term(self, source_term, target_term, source_lang, target_lang, category=None):
        """
        ユーザー辞書に用語を追加
        
        Args:
            source_term: 元の用語
            target_term: 翻訳先の用語
            source_lang: 元言語
            target_lang: 翻訳先言語
            category: カテゴリ（オプション）
            
        Returns:
            追加されたレコードのID（既存の場合は更新）
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            # 既存チェック
            cursor.execute('''
                SELECT id FROM user_dictionary 
                WHERE source_term = ? AND source_lang = ? AND target_lang = ?
            ''', (source_term, source_lang, target_lang))
            
            existing = cursor.fetchone()
            
            if existing:
                # 更新
                cursor.execute('''
                    UPDATE user_dictionary 
                    SET target_term = ?, category = ?, created_at = ?
                    WHERE id = ?
                ''', (target_term, category, timestamp, existing[0]))
                logger.info(f"📝 ユーザー辞書更新: {source_term} → {target_term}")
                return existing[0]
            else:
                # 新規追加
                cursor.execute('''
                    INSERT INTO user_dictionary 
                    (source_term, target_term, source_lang, target_lang, category, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (source_term, target_term, source_lang, target_lang, category, timestamp))
                conn.commit()
                record_id = cursor.lastrowid
                logger.info(f"📝 ユーザー辞書追加: {source_term} → {target_term}")
                return record_id
    
    def get_user_dictionary(self, source_lang=None, target_lang=None):
        """
        ユーザー辞書を取得
        
        Args:
            source_lang: 元言語でフィルタ
            target_lang: 翻訳先言語でフィルタ
            
        Returns:
            辞書のリスト
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM user_dictionary WHERE 1=1'
            params = []
            
            if source_lang:
                query += ' AND source_lang = ?'
                params.append(source_lang)
            
            if target_lang:
                query += ' AND target_lang = ?'
                params.append(target_lang)
            
            query += ' ORDER BY created_at DESC'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def delete_user_term(self, term_id):
        """
        ユーザー辞書から用語を削除
        
        Args:
            term_id: 削除する用語のID
            
        Returns:
            削除成功ならTrue
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM user_dictionary WHERE id = ?', (term_id,))
            conn.commit()
            logger.info(f"🗑️ ユーザー辞書削除: ID={term_id}")
            return cursor.rowcount > 0
    
    def save_batch_history(self, batch_id, total_files, completed_files, 
                          failed_files, total_time, status):
        """
        バッチ処理履歴を保存
        
        Args:
            batch_id: バッチID
            total_files: 総ファイル数
            completed_files: 完了ファイル数
            failed_files: 失敗ファイル数
            total_time: 総処理時間
            status: ステータス
            
        Returns:
            保存されたレコードのID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO batch_history 
                (batch_id, timestamp, total_files, completed_files, failed_files, total_time, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (batch_id, timestamp, total_files, completed_files, failed_files, total_time, status))
            
            conn.commit()
            record_id = cursor.lastrowid
            logger.info(f"📦 バッチ履歴保存: ID={record_id}, {completed_files}/{total_files}完了")
            return record_id
    
    def get_batch_history(self, limit=20):
        """
        バッチ処理履歴を取得
        
        Args:
            limit: 取得件数
            
        Returns:
            バッチ履歴のリスト
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM batch_history 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_statistics(self):
        """
        統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 翻訳総数
            cursor.execute('SELECT COUNT(*) FROM translation_history')
            total_translations = cursor.fetchone()[0]
            
            # 言語ペア別統計
            cursor.execute('''
                SELECT source_lang, target_lang, COUNT(*) as count
                FROM translation_history
                GROUP BY source_lang, target_lang
                ORDER BY count DESC
                LIMIT 5
            ''')
            top_language_pairs = [
                {'source': row[0], 'target': row[1], 'count': row[2]}
                for row in cursor.fetchall()
            ]
            
            # 平均品質スコア
            cursor.execute('SELECT AVG(quality_score) FROM translation_history WHERE quality_score IS NOT NULL')
            avg_quality = cursor.fetchone()[0] or 0
            
            # ユーザー辞書エントリ数
            cursor.execute('SELECT COUNT(*) FROM user_dictionary')
            user_terms = cursor.fetchone()[0]
            
            # バッチ処理総数
            cursor.execute('SELECT COUNT(*) FROM batch_history')
            total_batches = cursor.fetchone()[0]
            
            return {
                'total_translations': total_translations,
                'top_language_pairs': top_language_pairs,
                'avg_quality_score': round(avg_quality, 2),
                'user_dictionary_terms': user_terms,
                'total_batches': total_batches
            }
    
    # ⭐ ユーザー管理機能
    def create_user(self, username, password_hash, email=None):
        """
        新規ユーザー作成
        
        Args:
            username: ユーザー名
            password_hash: パスワードハッシュ
            email: メールアドレス（オプション）
            
        Returns:
            ユーザーID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            try:
                cursor.execute('''
                    INSERT INTO users (username, password_hash, email, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (username, password_hash, email, timestamp))
                conn.commit()
                user_id = cursor.lastrowid
                logger.info(f"👤 ユーザー作成: {username} (ID={user_id})")
                return user_id
            except sqlite3.IntegrityError:
                logger.warning(f"⚠️ ユーザー名重複: {username}")
                return None
    
    def get_user_by_username(self, username):
        """ユーザー名でユーザーを取得"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_user_login(self, user_id):
        """最終ログイン時刻を更新"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', (timestamp, user_id))
            conn.commit()
    
    def update_user_settings(self, user_id, settings):
        """
        ユーザー設定を更新
        
        Args:
            user_id: ユーザーID
            settings: 設定辞書
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET settings = ? WHERE id = ?', 
                         (json.dumps(settings, ensure_ascii=False), user_id))
            conn.commit()
            logger.info(f"⚙️ ユーザー設定更新: ID={user_id}")


# グローバルデータベースインスタンス
_db_instance = None

def get_db():
    """データベースインスタンスを取得（シングルトン）"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
