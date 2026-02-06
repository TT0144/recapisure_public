%%writefile kaggle_api_server.py
"""
Kaggle Notebook用 Apertus-8B APIサーバー

⚠️ このファイルをKaggle Notebookにコピーして使用してください
⚠️ 以下の変数を自分の値に置き換えてください：
    - API_KEY: 自分で生成したランダムなAPIキー（32文字以上推奨）
    - hf_token: HuggingFaceのアクセストークン
"""
from flask import Flask, request, jsonify
import torch
import os
import time
from functools import wraps

# ⭐ 進捗バーとログを抑制
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)

app = Flask(__name__)

tokenizer = None
model = None

# 🔒 重要: このAPIキーを自分で生成したランダムな文字列に置き換えてください
# 例: import secrets; print(secrets.token_urlsafe(32))
API_KEY = "YOUR_RANDOM_API_KEY_HERE_REPLACE_ME"

def require_api_key(f):
    """API認証デコレータ"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"success": False, "error": "Unauthorized"}), 401
        token = auth_header.replace('Bearer ', '')
        if token != API_KEY:
            return jsonify({"success": False, "error": "Invalid API key"}), 403
        return f(*args, **kwargs)
    return decorated_function

def load_model():
    """Apertus-8Bモデルをロード"""
    global tokenizer, model
    print("🔐 Apertus-8B モデルロード中...")
    
    # 🔒 重要: HuggingFaceトークンをhttps://huggingface.co/settings/tokensで取得して置き換えてください
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    model_id = "swiss-ai/Apertus-8B-Instruct-2509"
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    transformers.logging.set_verbosity_error()
    
    print("📦 Tokenizerロード中...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, trust_remote_code=True
        )
    except Exception as e:
        print(f"⚠️ AutoTokenizer失敗: {e}")
        from transformers import PreTrainedTokenizerFast
        from huggingface_hub import hf_hub_download
        tokenizer_file = hf_hub_download(
            repo_id=model_id, filename="tokenizer.json", token=hf_token
        )
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
    
    print("📦 Apertus-8Bモデルロード中... (3-5分)")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    
    print("✅ モデルロード完了!")
    print(f"📊 dtype: {model.dtype}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM使用量: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

def build_instruct_prompt(task: str, content: str, language: str = "Japanese") -> str:
    """Apertus-8B Instruct形式のプロンプトを構築"""
    
    native_system_prompts = {
        "Japanese": """あなたはSwiss AIが開発した多言語AIアシスタント「Apertus」です。
1,811言語に対応しており、要約と翻訳が得意です。

【最重要ルール】
- 回答は必ず日本語で行うこと
- 英語やその他の言語は絶対に使用禁止
- 要約結果のみを出力し、余計な説明は不要""",

        "English": """You are Apertus, a multilingual AI assistant developed by Swiss AI.
Supporting 1,811 languages, specializing in summarization and translation.

【Critical Rules】
- Always respond in English
- No other languages allowed
- Output only the summary without extra explanations""",

        "Chinese": """你是由Swiss AI开发的多语言AI助手「Apertus」。
支持1,811种语言，擅长摘要和翻译。

【最重要规则】
- 必须用中文回答
- 禁止使用其他语言
- 只输出摘要结果，无需额外说明""",
    }
    
    system_prompt = native_system_prompts.get(language, native_system_prompts["Japanese"])
    
    user_content = f"{task}\n\n{content}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt
    except Exception as e:
        print(f"⚠️ Chat template適用失敗: {e}")
        return f"{system_prompt}\n\n{user_content}"

@app.route('/health', methods=['GET'])
def health_check():
    """ヘルスチェック"""
    global model, tokenizer
    return jsonify({
        "success": True,
        "status": "running",
        "model_loaded": model is not None and tokenizer is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "model": "swiss-ai/Apertus-8B-Instruct"
    })

@app.route('/summarize', methods=['POST'])
@require_api_key
def summarize_text():
    """要約エンドポイント"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        max_length = data.get('max_length', 400)
        source_lang = data.get('source_lang', 'auto-detect')
        target_lang = data.get('target_lang', 'Japanese')
        style = data.get('style', 'balanced')
        summary_mode = data.get('summary_mode', 'short')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        # キャラクター数制限
        if len(text) > 10000:
            text = text[:10000]
        
        # 要約タスク構築
        if summary_mode == 'long':
            task = f"""以下のテキストを{target_lang}で800-1000文字で詳細に要約してください。

【要約スタイル: {style}】
- balanced: バランスの取れた標準的な要約
- detailed: より詳細で包括的な要約
- concise: 簡潔で要点を絞った要約
- tech_doc: 技術文書向けの専門的な要約

重要: 要約結果のみを出力し、前置きや説明は不要です。"""
        else:
            task = f"""以下のテキストを{target_lang}で200-400文字で簡潔に要約してください。

【要約スタイル: {style}】

重要: 要約結果のみを出力し、前置きや説明は不要です。"""
        
        prompt = build_instruct_prompt(task, text, target_lang)
        
        # トークナイズ
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length * 3,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # デコード
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # プロンプト部分を除去
        if "<|assistant|>" in generated_text:
            summary = generated_text.split("<|assistant|>")[-1].strip()
        else:
            summary = generated_text[len(prompt):].strip()
        
        return jsonify({
            "success": True,
            "summary": summary,
            "model": "Apertus-8B-Instruct",
            "source_lang": source_lang,
            "target_lang": target_lang
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/expand', methods=['POST'])
@require_api_key
def expand_text():
    """文章展開エンドポイント"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_length = data.get('target_length', 500)
        target_lang = data.get('target_lang', 'Japanese')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        task = f"""以下の短文を{target_lang}で{target_length}文字程度に詳細に展開してください。

重要: 展開結果のみを出力し、前置きや説明は不要です。"""
        
        prompt = build_instruct_prompt(task, text, target_lang)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=target_length * 3,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|assistant|>" in generated_text:
            result = generated_text.split("<|assistant|>")[-1].strip()
        else:
            result = generated_text[len(prompt):].strip()
        
        return jsonify({
            "success": True,
            "result": result,
            "model": "Apertus-8B-Instruct"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# モデルロード
load_model()

# ngrokまたはpyngrokを使用してトンネルを作成
print("\n🚀 Flaskサーバーを起動します...")
print("⚠️ ngrokでトンネルを作成してURLを取得してください\n")

# Kaggleではapp.run()を使用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
