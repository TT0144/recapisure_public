(() => {
  const appLogger = window.console || { log: () => {} };
  appLogger.log('recapisure frontend: app.js initialized');

  // グローバルに公開するユーティリティが必要になった場合に備えて、名前空間を確保
  window.recapisure = window.recapisure || {};
  
  // ==================== ダークモード ====================
  const darkModeToggle = document.getElementById('darkModeToggle');
  const darkModeIcon = document.getElementById('darkModeIcon');
  
  // 保存されたテーマを読み込み
  const savedTheme = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateDarkModeIcon(savedTheme);
  
  if (darkModeToggle) {
    darkModeToggle.addEventListener('click', toggleDarkMode);
  }
  
  function toggleDarkMode() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateDarkModeIcon(newTheme);
    
    showToast(newTheme === 'dark' ? '🌙 ダークモードに切り替えました' : '☀️ ライトモードに切り替えました', 'info');
  }
  
  function updateDarkModeIcon(theme) {
    if (darkModeIcon) {
      darkModeIcon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }
  }
  
  // ==================== ショートカットキー ====================
  // 入力欄にフォーカス中かチェック
  function isTyping() {
    const active = document.activeElement;
    if (!active) return false;
    const tag = active.tagName.toUpperCase();
    // テキスト入力欄、テキストエリア、contenteditable要素はスキップ
    if (tag === 'INPUT' || tag === 'TEXTAREA') return true;
    if (active.isContentEditable) return true;
    return false;
  }
  
  document.addEventListener('keydown', (e) => {
    // Ctrl+Enter: 要約実行（入力中でも許可 - 送信系なので）
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      const processBtn = document.getElementById('processBtn');
      const fetchUrlBtn = document.getElementById('fetchUrlBtn');
      const urlMode = document.getElementById('modeUrl');
      
      if (urlMode && urlMode.checked && fetchUrlBtn && fetchUrlBtn.offsetParent !== null) {
        fetchUrlBtn.click();
      } else if (processBtn) {
        processBtn.click();
      }
      showShortcutHint('Ctrl+Enter → 要約実行');
      return;
    }
    
    // 以下のショートカットは入力中は無効
    if (isTyping()) return;
    
    // Ctrl+D: ダークモード切り替え
    if (e.ctrlKey && e.key === 'd') {
      e.preventDefault();
      toggleDarkMode();
      showShortcutHint('Ctrl+D → ダークモード切替');
    }
    
    // Ctrl+Shift+C: 結果をコピー
    if (e.ctrlKey && e.shiftKey && e.key === 'C') {
      e.preventDefault();
      if (typeof copyResult === 'function') {
        copyResult();
      }
      showShortcutHint('Ctrl+Shift+C → 結果コピー');
    }
    
    // Ctrl+N: 新しい処理
    if (e.ctrlKey && e.key === 'n') {
      e.preventDefault();
      if (typeof newProcess === 'function') {
        newProcess();
      }
      showShortcutHint('Ctrl+N → 新しい処理');
    }
    
    // Escape: 入力欄をクリア
    if (e.key === 'Escape') {
      const inputText = document.getElementById('inputText');
      if (inputText && document.activeElement === inputText) {
        inputText.value = '';
        showShortcutHint('Esc → 入力クリア');
      }
    }
    
    // ?: ショートカットヘルプ表示
    if (e.key === '?' && !e.ctrlKey && !e.altKey) {
      const activeElement = document.activeElement;
      if (activeElement.tagName !== 'INPUT' && activeElement.tagName !== 'TEXTAREA') {
        e.preventDefault();
        showShortcutHelp();
      }
    }
  });
  
  // ショートカットヒント表示
  function showShortcutHint(message) {
    let hint = document.querySelector('.shortcut-hint');
    if (!hint) {
      hint = document.createElement('div');
      hint.className = 'shortcut-hint';
      document.body.appendChild(hint);
    }
    hint.textContent = message;
    hint.classList.add('show');
    
    setTimeout(() => {
      hint.classList.remove('show');
    }, 1500);
  }
  
  // ショートカットヘルプモーダル表示
  function showShortcutHelp() {
    const shortcuts = [
      { key: 'Ctrl + Enter', desc: '要約実行' },
      { key: 'Ctrl + D', desc: 'ダークモード切替' },
      { key: 'Ctrl + Shift + C', desc: '結果をコピー' },
      { key: 'Ctrl + N', desc: '新しい処理' },
      { key: 'Esc', desc: '入力欄クリア' },
      { key: '?', desc: 'ショートカット一覧' },
    ];
    
    const html = shortcuts.map(s => 
      `<div class="d-flex justify-content-between mb-2">
        <kbd>${s.key}</kbd>
        <span>${s.desc}</span>
      </div>`
    ).join('');
    
    showToast(`<strong>⌨️ ショートカットキー</strong><hr class="my-2">${html}`, 'info', 5000);
  }
  
  // ==================== コピー機能の強化 ====================
  window.copyResult = function() {
    const resultText = document.getElementById('resultText');
    if (!resultText) return;
    
    const text = resultText.innerText || resultText.textContent;
    if (!text || text.trim() === '') {
      showToast('コピーする結果がありません', 'warning');
      return;
    }
    
    navigator.clipboard.writeText(text).then(() => {
      const copyBtn = document.querySelector('[onclick="copyResult()"]');
      const copyBtnText = document.getElementById('copyBtnText');
      
      if (copyBtn) {
        copyBtn.classList.remove('btn-primary');
        copyBtn.classList.add('btn-success');
        if (copyBtnText) copyBtnText.textContent = 'コピーしました！';
        
        setTimeout(() => {
          copyBtn.classList.remove('btn-success');
          copyBtn.classList.add('btn-primary');
          if (copyBtnText) copyBtnText.textContent = '結果をコピー';
        }, 2000);
      }
      
      showToast('📋 結果をクリップボードにコピーしました', 'success');
    }).catch(err => {
      appLogger.error('Copy failed:', err);
      showToast('コピーに失敗しました', 'danger');
    });
  };
  
  // ==================== キーワード抽出 ====================
  window.extractKeywords = function(text, maxKeywords = 8) {
    if (!text || text.length < 10) return [];
    
    // 日本語のストップワード
    const stopWords = new Set([
      'これ', 'それ', 'あれ', 'この', 'その', 'あの', 'ここ', 'そこ', 'あそこ',
      'こちら', 'どこ', 'だれ', 'なに', 'なん', '何', 'ある', 'いる', 'する',
      'なる', 'できる', 'ない', 'ます', 'です', 'である', 'という', 'こと',
      'もの', 'ため', 'よう', 'など', 'について', 'として', 'における', 'による',
      'において', 'に対して', 'のため', 'によって', 'からの', 'への', 'での',
      '必要', '可能', '重要', '場合', '方法', '結果', '内容', '情報', '問題',
      '一つ', '二つ', '一方', '他方', '以下', '以上', '以外', '以内', '同様',
      'また', 'および', 'または', 'しかし', 'ただし', 'なお', 'すなわち',
      'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
      'should', 'may', 'might', 'must', 'and', 'or', 'but', 'if', 'then',
      'else', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
      'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
      'own', 'same', 'so', 'than', 'too', 'very', 'just', 'can', 'will',
      'with', 'from', 'this', 'that', 'these', 'those', 'which', 'what'
    ]);
    
    // テキストから単語を抽出
    // 日本語: カタカナ語、漢字語を優先
    const japanesePattern = /[ァ-ヴー]{3,}|[一-龯]{2,}/g;
    const englishPattern = /[A-Za-z]{4,}/g;
    
    const japaneseMatches = text.match(japanesePattern) || [];
    const englishMatches = text.match(englishPattern) || [];
    
    // 単語の出現回数をカウント
    const wordCount = new Map();
    
    [...japaneseMatches, ...englishMatches].forEach(word => {
      const normalized = word.toLowerCase();
      if (!stopWords.has(normalized) && !stopWords.has(word)) {
        wordCount.set(word, (wordCount.get(word) || 0) + 1);
      }
    });
    
    // 出現回数でソートして上位を返す
    const sorted = Array.from(wordCount.entries())
      .filter(([word, count]) => count >= 1 && word.length >= 2)
      .sort((a, b) => b[1] - a[1])
      .slice(0, maxKeywords)
      .map(([word]) => word);
    
    return sorted;
  };
  
  // キーワードを表示
  window.displayKeywords = function(keywords) {
    const container = document.getElementById('keywordsContainer');
    const area = document.getElementById('keywordsArea');
    
    if (!container || !area) return;
    
    if (!keywords || keywords.length === 0) {
      area.style.display = 'none';
      return;
    }
    
    container.innerHTML = keywords.map(kw => 
      `<span class="keyword-badge">${escapeHtml(kw)}</span>`
    ).join('');
    
    area.style.display = 'block';
  };
  
  // HTMLエスケープ
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
  
  // ==================== トースト通知 ====================
  window.showToast = function(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast show align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
      <div class="d-flex">
        <div class="toast-body">${message}</div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
      </div>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, duration);
  };
  
  // ==================== 結果表示時にキーワード抽出 ====================
  // MutationObserverで結果表示を監視
  const resultText = document.getElementById('resultText');
  if (resultText) {
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList' || mutation.type === 'characterData') {
          const text = resultText.innerText || resultText.textContent;
          if (text && text.trim().length > 20) {
            const keywords = extractKeywords(text);
            displayKeywords(keywords);
          }
        }
      });
    });
    
    observer.observe(resultText, { 
      childList: true, 
      characterData: true, 
      subtree: true 
    });
  }
  
  appLogger.log('recapisure: Dark mode, shortcuts, keywords enabled');
})();
