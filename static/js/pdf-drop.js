/**
 * recapisure - ファイルドラッグ&ドロップ機能
 * PDF・画像ファイルをブラウザ内にドロップすると、自動的にテキストを抽出して入力エリアに挿入
 * ⭐ 画像対応版: PNG, JPG, JPEG, GIF, BMP, WEBP
 */

// ⭐ 対応ファイル形式の定義
const SUPPORTED_FILE_TYPES = {
    pdf: ['application/pdf'],
    image: ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'],
    text: ['text/plain']
};

// ⭐ ファイルドラッグ&ドロップ機能の初期化
function initPdfDropZone() {
    const dropZone = document.getElementById('pdf-drop-zone');
    const inputText = document.getElementById('inputText');
    
    if (!dropZone || !inputText) {
        console.warn('PDF drop zone or input text area not found');
        return;
    }
    
    // ドラッグオーバー（ファイルが上に来た時）
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
    });
    
    // ドラッグリーブ（ファイルが離れた時）
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
    });
    
    // ドロップ（ファイルが落とされた時）
    dropZone.addEventListener('drop', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length === 0) {
            return;
        }
        
        const file = files[0];
        await processDroppedFile(file);
    });
    
    // クリックでファイル選択
    dropZone.addEventListener('click', () => {
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = '.pdf,.txt,.png,.jpg,.jpeg,.gif,.bmp,.webp';
        fileInput.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            await processDroppedFile(file);
        };
        fileInput.click();
    });
}

// ⭐ ドロップされたファイルを処理
async function processDroppedFile(file) {
    const fileType = file.type;
    const fileName = file.name.toLowerCase();
    
    // PDFファイル
    if (SUPPORTED_FILE_TYPES.pdf.includes(fileType) || fileName.endsWith('.pdf')) {
        await handlePdfFile(file);
    } 
    // 画像ファイル（PNG, JPG等）
    else if (SUPPORTED_FILE_TYPES.image.includes(fileType) || 
             fileName.match(/\.(png|jpg|jpeg|gif|bmp|webp)$/)) {
        await handleImageFile(file);
    } 
    // テキストファイル
    else if (SUPPORTED_FILE_TYPES.text.includes(fileType) || fileName.endsWith('.txt')) {
        await handleTextFile(file);
    } 
    // その他のファイル
    else {
        alert('対応していないファイル形式です。\\n\\n対応形式:\\n📄 PDF, TXT\\n📷 PNG, JPG, JPEG, GIF, BMP, WEBP');
    }
}

// ⭐ PDFファイルの処理（サーバー側で抽出）
async function handlePdfFile(file) {
    try {
        showToast('📄 PDFをサーバーにアップロード中...', 'info');
        
        // FormDataを作成してPDFファイルをサーバーに送信
        const formData = new FormData();
        formData.append('file', file);
        
        // 進捗表示
        const progressToast = showProgressToast('PDFをアップロード中...');
        
        // サーバーにPDFを送信してテキスト抽出
        const response = await fetch('/api/upload-pdf', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`サーバーエラー: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        // テキストエリアに挿入
        $('#inputText').val(result.text.trim());
        $('#charCount').text(result.text.trim().length);
        
        // 進捗完了
        closeProgressToast(progressToast);
        showToast(`✅ PDFを読み込みました (${result.text.trim().length}文字)`, 'success');
        
    } catch (error) {
        console.error('PDF読み込みエラー:', error);
        showToast('❌ PDFの読み込みに失敗しました: ' + error.message, 'error');
    }
}

// ⭐ 画像ファイルの処理（OCRでテキスト抽出）
async function handleImageFile(file) {
    try {
        showToast('📷 画像をアップロード中...', 'info');
        
        // FormDataを作成して画像ファイルをサーバーに送信
        const formData = new FormData();
        formData.append('file', file);
        
        // 進捗表示
        const progressToast = showProgressToast('画像をOCR処理中... (文字認識には時間がかかる場合があります)');
        
        // サーバーに画像を送信してOCRテキスト抽出
        const response = await fetch('/api/upload-image', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`サーバーエラー: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || '画像の処理に失敗しました');
        }
        
        // テキストエリアに挿入
        $('#inputText').val(result.text.trim());
        $('#charCount').text(result.text.trim().length);
        
        // 進捗完了
        closeProgressToast(progressToast);
        showToast(`✅ 画像からテキストを抽出しました (${result.text.trim().length}文字)`, 'success');
        
    } catch (error) {
        console.error('画像読み込みエラー:', error);
        showToast('❌ 画像の処理に失敗しました: ' + error.message, 'error');
    }
}

// ⭐ テキストファイルの処理
async function handleTextFile(file) {
    try {
        showToast('📝 テキストファイルを読み込み中...', 'info');
        
        const text = await file.text();
        $('#inputText').val(text.trim());
        $('#charCount').text(text.trim().length);
        
        showToast(`✅ テキストファイルを読み込みました (${text.trim().length}文字)`, 'success');
        
    } catch (error) {
        console.error('テキストファイル読み込みエラー:', error);
        showToast('❌ ファイルの読み込みに失敗しました: ' + error.message, 'error');
    }
}

// ⭐ 進捗トーストを表示
function showProgressToast(message) {
    const toastId = 'progress-toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white bg-info border-0 show" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-spinner fa-spin me-2"></i>${message}
                </div>
            </div>
        </div>
    `;
    $('#toastContainer').append(toastHtml);
    return toastId;
}

// ⭐ 進捗トーストを更新
function updateProgressToast(toastId, message) {
    $(`#${toastId} .toast-body`).html(`<i class="fas fa-spinner fa-spin me-2"></i>${message}`);
}

// ⭐ 進捗トーストを閉じる
function closeProgressToast(toastId) {
    $(`#${toastId}`).fadeOut(300, function() {
        $(this).remove();
    });
}

// ⭐ 設定アコーディオンの状態を保存
function saveAccordionState() {
    const basicSettingsOpen = $('#collapseBasicSettings').hasClass('show');
    const advancedSettingsOpen = $('#collapseAdvancedSettings').hasClass('show');
    
    localStorage.setItem('basicSettingsOpen', basicSettingsOpen);
    localStorage.setItem('advancedSettingsOpen', advancedSettingsOpen);
}

// ⭐ 設定アコーディオンの状態を復元
function restoreAccordionState() {
    const basicSettingsOpen = localStorage.getItem('basicSettingsOpen') !== 'false'; // デフォルトtrue
    const advancedSettingsOpen = localStorage.getItem('advancedSettingsOpen') === 'true'; // デフォルトfalse
    
    if (basicSettingsOpen) {
        $('#collapseBasicSettings').addClass('show');
    }
    
    if (advancedSettingsOpen) {
        $('#collapseAdvancedSettings').addClass('show');
    }
}

// ⭐ ページ読み込み時に初期化
$(document).ready(function() {
    // PDFドラッグ&ドロップ機能を初期化
    initPdfDropZone();
    
    // アコーディオン状態を復元
    restoreAccordionState();
    
    // アコーディオンの開閉時に状態を保存
    $('.settings-accordion .accordion-collapse').on('shown.bs.collapse hidden.bs.collapse', saveAccordionState);
});
