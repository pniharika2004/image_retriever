import { getDocument, GlobalWorkerOptions } from 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.min.mjs';

GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.worker.min.mjs';

const uploadForm = document.getElementById('upload-form');
const pdfFileInput = document.getElementById('pdf-file');
const docStatus = document.getElementById('doc-status');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const messagesDiv = document.getElementById('messages');

const state = {
  docId: null,
  pdfUrl: null,
  pdfDoc: null,
  numPages: 0
};

// Deduplicate rendered page images across the whole chat session
const renderedKeysGlobal = new Set();

function addMessage(sender, text) {
  const wrapper = document.createElement('div');
  wrapper.className = `msg ${sender}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrapper.appendChild(bubble);
  messagesDiv.appendChild(wrapper);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function addImageCanvas(canvas) {
  const wrapper = document.createElement('div');
  wrapper.className = 'msg image';
  wrapper.appendChild(canvas);
  messagesDiv.appendChild(wrapper);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function renderPageToCanvas(pageNumber, pdfDoc) {
  const doc = pdfDoc || state.pdfDoc;
  if (!doc) return null;
  const page = await doc.getPage(pageNumber);
  const viewport = page.getViewport({ scale: 1.2 });
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  canvas.width = viewport.width;
  canvas.height = viewport.height;
  await page.render({ canvasContext: context, viewport }).promise;
  canvas.style.maxWidth = '100%';
  return canvas;
}

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = pdfFileInput.files[0];
  if (!file) return;
  docStatus.textContent = 'Uploading and indexing...';

  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch('/api/upload', { method: 'POST', body: formData });
  if (!response.ok) {
    docStatus.textContent = 'Upload failed';
    return;
  }
  const { docId, pdfUrl, numPages } = await response.json();
  state.docId = docId;
  state.pdfUrl = pdfUrl;
  state.numPages = numPages;
  state.pdfDoc = await getDocument(pdfUrl).promise;
  // Reset rendered image cache for new document
  renderedKeysGlobal.clear();
  docStatus.textContent = `Loaded ${numPages} pages`;
  addMessage('system', 'PDF loaded. Ask a question.');
});

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = userInput.value.trim();
  if (!question) return;
  addMessage('user', question);
  userInput.value = '';

  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: question, docId: state.docId })
  });
  if (!res.ok) {
    addMessage('assistant', 'Sorry, something went wrong.');
    return;
  }
  const data = await res.json();
  const answer = data.answer || '';
  addMessage('assistant', answer);

  // Multi-document images support
  if (data.needs_image) {
    if (Array.isArray(data.images) && data.images.length > 0) {
      for (const item of data.images) {
        const pdfUrl = item.pdf_url;
        const pages = Array.isArray(item.pages) ? item.pages : [];
        if (!pdfUrl || pages.length === 0) continue;
        try {
          const pdfDoc = await getDocument(pdfUrl).promise;
          for (const pageNum of pages) {
            const key = `${item.docId || pdfUrl}|${pageNum}`;
            if (renderedKeysGlobal.has(key)) continue;
            renderedKeysGlobal.add(key);
            try {
              const canvas = await renderPageToCanvas(pageNum, pdfDoc);
              if (canvas) addImageCanvas(canvas);
            } catch (err) {
              console.warn('Render error:', err);
            }
          }
        } catch (err) {
          console.warn('PDF load error:', err);
        }
      }
    } else if (Array.isArray(data.related_pages)) {
      // Backward compatible single-document path
      const pages = data.related_pages;
      for (const pageNum of pages) {
        const key = `${state.docId || 'current'}|${pageNum}`;
        if (renderedKeysGlobal.has(key)) continue;
        renderedKeysGlobal.add(key);
        try {
          const canvas = await renderPageToCanvas(pageNum);
          if (canvas) addImageCanvas(canvas);
        } catch (err) {
          console.warn('Render error:', err);
        }
      }
    }
  }
});


