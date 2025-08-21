import io
import re
import uuid
import numpy as np
import faiss
from flask import Blueprint, jsonify, render_template, request, session as flask_session
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'html'}

document_bp = Blueprint('document', __name__)

# Simple in-memory session store (kept identical to previous behavior)
user_sessions = {}


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def chunk_text(text, max_tokens=500):
	sentences = re.split(r'(?<=[.?!])\s+', text)
	chunks = []
	current_chunk = []
	current_length = 0
	for sentence in sentences:
		length = len(sentence.split())
		if current_length + length > max_tokens:
			chunks.append(" ".join(current_chunk))
			current_chunk = [sentence]
			current_length = length
		else:
			current_chunk.append(sentence)
			current_length += length
	if current_chunk:
		chunks.append(" ".join(current_chunk))
	return chunks


def parse_uploaded_file(file_storage):
	filename = secure_filename(file_storage.filename)
	ext = filename.rsplit('.', 1)[1].lower()
	file_bytes = file_storage.read()
	text = ""
	if ext in ['txt', 'html']:
		text = file_bytes.decode('utf-8', errors='ignore')
	elif ext == 'pdf':
		reader = PdfReader(io.BytesIO(file_bytes))
		pages = []
		for page in reader.pages:
			page_text = page.extract_text()
			if page_text.strip():
				pages.append(page_text)
		text = "\n".join(pages)
	elif ext == 'docx':
		doc = Document(io.BytesIO(file_bytes))
		paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
		text = "\n".join(paragraphs)
	else:
		raise ValueError(f"Unsupported file type: {ext}")
	if not text.strip():
		raise ValueError("No text content extracted from file")
	return text, filename


@document_bp.route('/document-parser', methods=['GET'])
def document_parser():
	return render_template('document_parser.html')


@document_bp.route('/upload-document', methods=['POST'])
def upload_document():
	try:
		if 'file' not in request.files:
			return jsonify({'error': 'No file part'}), 400
		file = request.files['file']
		if file.filename == '':
			return jsonify({'error': 'No selected file'}), 400
		if not allowed_file(file.filename):
			return jsonify({'error': 'Unsupported file type'}), 400

		text, filename = parse_uploaded_file(file)
		chunks = chunk_text(text)
		if not chunks:
			return jsonify({'error': 'No content could be extracted from the file'}), 400

		chunk_data = [{
			"ticker": "USER_UPLOAD",
			"filing_type": filename.rsplit('.', 1)[1].lower(),
			"filing_id": filename,
			"chunk_id": idx,
			"chunk_text": chunk
		} for idx, chunk in enumerate(chunks)]

		model = SentenceTransformer('all-MiniLM-L6-v2')
		texts = [entry["chunk_text"] for entry in chunk_data]
		embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
		matrix = np.array(embeddings).astype('float32')
		faiss.normalize_L2(matrix)
		index = faiss.IndexFlatIP(matrix.shape[1])
		index.add(matrix)

		if 'session_id' not in flask_session:
			flask_session['session_id'] = str(uuid.uuid4())
		session_id = flask_session['session_id']
		user_sessions[session_id] = {
			'chunks': chunk_data,
			'texts': texts,
			'index': index,
			'embeddings': matrix,
			'model': model,
			'filename': filename
		}

		return jsonify({'success': True, 'num_chunks': len(chunks), 'filename': filename, 'total_text_length': len(text)})
	except Exception as e:
		return jsonify({'error': str(e)}), 500


@document_bp.route('/rag-query', methods=['POST'])
def rag_query():
	try:
		data = request.get_json()
		query = data.get('query', '')
		top_k = int(data.get('top_k', 5))
		if not query.strip():
			return jsonify({'error': 'Query cannot be empty'}), 400
		session_id = flask_session.get('session_id')
		if not session_id or session_id not in user_sessions:
			return jsonify({'error': 'No document uploaded for this session'}), 400
		sess = user_sessions[session_id]
		model = sess['model']
		index = sess['index']
		chunk_data = sess['chunks']
		texts = sess['texts']
		qv = model.encode([query], convert_to_numpy=True)
		faiss.normalize_L2(qv)
		distances, indices = index.search(qv, top_k)
		results = []
		top_chunk_texts = []
		for rank, idx in enumerate(indices[0]):
			if idx >= len(chunk_data):
				continue
			chunk_info = chunk_data[idx]
			text = chunk_info["chunk_text"]
			score = float(distances[0][rank])
			results.append({'rank': rank+1, 'score': round(score, 4), 'preview': text[:500] + ("..." if len(text) > 500 else ""), 'chunk_id': chunk_info["chunk_id"], 'source': chunk_info["filing_id"]})
			top_chunk_texts.append(text)
		# simple extractive summary
		sentences = []
		for t in top_chunk_texts:
			for s in re.split(r'(?<=[.!?])\s+', t):
				if s.strip():
					sentences.append(s.strip())
					if len(sentences) >= 3:
						break
			if len(sentences) >= 3:
				break
		ai_summary = ' '.join(sentences)
		# key phrases (regex fallback)
		phrases = list(set(re.findall(r'\b\w{4,}\b', ' '.join(top_chunk_texts))))[:10]
		return jsonify({'success': True, 'results': results, 'query': query, 'total_results': len(results), 'ai_summary': ai_summary, 'insight_phrases': phrases})
	except Exception as e:
		return jsonify({'error': str(e)}), 500


@document_bp.route('/clear-session', methods=['POST'])
def clear_session():
	try:
		session_id = flask_session.get('session_id')
		if session_id in user_sessions:
			del user_sessions[session_id]
		flask_session.pop('session_id', None)
		return jsonify({'success': True, 'message': 'Session cleared successfully'})
	except Exception as e:
		return jsonify({'error': str(e)}), 500


@document_bp.route('/get-session-info', methods=['GET'])
def get_session_info():
	try:
		session_id = flask_session.get('session_id')
		if session_id in user_sessions:
			sess = user_sessions[session_id]
			return jsonify({'success': True, 'filename': sess.get('filename', 'Unknown'), 'num_chunks': len(sess.get('chunks', [])), 'has_document': True})
		else:
			return jsonify({'success': True, 'has_document': False})
	except Exception as e:
		return jsonify({'error': str(e)}), 500



