from flask import Flask, render_template, request, jsonify, session as flask_session
import requests
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from textblob import TextBlob
import yfinance as yf
from sec_edgar_downloader import Downloader
import tempfile
import zipfile
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from werkzeug.utils import secure_filename
import faiss
from sentence_transformers import SentenceTransformer
import threading
import uuid
from transformers.pipelines import pipeline
import torch
from sklearn.preprocessing import StandardScaler
import time
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from finance_agent import ask_finance_agent
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
except ImportError:
    nlp = None

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

# Finnhub API Key
FINNHUB_API_KEY = "d1hum6hr01qhsrhc8ftgd1hum6hr01qhsrhc8fu0"

# India FY 2025-26 (AY 2026-27) Tax Slabs
# New Regime FY 2025-26 (per KPMG flash alert)
NEW_REGIME_SLABS = [
    (400000, 0),
    (800000, 5),
    (1200000, 10),
    (1600000, 15),
    (2000000, 20),
    (2400000, 25),
    (float('inf'), 30)
]

# Old Regime Slabs
OLD_REGIME_SLABS_UNDER_60 = [
    (250000, 0),
    (500000, 5),
    (1000000, 20),
    (float('inf'), 30)
]

OLD_REGIME_SLABS_60_TO_80 = [
    (300000, 0),
    (500000, 5),
    (1000000, 20),
    (float('inf'), 30)
]

OLD_REGIME_SLABS_80_PLUS = [
    (500000, 0),
    (1000000, 20),
    (float('inf'), 30)
]

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'html'}

# In-memory session storage for uploaded doc chunks and FAISS index
user_sessions = {}

# Initialize FinBERT sentiment pipeline (load once)
try:
    finbert_pipe = pipeline("text-classification", model="ProsusAI/finbert")
except Exception as e:
    finbert_pipe = None
    print(f"Warning: Could not load FinBERT: {e}")

SENTIMENT_MODEL_PATH = 'results/saved_model'
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Helper: check allowed file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ======= Tax Calculation Functions =======
def compute_slab_tax(income, slabs):
    """Compute tax based on income slabs"""
    remaining = income
    lower = 0
    tax = 0
    steps = []
    
    for upto, rate in slabs:
        band_amt = max(0, min(remaining, upto - lower))
        if band_amt > 0:
            t = band_amt * rate / 100
            tax += t
            steps.append({
                'band': [lower + 1, upto],
                'amount': band_amt,
                'rate': rate,
                'tax': t
            })
            remaining -= band_amt
            lower = upto
        if remaining <= 0:
            break
    
    return {'tax': tax, 'steps': steps}

def compute_surcharge(base_tax, total_income, regime):
    """Compute surcharge based on total income and regime"""
    t = total_income
    rate = 0
    
    if t > 50000000 and regime == "old":
        rate = 37  # old regime only
    if t > 20000000:
        rate = max(rate, 25)
    if t > 10000000:
        rate = max(rate, 15)
    if t > 5000000:
        rate = max(rate, 10)
    if regime == "new" and rate == 37:
        rate = 25  # safety cap
    
    surcharge = base_tax * rate / 100
    return {'rate': rate, 'surcharge': surcharge}

def apply_87a_rebate_new_regime(base_tax, taxable_excl_special):
    """Apply 87A rebate for new regime"""
    # Rebate up to Rs. 60,000 if taxable income (excluding special rate income) <= Rs. 12,00,000
    if taxable_excl_special <= 1200000:
        return max(0, base_tax - 60000)
    return base_tax

def calc_new_regime(inputs):
    """Calculate tax under new regime"""
    std_ded = 75000  # standard deduction (salary/pension) new regime
    family_pension_cap = 25000
    
    normal_income = max(0, inputs['salary_income'] + inputs['other_income'] - std_ded - 
                       min(inputs['family_pension_deduction'], family_pension_cap) - 
                       inputs['employer_nps_80ccd2'])
    
    # Slab tax on normal income
    slab_result = compute_slab_tax(normal_income, NEW_REGIME_SLABS)
    slab_tax = slab_result['tax']
    steps = slab_result['steps']
    
    # Special rate incomes
    stcg_tax = inputs['stcg_111a'] * 0.15  # 15%
    ltcg_eq_exemption = max(0, inputs['ltcg_112a'] - 100000)
    ltcg_eq_tax = ltcg_eq_exemption * 0.10  # 10% over Rs. 1L
    other_ltcg_tax = inputs['other_ltcg_112'] * (inputs['other_ltcg_112_rate'] / 100)
    
    # Apply 87A only to slab portion, excluding special incomes
    slab_tax_after_rebate = apply_87a_rebate_new_regime(slab_tax, normal_income)
    base_tax = slab_tax_after_rebate + stcg_tax + ltcg_eq_tax + other_ltcg_tax
    
    # Surcharge
    gross_total_income = (inputs['salary_income'] + inputs['other_income'] + 
                          inputs['stcg_111a'] + inputs['ltcg_112a'] + inputs['other_ltcg_112'])
    surcharge_result = compute_surcharge(base_tax, gross_total_income, "new")
    surcharge_rate = surcharge_result['rate']
    surcharge = surcharge_result['surcharge']
    
    cess = 0.04 * (base_tax + surcharge)
    total_tax = max(0, base_tax + surcharge + cess)
    
    return {
        'regime': 'new',
        'normal_income': normal_income,
        'steps': steps,
        'slab_tax': slab_tax,
        'slab_tax_after_rebate': slab_tax_after_rebate,
        'stcg_tax': stcg_tax,
        'ltcg_eq_tax': ltcg_eq_tax,
        'other_ltcg_tax': other_ltcg_tax,
        'surcharge_rate': surcharge_rate,
        'surcharge': surcharge,
        'cess': cess,
        'total_tax': total_tax
    }

def calc_old_regime(inputs):
    """Calculate tax under old regime"""
    std_ded_old = 50000 if (inputs['old_std_ded'] and inputs['salary_income'] > 0) else 0
    cap_80c = min(inputs['ded_80c'], 150000)
    cap_80d = min(inputs['ded_80d'], 75000)
    cap_ccd1b = min(inputs['ded_80ccd1b'], 50000)
    hp_loss = -1 * min(inputs['housing_loan_interest'], 200000)  # negative (loss from house property)
    other_old = max(0, inputs['other_deductions_old'])
    gross = (inputs['salary_income'] + inputs['other_income'] + 
             inputs['stcg_111a'] + inputs['ltcg_112a'] + inputs['other_ltcg_112'])
    
    normal_gti = max(0, inputs['salary_income'] + inputs['other_income'] - std_ded_old + hp_loss)
    deductions = cap_80c + cap_80d + cap_ccd1b + other_old
    normal_income = max(0, normal_gti - deductions)
    
    # Old regime slabs depend on age
    if inputs['age'] == "<60":
        slabs = OLD_REGIME_SLABS_UNDER_60
    elif inputs['age'] == "60-80":
        slabs = OLD_REGIME_SLABS_60_TO_80
    else:
        slabs = OLD_REGIME_SLABS_80_PLUS
    
    slab_result = compute_slab_tax(normal_income, slabs)
    slab_tax = slab_result['tax']
    steps = slab_result['steps']
    
    # Special rate incomes (same as new)
    stcg_tax = inputs['stcg_111a'] * 0.15  # 15%
    ltcg_eq_exemption = max(0, inputs['ltcg_112a'] - 100000)
    ltcg_eq_tax = ltcg_eq_exemption * 0.10
    other_ltcg_tax = inputs['other_ltcg_112'] * (inputs['other_ltcg_112_rate'] / 100)
    
    # Section 87A rebate under old regime
    slab_tax_after_rebate = slab_tax
    if (inputs['resident'] and normal_income <= 500000 and 
        inputs['stcg_111a'] == 0 and inputs['ltcg_112a'] == 0 and inputs['other_ltcg_112'] == 0):
        slab_tax_after_rebate = max(0, slab_tax - 12500)
    
    base_tax = slab_tax_after_rebate + stcg_tax + ltcg_eq_tax + other_ltcg_tax
    gross_total_income = gross
    surcharge_result = compute_surcharge(base_tax, gross_total_income, "old")
    surcharge_rate = surcharge_result['rate']
    surcharge = surcharge_result['surcharge']
    cess = 0.04 * (base_tax + surcharge)
    total_tax = max(0, base_tax + surcharge + cess)
    
    return {
        'regime': 'old',
        'normal_income': normal_income,
        'steps': steps,
        'slab_tax': slab_tax,
        'slab_tax_after_rebate': slab_tax_after_rebate,
        'stcg_tax': stcg_tax,
        'ltcg_eq_tax': ltcg_eq_tax,
        'other_ltcg_tax': other_ltcg_tax,
        'surcharge_rate': surcharge_rate,
        'surcharge': surcharge,
        'cess': cess,
        'total_tax': total_tax,
        'details': {
            'std_ded_old': std_ded_old,
            'cap_80c': cap_80c,
            'cap_80d': cap_80d,
            'cap_ccd1b': cap_ccd1b,
            'hp_loss': hp_loss,
            'other_old': other_old,
            'deductions': deductions
        }
    }

def suggest_optimization(inputs):
    """Suggest optimization for old regime"""
    old_res = calc_old_regime(inputs)
    remaining_80c = max(0, 150000 - min(inputs['ded_80c'], 150000))
    remaining_ccd1b = max(0, 50000 - min(inputs['ded_80ccd1b'], 50000))
    budget = inputs['investment_budget']
    to_80c = min(remaining_80c, budget)
    to_ccd1b = min(remaining_ccd1b, budget - to_80c)
    total_add = to_80c + to_ccd1b
    
    # Estimate marginal rate = last slab rate in old regime
    if inputs['age'] == "<60":
        slabs = OLD_REGIME_SLABS_UNDER_60
    elif inputs['age'] == "60-80":
        slabs = OLD_REGIME_SLABS_60_TO_80
    else:
        slabs = OLD_REGIME_SLABS_80_PLUS
    
    last_rate = 30
    for upto, rate in slabs:
        if old_res['normal_income'] <= upto:
            last_rate = rate
            break
    
    est_savings = total_add * last_rate / 100  # ignore cess/surcharge for quick estimate
    
    return {
        'to_80c': to_80c,
        'to_ccd1b': to_ccd1b,
        'total_add': total_add,
        'est_savings': est_savings,
        'last_rate': last_rate,
        'note': "Investing the suggested amounts under the old regime could lower taxable income; compare regimes after applying."
    }

# Helper: chunk text
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

# Helper: parse uploaded file
from PyPDF2 import PdfReader
from docx import Document

def parse_uploaded_file(file_storage):
    """Parse uploaded file and extract text content"""
    filename = secure_filename(file_storage.filename)
    ext = filename.rsplit('.', 1)[1].lower()
    file_bytes = file_storage.read()
    text = ""
    
    try:
        if ext == 'txt' or ext == 'html':
            text = file_bytes.decode('utf-8', errors='ignore')
        elif ext == 'pdf':
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    pages.append(page_text)
            text = "\n".join(pages)
        elif ext == 'docx':
            doc = Document(io.BytesIO(file_bytes))
            paragraphs = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():  # Only add non-empty paragraphs
                    paragraphs.append(paragraph.text)
            text = "\n".join(paragraphs)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        if not text.strip():
            raise ValueError("No text content extracted from file")
            
        return text, filename
    except Exception as e:
        raise ValueError(f"Error parsing file: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

# 1. Document Parser Functionality
@app.route('/document-parser', methods=['GET'])
def document_parser():
    return render_template('document_parser.html')

@app.route('/upload-document', methods=['POST'])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # Parse the uploaded file
        text, filename = parse_uploaded_file(file)
        
        # Create chunks
        chunks = chunk_text(text)
        if not chunks:
            return jsonify({'error': 'No content could be extracted from the file'}), 400
        
        # Create chunk data with metadata (similar to chunk_maker.ipynb)
        chunk_data = []
        for idx, chunk in enumerate(chunks):
            chunk_data.append({
                "ticker": "USER_UPLOAD",
                "filing_type": filename.rsplit('.', 1)[1].lower(),
                "filing_id": filename,
                "chunk_id": idx,
                "chunk_text": chunk
            })
        
        # Initialize sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings
        texts = [entry["chunk_text"] for entry in chunk_data]
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
        matrix = np.array(embeddings).astype('float32')
        
        # Build FAISS index (normalized, cosine similarity)
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        
        # Use Flask session for session ID
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
        
        return jsonify({
            'success': True, 
            'num_chunks': len(chunks),
            'filename': filename,
            'total_text_length': len(text)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rag-query', methods=['POST'])
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
        
        # Encode query
        qv = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(qv)
        
        # Search
        distances, indices = index.search(qv, top_k)
        
        results = []
        top_chunk_texts = []
        for rank, idx in enumerate(indices[0]):
            if idx >= len(chunk_data):
                continue
            chunk_info = chunk_data[idx]
            text = chunk_info["chunk_text"]
            score = float(distances[0][rank])
            results.append({
                'rank': rank+1,
                'score': round(score, 4),
                'preview': text[:500] + ("..." if len(text) > 500 else ""),
                'chunk_id': chunk_info["chunk_id"],
                'source': chunk_info["filing_id"]
            })
            top_chunk_texts.append(text)
        # --- Extractive Summary from Chunks Only ---
        def extract_summary(texts, max_sentences=3):
            sentences = []
            for t in texts:
                sents = re.split(r'(?<=[.!?])\s+', t)
                for s in sents:
                    if s.strip():
                        sentences.append(s.strip())
                    if len(sentences) >= max_sentences:
                        break
                if len(sentences) >= max_sentences:
                    break
            return ' '.join(sentences)
        ai_summary = extract_summary(top_chunk_texts, max_sentences=3)
        # --- Insight Cards (key phrases/entities) ---
        def extract_key_phrases(texts):
            phrases = []
            if nlp:
                for t in texts:
                    doc = nlp(t)
                    phrases.extend([chunk.text for chunk in doc.noun_chunks])
                    phrases.extend([token.text for token in doc if token.pos_ == 'ADJ'])
            else:
                for t in texts:
                    phrases.extend(re.findall(r'\b\w{4,}\b', t))
            return list(set(phrases))[:10]
        insight_phrases = extract_key_phrases(top_chunk_texts)
        return jsonify({
            'success': True, 
            'results': results,
            'query': query,
            'total_results': len(results),
            'ai_summary': ai_summary,
            'insight_phrases': insight_phrases
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear-session', methods=['POST'])
def clear_session():
    """Clear the current session's uploaded document data"""
    try:
        session_id = flask_session.get('session_id')
        if session_id in user_sessions:
            del user_sessions[session_id]
        flask_session.pop('session_id', None)
        return jsonify({'success': True, 'message': 'Session cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-session-info', methods=['GET'])
def get_session_info():
    """Get information about the current session's uploaded document"""
    try:
        session_id = flask_session.get('session_id')
        if session_id in user_sessions:
            sess = user_sessions[session_id]
            return jsonify({
                'success': True,
                'filename': sess.get('filename', 'Unknown'),
                'num_chunks': len(sess.get('chunks', [])),
                'has_document': True
            })
        else:
            return jsonify({
                'success': True,
                'has_document': False
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 2. Indian Tax Calculator Functionality
@app.route('/tax-calculator')
def tax_calculator():
    return render_template('tax_calculator.html')

@app.route('/calculate-tax', methods=['POST'])
def calculate_tax():
    try:
        data = request.get_json()
        
        # Extract inputs with defaults
        inputs = {
            'resident': data.get('resident', True),
            'age': data.get('age', '<60'),
            'salary_income': float(data.get('salary_income', 0)),
            'other_income': float(data.get('other_income', 0)),
            'stcg_111a': float(data.get('stcg_111a', 0)),
            'ltcg_112a': float(data.get('ltcg_112a', 0)),
            'other_ltcg_112': float(data.get('other_ltcg_112', 0)),
            'other_ltcg_112_rate': float(data.get('other_ltcg_112_rate', 20)),
            'old_std_ded': data.get('old_std_ded', True),
            'ded_80c': float(data.get('ded_80c', 0)),
            'ded_80d': float(data.get('ded_80d', 0)),
            'ded_80ccd1b': float(data.get('ded_80ccd1b', 0)),
            'housing_loan_interest': float(data.get('housing_loan_interest', 0)),
            'other_deductions_old': float(data.get('other_deductions_old', 0)),
            'employer_nps_80ccd2': float(data.get('employer_nps_80ccd2', 0)),
            'family_pension_deduction': float(data.get('family_pension_deduction', 0)),
            'investment_budget': float(data.get('investment_budget', 0))
        }
        
        # Calculate taxes for both regimes
        new_regime_result = calc_new_regime(inputs)
        old_regime_result = calc_old_regime(inputs)
        
        # Determine which regime is better
        better_regime = "new" if new_regime_result['total_tax'] <= old_regime_result['total_tax'] else "old"
        tax_difference = abs(new_regime_result['total_tax'] - old_regime_result['total_tax'])
        
        # Get optimization suggestions
        optimization = suggest_optimization(inputs)
        
        # Clean up steps data to handle infinity values
        def clean_steps(steps):
            cleaned_steps = []
            for step in steps:
                cleaned_step = {
                    'band': [
                        step['band'][0] if step['band'][0] != float('inf') else 999999999,
                        step['band'][1] if step['band'][1] != float('inf') else 999999999
                    ],
                    'amount': step['amount'],
                    'rate': step['rate'],
                    'tax': step['tax']
                }
                cleaned_steps.append(cleaned_step)
            return cleaned_steps

        return jsonify({
            'success': True,
            'new_regime': {
                'total_tax': round(new_regime_result['total_tax'], 2),
                'normal_income': round(new_regime_result['normal_income'], 2),
                'slab_tax': round(new_regime_result['slab_tax'], 2),
                'slab_tax_after_rebate': round(new_regime_result['slab_tax_after_rebate'], 2),
                'stcg_tax': round(new_regime_result['stcg_tax'], 2),
                'ltcg_eq_tax': round(new_regime_result['ltcg_eq_tax'], 2),
                'other_ltcg_tax': round(new_regime_result['other_ltcg_tax'], 2),
                'surcharge_rate': new_regime_result['surcharge_rate'],
                'surcharge': round(new_regime_result['surcharge'], 2),
                'cess': round(new_regime_result['cess'], 2),
                'steps': clean_steps(new_regime_result['steps'])
            },
            'old_regime': {
                'total_tax': round(old_regime_result['total_tax'], 2),
                'normal_income': round(old_regime_result['normal_income'], 2),
                'slab_tax': round(old_regime_result['slab_tax'], 2),
                'slab_tax_after_rebate': round(old_regime_result['slab_tax_after_rebate'], 2),
                'stcg_tax': round(old_regime_result['stcg_tax'], 2),
                'ltcg_eq_tax': round(old_regime_result['ltcg_eq_tax'], 2),
                'other_ltcg_tax': round(old_regime_result['other_ltcg_tax'], 2),
                'surcharge_rate': old_regime_result['surcharge_rate'],
                'surcharge': round(old_regime_result['surcharge'], 2),
                'cess': round(old_regime_result['cess'], 2),
                'steps': clean_steps(old_regime_result['steps']),
                'deductions': old_regime_result['details']['deductions']
            },
            'comparison': {
                'better_regime': better_regime,
                'tax_difference': round(tax_difference, 2),
                'savings_with_better': round(tax_difference, 2)
            },
            'optimization': {
                'to_80c': round(optimization['to_80c'], 2),
                'to_ccd1b': round(optimization['to_ccd1b'], 2),
                'total_add': round(optimization['total_add'], 2),
                'est_savings': round(optimization['est_savings'], 2),
                'last_rate': optimization['last_rate'],
                'note': optimization['note']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 3. Live Stock Chart Functionality
@app.route('/stock-chart')
def stock_chart():
    return render_template('stock_chart.html')

@app.route('/get-stock-data', methods=['POST'])
def get_stock_data():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL').upper()

        # Get stock data from yfinance
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1mo')  # Last 30 days
        if hist.empty:
            return jsonify({'error': 'No data available for this symbol'})
        dates = [d.strftime('%Y-%m-%d') for d in hist.index]
        prices = hist['Close'].tolist()
        volumes = hist['Volume'].tolist()

        # Get company info from yfinance
        info = ticker.info
        company_info = {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A')
        }

        # --- Analyze past week trend and generate recommendation ---
        recommendation = generate_stock_recommendation(prices, company_info)

        return jsonify({
            'success': True,
            'symbol': symbol,
            'dates': dates,
            'prices': prices,
            'volumes': volumes,
            'company_info': company_info,
            'recommendation': recommendation
        })
    except Exception as e:
        return jsonify({'error': str(e)})

def get_company_info(symbol):
    """Get basic company information"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A')
        }
    except:
        return {
            'name': symbol,
            'sector': 'N/A',
            'industry': 'N/A',
            'market_cap': 'N/A',
            'pe_ratio': 'N/A'
        }

# --- New: Helper to generate stock recommendation based on past week trend ---
def generate_stock_recommendation(prices, company_info):
    if len(prices) < 7:
        return "Not enough data for weekly trend analysis."
    last_week = prices[-7:]
    change = last_week[-1] - last_week[0]
    pct_change = (change / last_week[0]) * 100 if last_week[0] != 0 else 0
    sector = company_info.get('sector', 'N/A')
    industry = company_info.get('industry', 'N/A')
    
    if pct_change > 2:
        trend = "Uptrend"
        rec = f"Uptrend detected in the past week (+{pct_change:.2f}%). {sector} sector. Consider for short-term momentum if fundamentals are strong."
    elif pct_change < -2:
        trend = "Downtrend"
        rec = f"Downtrend detected in the past week ({pct_change:.2f}%). {sector} sector. Exercise caution or look for reversal signals."
    else:
        trend = "Stable"
        rec = f"Stable price movement this week ({pct_change:.2f}%). {sector} sector. Consider for diversification or long-term holding."
    
    # Add a sector/industry-based flavor
    if sector != 'N/A' and industry != 'N/A':
        rec += f" ({industry})"
    return rec

# 4. Sentiment Analysis Functionality
@app.route('/sentiment-analysis')
def sentiment_analysis():
    return render_template('sentiment_analysis.html')

@app.route('/get-financial-news', methods=['GET'])
def get_financial_news():
    """Fetch company news from Finnhub API (AAPL as default)"""
    try:
        from datetime import datetime, timedelta
        symbol = 'AAPL'  # Default symbol for demo; can be parameterized
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=30)
        url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}'
        response = requests.get(url)
        data = response.json()
        print("Finnhub company-news response:", data)  # DEBUG
        if not isinstance(data, list):
            return jsonify({'error': 'No news data available'})
        news_items = []
        for item in data[:15]:  # Limit to 15 items
            title = item.get('headline', '')
            summary = item.get('summary', '')
            text_to_analyze = f"{title}. {summary}"
            sentiment_result = analyze_text_sentiment(text_to_analyze)
            # Use categorize_news to determine the category
            category = categorize_news(title, summary)
            news_items.append({
                'title': title,
                'summary': summary[:200] + "..." if len(summary) > 200 else summary,
                'url': item.get('url', ''),
                'time_published': item.get('datetime', ''),
                'source': item.get('source', ''),
                'sentiment': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'category': category,
                'tickers': [symbol]
            })
        return jsonify({
            'success': True,
            'news': news_items,
            'total_count': len(news_items)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/analyze-news-sentiment', methods=['POST'])
def analyze_news_sentiment():
    """Analyze sentiment of specific news text"""
    try:
        data = request.get_json()
        news_text = data.get('text', '')
        
        if not news_text:
            return jsonify({'error': 'Please provide news text'})
        
        result = analyze_text_sentiment(news_text)
        return jsonify({
            'success': True,
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'key_phrases': result['key_phrases'],
            'insights': result['insights']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get-news-sentiment-summary', methods=['GET'])
def get_news_sentiment_summary():
    """Get overall sentiment statistics for recent company news from Finnhub (AAPL)"""
    try:
        from datetime import datetime, timedelta
        symbol = 'AAPL'
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=30)
        url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}'
        response = requests.get(url)
        data = response.json()
        if not isinstance(data, list):
            return jsonify({'error': 'No news data available'})
        sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
        categories = {}
        for item in data[:30]:
            title = item.get('headline', '')
            summary = item.get('summary', '')
            text = f"{title}. {summary}"
            sentiment_result = analyze_text_sentiment(text)
            sentiment = sentiment_result['sentiment'].lower()
            # Use categorize_news to determine the category
            category = categorize_news(title, summary)
            if category not in categories:
                categories[category] = {'positive': 0, 'neutral': 0, 'negative': 0}
            categories[category][sentiment] += 1
            sentiments[sentiment] += 1
        total = sum(sentiments.values())
        if total > 0:
            sentiment_percentages = {
                'positive': round((sentiments['positive'] / total) * 100, 1),
                'neutral': round((sentiments['neutral'] / total) * 100, 1),
                'negative': round((sentiments['negative'] / total) * 100, 1)
            }
        else:
            sentiment_percentages = {'positive': 0, 'neutral': 0, 'negative': 0}
        if sentiment_percentages['positive'] > 60:
            overall_sentiment = "Bullish"
            sentiment_description = "Market sentiment is positive with strong optimism"
        elif sentiment_percentages['negative'] > 60:
            overall_sentiment = "Bearish"
            sentiment_description = "Market sentiment is negative with concerns"
        else:
            overall_sentiment = "Neutral"
            sentiment_description = "Market sentiment is mixed with balanced views"
        return jsonify({
            'success': True,
            'overall_sentiment': overall_sentiment,
            'sentiment_description': sentiment_description,
            'sentiment_percentages': sentiment_percentages,
            'category_breakdown': categories,
            'total_news_analyzed': total
        })
    except Exception as e:
        return jsonify({'error': str(e)})

def analyze_text_sentiment(text):
    """Analyze sentiment of text using RoBERTa model"""
    try:
        inputs = sentiment_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()[0]
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))
            sentiment = label_map[pred_id]
            confidence = float(probs[pred_id])
        
        # Extract key phrases
        key_phrases = extract_key_phrases_spacy(text)
        
        # Generate insights
        insights = generate_sentiment_insights_roberta(sentiment, confidence, text)
        
        return {
            'sentiment': sentiment,
            'confidence': round(confidence * 100, 2),
            'key_phrases': key_phrases,
            'insights': insights
        }
    except Exception as e:
        return {
            'sentiment': 'Neutral',
            'confidence': 50.0,
            'key_phrases': [],
            'insights': ['Analysis error occurred']
        }

def categorize_news(title, summary):
    """Categorize news based on content"""
    text = f"{title} {summary}".lower()
    
    if any(word in text for word in ['earnings', 'quarterly', 'revenue', 'profit', 'loss']):
        return 'Earnings'
    elif any(word in text for word in ['fed', 'federal reserve', 'interest rate', 'inflation']):
        return 'Monetary Policy'
    elif any(word in text for word in ['merger', 'acquisition', 'buyout', 'deal']):
        return 'M&A'
    elif any(word in text for word in ['ipo', 'initial public offering', 'listing']):
        return 'IPO'
    elif any(word in text for word in ['crypto', 'bitcoin', 'ethereum', 'blockchain']):
        return 'Cryptocurrency'
    elif any(word in text for word in ['oil', 'energy', 'gas', 'renewable']):
        return 'Energy'
    elif any(word in text for word in ['tech', 'technology', 'software', 'ai', 'artificial intelligence']):
        return 'Technology'
    elif any(word in text for word in ['healthcare', 'medical', 'pharma', 'biotech']):
        return 'Healthcare'
    else:
        return 'Market News'

# --- Helper: Key phrase extraction using spaCy or fallback ---
def extract_key_phrases_spacy(text):
    phrases = []
    if nlp:
        doc = nlp(text)
        # Noun phrases
        phrases.extend([chunk.text for chunk in doc.noun_chunks])
        # Adjectives
        phrases.extend([token.text for token in doc if token.pos_ == 'ADJ'])
    else:
        # Fallback: simple regex for noun-like words and adjectives
        import re
        phrases.extend(re.findall(r'\b\w{4,}\b', text))
    return list(set(phrases))[:10]

# --- Helper: Generate insights for RoBERTa ---
def generate_sentiment_insights_roberta(sentiment, confidence, text):
    insights = []
    if sentiment == 'Positive' and confidence > 0.7:
        insights.append('Strong positive sentiment detected.')
    elif sentiment == 'Negative' and confidence > 0.7:
        insights.append('Strong negative sentiment detected.')
    elif sentiment == 'Neutral' or confidence < 0.5:
        insights.append('Sentiment is neutral or uncertain.')
    # Word count analysis
    word_count = len(text.split())
    if word_count > 100:
        insights.append(f'Detailed analysis ({word_count} words).')
    elif word_count < 20:
        insights.append('Brief statement.')
    return insights

@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'Please provide text to analyze'})
        
        result = analyze_text_sentiment(text)
        
        # Map sentiment to score for UI compatibility
        score_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
        
        return jsonify({
            'success': True,
            'sentiment': result['sentiment'],
            'sentiment_score': score_map.get(result['sentiment'], 0),
            'confidence': result['confidence'],
            'key_phrases': result['key_phrases'],
            'insights': result['insights']
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# 5. Portfolio Optimizer (Curated from fin-mar.ipynb and fin-risk.ipynb)
@app.route('/portfolio-optimizer')
def portfolio_optimizer():
    return render_template('portfolio_optimizer.html')

@app.route('/optimize-portfolio', methods=['POST'])
def optimize_portfolio():
    try:
        data = request.get_json()
        stocks = data.get('stocks', [])
        risk_tolerance = data.get('risk_tolerance', 'moderate')
        if not stocks:
            return jsonify({'error': 'Please provide stock symbols'})
        
        # --- Step 1: Gather Data ---
        portfolio_data = {}
        sentiment_scores = {}
        risk_scores = {}
        diagnostics = {}
        scaler = StandardScaler()
        
        for stock in stocks[:10]:
            try:
                # Get historical returns
                ticker = yf.Ticker(stock)
                hist = ticker.history(period='1y')
                if hist.empty:
                    continue
                returns = hist['Close'].pct_change().dropna()
                portfolio_data[stock] = returns
                
                # --- Step 2: Sentiment Analysis (FinBERT) ---
                headlines = get_news_headlines_yahoo(stock)
                sentiment, confidence, keywords = analyze_sentiment_finbert(headlines)
                sentiment_scores[stock] = {'sentiment': sentiment, 'confidence': confidence, 'keywords': keywords}
                
                # --- Step 3: Risk Proxy (Volatility) ---
                # If XGBoost model is available, use it; else use volatility as risk score
                risk_score = float(returns.std())
                risk_scores[stock] = risk_score
                
                diagnostics[stock] = {
                    'volatility': risk_score,
                    'sentiment': sentiment,
                    'sentiment_confidence': confidence,
                    'keywords': keywords,
                    'n_headlines': len(headlines)
                }
            except Exception as e:
                diagnostics[stock] = {'error': str(e)}
                continue
        
        if len(portfolio_data) < 2:
            return jsonify({'error': 'Need at least 2 valid stocks for optimization'})
        
        # --- Step 4: Risk-Adjusted Returns ---
        returns_df = pd.DataFrame(portfolio_data)
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        # Ensure mean_returns is always a Series (even for one stock)
        if not isinstance(mean_returns, pd.Series):
            mean_returns = pd.Series([mean_returns], index=list(portfolio_data.keys()))
        adj_returns = mean_returns.copy() if isinstance(mean_returns, pd.Series) else pd.Series([mean_returns], index=list(portfolio_data.keys()))
        for stock in adj_returns.index:
            # Sentiment boost: +10% for positive, -10% for negative
            sentiment_adj = 1.0
            if sentiment_scores[stock]['sentiment'] == 'positive':
                sentiment_adj += 0.10
            elif sentiment_scores[stock]['sentiment'] == 'negative':
                sentiment_adj -= 0.10
            # Risk penalty: penalize high volatility
            risk_penalty = 1.0 - min(risk_scores[stock] * 2, 0.5)  # Cap penalty at 50%
            adj_returns[stock] = mean_returns[stock] * sentiment_adj * risk_penalty
        
        # --- Step 5: Portfolio Optimization ---
        if risk_tolerance == 'conservative':
            weights = optimize_conservative(adj_returns, cov_matrix)
        elif risk_tolerance == 'aggressive':
            weights = optimize_aggressive(adj_returns, cov_matrix)
        else:
            weights = optimize_moderate(adj_returns, cov_matrix)
        
        # --- Step 6: Portfolio Metrics ---
        portfolio_return = (mean_returns * weights).sum()
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        return jsonify({
            'success': True,
            'weights': {stock: round(float(weights[stock]), 3) for stock in weights.index},
            'expected_return': round(float(portfolio_return * 252), 3),  # Annualized
            'expected_risk': round(float(portfolio_risk * np.sqrt(252)), 3),  # Annualized
            'sharpe_ratio': round(float(sharpe_ratio * np.sqrt(252)), 3),
            'diagnostics': diagnostics
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# --- Helper: Yahoo Finance News Headlines ---
def get_news_headlines_yahoo(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/latest-news/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = []
        for item in soup.select('h3, h2, a'):  # Try common headline tags
            text = item.get_text(strip=True)
            if text and len(text) > 10:
                headlines.append(text)
        return headlines[:10] if headlines else [f"{ticker} stock news"]
    except Exception:
        return [f"{ticker} stock news"]

# --- Helper: FinBERT Sentiment ---
def analyze_sentiment_finbert(headlines):
    if not headlines or not finbert_pipe:
        return "neutral", 0.5, []
    try:
        sentiments = []
        keywords = []
        for headline in headlines:
            if len(headline.strip()) < 5:
                continue
            result = finbert_pipe(headline)[0]
            sentiment_label = result['label'].lower()
            confidence = result['score']
            sentiments.append({'sentiment': sentiment_label, 'confidence': confidence})
            words = re.findall(r'\b[A-Za-z]{4,}\b', headline.lower())
            keywords.extend([w for w in words if w not in ['said', 'says', 'will', 'would', 'could', 'should']])
        if not sentiments:
            return "neutral", 0.5, []
        positive_score = sum(s['confidence'] for s in sentiments if s['sentiment'] == 'positive')
        negative_score = sum(s['confidence'] for s in sentiments if s['sentiment'] == 'negative')
        neutral_score = sum(s['confidence'] for s in sentiments if s['sentiment'] == 'neutral')
        total_score = positive_score + negative_score + neutral_score
        if total_score == 0:
            return "neutral", 0.5, list(set(keywords[:5]))
        if positive_score > negative_score and positive_score > neutral_score:
            overall_sentiment = "positive"
            confidence = positive_score / total_score
        elif negative_score > positive_score and negative_score > neutral_score:
            overall_sentiment = "negative"
            confidence = negative_score / total_score
        else:
            overall_sentiment = "neutral"
            confidence = neutral_score / total_score if neutral_score > 0 else 0.5
        return overall_sentiment, confidence, list(set(keywords[:5]))
    except Exception:
        return "neutral", 0.5, []

# --- Portfolio Optimization Functions (unchanged) ---
def optimize_conservative(returns, cov):
    weights = np.ones(len(returns)) / len(returns)
    volatilities = np.sqrt(np.diag(cov))
    low_vol_indices = np.argsort(volatilities)[:len(returns)//2]
    weights[low_vol_indices] *= 1.2
    return pd.Series(weights / weights.sum(), index=returns.index)

def optimize_moderate(returns, cov):
    weights = np.ones(len(returns)) / len(returns)
    return pd.Series(weights, index=returns.index)

def optimize_aggressive(returns, cov):
    weights = returns / returns.sum()
    return pd.Series(weights, index=returns.index)

@app.route('/ask-agent', methods=['POST'])
def ask_agent():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    try:
        answer = ask_finance_agent(query)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/finance-agent')
def finance_agent():
    return render_template('finance_agent.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 