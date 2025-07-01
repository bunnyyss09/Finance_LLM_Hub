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
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
except ImportError:
    nlp = None

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "2VDHJUTDISNC6NHT"

# Tax slabs for India (Old Regime)
TAX_SLABS = [
    (250000, 0.0),
    (500000, 0.05),
    (1000000, 0.2),
    (float('inf'), 0.3)
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

# Load fine-tuned RoBERTa model and tokenizer once
SENTIMENT_MODEL_PATH = 'results/saved_model'
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Helper: check allowed file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

# 2. Tax Planner Functionality
@app.route('/tax-planner')
def tax_planner():
    return render_template('tax_planner.html')

@app.route('/calculate-tax', methods=['POST'])
def calculate_tax():
    try:
        data = request.get_json()
        income = float(data.get('income', 0))
        deductions = float(data.get('deductions', 0))
        
        # Calculate tax
        taxable_income = max(0, income - deductions)
        tax = 0.0
        prev_limit = 0
        
        for limit, rate in TAX_SLABS:
            if taxable_income > prev_limit:
                slab_amount = min(taxable_income, limit) - prev_limit
                tax += slab_amount * rate
                prev_limit = limit
            else:
                break
        
        # Calculate effective tax rate
        effective_rate = (tax / income * 100) if income > 0 else 0
        
        # Generate recommendations
        recommendations = generate_tax_recommendations(income, deductions, tax)
        
        return jsonify({
            'success': True,
            'taxable_income': round(taxable_income, 2),
            'tax_amount': round(tax, 2),
            'effective_rate': round(effective_rate, 2),
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

def generate_tax_recommendations(income, deductions, tax):
    """Generate personalized tax recommendations"""
    recommendations = []
    
    if deductions < 150000:  # Standard deduction limit
        recommendations.append("Consider maximizing 80C deductions (ELSS, PPF, LIC) up to ₹1.5L")
    
    if income > 1000000:
        recommendations.append("Consider NPS contribution for additional ₹50K deduction under 80CCD(1B)")
    
    if deductions < income * 0.3:
        recommendations.append("Explore home loan interest deduction under Section 24(b)")
    
    if not recommendations:
        recommendations.append("Your tax planning looks good! Consider consulting a tax advisor for optimization.")
    
    return recommendations

# 3. Live Stock Chart Functionality
@app.route('/stock-chart')
def stock_chart():
    return render_template('stock_chart.html')

@app.route('/get-stock-data', methods=['POST'])
def get_stock_data():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL').upper()
        
        # Get stock data from Alpha Vantage
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()
        
        if 'Error Message' in data:
            return jsonify({'error': 'Invalid symbol or API error'})
        
        time_series = data.get('Time Series (Daily)', {})
        if not time_series:
            return jsonify({'error': 'No data available for this symbol'})
        
        # Process data for charting
        dates = []
        prices = []
        volumes = []
        
        for date, values in list(time_series.items())[:30]:  # Last 30 days
            dates.append(date)
            prices.append(float(values['4. close']))
            volumes.append(int(values['5. volume']))
        
        # Reverse to get chronological order
        dates.reverse()
        prices.reverse()
        volumes.reverse()
        
        # Get company info
        company_info = get_company_info(symbol)
        
        # --- New: Analyze past week trend and generate recommendation ---
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
    """Fetch financial news from Alpha Vantage API"""
    try:
        # Get news from Alpha Vantage
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={ALPHA_VANTAGE_API_KEY}&limit=20'
        response = requests.get(url)
        data = response.json()
        
        if 'feed' not in data:
            return jsonify({'error': 'No news data available'})
        
        news_items = []
        for item in data['feed'][:15]:  # Limit to 15 items
            # Analyze sentiment using our RoBERTa model
            title = item.get('title', '')
            summary = item.get('summary', '')
            text_to_analyze = f"{title}. {summary}"
            
            # Get sentiment analysis
            sentiment_result = analyze_text_sentiment(text_to_analyze)
            
            # Categorize news
            category = categorize_news(title, summary)
            
            news_items.append({
                'title': title,
                'summary': summary[:200] + "..." if len(summary) > 200 else summary,
                'url': item.get('url', ''),
                'time_published': item.get('time_published', ''),
                'source': item.get('source', ''),
                'sentiment': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'category': category,
                'tickers': item.get('ticker_sentiment', [])[:3]  # Top 3 tickers
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
    """Get overall sentiment statistics for recent news"""
    try:
        # Fetch recent news and analyze
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={ALPHA_VANTAGE_API_KEY}&limit=50'
        response = requests.get(url)
        data = response.json()
        
        if 'feed' not in data:
            return jsonify({'error': 'No news data available'})
        
        sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
        categories = {}
        
        for item in data['feed'][:30]:
            title = item.get('title', '')
            summary = item.get('summary', '')
            text = f"{title}. {summary}"
            
            sentiment_result = analyze_text_sentiment(text)
            sentiment = sentiment_result['sentiment'].lower()
            sentiments[sentiment] += 1
            
            category = categorize_news(title, summary)
            if category not in categories:
                categories[category] = {'positive': 0, 'neutral': 0, 'negative': 0}
            categories[category][sentiment] += 1
        
        total = sum(sentiments.values())
        if total > 0:
            sentiment_percentages = {
                'positive': round((sentiments['positive'] / total) * 100, 1),
                'neutral': round((sentiments['neutral'] / total) * 100, 1),
                'negative': round((sentiments['negative'] / total) * 100, 1)
            }
        else:
            sentiment_percentages = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        # Determine overall market sentiment
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

# 6. Financial News Analyzer (Additional Functionality)
@app.route('/news-analyzer')
def news_analyzer():
    return render_template('news_analyzer.html')

@app.route('/analyze-news', methods=['POST'])
def analyze_news():
    try:
        data = request.get_json()
        news_text = data.get('news_text', '')
        
        if not news_text:
            return jsonify({'error': 'Please provide news text'})
        
        # Analyze sentiment
        blob = TextBlob(news_text)
        sentiment_score = blob.sentiment.polarity
        
        # Extract financial entities
        financial_entities = extract_financial_entities(news_text)
        
        # Analyze market impact
        market_impact = analyze_market_impact(sentiment_score, financial_entities)
        
        # Generate trading insights
        trading_insights = generate_trading_insights(sentiment_score, financial_entities)
        
        return jsonify({
            'success': True,
            'sentiment_score': round(sentiment_score, 3),
            'financial_entities': financial_entities,
            'market_impact': market_impact,
            'trading_insights': trading_insights
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

def extract_financial_entities(text):
    """Extract financial entities from text"""
    entities = {
        'companies': [],
        'currencies': [],
        'numbers': [],
        'keywords': []
    }
    
    # Extract company names (simple pattern)
    company_pattern = r'\b[A-Z]{2,}\b'
    entities['companies'] = re.findall(company_pattern, text)[:5]
    
    # Extract currency mentions
    currency_pattern = r'\$\d+|\d+\s*(?:USD|EUR|GBP|INR)'
    entities['currencies'] = re.findall(currency_pattern, text)
    
    # Extract numbers
    number_pattern = r'\d+(?:\.\d+)?%?'
    entities['numbers'] = re.findall(number_pattern, text)[:10]
    
    # Extract financial keywords
    keywords = ['revenue', 'profit', 'loss', 'earnings', 'stock', 'market', 'investment', 'dividend']
    entities['keywords'] = [word for word in keywords if word.lower() in text.lower()]
    
    return entities

def analyze_market_impact(sentiment_score, entities):
    """Analyze potential market impact"""
    impact = "Neutral"
    
    if sentiment_score > 0.3:
        impact = "Potentially Bullish"
    elif sentiment_score < -0.3:
        impact = "Potentially Bearish"
    
    if len(entities['companies']) > 0:
        impact += f" - Affects: {', '.join(entities['companies'][:3])}"
    
    return impact

def generate_trading_insights(sentiment_score, entities):
    """Generate trading insights"""
    insights = []
    
    if sentiment_score > 0.5:
        insights.append("Consider bullish positions on mentioned companies")
    elif sentiment_score < -0.5:
        insights.append("Consider defensive positions or short opportunities")
    
    if 'earnings' in entities['keywords']:
        insights.append("Monitor earnings announcements and guidance")
    
    if 'dividend' in entities['keywords']:
        insights.append("Watch for dividend-related news and policy changes")
    
    if not insights:
        insights.append("Monitor for follow-up news and market reactions")
    
    return insights

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 