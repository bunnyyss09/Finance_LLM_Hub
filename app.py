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
        
        return jsonify({
            'success': True, 
            'results': results,
            'query': query,
            'total_results': len(results)
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
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'dates': dates,
            'prices': prices,
            'volumes': volumes,
            'company_info': company_info
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

# 4. Sentiment Analysis Functionality
@app.route('/sentiment-analysis')
def sentiment_analysis():
    return render_template('sentiment_analysis.html')

@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Please provide text to analyze'})
        
        # Analyze sentiment using TextBlob
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        subjectivity_score = blob.sentiment.subjectivity
        
        # Categorize sentiment
        if sentiment_score > 0.1:
            sentiment = 'Positive'
        elif sentiment_score < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        # Extract key phrases
        key_phrases = extract_key_phrases(text)
        
        # Generate sentiment insights
        insights = generate_sentiment_insights(sentiment_score, subjectivity_score, text)
        
        return jsonify({
            'success': True,
            'sentiment': sentiment,
            'sentiment_score': round(sentiment_score, 3),
            'subjectivity_score': round(subjectivity_score, 3),
            'key_phrases': key_phrases,
            'insights': insights
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

def extract_key_phrases(text):
    """Extract key phrases from text"""
    blob = TextBlob(text)
    # Get noun phrases and adjectives
    phrases = []
    for phrase in blob.noun_phrases[:5]:
        phrases.append(phrase)
    for word, tag in blob.tags:
        if tag.startswith('JJ'):  # Adjectives
            phrases.append(word)
    return list(set(phrases))[:10]

def generate_sentiment_insights(sentiment_score, subjectivity_score, text):
    """Generate insights based on sentiment analysis"""
    insights = []
    
    if sentiment_score > 0.5:
        insights.append("Strong positive sentiment detected")
    elif sentiment_score < -0.5:
        insights.append("Strong negative sentiment detected")
    
    if subjectivity_score > 0.7:
        insights.append("Text contains subjective opinions")
    elif subjectivity_score < 0.3:
        insights.append("Text appears to be factual")
    
    # Word count analysis
    word_count = len(text.split())
    if word_count > 100:
        insights.append(f"Detailed analysis ({word_count} words)")
    elif word_count < 20:
        insights.append("Brief statement")
    
    return insights

# 5. Portfolio Optimizer (Additional Functionality)
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
        
        # Get historical data for portfolio optimization
        portfolio_data = {}
        for stock in stocks[:10]:  # Limit to 10 stocks
            try:
                ticker = yf.Ticker(stock)
                hist = ticker.history(period='1y')
                if not hist.empty:
                    portfolio_data[stock] = hist['Close'].pct_change().dropna()
            except:
                continue
        
        if len(portfolio_data) < 2:
            return jsonify({'error': 'Need at least 2 valid stocks for optimization'})
        
        # Calculate optimal weights using Markowitz optimization
        returns_df = pd.DataFrame(portfolio_data)
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Simple optimization based on risk tolerance
        if risk_tolerance == 'conservative':
            weights = optimize_conservative(mean_returns, cov_matrix)
        elif risk_tolerance == 'aggressive':
            weights = optimize_aggressive(mean_returns, cov_matrix)
        else:
            weights = optimize_moderate(mean_returns, cov_matrix)
        
        # Calculate portfolio metrics
        portfolio_return = (mean_returns * weights).sum()
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        return jsonify({
            'success': True,
            'weights': {stock: round(weight, 3) for stock, weight in weights.items()},
            'expected_return': round(portfolio_return * 252, 3),  # Annualized
            'expected_risk': round(portfolio_risk * np.sqrt(252), 3),  # Annualized
            'sharpe_ratio': round(sharpe_ratio * np.sqrt(252), 3)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

def optimize_conservative(returns, cov):
    """Conservative portfolio optimization"""
    # Equal weight with slight bias to lower volatility
    weights = np.ones(len(returns)) / len(returns)
    volatilities = np.sqrt(np.diag(cov))
    low_vol_indices = np.argsort(volatilities)[:len(returns)//2]
    weights[low_vol_indices] *= 1.2
    return pd.Series(weights / weights.sum(), index=returns.index)

def optimize_moderate(returns, cov):
    """Moderate portfolio optimization"""
    # Equal weight
    weights = np.ones(len(returns)) / len(returns)
    return pd.Series(weights, index=returns.index)

def optimize_aggressive(returns, cov):
    """Aggressive portfolio optimization"""
    # Weight by expected returns
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