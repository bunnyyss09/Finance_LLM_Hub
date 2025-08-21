from flask import Blueprint, jsonify, render_template, request
import requests
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
from config import FINNHUB_API_KEY, SENTIMENT_MODEL_PATH

sentiment_bp = Blueprint('sentiment', __name__)

sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}


@sentiment_bp.route('/sentiment-analysis')
def sentiment_analysis():
	return render_template('sentiment_analysis.html')


@sentiment_bp.route('/get-financial-news', methods=['GET'])
def get_financial_news():
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
		news_items = []
		for item in data[:15]:
			title = item.get('headline', '')
			summary = item.get('summary', '')
			text_to_analyze = f"{title}. {summary}"
			result = _analyze_text_sentiment(text_to_analyze)
			category = _categorize_news(title, summary)
			news_items.append({'title': title, 'summary': summary[:200] + "..." if len(summary) > 200 else summary, 'url': item.get('url', ''), 'time_published': item.get('datetime', ''), 'source': item.get('source', ''), 'sentiment': result['sentiment'], 'confidence': result['confidence'], 'category': category, 'tickers': [symbol]})
		return jsonify({'success': True, 'news': news_items, 'total_count': len(news_items)})
	except Exception as e:
		return jsonify({'error': str(e)})


@sentiment_bp.route('/analyze-news-sentiment', methods=['POST'])
def analyze_news_sentiment():
	try:
		data = request.get_json()
		news_text = data.get('text', '')
		if not news_text:
			return jsonify({'error': 'Please provide news text'})
		result = _analyze_text_sentiment(news_text)
		return jsonify({'success': True, 'sentiment': result['sentiment'], 'confidence': result['confidence'], 'key_phrases': result['key_phrases'], 'insights': result['insights']})
	except Exception as e:
		return jsonify({'error': str(e)})


@sentiment_bp.route('/get-news-sentiment-summary', methods=['GET'])
def get_news_sentiment_summary():
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
			res = _analyze_text_sentiment(text)
			sentiment = res['sentiment'].lower()
			category = _categorize_news(title, summary)
			if category not in categories:
				categories[category] = {'positive': 0, 'neutral': 0, 'negative': 0}
			categories[category][sentiment] += 1
			sentiments[sentiment] += 1
		total = sum(sentiments.values())
		if total > 0:
			percentages = {k: round((v / total) * 100, 1) for k, v in sentiments.items()}
		else:
			percentages = {'positive': 0, 'neutral': 0, 'negative': 0}
		if percentages['positive'] > 60:
			overall = "Bullish"; desc = "Market sentiment is positive with strong optimism"
		elif percentages['negative'] > 60:
			overall = "Bearish"; desc = "Market sentiment is negative with concerns"
		else:
			overall = "Neutral"; desc = "Market sentiment is mixed with balanced views"
		return jsonify({'success': True, 'overall_sentiment': overall, 'sentiment_description': desc, 'sentiment_percentages': percentages, 'category_breakdown': categories, 'total_news_analyzed': total})
	except Exception as e:
		return jsonify({'error': str(e)})


def _analyze_text_sentiment(text: str):
	inputs = sentiment_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
	with torch.no_grad():
		outputs = sentiment_model(**inputs)
		probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
		pred_id = int(np.argmax(probs))
	sentiment = label_map[pred_id]
	confidence = float(probs[pred_id])
	# simple key phrase extraction fallback
	key_phrases = []
	for w in text.split():
		if len(w) >= 4:
			key_phrases.append(w.strip('.,:;!?').lower())
	key_phrases = list({k for k in key_phrases})[:10]
	insights = []
	if sentiment == 'Positive' and confidence > 0.7:
		insights.append('Strong positive sentiment detected.')
	elif sentiment == 'Negative' and confidence > 0.7:
		insights.append('Strong negative sentiment detected.')
	elif sentiment == 'Neutral' or confidence < 0.5:
		insights.append('Sentiment is neutral or uncertain.')
	return {'sentiment': sentiment, 'confidence': round(confidence * 100, 2), 'key_phrases': key_phrases, 'insights': insights}


def _categorize_news(title, summary):
	text = f"{title} {summary}".lower()
	if any(w in text for w in ['earnings', 'quarterly', 'revenue', 'profit', 'loss']):
		return 'Earnings'
	if any(w in text for w in ['fed', 'federal reserve', 'interest rate', 'inflation']):
		return 'Monetary Policy'
	if any(w in text for w in ['merger', 'acquisition', 'buyout', 'deal']):
		return 'M&A'
	if any(w in text for w in ['ipo', 'initial public offering', 'listing']):
		return 'IPO'
	if any(w in text for w in ['crypto', 'bitcoin', 'ethereum', 'blockchain']):
		return 'Cryptocurrency'
	if any(w in text for w in ['oil', 'energy', 'gas', 'renewable']):
		return 'Energy'
	if any(w in text for w in ['tech', 'technology', 'software', 'ai', 'artificial intelligence']):
		return 'Technology'
	if any(w in text for w in ['healthcare', 'medical', 'pharma', 'biotech']):
		return 'Healthcare'
	return 'Market News'



@sentiment_bp.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
	try:
		data = request.get_json()
		text = data.get('text', '')
		if not text:
			return jsonify({'error': 'Please provide text to analyze'})
		res = _analyze_text_sentiment(text)
		score_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
		return jsonify({'success': True, 'sentiment': res['sentiment'], 'sentiment_score': score_map.get(res['sentiment'], 0), 'confidence': res['confidence'], 'key_phrases': res['key_phrases'], 'insights': res['insights']})
	except Exception as e:
		return jsonify({'error': str(e)})

