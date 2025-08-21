from flask import Blueprint, jsonify, render_template, request
import yfinance as yf

stocks_bp = Blueprint('stocks', __name__)


@stocks_bp.route('/stock-chart')
def stock_chart():
	return render_template('stock_chart.html')


@stocks_bp.route('/get-stock-data', methods=['POST'])
def get_stock_data():
	try:
		data = request.get_json()
		symbol = data.get('symbol', 'AAPL').upper()
		ticker = yf.Ticker(symbol)
		hist = ticker.history(period='1mo')
		if hist.empty:
			return jsonify({'error': 'No data available for this symbol'})
		dates = [d.strftime('%Y-%m-%d') for d in hist.index]
		prices = hist['Close'].tolist()
		volumes = hist['Volume'].tolist()
		info = ticker.info
		company_info = {
			'name': info.get('longName', symbol),
			'sector': info.get('sector', 'N/A'),
			'industry': info.get('industry', 'N/A'),
			'market_cap': info.get('marketCap', 'N/A'),
			'pe_ratio': info.get('trailingPE', 'N/A')
		}
		recommendation = _generate_stock_recommendation(prices, company_info)
		return jsonify({'success': True, 'symbol': symbol, 'dates': dates, 'prices': prices, 'volumes': volumes, 'company_info': company_info, 'recommendation': recommendation})
	except Exception as e:
		return jsonify({'error': str(e)})


def _generate_stock_recommendation(prices, company_info):
	if len(prices) < 7:
		return "Not enough data for weekly trend analysis."
	last_week = prices[-7:]
	change = last_week[-1] - last_week[0]
	pct_change = (change / last_week[0]) * 100 if last_week[0] != 0 else 0
	sector = company_info.get('sector', 'N/A')
	industry = company_info.get('industry', 'N/A')
	if pct_change > 2:
		rec = f"Uptrend detected in the past week (+{pct_change:.2f}%). {sector} sector. Consider for short-term momentum if fundamentals are strong."
	elif pct_change < -2:
		rec = f"Downtrend detected in the past week ({pct_change:.2f}%). {sector} sector. Exercise caution or look for reversal signals."
	else:
		rec = f"Stable price movement this week ({pct_change:.2f}%). {sector} sector. Consider for diversification or long-term holding."
	if sector != 'N/A' and industry != 'N/A':
		rec += f" ({industry})"
	return rec



