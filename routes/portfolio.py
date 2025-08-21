from flask import Blueprint, jsonify, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf

portfolio_bp = Blueprint('portfolio', __name__)


@portfolio_bp.route('/portfolio-optimizer')
def portfolio_optimizer():
	return render_template('portfolio_optimizer.html')


@portfolio_bp.route('/optimize-portfolio', methods=['POST'])
def optimize_portfolio():
	try:
		data = request.get_json()
		stocks = data.get('stocks', [])
		risk_tolerance = data.get('risk_tolerance', 'moderate')
		if not stocks:
			return jsonify({'error': 'Please provide stock symbols'})
		portfolio_data = {}
		for stock in stocks[:10]:
			try:
				ticker = yf.Ticker(stock)
				hist = ticker.history(period='1y')
				if hist.empty:
					continue
				portfolio_data[stock] = hist['Close'].pct_change().dropna()
			except Exception:
				continue
		if len(portfolio_data) < 2:
			return jsonify({'error': 'Need at least 2 valid stocks for optimization'})
		returns_df = pd.DataFrame(portfolio_data)
		mean_returns = returns_df.mean()
		cov_matrix = returns_df.cov()
		adj_returns = mean_returns.copy()
		if risk_tolerance == 'conservative':
			weights = _optimize_conservative(adj_returns, cov_matrix)
		elif risk_tolerance == 'aggressive':
			weights = _optimize_aggressive(adj_returns, cov_matrix)
		else:
			weights = _optimize_moderate(adj_returns, cov_matrix)
		portfolio_return = (mean_returns * weights).sum()
		portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
		sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
		return jsonify({'success': True, 'weights': {s: round(float(weights[s]), 3) for s in weights.index}, 'expected_return': round(float(portfolio_return * 252), 3), 'expected_risk': round(float(portfolio_risk * np.sqrt(252)), 3), 'sharpe_ratio': round(float(sharpe_ratio * np.sqrt(252)), 3)})
	except Exception as e:
		return jsonify({'error': str(e)})


def _optimize_conservative(returns, cov):
	weights = np.ones(len(returns)) / len(returns)
	volatilities = np.sqrt(np.diag(cov))
	low_vol_indices = np.argsort(volatilities)[:len(returns)//2]
	weights[low_vol_indices] *= 1.2
	return pd.Series(weights / weights.sum(), index=returns.index)


def _optimize_moderate(returns, cov):
	weights = np.ones(len(returns)) / len(returns)
	return pd.Series(weights, index=returns.index)


def _optimize_aggressive(returns, cov):
	weights = returns / returns.sum()
	return pd.Series(weights, index=returns.index)



