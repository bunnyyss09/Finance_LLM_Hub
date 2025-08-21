from flask import Blueprint, jsonify, render_template, request
from financial_advisor import ask_financial_advisor

advisor_bp = Blueprint('advisor', __name__)


@advisor_bp.route('/financial-advisor')
def financial_advisor_page():
	return render_template('financial_advisor.html')


@advisor_bp.route('/ask-advisor', methods=['POST'])
def ask_advisor():
	data = request.get_json()
	query = data.get('query', '')
	if not query:
		return jsonify({'error': 'No query provided'}), 400
	try:
		answer = ask_financial_advisor(query)
		return jsonify({'answer': answer})
	except Exception as e:
		return jsonify({'error': str(e)}), 500



