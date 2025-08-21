from flask import Flask
from config import SECRET_KEY, MAX_CONTENT_LENGTH
from routes.core import core_bp
from routes.document_parser import document_bp
from routes.tax import tax_bp
from routes.stocks import stocks_bp
from routes.sentiment import sentiment_bp
from routes.advisor import advisor_bp
from routes.portfolio import portfolio_bp

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Register feature blueprints (routes preserved)
app.register_blueprint(core_bp)
app.register_blueprint(document_bp)
app.register_blueprint(tax_bp)
app.register_blueprint(stocks_bp)
app.register_blueprint(sentiment_bp)
app.register_blueprint(advisor_bp)
app.register_blueprint(portfolio_bp)

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=5000)


