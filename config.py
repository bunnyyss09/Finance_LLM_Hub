import os

# Flask
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "your-secret-key-here")
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 50 * 1024 * 1024))  # 50MB

# External APIs / Models
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "d1hum6hr01qhsrhc8ftgd1hum6hr01qhsrhc8fu0")
SENTIMENT_MODEL_PATH = os.environ.get("SENTIMENT_MODEL_PATH", "results/saved_model")



