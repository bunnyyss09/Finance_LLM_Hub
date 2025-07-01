# Finance LLM Hub - AI-Powered Financial Analysis Platform

A comprehensive web application that leverages AI and machine learning to provide advanced financial analysis tools. Built with Flask, featuring 6 core functionalities and modern UI design.

## 🚀 Features

### 1. **Document Parser**
- SEC 10-K filing analysis and extraction
- Key section identification (Business Overview, Risk Factors, Financial Data, Management Discussion)
- AI-powered summarization of complex financial documents
- Support for multiple document formats

### 2. **Tax Planner**
- Indian tax calculation (Old Regime)
- AI-powered deduction optimization recommendations
- Real-time tax slab analysis
- Personalized investment suggestions

### 3. **Live Stock Charts**
- Real-time stock data from Alpha Vantage API
- Interactive price and volume charts
- Company information and metrics
- 30-day historical data visualization

### 4. **Sentiment Analysis**
- Advanced NLP-based sentiment analysis
- Key phrase extraction
- Subjectivity assessment
- AI-generated insights

### 5. **Portfolio Optimizer**
- Markowitz Modern Portfolio Theory implementation
- Risk-adjusted return optimization
- Multiple risk tolerance levels (Conservative, Moderate, Aggressive)
- Sharpe ratio calculation and visualization

### 6. **News Analyzer**
- Financial news sentiment analysis
- Entity extraction (companies, currencies, numbers, keywords)
- Market impact assessment
- Trading insights generation

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd finance_llm_web
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## 📊 API Keys Required

### Alpha Vantage API
- Get your free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
- Update the `ALPHA_VANTAGE_API_KEY` variable in `app.py`

## 🎯 Usage Examples

### Document Parser
1. Navigate to `/document-parser`
2. Enter a stock symbol (e.g., AAPL, MSFT)
3. Click "Analyze Filing" to get SEC filing insights

### Tax Planner
1. Go to `/tax-planner`
2. Enter your annual income and deductions
3. Get AI-powered tax optimization recommendations

### Stock Charts
1. Visit `/stock-chart`
2. Enter a stock symbol
3. View interactive charts and company data

### Sentiment Analysis
1. Access `/sentiment-analysis`
2. Paste any text for sentiment analysis
3. Get detailed insights and key phrases

### Portfolio Optimizer
1. Go to `/portfolio-optimizer`
2. Add multiple stock symbols
3. Choose risk tolerance level
4. Get optimized portfolio weights

### News Analyzer
1. Navigate to `/news-analyzer`
2. Paste financial news content
3. Get market impact analysis and trading insights

## 🏗️ Architecture

```
finance_llm_web/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── templates/            # HTML templates
│   ├── index.html        # Main landing page
│   ├── document_parser.html
│   ├── tax_planner.html
│   ├── stock_chart.html
│   ├── sentiment_analysis.html
│   ├── portfolio_optimizer.html
│   └── news_analyzer.html
└── notebooks/            # Original Jupyter notebooks
    ├── chunk_maker.ipynb
    ├── tax-planner.ipynb
    └── live_StockChat.ipynb
```

## 🔧 Technologies Used

- **Backend**: Flask, Python
- **Frontend**: Bootstrap 5, Chart.js, Font Awesome
- **AI/ML**: TextBlob, scikit-learn, TensorFlow
- **Data Sources**: Alpha Vantage API, Yahoo Finance, SEC EDGAR
- **Visualization**: Matplotlib, Seaborn, Chart.js

## 🎨 UI Features

- Modern gradient backgrounds
- Responsive design for all devices
- Interactive charts and visualizations
- Real-time data updates
- Loading animations and feedback
- Sample data for testing

## 🔒 Security Features

- Input validation and sanitization
- Error handling and user feedback
- Secure API key management
- Session management

## 📈 Performance Optimizations

- Efficient data processing
- Caching for API responses
- Optimized chart rendering
- Minimal dependencies

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
1. Set up a production WSGI server (Gunicorn)
2. Configure environment variables
3. Set up reverse proxy (Nginx)
4. Enable HTTPS

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Alpha Vantage for financial data API
- SEC EDGAR for filing data
- Yahoo Finance for stock information
- Bootstrap and Chart.js for UI components

## 📞 Support

For questions or support, please open an issue in the repository.

---

**Note**: This is a demonstration project. For production use, ensure proper security measures, API rate limiting, and data validation are implemented. 