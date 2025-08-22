# Finance LLM Hub

AI-powered financial analysis platform built with Flask, featuring 7 core functionalities with modern UI design and intelligent financial insights.

## Features

### 1. Document Parser & RAG System
- Upload and analyze financial documents (PDF, DOCX, TXT, HTML)
- AI-powered document chunking and semantic search using FAISS vector database
- Sentence-BERT embeddings for context-aware document retrieval
- Interactive Q&A system for uploaded financial documents

### 2. Tax Calculator & Planner
- Comprehensive Indian tax calculation (Old & New Regime)
- Age-based tax slab calculations (Under 60, 60-80, 80+)
- Advanced deduction optimization (80C, 80D, 80CCD1B)
- Capital gains tax calculations (STCG, LTCG)

### 3. Live Stock Analysis
- Real-time stock data integration using Yahoo Finance API
- Interactive price and volume charts with 30-day historical data
- Company fundamentals and key metrics display
- AI-generated stock recommendations based on trend analysis

### 4. Advanced Sentiment Analysis
- Fine-tuned RoBERTa model for financial sentiment classification
- Real-time financial news sentiment analysis using Finnhub API
- Key phrase extraction and subjectivity assessment
- Market impact analysis and trading insights

### 5. Portfolio Optimizer
- Markowitz Modern Portfolio Theory implementation
- Risk-adjusted return optimization with multiple risk profiles
- Sharpe ratio calculation and visualization
- Support for Conservative, Moderate, and Aggressive strategies

### 6. Financial News Analyzer
- Automated news sentiment classification
- Entity extraction (companies, currencies, financial terms)
- Market impact assessment and trend analysis
- Integration with financial news APIs

### 7. AI Financial Advisor
- LLaMA-3-8B powered financial advisory system
- Personalized investment recommendations
- Risk assessment and portfolio suggestions
- Integration with Hugging Face Inference API

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/finance_llm_web.git
   cd finance_llm_web
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file:
   ```env
   FLASK_SECRET_KEY=your-secret-key-here
   FINNHUB_API_KEY=your-finnhub-api-key
   HUGGINGFACE_API_TOKEN=your-hf-token
   FIN_ADVISOR_MODEL_ID=meta-llama/Meta-Llama-3-8B
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## API Keys Required

- **Finnhub API**: Get your free API key from [Finnhub](https://finnhub.io/)
- **Hugging Face API**: Get your API token from [Hugging Face](https://huggingface.co/settings/tokens)
- **Yahoo Finance**: No API key required (uses yfinance library)

## Project Structure

```
finance_llm_web/
├── app.py                    # Main Flask application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── financial_advisor.py      # AI advisor implementation
├── routes/                   # Feature-specific routes
│   ├── core.py              # Main routes
│   ├── document_parser.py   # RAG system & document analysis
│   ├── tax.py               # Tax calculations
│   ├── stocks.py            # Stock data & analysis
│   ├── sentiment.py         # Sentiment analysis
│   ├── portfolio.py         # Portfolio optimization
│   └── advisor.py           # AI financial advisor
├── templates/               # HTML templates
│   ├── index.html
│   ├── document_parser.html
│   ├── tax_calculator.html
│   ├── stock_chart.html
│   ├── sentiment_analysis.html
│   ├── portfolio_optimizer.html
│   └── financial_advisor.html
└── results/                 # Fine-tuned models
    └── saved_model/         # RoBERTa sentiment model
```

## Technologies

- **Backend**: Flask, Python 3.8+
- **AI/ML**: Hugging Face Transformers, Sentence-BERT, FAISS, RoBERTa, TextBlob
- **Data Sources**: Yahoo Finance (yfinance), Finnhub API, Hugging Face Hub
- **Frontend**: Bootstrap 5, Chart.js, Font Awesome
- **Data Processing**: NumPy, Pandas, PyPDF2, python-docx

## Future Enhancements

- [ ] Real-time portfolio tracking
- [ ] Cryptocurrency analysis integration
- [ ] Advanced technical indicators
- [ ] Multi-language support
- [ ] Mobile app development

---

**⚠️ Disclaimer**: This application is for educational and research purposes. Always consult with qualified financial advisors before making investment decisions.
