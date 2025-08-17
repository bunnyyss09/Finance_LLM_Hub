import os
import re
import requests
import yfinance as yf
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool, CodeAgent, ToolCallingAgent, InferenceClientModel, WebSearchTool, LiteLLMModel
from sklearn.linear_model import LogisticRegression
import urllib.parse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Set up sentiment model
sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
sentiment_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

# Alpha Vantage API Key (should be set as env var in production)
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "XF16KZY6ZHWNB09L")

@tool
def visit_webpage(url: str) -> str:
    """
    Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        markdown_content = markdownify(response.text).strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return markdown_content
    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def get_stock_price(ticker: str, historical: bool = False, period: str = "1mo", start_date: str = None, end_date: str = None) -> str:
    """
    Fetches current and/or historical stock price data for the given ticker symbol.

    Args:
        ticker: The stock ticker symbol (e.g., AAPL for Apple, MSFT for Microsoft).
        historical: Whether to include historical data (default: False for current price only).
        period: Time period for historical data. Valid options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (default: 1mo).
        start_date: Start date for historical data in YYYY-MM-DD format (optional, overrides period).
        end_date: End date for historical data in YYYY-MM-DD format (optional, defaults to today).

    Returns:
        A string with current price and/or historical data summary, or fallback search result.
    """
    try:
        stock = yf.Ticker(ticker)
        if not historical:
            price = stock.history(period="1d")['Close'].iloc[-1]
            return f"The current price of {ticker.upper()} is ${price:.2f}."
        else:
            if start_date and end_date:
                hist_data = stock.history(start=start_date, end=end_date)
            elif start_date:
                hist_data = stock.history(start=start_date)
            else:
                hist_data = stock.history(period=period)
            if hist_data.empty:
                return f"No historical data found for {ticker.upper()}."
            current_price = hist_data['Close'].iloc[-1]
            oldest_price = hist_data['Close'].iloc[0]
            highest_price = hist_data['High'].max()
            lowest_price = hist_data['Low'].min()
            avg_volume = hist_data['Volume'].mean()
            price_change = ((current_price - oldest_price) / oldest_price) * 100
            start_str = hist_data.index[0].strftime('%Y-%m-%d')
            end_str = hist_data.index[-1].strftime('%Y-%m-%d')
            return (
                f"Historical data for {ticker.upper()} ({start_str} to {end_str}):\n"
                f"Current price: ${current_price:.2f}\n"
                f"Period change: {price_change:+.2f}%\n"
                f"Highest price: ${highest_price:.2f}\n"
                f"Lowest price: ${lowest_price:.2f}\n"
                f"Average daily volume: {avg_volume:,.0f}\n"
                f"Total trading days: {len(hist_data)}"
            )
    except Exception:
        try:
            if historical:
                url = (
                    f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
                    f"&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
                )
                response = requests.get(url, timeout=10)
                data = response.json()
                if "Time Series (Daily)" in data:
                    time_series = data["Time Series (Daily)"]
                    dates = sorted(time_series.keys(), reverse=True)
                    if len(dates) >= 30:
                        current_price = float(time_series[dates[0]]["4. close"])
                        month_ago_price = float(time_series[dates[29]]["4. close"])
                        price_change = ((current_price - month_ago_price) / month_ago_price) * 100
                        return (
                            f"Historical data for {ticker.upper()} (Alpha Vantage - last 30 days):\n"
                            f"Current price: ${current_price:.2f}\n"
                            f"30-day change: {price_change:+.2f}%"
                        )
                return f"Limited historical data available for {ticker.upper()}"
            else:
                url = (
                    f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
                    f"&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
                )
                response = requests.get(url, timeout=10)
                data = response.json()
                price = float(data["Global Quote"]["05. price"])
                return f"The current price of {ticker.upper()} is ${price:.2f}."
        except Exception:
            search_query = f"{ticker} current stock price"
            if historical:
                search_query += " historical data"
            search_query += " site:finance.yahoo.com OR site:marketwatch.com"
            return f"Unable to fetch data for {ticker.upper()}. Please search manually: {search_query}"

@tool
def market_trend_predictor(ticker: str) -> str:
    """
    Predicts the short-term market trend (up/down) for a stock using recent price changes.
    Falls back to a web search if data unavailable.

    Args:
        ticker: The stock ticker symbol (e.g., AAPL for Apple, MSFT for Microsoft).

    Returns:
        A string indicating the predicted trend or web search fallback result.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo", interval="1d")
        if df.empty:
            raise Exception("No yfinance data")
    except Exception:
        try:
            url = (
                f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
                f"&symbol={ticker}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
            )
            response = requests.get(url)
            data = response.json()
            if "Time Series (Daily)" not in data:
                raise Exception("No Alpha Vantage data")
            import pandas as pd
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index").astype(float)
            df.sort_index(inplace=True)
            df.rename(columns={"5. adjusted close": "Close"}, inplace=True)
            df = df[["Close"]]
        except Exception:
            search_query = f"{ticker} short term market trend"
            return visit_webpage(search_query)
    df["pct_change"] = df["Close"].pct_change()
    df["target"] = (df["pct_change"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    if len(df) < 30:
        search_query = f"{ticker} short term market trend"
        return visit_webpage(search_query)
    model = LogisticRegression()
    model.fit(df[["pct_change"]].values, df["target"].values)
    latest_change = df["pct_change"].iloc[-1]
    pred = model.predict([[latest_change]])[0]
    confidence = model.predict_proba([[latest_change]])[0][pred]
    trend = "UP" if pred == 1 else "DOWN"
    return (
        f"Predicted short-term trend for {ticker.upper()}: *{trend}* "
        f"(Confidence: {confidence:.2%})"
    )

@tool
def financial_sentiment_analyzer(news_text: str) -> str:
    """
    Analyzes the sentiment of financial or stock market-related news text.

    Args:
        news_text: The text of the news article, headline, or report.

    Returns:
        The detected sentiment (Positive, Negative, Neutral) along with confidence.
    """
    try:
        result = sentiment_pipeline(news_text)[0]
        label = result['label']
        score = result['score']
        return f"Sentiment: {label} (Confidence: {score:.2%})"
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"

@tool
def search_financial_news_web(query: str) -> str:
    """
    Searches for financial news using web search as a fallback.
    
    Args:
        query: The financial topic to search for.
    
    Returns:
        Search results from web search as markdown, or an error message if the search fails.
    """
    try:
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query + ' latest financial news')}"
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        markdown_content = markdownify(response.text).strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return markdown_content
    except Exception as e:
        return f"Error searching web for '{query}': {str(e)}"

# Model and agent setup
model_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
model = InferenceClientModel(model_id=model_id)

web_agent = ToolCallingAgent(
    tools=[WebSearchTool(), visit_webpage],
    model=model,
    max_steps=2,
    name="web_search_agent",
    description="Runs web searches for you.",
)

finance_agent = ToolCallingAgent(
    tools=[get_stock_price, market_trend_predictor, financial_sentiment_analyzer, search_financial_news_web],
    model=model,
    max_steps=2,
    name="finance_agent",
    description="Handles stock prices and market summary tasks.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent, finance_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)

def ask_finance_agent(query: str) -> str:
    """
    Run the manager agent with the given query and return the answer as a string.
    """
    return manager_agent.run(query)