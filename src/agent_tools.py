""" agent tools for use by tooling capable models """
import typing
from ddgs import DDGS
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf

class DuckDuckGoSearchInput(BaseModel):
    """Input schema for DuckDuckGo web search"""
    query: str = Field(description='Search query to look up')

class DuckDuckGoSearchTool(BaseTool):
    """Search DuckDuckGo for current information"""
    name: str = 'duckduckgo_search'
    description: str = 'Search DuckDuckGo for current information'
    args_schema: typing.Type[BaseModel] = DuckDuckGoSearchInput

    # pylint: disable-next=arguments-differ
    def _run(self, query: str) -> str:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=5))
        formatted_results = []
        for result in results:
            formatted_results.append(f"**{result['title']}**\n{result['body']}"
                                     f"\nSource: {result['href']}")
        return '\n\n'.join(formatted_results)


class StockPriceInput(BaseModel):
    """Input schema for stock price lookup"""
    ticker: str = Field(description='Stock ticker symbol (e.g., AAPL, PANW, GOOGL)')

class StockPriceTool(BaseTool):
    """Get current stock price and market data using Yahoo Finance.

    Use this tool when users ask about:
    - Stock prices
    - Share prices
    - Ticker values
    - Current market data for a specific company

    This is FREE and fast compared to web search.
    """
    name: str = 'stock_price'
    description: str = (
        "Get current stock price, daily change, and market data for a company "
        "ticker symbol. Use when users ask about stock/share/ticker prices."
    )
    args_schema: typing.Type[BaseModel] = StockPriceInput

    # pylint: disable-next=arguments-differ
    def _run(self, ticker: str) -> str:
        stock = yf.Ticker(ticker.upper().strip())
        info = stock.info

        # Extract relevant fields with fallbacks
        price = info.get('regularMarketPrice') or info.get('currentPrice')
        prev_close = info.get('previousClose') or info.get('regularPreviousClose')

        if price is None:
            return f"Could not find stock data for ticker '{ticker.upper()}'. Please verify the ticker symbol."

        name = info.get('shortName') or info.get('longName', ticker.upper())

        # Calculate change if not provided
        change = info.get('regularMarketChange')
        if change is None and prev_close:
            try:
                change = float(price) - float(prev_close)
            except (TypeError, ValueError):
                change = None

        # Calculate change percentage
        change_pct = info.get('regularMarketChangePercent')
        if change_pct is None and prev_close and change:
            try:
                change_pct = (float(change) / float(prev_close)) * 100
            except (TypeError, ValueError):
                change_pct = None

        # Build response
        lines = [f"Stock Data for {ticker.upper()} ({name}):"]
        lines.append(f"- Current Price: ${price:.2f}")

        if change is not None:
            sign = '+' if change >= 0 else ''
            lines.append(f"- Change: {sign}{change:.2f}")
        if change_pct is not None:
            sign = '+' if change_pct >= 0 else ''
            lines.append(f"- Change %: {sign}{change_pct:.2f}%")

        open_price = info.get('regularMarketOpen')
        if open_price:
            lines.append(f"- Open: ${open_price:.2f}")

        high = info.get('regularMarketDayHigh')
        low = info.get('regularMarketDayLow')
        if high:
            lines.append(f"- Day High: ${high:.2f}")
        if low:
            lines.append(f"- Day Low: ${low:.2f}")

        if prev_close:
            lines.append(f"- Previous Close: ${prev_close:.2f}")

        volume = info.get('regularMarketVolume')
        if volume:
            lines.append(f"- Volume: {volume:,}")

        return '\n'.join(lines)
