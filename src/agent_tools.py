""" agent tools for use by tooling capable models """
import typing
from ddgs import DDGS
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class DuckDuckGoSearchInput(BaseModel):
    """
    Docstring for DuckDuckGoSearchInput
    """
    query: str = Field(description="Search query to look up")

class DuckDuckGoSearchTool(BaseTool):
    """
    Docstring for DuckDuckGoSearchTool
    """
    name: str = "duckduckgo_search"  # Added type annotation
    description: str = "Search DuckDuckGo for current information"  # Added type annotation
    args_schema: typing.Type[BaseModel] = DuckDuckGoSearchInput

    # pylint: disable-next=arguments-differ
    def _run(self, query: str) -> str:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=5))
        formatted_results = []
        for result in results:
            formatted_results.append(f"**{result['title']}**\n{result['body']}"
                                     f"\nSource: {result['href']}")
        return "\n\n".join(formatted_results)
