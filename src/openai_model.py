""" Module responsible for instancing the OpenAI connection to the LLM """
from langchain_openai import ChatOpenAI

class OpenAIModel():
    """
    Instantiate the connection
    """
    def __init__(self, **kwargs):
        self.llm = ChatOpenAI(**kwargs)

    def llm_query(self, prompt_template)->object:
        """ query the llm with a message, without streaming """
        return self.llm.invoke(prompt_template)
