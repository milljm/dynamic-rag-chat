""" Provides direct access to an LLM (no streaming) """
from langchain_ollama import ChatOllama

class OllamaModel():
    """
    Responsible for dealing directly with LLMs,
    out side of the realm of the Chat class
    """
    def __init__(self, base_url):
        self.base_url = base_url

    def llm_query(self, model, prompt_template, temp=1.0)->object:
        """ query the llm with a message, without streaming """
        llm = ChatOllama(model=model,
                         temperature=temp,
                         base_url=self.base_url,
                         repeat_penalty=1.1,
                         streaming=False)
        return llm.invoke(prompt_template)
