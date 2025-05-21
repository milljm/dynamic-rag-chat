""" Provides direct access to an LLM (no streaming) """
from langchain_ollama import ChatOllama

class OllamaModel():
    """
    Responsible for dealing directly with LLMs,
    out side of the realm of the Chat class
    """
    def __init__(self, base_url, **kwargs):
        self.llm = ChatOllama(base_url=base_url,
                              model=kwargs['preconditioner'],
                              temperature=0.5,
                              repeat_penalty=1.1,
                              streaming=False,
                              num_ctx=kwargs['num_ctx'])

    def llm_query(self, prompt_template)->object:
        """ query the llm with a message, without streaming """
        return self.llm.invoke(prompt_template)
