""" Provides direct access to an LLM (no streaming) """
from langchain_ollama import ChatOllama

class OllamaModel():
    """
    Responsible for dealing directly with LLMs,
    out side of the realm of the Chat class
    """
    def __init__(self, base_url):
        self.base_url = base_url

    def llm_query(self, model, prompt_template, temp=0.3)->object:
        """ query the llm with a message, without streaming """
        llm = ChatOllama(model=model,
                         temperature=temp,
                         base_url=self.base_url,
                         streaming=False)
        # I've seen failures at this step. I suspect bad tags, but I need to them
        with open('debugging_output.log', '+a', encoding='utf-8') as f:
            f.write(str(prompt_template))
        response = llm.invoke(prompt_template, stop=["\n\n", "###", "Conclusion"])
        return response
