"""
ContextManager aims at handeling everything relating to the context
being supplied to the LLM. It utilizing several methods:

    Emoji removal
    Fuzzy match sentences
    list[] -> set() removes any matches from the RAG.
"""
import re
import os
import hashlib
import threading
from langchain.prompts import ChatPromptTemplate
from RAGTagManager import RAGTagManager, RAG
from OllamaModel import OllamaModel
from PromptManager import PromptManager
current_dir = os.path.dirname(os.path.abspath(__file__))
class ContextManager(PromptManager):
    """ A collection of methods aimed at producing/reducing the context """
    def __init__(self, console, **kwargs):
        super().__init__(console)
        self.console = console
        self.prompts = PromptManager(console, debug=self.debug)
        self.rag = RAG(console, **kwargs)
        self.rag_tagger = RAGTagManager(console, **kwargs)
        self.host = kwargs['host']
        self.matches = kwargs['matches']
        self.preconditioner = kwargs['preconditioner']
        self.debug = kwargs['debug']
        self.prompts.build_prompts()

    def pre_processor(self, query, tagging=False, collection='default')->tuple:
        """
        lightweight LLM as a summarization/tagging pre-processor
        """
        prompts = self.prompts
        pre_llm = OllamaModel(self.host)
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        if tagging:
            system_prompt = (prompts.get_prompt(f'{prompts.tag_prompt_file}_system.txt')
                             if self.debug else prompts.tag_prompt_system)
            human_prompt = (prompts.get_prompt(f'{prompts.tag_prompt_file}_human.txt')
                            if self.debug else prompts.tag_prompt_human)
            docs = self.rag.retrieve_data(query, collection, matches=self.matches)
            context = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())
        else:
            system_prompt = (prompts.get_prompt(f'{prompts.pre_prompt_file}_system.txt')
                             if self.debug else prompts.pre_prompt_system)
            human_prompt = (prompts.get_prompt(f'{prompts.pre_prompt_file}_human.txt')
                            if self.debug else prompts.pre_prompt_human)
            context = query

        prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", human_prompt)
                ])
        prompt = prompt_template.format_messages(context=context)
        if self.debug:
            self.console.print(f'PRE-PROCESSOR PROMPT:\n{prompt}\n\n',
                                style='color(233)', highlight=False)
        # pylint: enable=no-member
        content = pre_llm.llm_query(self.preconditioner, prompt).content
        if self.debug:
            self.console.print(f'PRE-PROCESSOR RESPONSE:\n{content}\n\n',
                                style='color(233)', highlight=False)
        tags = self.rag_tagger.get_tags(content, debug=self.debug)
        return (content, tags)

    def post_processing(self, response):
        """
        Send LLM's response off for post processing as a thread. This allows the
        user to begin formulating a response (and even sending a new message
        before this has completed). This step will begin to shine as more and
        more vectors are established, allowing the pre-processing step to draw
        in more nuanced context.
        """
        prompts = self.prompts
        # pylint: disable=no-member # dynamic prompts (see self.__build_prompts)
        post_prompt = (prompts.get_prompt(f'{prompts.tag_prompt_file}_system.txt')
                        if self.debug else prompts.tag_prompt_system)
        prompt_template = ChatPromptTemplate.from_messages([
                    ("system", post_prompt),
                    ("human", '{context}')
                ])
        prompt = prompt_template.format_messages(context=response, question='')
        if self.debug:
            self.console.print(f'POST PROCESS PROMPT TEMPLATE:\n{prompt}\n\n',
                               style='color(233)', highlight=False)
        # pylint: enable=no-member
        threading.Thread(target=self.rag_tagger.update_rag,
                         args=(self.host, self.preconditioner, prompt),
                         kwargs={'debug': self.debug}).start()

    @staticmethod
    def token_retreiver(context_list: list)->int:
        """ iterate over list and do a word count (token) """
        _token_cnt = 0
        if isinstance(context_list, list):
            for sentence in context_list:
                _token_cnt += len(sentence.split())
        return _token_cnt

    @staticmethod
    def normalize_for_dedup(text: str) -> str:
        """ remove emojis and other markdown """
        text = re.sub(r'[\U0001F600-\U0001F64F\u2600-\u26FF\u2700-\u27BF]', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.lower().split())

    def deduplicate_texts(self, texts: list[str]) -> list[str]:
        """ attempt to remove fuzzy duplicate sentences """
        seen = set()
        unique = []
        for text in texts:
            norm = self.normalize_for_dedup(text)
            key = hashlib.md5(norm.encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(text)
        return unique

    def doc_summarizer(self, documents)->tuple:
        """
        Take incomming documents and remove the fluff. Return both the
        final product designed to be served to the LLM, and the token
        counts (before/after):
             .doc_summarizer(documents) -> (str, pre_count, post_count)
        """
        pre = self.token_retreiver(list(map(lambda doc: doc.page_content, documents)))
        context = list(set(list(map(lambda doc: doc.page_content, documents))))
        context = self.deduplicate_texts(context)
        # TODO: get this working better. It's stripping too much context
        # (context, _) = self.pre_processor(context)
        post = self.token_retreiver(context)
        return (context, pre, post)

    def handle_context(self, data, direction='query')->tuple[list,list,int]:
        """ Method to handle all the lovely context """
        # Retrieve context from AI and User RAG
        if direction == 'query':
            ai_documents = []
            user_documents = []
            for collection in ['ai_response', 'user_queries']:
                ai_documents.extend(self.rag.retrieve_data(data,
                                                           collection,
                                                           matches=self.matches))
            # AI Tags/Documents
            (_, ai_tags) = self.pre_processor(data, tagging=True, collection='ai_response')
            for tag in ai_tags:
                if self.debug:
                    self.console.print(f'AI RAG/TAG Collection:{tag.tag}',
                                       style='color(233)',
                                       highlight=False)
                ai_documents.extend(self.rag.retrieve_data(tag.content,
                                                           tag.tag,
                                                           matches=self.matches))
                # Search once more in the default RAG (brings in some nuances)
                ai_documents.extend(self.rag.retrieve_data(tag.content,
                                                           'ai_response',
                                                            matches=self.matches))
            (ai_context, ai_pre, ai_post) = self.doc_summarizer(ai_documents)

            # User Tags/Documents
            (_, user_tags) = self.pre_processor(data, tagging=True, collection='user_queries')
            for tag in user_tags:
                if self.debug:
                    self.console.print(f'USER RAG/TAG Collection:{tag.tag}',
                                       style='color(233)',
                                       highlight=False)
                user_documents.extend(self.rag.retrieve_data(tag.content,
                                                             tag.tag,
                                                             matches=self.matches))
            (user_context, user_pre, user_post) = self.doc_summarizer(user_documents)

            # token math
            pre_total = ai_pre + user_pre
            post_total = ai_post + user_post
            token_reduction = pre_total - post_total

            if token_reduction and self.debug:
                self.console.print(f'RAG TOKEN REDUCTION:\n{pre_total} --> '
                                f'{post_total} = {token_reduction}\n\n',
                                style='color(233)',
                                highlight=False)

            # Store the users query to their RAG, now that we are done pre-processing
            self.rag.store_data(data,
                                collection='user_queries',
                                chunk_size=150,
                                chunk_overlap=50)

            # If the context is empty (no documents found), well, wow. Nothing.
            if not ai_context and not user_context:
                return ('', '', token_reduction)

            # LLMs prefer strings separated by \n\n
            return ('\n\n'.join(ai_context), '\n\n'.join(user_context), token_reduction)

        # Store Context to AI RAG
        # Aggresively fragment the response from heavy-weight LLM responses
        # We can afford to do this due to the dynamic RAG/Tagging in post_processing
        self.rag.store_data(data, collection='ai_response', chunk_size=150, chunk_overlap=50)
        # dynamic RAG creation starts here (non-blocking)
        self.post_processing(data)
