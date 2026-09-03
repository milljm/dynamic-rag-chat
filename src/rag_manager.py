"""
rag_manager aims at handling RAG operations
"""
import os
import re
import logging
import shutil
from uuid import uuid4
from typing import Any, List
from rank_bm25 import BM25Okapi
from pydantic import ConfigDict, Field
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
try:
    from .chat_utils import CommonUtils, ChatOptions, RAGTag  # for Type Hinting
    from .filter_builder import metadata_matches
    from .attachment_store import get_attachment, put_attachment, delete_attachment
    from .rerank import configured as rerank_configured
    from .rerank import post_rerank, reorder as rerank_reorder
except ImportError:
    from chat_utils import CommonUtils, ChatOptions, RAGTag
    from filter_builder import metadata_matches
    from attachment_store import get_attachment, put_attachment, delete_attachment
    from rerank import configured as rerank_configured
    from rerank import post_rerank, reorder as rerank_reorder
# Silence initial RAG database being empty
logging.getLogger('chromadb').setLevel(logging.ERROR)


class BM25Retriever(BaseRetriever):
    """rank_bm25 wrapper. langchain-community's copy is sunsetting."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    vectorizer: Any = None
    docs: list[Document] = Field(default_factory=list)
    k: int = 4

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Case-fold alphanumerics; 'Login,' and 'login' are the same token."""
        return re.findall(r'\w+', (text or '').lower())

    @classmethod
    def from_documents(cls, documents: list[Document], **kwargs) -> 'BM25Retriever':
        """Build a BM25 index over Document.page_content."""
        docs = list(documents)
        if not docs:
            return cls(vectorizer=None, docs=[], **kwargs)
        tokenized = [cls.tokenize(d.page_content) for d in docs]
        return cls(vectorizer=BM25Okapi(tokenized), docs=docs, **kwargs)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """Rank documents for query; run_manager is unused (BaseRetriever API)."""
        del run_manager
        if not self.docs:
            return []
        return list(self.vectorizer.get_top_n(self.tokenize(query), self.docs, n=self.k))

class RAG():
    """
    ### RAG

    Responsible for RAG operations.

    *Class init args:*
        .. code-block:: python
            console: Rich.console  # Top level Rich console object
            common: CommonUtils    # Needed for all the metadata tagging/regex involved
            args: ChatOptions      # Arguments in the form of ChatOption dataclass

    *Usage:*
        - instance RAG:
            .. code-block:: python
                rag = RAG(console, common, args)

        - retrieve data from RAG:
            .. code-block:: python
                rag.retrieve(query, collection, metadatas={})

        - store data to RAG:
            .. code-block:: python
                rag.store(query, collection, metadatas={})

    - *query:* Required. string containing user's question.
    - *collection:* Required. RAG collection to pull/write from/to.
    - *metadatas:* Optional. Use field-filtering matching if set.
    """
    def __init__(self, console, common: CommonUtils, args: ChatOptions):
        self.console = console
        self.common = common
        self.opts = args
        self.retriever_id = 0
        self.embeddings = self._make_embeddings()
        # Opening Chroma(persist_directory=…) per call was ~400ms of the
        # 11ms-embed + 400ms-gap RAG loop. Reuse one client per collection.
        self._chroma: dict[str, Chroma] = {}
        self._pdr: dict[str, ParentDocumentRetriever] = {}

        p_chunk_size = 1000
        p_chunk_overlap = 500
        p_separators = ['\n\n']
        c_chunk_size = 100
        c_chunk_overlap = 50
        c_separators = ['.']
        if self.opts.assistant_mode:
            p_chunk_size = 2000
            p_chunk_overlap = 1000
            c_chunk_size = 1000
            c_chunk_overlap = 500

        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=p_chunk_size,
                                                              chunk_overlap=p_chunk_overlap,
                                                              separators=p_separators)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=c_chunk_size,
                                                             chunk_overlap=c_chunk_overlap,
                                                             separators=c_separators)

    def _make_embeddings(self):
        """Ollama vs OpenAI embeddings. None until Settings has a server + model."""
        host = str(getattr(self.opts, 'emb_host', None) or '')
        model = getattr(self.opts, 'embeddings', None)
        if not host or not model:
            return None
        try:
            if ':11434' in host:
                # https://someaddress.com/v1 --> someaddress:11434
                found = re.findall(r'([\w+\.-]+:[0-9]+)', host)
                if not found:
                    return None
                return OllamaEmbeddings(base_url=found[0], model=model)
            return OpenAIEmbeddings(
                base_url=host,
                model=model,
                api_key=self.opts.api_key,
                check_embedding_ctx_length=False,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    def _embeddings_ready(self) -> bool:
        """False until Settings fills in an embedding server."""
        return self.embeddings is not None

    def _rerank_ready(self) -> bool:
        return rerank_configured(self.opts)

    def _recall_k(self, tagged: bool, n_values: int = 1) -> int:
        """Wide net when a reranker will cut back to ``matches``."""
        k = max(int(self.opts.matches), 0)
        if tagged:
            k = max(k * 4 * max(n_values, 1), 8)
        if self._rerank_ready() and k > 0:
            return min(50, max(k, int(self.opts.matches) * 12, 24))
        return k

    def _apply_rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Cross-encoder reorder. On failure, keep the first ``matches``."""
        keep = max(int(self.opts.matches), 0)
        if not documents or keep <= 0 or not self._rerank_ready():
            return documents
        texts = [d.page_content[:1500] for d in documents]
        order = post_rerank(
            self.opts.rerank_host,
            self.opts.rerank_llm,
            query,
            texts,
            keep,
            api_key=getattr(self.opts, 'api_key', 'none') or 'none',
            timeout=float(getattr(self.opts, 'rerank_timeout', 8.0) or 8.0),
        )
        if self.opts.debug and self.console:
            self.console.print(
                f'Rerank {len(documents)} → {keep} ({"ok" if order else "skip"})',
                style=f'color({self.opts.color})',
                highlight=False,
            )
        if not order:
            return documents if len(documents) <= keep else documents[:keep]
        return rerank_reorder(documents, order, keep)

    @staticmethod
    def _normalize_collection_name(name: str,
                                   min_length: int = 3,
                                   max_length: int = 63,
                                   pad_char: str = 'x') -> str:
        """ pad/sanitize the could-be-invalid collection names """
        # Replace all invalid characters with dashes
        name = re.sub(r'[^a-zA-Z0-9_-]', '-', name)
        # Remove leading/trailing non-alphanumerics to meet start/end rule
        name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
        name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
        # Replace multiple dashes/underscores if needed (optional cleanup)
        name = re.sub(r'[-_]{2,}', '-', name)
        # Avoid names that look like IP addresses
        if re.fullmatch(r'\d{1,3}(\.\d{1,3}){3}', name):
            name = f"col-{name.replace('.', '-')}"
        # Enforce length limits
        if len(name) < min_length:
            name = name.ljust(min_length, pad_char)
        elif len(name) > max_length:
            name = name[:max_length]
        return name

    def _forget_collection(self, collection: str) -> None:
        """Drop cached Chroma / ParentDocumentRetriever for a collection name."""
        key = self._normalize_collection_name(collection)
        self._chroma.pop(key, None)
        self._pdr.pop(key, None)

    def _parent_retriever(self, collection: str)->ParentDocumentRetriever:
        """ Return ParentDocumentRetriever for provided collection """
        collection = self._normalize_collection_name(collection)
        cached = self._pdr.get(collection)
        if cached is not None:
            return cached
        fs = LocalFileStore(os.path.join(self.opts.vector_dir, collection))
        store = create_kv_docstore(fs)
        retriever = ParentDocumentRetriever(
                    vectorstore=self._vector_store(collection),
                    docstore=store,
                    child_splitter=self.child_splitter,
                    parent_splitter=self.parent_splitter)
        self._pdr[collection] = retriever
        return retriever

    @staticmethod
    def _file_key(filename: str) -> str:
        """Stable docstore id for a whole gold file."""
        return f'file:{filename.lower()}'

    def _docstore(self, collection: str):
        """KV docstore used by ParentDocumentRetriever for this collection."""
        retriever = self._pdr.get(self._normalize_collection_name(collection))
        if retriever is not None:
            return retriever.docstore
        collection = self._normalize_collection_name(collection)
        fs = LocalFileStore(os.path.join(self.opts.vector_dir, collection))
        return create_kv_docstore(fs)

    def store_full_file(self, collection: str, filename: str, text: str) -> None:
        """Keep the unsplit file so a later filename mention can retrieve all of it."""
        if not filename or not (text or '').strip():
            return
        key = self._file_key(filename)
        body = text
        if not body.startswith('ATTACHED FILE:') and not body.startswith('ATTACHED IMAGE:'):
            body = f'ATTACHED FILE: {filename}\n\n{body}'
        try:
            put_attachment(self.opts.vector_dir, filename, body)
        except OSError:
            pass
        doc = Document(
            page_content=body,
            metadata={'filename': filename.lower(), 'whole_file': 'true'},
        )
        try:
            self._docstore(collection).mset([(key, doc)])
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    def get_full_file(self, collection: str, filename: str) -> Document | None:
        """Return the unsplit gold file, or None.

        Directory first (vector_dir/attachments), then the docstore key.
        """
        text = get_attachment(self.opts.vector_dir, filename)
        if text:
            return Document(
                page_content=text,
                metadata={'filename': filename.lower(), 'whole_file': 'true'},
            )
        try:
            got = self._docstore(collection).mget([self._file_key(filename)])
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        if got and got[0] is not None:
            return got[0]
        return None

    def delete_named_file(self, collection: str, filename: str) -> bool:
        """Remove a whole file from the cabinet and its gold chunks."""
        name = (filename or '').strip()
        if not name:
            return False
        removed = delete_attachment(self.opts.vector_dir, name)
        try:
            self._docstore(collection).mdelete([self._file_key(name)])
            removed = True
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        try:
            vector = self._vector_store(collection)
            # pylint: disable=protected-access
            payload = vector._collection.get(
                where={'filename': name.lower()},
                include=['metadatas'],
            )
            ids = list(payload.get('ids') or [])
            parent_ids = []
            seen = set()
            for meta in payload.get('metadatas') or []:
                if not isinstance(meta, dict):
                    continue
                parent = meta.get('doc_id')
                if parent and parent not in seen:
                    seen.add(parent)
                    parent_ids.append(parent)
            if ids:
                vector._collection.delete(ids=ids)
                removed = True
            if parent_ids:
                self._docstore(collection).mdelete(parent_ids)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return removed

    def retrieve_named_files(self, query: str, collection: str) -> list[Document]:
        """If the query names a gold file, return that file in full."""
        names = CommonUtils.extract_filenames(query)[:2]
        if not names:
            return []
        found = []
        seen = set()
        for name in names:
            doc = self.get_full_file(collection, name)
            if doc is None:
                doc = self._parents_for_filename(collection, name)
            if doc is None:
                continue
            key = getattr(doc, 'page_content', '')[:80]
            if key in seen:
                continue
            seen.add(key)
            found.append(doc)
        return found

    def _parents_for_filename(self, collection: str, filename: str) -> Document | None:
        """Fallback: stitch parent chunks tagged with this filename."""
        try:
            vector = self._vector_store(collection)
            # pylint: disable=protected-access
            payload = vector._collection.get(
                where={'filename': filename.lower()},
                include=['metadatas'],
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        ids = []
        seen = set()
        for meta in payload.get('metadatas') or []:
            if not isinstance(meta, dict):
                continue
            parent = meta.get('doc_id')
            if not parent or parent in seen:
                continue
            seen.add(parent)
            ids.append(parent)
        if not ids:
            return None
        try:
            parts = self._docstore(collection).mget(ids)
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        texts = [p.page_content for p in parts if p is not None]
        if not texts:
            return None
        return Document(
            page_content='\n\n'.join(texts),
            metadata={'filename': filename.lower(), 'whole_file': 'true'},
        )

    def _vector_store(self, collection: str)->Chroma:
        """ Return our Chroma Collections Database """
        collection = self._normalize_collection_name(collection)
        cached = self._chroma.get(collection)
        if cached is not None:
            return cached
        chroma = Chroma(persist_directory=self.opts.vector_dir,
                        embedding_function=self.embeddings,
                        collection_name=collection)
        self._chroma[collection] = chroma
        return chroma

    def _bm25_retriever(self, documents: list[Document])->BM25Retriever:
        """
        ### BM25 Retriever

        Returns a BM25 retriever object.

        *Key init args:*
            .. code-block:: python
                documents: list[Document]  # list of Document objects
        *Returns:*
            .. code-block:: python
                return retrievers
        """
        _retriever = BM25Retriever.from_documents(documents)
        _retriever.k = (
            len(documents) if self._rerank_ready() else self.opts.matches
        )
        return _retriever

    def _chroma_retriever(self, collection: str, kwargs)->BaseRetriever:
        """
        ### Chroma Retriever

        Returns a Chroma retriever object.

        *Key init args:*
            .. code-block:: python
                documents: list[Document]  # list of Document objects
                id: int                    # An id to tag this retriever with
        *Returns:*
            .. code-block:: python
                return retrievers
        """
        chroma = self._vector_store(collection)
        _retriever = chroma.as_retriever(search_type='similarity',
                                         search_kwargs=kwargs)
        return _retriever

    @staticmethod
    def _filter_spec(metadatas: dict | None) -> tuple[str | None, list[str]]:
        """Unpack FilterBuilder `{field, values}` without treating it as a Chroma where."""
        if not isinstance(metadatas, dict):
            return None, []
        values = metadatas.get('values')
        field = metadatas.get('field')
        if field and isinstance(values, list):
            return str(field), [str(v) for v in values if str(v).strip()]
        return None, []

    def _similar_docs(self, query: str, collection: str, k: int) -> list[Document]:
        """Similarity search with a specific k."""
        retriever = self._chroma_retriever(collection, {'k': k})
        return retriever.invoke(query)

    @staticmethod
    def _parent_ids(documents: list[Document]) -> list[str]:
        """Unique ParentDocumentRetriever ids from child metadata."""
        ids = []
        seen = set()
        for doc in documents:
            meta = getattr(doc, 'metadata', None) or {}
            pid = meta.get('doc_id')
            if not pid or pid in seen:
                continue
            seen.add(pid)
            ids.append(pid)
        return ids

    def _promote_parents(self, children: list[Document], collection: str) -> list[Document]:
        """Swap winners for their parent docs. One parent per doc_id.

        Rerank (and BM25) score *children*. Promoting after the cut stops a
        parent from crowding siblings out of ``matches``.
        """
        parents = {}
        ids = self._parent_ids(children)
        if ids:
            try:
                got = self._docstore(collection).mget(ids)
            except Exception:  # pylint: disable=broad-exception-caught
                got = []
            for pid, parent in zip(ids, got or []):
                if parent is not None:
                    parents[pid] = parent
        out: list[Document] = []
        seen: set[str] = set()
        for child in children:
            meta = getattr(child, 'metadata', None) or {}
            pid = meta.get('doc_id')
            if pid and pid in parents:
                if pid in seen:
                    continue
                seen.add(pid)
                out.append(parents[pid])
                continue
            key = f'child:{id(child)}'
            if key in seen:
                continue
            seen.add(key)
            out.append(child)
        return out

    def retrieve(self, query: str, collection: str, metadatas: dict = None) -> list[Document]:
        """Similarity, then Python-side field membership, then BM25.

        List tags are stored as comma-joined strings, so Chroma `$in` never
        matches a single name. Filter those in Python instead.

        Cross-encoder rerank (optional) cuts to ``matches``, then each winner
        is swapped for its parent document when one exists.
        """
        if not self._embeddings_ready() or self.opts.matches == 0:
            return []
        field, values = self._filter_spec(metadatas)
        k = self._recall_k(bool(values), len(values) if values else 1)
        try:
            documents = self._similar_docs(query, collection, k)
            if values:
                hit = [d for d in documents
                       if metadata_matches(d.metadata, field, values)]
                if hit:
                    documents = hit
            if not documents:
                return documents
            documents = self._bm25_retriever(documents).invoke(query)
        except ValueError:
            if metadatas:
                return self.retrieve(query, collection, metadatas=None)
            return []
        documents = self._apply_rerank(query, documents)
        return self._promote_parents(documents, collection)

    def store_data(self, data,
                         tags_metadata: list[RAGTag[str,str|list]] = None,
                         collection: str = '',
                         quiet: bool = False)->None:
        """ store data into the RAG with optional metadata tagged with it """
        if not collection:
            collection = self.common.attributes.collections['ai']
        if not self._embeddings_ready():
            return
        # Remove metadata tagging information from data
        data = self.common.sanitize_response(data, strip=True)
        if tags_metadata is None:
            tags_metadata = {}
        meta_dict = dict(tags_metadata)
        meta_dict = self.common.normalize_metadata_for_rag(meta_dict)
        if self.opts.debug:
            self.console.print(f'\nSTORE DATA >>>{data}<<<\nTAGS:\n{meta_dict}'
                               f'\nTO COLLECTION:{collection}',
                               style=f'color({self.opts.color})',
                               highlight=False)
        doc = Document(data, metadata=meta_dict)
        retriever = self._parent_retriever(collection)
        try:
            retriever.add_documents([doc])
        # pylint: disable=bare-except  # Sometimes this can fail for a variety of reasons
        except:
            if not quiet:
                print(f'\nERROR STORING DATA:\n{data}\n\nTAGS:\n{meta_dict}\n\n'
                    'Check for malformed TAGS (no list items is usually the culprit)')
        # pylint: enable=bare-except

    def delete_collection(self, source: str)->None:
        """
        Docstring for delete_collection

        :param self: Description
        :param collection: Description
        :type collection: str
        """
        collection_list = [self.common.attributes.collections[x]
                           for x in self.common.attributes.collections]
        for collection in collection_list:
            if collection == 'gold_documents':
                continue
            f_source = f'{source}_{collection}'
            if self.opts.debug:
                self.console.print(f'\nSOURCE COLLECTION >>>{f_source}<<<\n',
                                    style=f'color({self.opts.color})',
                                    highlight=False)

            src_vs = self._vector_store(f_source)
            # pylint: disable=protected-access
            client = src_vs._client
            # pylint: enable=protected-access
            client.delete_collection(f_source)
            self._forget_collection(f_source)

    def _clone_chroma_payload(self, f_source: str, f_target: str, overwrite: bool) -> None:
        """Copy one Chroma collection's documents/metadatas/embeddings."""
        src_vs = self._vector_store(f_source)
        dst_vs = self._vector_store(f_target)
        # pylint: disable=protected-access
        src_col = src_vs._collection
        dst_col = dst_vs._collection
        # pylint: enable=protected-access
        if overwrite:
            try:
                ids = (dst_col.get() or {}).get('ids') or []
                if ids:
                    dst_col.delete(ids=ids)
            except (ValueError, RuntimeError, OSError):
                pass
        try:
            payload = src_col.get(include=['documents', 'metadatas', 'embeddings'])
        except (ValueError, RuntimeError, OSError) as exc:
            raise RuntimeError(
                f"Failed to read f_source collection '{f_source}': {exc}",
            ) from exc
        ids = payload.get('ids') or []
        if not ids:
            return
        docs = payload.get('documents') or [None] * len(ids)
        metas = payload.get('metadatas') or [{} for _ in ids]
        embs = payload.get('embeddings')
        new_ids = [f'{f_target}_{i}_{uuid4().hex}' for i, _ in enumerate(ids)]
        try:
            if embs is not None:
                dst_col.add(
                    ids=new_ids, documents=docs, metadatas=metas, embeddings=embs,
                )
            else:
                dst_col.add(ids=new_ids, documents=docs, metadatas=metas)
        except (ValueError, RuntimeError, OSError) as exc:
            raise RuntimeError(
                f"Failed to write f_target collection '{f_target}': {exc}",
            ) from exc

    def _clone_docstore_dir(self, f_source: str, f_target: str, overwrite: bool) -> None:
        """Mirror the ParentDocumentRetriever on-disk store."""
        src_store_dir = os.path.join(self.opts.vector_dir, f_source)
        dst_store_dir = os.path.join(self.opts.vector_dir, f_target)
        try:
            if os.path.exists(dst_store_dir) and overwrite:
                shutil.rmtree(dst_store_dir)
            if os.path.exists(src_store_dir):
                shutil.copytree(src_store_dir, dst_store_dir, dirs_exist_ok=True)
            else:
                os.makedirs(dst_store_dir, exist_ok=True)
        except (OSError, PermissionError) as exc:
            raise RuntimeError(
                f"Failed cloning docstore '{src_store_dir}' -> "
                f"'{dst_store_dir}': {exc}",
            ) from exc

    def clone_collection(self, source: str, target: str, *, overwrite: bool = False) -> None:
        """Clone Chroma + parent-docstore from source branch to target."""
        if not source or not target or source == target:
            raise ValueError(
                'clone_collection: source/target must be different, non-empty names.',
            )
        collection_list = [
            self.common.attributes.collections[x]
            for x in self.common.attributes.collections
        ]
        for collection in collection_list:
            if collection == 'gold_documents':
                continue
            f_source = f'{source}_{collection}'
            f_target = f'{target}_{collection}'
            if self.opts.debug:
                self.console.print(
                    f'\nSOURCE COLLECTION >>>{source}_{collection}<<<\n'
                    f'TARGET COLLECTION: >>>{target}_{collection}<<<',
                    style=f'color({self.opts.color})',
                    highlight=False,
                )
            self._clone_chroma_payload(f_source, f_target, overwrite)
            self._clone_docstore_dir(f_source, f_target, overwrite)
            self._forget_collection(f_target)
            if getattr(self.opts, 'debug', False):
                self.console.print(
                    f"[green]Cloned RAG collection[/green] '{f_source}' "
                    f"➜ '{f_target}'",
                    highlight=False,
                )

    def build_collection_from_texts(self,
                                    target: str,
                                    texts: List[str],
                                    overwrite: bool = True) -> None:
        """
        Rebuild `target` collection from raw turn texts; also reset the docstore folder.
        """
        if not target:
            raise ValueError('build_collection_from_texts: target name cannot be empty.')
        collection_list = [self.common.attributes.collections[x]
                           for x in self.common.attributes.collections]
        for collection in collection_list:
            if collection == 'gold_documents':
                continue
            if self.opts.debug:
                self.console.print(f'TARGET COLLECTION: >>>{target}_{collection}<<<',
                                   style=f'color({self.opts.color})',
                                   highlight=False)
            f_target = f'{target}_{collection}'
            self._forget_collection(f_target)
            vs = self._vector_store(f_target)
            # pylint: disable=protected-access
            col = vs._collection
            # pylint: enable=protected-access

            # Clear f_target collection (ids are always returned with .get())
            if overwrite:
                try:
                    existing = col.get()
                    old_ids = existing.get('ids') or []
                    if old_ids:
                        col.delete(ids=old_ids)
                except (ValueError, RuntimeError, OSError):
                    pass

            # Reset docstore path
            store_dir = os.path.join(self.opts.vector_dir, f_target)
            try:
                if overwrite and os.path.exists(store_dir):
                    shutil.rmtree(store_dir)
                os.makedirs(store_dir, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise RuntimeError(f"Failed to prepare docstore '{store_dir}': {e}") from e

            # Add texts
            ids, docs, metas = [], [], []
            for i, text in enumerate(texts, start=1):
                doc_id = f'{f_target}_{i}_{uuid4().hex}'
                ids.append(doc_id)
                docs.append(text)
                metas.append({'turn': i})
                # (optional) persist each turn into the docstore for ParentDocumentRetriever parity
                try:
                    with open(os.path.join(store_dir,
                                        f'{i:05d}_{doc_id}.txt'),
                                        'w', encoding='utf-8') as f:
                        f.write(text)
                except (OSError, PermissionError) as e:
                    raise RuntimeError(f'Failed writing turn {i} to '
                                       f"docstore '{store_dir}': {e}") from e

            if ids:
                try:
                    # Let Chroma embed via the collection's embedding
                    # function; don't pass embeddings.
                    col.add(ids=ids, documents=docs, metadatas=metas)
                except (ValueError, RuntimeError, OSError) as e:
                    raise RuntimeError(f"Failed adding documents to '{f_target}': {e}") from e

            if getattr(self.opts, 'debug', False):
                self.console.print(
                    f"[green]Built RAG collection '{f_target}' from {len(texts)} texts.[/green]",
                    highlight=False
                )
