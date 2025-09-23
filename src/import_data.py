""" Handle importing of data into a vector database """
import os
import sys
import time
from datetime import timedelta
import argparse # for type hinting
from bs4 import BeautifulSoup
import requests
from langchain_community.document_loaders import PyPDFLoader
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
import pypdf # for error handling of PyPDFLoader
from src import RAGTag

class ImportData:
    """ handle incoming data and store it accordingly into the RAG """
    def __init__(self, session):
        self.d_session = session
        self.state = {}    # rich live panel stateful properties
        self.live = None   # rich live object
        self.parent_splitter = self.d_session.rag.parent_splitter
        self.child_splitter = self.d_session.rag.child_splitter
        try:
            # pylint: disable=protected-access  # protected by try
            self.parent_split = (f'Parent=[bold]{self.parent_splitter._chunk_size}[/]'
                            f'/[bold]{self.parent_splitter._chunk_overlap}[/]'
                            f' Split: [bold]{repr(self.parent_splitter._separators[0])}[/]')
            self.child_split = (f'Child=[bold]{self.child_splitter._chunk_size}[/]'
                           f'/[bold]{self.child_splitter._chunk_overlap}[/]'
                           f' Split: [bold]{repr(self.child_splitter._separators[0])}[/]')
            # pylint: enable=protected-access
        except AttributeError:
            self.parent_split = f'Parent=[bold]2000[/]/[bold]1000[/] Split: [bold]{repr("\n\n")}[/]'
            self.child_split = f'Child=[bold]100[/]/[bold]100[/] Split: [bold]{repr(".")}[/]'

    def estimate_eta(self, start_time, processed, total):
        """Estimate time remaining based on progress"""
        elapsed = time.time() - start_time
        if processed == 0:
            return "Calculating..."
        rate = elapsed / processed
        remaining = total - processed
        eta_seconds = int(rate * remaining)
        return str(timedelta(seconds=eta_seconds))

    def make_full_status(self)->Group:
        """
        ### Build and return all Rich Panels

        *Key init args:*
            .. code-block:: python
                None
        *Returns:*
            .. code-block:: python
                return Group
        """
        eta = self.estimate_eta(self.state['start_time'],
                    self.state['file_idx'] + 1,
                    self.state['file_total'])

        file_table = Table.grid(padding=(0, 1))
        for file_path in self.state['processed_files']:
            file_table.add_row(f'[dim]File:[/] {os.path.basename(file_path)}')
        file_table.add_row('')  # Spacer
        file_table.add_row((f'[bold green]File Progress:[/] {self.state["file_idx"]+1} '
                            f'/ {self.state["file_total"]}'
                            f' [bold green]Estimated Time Remaining:[/] {eta}'))
        file_panel = Panel.fit(
            file_table,
            title='📂 File Import Status',
            border_style='green'
        )

        chunk_panel = self.state.get('chunk_panel')
        if not isinstance(chunk_panel, Panel):
            chunk_panel = Panel(chunk_panel or 'Awaiting chunk processing...',
                                border_style='blue',
                                title='🔄 Processing RAG Input')

        # 🛑 Error Panel
        error_lines = []
        for file, err in self.state.get('failed', set([])):
            short_error = str(err).split('\n', maxsplit=1)[0][:100]
            error_lines.append(f'[red]❌ {file}[/]: {short_error}')

        if error_lines:
            error_panel = Panel(
                '\n'.join(error_lines),
                title='❌ Failed Files',
                border_style='red'
            )
            return Group(file_panel, chunk_panel, error_panel)
        return Group(file_panel, chunk_panel)

    def make_status_table(self, parent: tuple[int,int,list],
                                child: tuple[int,int,int,int],
                                file_path: str,
                                **kwargs)->Panel:
        """
        ### Return Rich Live Panel with updated counts based on the current/totals

        *Key init args:*
            .. code-block:: python
                parent:           tuple(current, total, RAGTag)
                child:            tuple(current, total, keyword current, keyword total)
                data_file:        '/path/to/file being processed'
                processing: bool, if True 'LLM Pre-Processor/Tagging' else ''
                storing: bool,    if True 'RAG Update' else ''
                retry: bool,      if True '⚠️  Retrying' else ''
                message: str,     'displays any information in dimmed syntax'
        *Returns:*
            .. code-block:: python
                return Panel(Table)
        """
        k = kwargs # short hand
        table = Table.grid()
        proc = f'{" [italic red]LLM Pre-Processor/Tagging[/]" if k.get("processing",
                                                                       False) else ""}'
        proc = f'{proc} {"[yellow]⚠️  Retrying:[/]" if k.get("retry", False) else ""}'
        proc = f'{proc} [dim]{k.get("message", False)}[/]' if k.get("retry", False) else proc
        stor = f'{"[italic red]RAG Update[/]" if k.get("storing", False) else ""}'
        meta = f'[dim]Applying Meta Tags to Children: {parent[2][:4]}[/]'
        table.add_row(f'[bold]Current File:[/] {os.path.basename(file_path)}')
        table.add_row(f'[bold green]Parent Chunk:[/] {parent[0]+1} / {parent[1]}{proc}')
        table.add_row(f'[cyan]Child Chunks:[/] {child[0]}/{child[1]} '
                    f'keywords:{child[2]}/{child[3]} {stor}')
        table.add_row(f'[yellow]Chunk Sizes:[/] {self.parent_split} [blue]│[/] {self.child_split}')
        if parent[2]:
            table.add_row(f'{meta}')
        elif not parent[2] and k.get('storing', False):
            table.add_row('[dim]Warning: No Meta Data[/]')
            self.state['failed'].add((file_path, 'No Meta Data gathered'))
        return Panel(table, title="🔄 Processing RAG Input", border_style="blue")

    def do_parentdocs(self, data: str,
                            file_path: str)->tuple[bool,str]:
        """
        ### Parent Document Storage

        Iterate over parent documents and store them to the gold RAG. The text
        will be split into child documents accordingly, based on attributes set
        in ragtag_manager.py.

        *Key init args:*
            .. code-block:: python
                data:      'the text to store'
                file_path: '/path/to/file'
        *Returns tuple with success|failure, 'error string':*
            .. code-block:: python
                return (bool, '')
        """
        collections = self.d_session.common.attributes.collections  # short hand
        split_docs = self.parent_splitter.split_text(data)
        meta_tags = []
        for cnt, split_doc in enumerate(split_docs):
            child_docs = self.child_splitter.split_text(split_doc)
            if self.live:
                self.state['chunk_panel'] = self.make_status_table(
                                                            (cnt, len(split_docs), meta_tags),
                                                            (0, len(child_docs), 0, 0),
                                                            file_path,
                                                            processing=True
                                                            )
                self.live.update(self.make_full_status())
            # Some times the LLM/RAG Tagging does not gibe well. And a simple 'try again' works
            for attempt in range(2):
                try:
                    (message,
                     meta_tags,
                     status) = self.d_session.context.pre_processor(split_doc)
                    if not status:
                        raise RuntimeError(message)
                    _normal = self.d_session.common.normalize_for_dedup(split_doc)
                    self.d_session.rag.store_data(_normal,
                                                  tags_metadata=meta_tags,
                                                  collection=collections['gold'])
                    break
                # pylint: disable=broad-exception-caught  # handling lots of possibilities here
                except Exception as e:
                    if attempt == 1:
                        if self.live:
                            self.state['chunk_panel'] = self.make_status_table(
                                                            (cnt,len(split_docs),meta_tags),
                                                            (0, len(child_docs), 0, 0),
                                                            file_path,
                                                            )
                            self.state['failed'].add((file_path, e))
                            self.live.update(self.make_full_status())
                        return (False, f'Preprocessor/Tagging error: {e}')
                    if self.live:
                        self.state['chunk_panel'] = self.make_status_table(
                                                                (cnt,len(split_docs),meta_tags),
                                                                (0, len(child_docs), 0, 0),
                                                                file_path,
                                                                processing=True,
                                                                retry=True,
                                                                message=e
                                                                )
                        self.live.update(self.make_full_status())
                    time.sleep(0.5)
                # pylint enable=broad-exception-caught
            self._do_childdocs(child_docs, file_path, (cnt, len(split_docs)), meta_tags)
        return (True, '')

    def _do_childdocs(self, child_docs: list[str],
                            file_path: str,
                            parent_state: tuple[int,int],
                            meta_tags: list[RAGTag])->None:
        """
        #### private method

        Store child documents into the gold RAG. Should only be called from
        do_parentdocs, as that method will properly split parent documents into
        children.

        *Key init args:*
            .. code-block:: python
                child_docs:   ['split text', 'split text']
                file_path:    '/path/to/file'
                parent_state: (parent_cnt, parent_total)
                meta_tags:    [RAGTag]
        *Returns:*
            .. code-block:: python
                return None
        """
        c_cnt = 0
        k_cnt = 0
        _meta = list(meta_tags)
        _contents = []
        mode = 'document_topics' if self.d_session.common.opts.assistant_mode else 'entity'
        collections = self.d_session.common.attributes.collections  # short hand
        keyword_contents = [x for x in _meta if x.tag == mode]
        if keyword_contents:
            _meta.remove(keyword_contents[0])
            if isinstance(keyword_contents[0].content, list):
                _contents = keyword_contents[0].content
            else:
                _contents = [keyword_contents[0].content]

        # For each child document
        for c_cnt, child_doc in enumerate(child_docs):
            # For each value in keywords, we write to the RAG with this one value
            for k_cnt, content in enumerate(_contents):
                _tmp_meta = [RAGTag(mode, content), *_meta]
                self.d_session.rag.store_data(child_doc,
                                              tags_metadata=_tmp_meta,
                                              collection=collections['gold'])
                if self.live and self.state is not None:
                    self.state['chunk_panel'] = self.make_status_table(
                                        (parent_state[0], parent_state[1], _tmp_meta),
                                        (c_cnt+1, len(child_docs), k_cnt, len(_contents)),
                                        file_path,
                                        storing=True
                                        )
                    self.live.update(self.make_full_status())

        # Finish up by triggering one last uptime to the Rich Panel
        if self.live and child_docs:
            self.state['chunk_panel'] = self.make_status_table(
                                                (c_cnt, parent_state[1],[]),
                                                (k_cnt+1, len(child_docs), 0, 0),
                                                file_path
                                                )
            self.live.update(self.make_full_status())

    def store_data(self, data: str, file_path: str = '')->tuple[bool,str]:
        """
        ### Store Data

        Tag, process and split incoming documents into the Gold RAG.

        *Key init args:*
            .. code-block:: python
                data:      'large amounts of text to store'
                file_path: '/path/to/file'
        *Returns execution success tuple:*
            .. code-block:: python
                return (bool, 'error string')
        """
        return self.do_parentdocs(data, file_path)

    def extract_text_from_dir(self, directory: str)->None:
        """
        ### Recursive Import

        Walk through supplied directory recursively and store data to gold RAG
        collection.

        Currently supports loading: `*.md *.html *.txt *.pdf *.template`

        *Key init args:*
            .. code-block:: python
                directory: '/path/to/directory'
        *Exits when finished:*
            .. code-block:: python
                sys.exit()
        """
        files_to_process = []
        for fdir, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.md', '.html', '.txt', '.pdf', '.template')):
                    files_to_process.append((fdir, file))

        self.state = {
            'file_idx': 0,
            'file_total': len(files_to_process),
            'file_name': '',
            'processed_files': [],
            'chunk_panel': None,
            'failed': set([]),
            'start_time': time.time(),
        }

        with Live(self.make_full_status(), refresh_per_second=20) as self.live:
            for idx, (fdir, file) in enumerate(files_to_process):
                _file = os.path.join(fdir, file)
                self.state['file_idx'] = idx
                self.state['file_name'] = _file
                self.state['processed_files'].append(_file)
                max_history = 10
                if len(self.state['processed_files']) > max_history:
                    self.state['processed_files'] = self.state['processed_files'][-max_history:]
                self.live.update(self.make_full_status())
                if file.endswith('.pdf'):
                    directory = file
                    self.extract_text_from_pdf(directory, do_exit=False)
                    continue
                with open(_file, 'r', encoding='utf-16') as file_handle:
                    document_content = file_handle.read()
                if _file.endswith('.html'):
                    soup = BeautifulSoup(document_content, 'html.parser')
                    document_content = soup.get_text()
                self.store_data(document_content, _file)
        sys.exit()

    def extract_text_from_pdf(self, v_args: argparse.ArgumentParser, do_exit=True)->None:
        """ Store imported PDF text directly into the RAG """
        print(f'Importing document: {v_args.import_pdf}')
        loader = PyPDFLoader(v_args.import_pdf)
        pages = []
        try:
            for page in loader.lazy_load():
                pages.append(page)
            page_texts = list(map(lambda doc: doc.page_content, pages))
            for p_cnt, page_text in enumerate(page_texts):
                if page_text:
                    print(f'\tPage {p_cnt+1}/{len(page_texts)}')
                    self.store_data(page_text, v_args.import_pdf)
                else:
                    print(f'\tPage {p_cnt+1}/{len(page_texts)} blank')
        except pypdf.errors.PdfStreamError as e:
            print(f'Error loading PDF:\n\n\t{e}\n\nIs this a valid PDF?')
            sys.exit(1)
        if do_exit:
            sys.exit()

    def store_text(self, v_args: argparse.ArgumentParser)->None:
        """ Store imported text file directly into the RAG """
        print(f'Importing document: {v_args.import_txt}')
        with open(v_args.import_txt, 'r', encoding='utf-8') as file:
            document_content = file.read()
            if v_args.import_txt.endswith('.html'):
                soup = BeautifulSoup(document_content, 'html.parser')
                document_content = soup.get_text()
            self.store_data(document_content, v_args.import_txt)
        sys.exit()

    def extract_text_from_web(self, v_args: argparse.ArgumentParser)->None:
        """ extract plain text from web address """
        response = requests.get(v_args.import_web, timeout=300)
        if response.status_code == 200:
            print(f"Document loaded from: {v_args.import_web}")
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            self.store_data(text, '')
        else:
            print(f'Error obtaining webpage: {response.status_code}\n{response.raw}')
            sys.exit(1)
