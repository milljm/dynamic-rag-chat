"""Run with: python src/project_store_test.py"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project_store import (  # noqa: E402  # pylint: disable=wrong-import-position
    ProjectNeedFeed,
    delete_file,
    extract_named_fences,
    persist_named_fences,
    read_file,
    run_file,
    safe_relpath,
    take_project_tag,
    tree_listing,
    write_file,
)


class PathJailTest(unittest.TestCase):
    """No escape from the workspace."""

    def test_accepts_nested_source(self):
        self.assertEqual(safe_relpath('src/hello.py'), 'src/hello.py')
        self.assertEqual(safe_relpath('./app.js'), 'app.js')

    def test_rejects_traversal(self):
        self.assertIsNone(safe_relpath('../secret.py'))
        self.assertIsNone(safe_relpath('/etc/passwd'))
        self.assertIsNone(safe_relpath('foo/../../etc/passwd'))
        self.assertIsNone(safe_relpath('.env'))
        self.assertIsNone(safe_relpath(''))

    def test_rejects_no_extension(self):
        self.assertIsNone(safe_relpath('Makefile'))
        self.assertIsNone(safe_relpath('src/foo'))


class WorkspaceTest(unittest.TestCase):
    """Write / read / delete / fences / run."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

    def test_write_roundtrip(self):
        stored = write_file(self.root, 'src/hi.py', 'print(1)\n')
        self.assertEqual(stored, 'src/hi.py')
        self.assertEqual(read_file(self.root, 'src/hi.py'), 'print(1)\n')
        listing = tree_listing(self.root)
        self.assertIn('src/hi.py', listing)

    def test_delete_prunes_empty_dirs(self):
        write_file(self.root, 'src/hi.py', 'x\n')
        self.assertTrue(delete_file(self.root, 'src/hi.py'))
        self.assertIsNone(read_file(self.root, 'src/hi.py'))
        workspace = Path(self.root) / 'projects' / 'workspace'
        self.assertFalse((workspace / 'src').exists())

    def test_named_fences_persist(self):
        text = (
            'Here you go.\n'
            '```python src/hello.py\n'
            'print("hi")\n'
            '```\n'
            'and\n'
            '```js:app.js\n'
            'console.log(1)\n'
            '```\n'
        )
        arts = extract_named_fences(text)
        names = {a['file'] for a in arts}
        self.assertEqual(names, {'src/hello.py', 'app.js'})
        written = persist_named_fences(self.root, text)
        self.assertEqual(sorted(written), ['app.js', 'src/hello.py'])
        self.assertIn('print("hi")', read_file(self.root, 'src/hello.py') or '')

    def test_unnamed_fence_is_ignored(self):
        self.assertEqual(extract_named_fences('```python\nprint(1)\n```'), [])

    def test_run_python(self):
        write_file(self.root, 'hi.py', 'print("hello-project")\n')
        result = run_file(self.root, 'hi.py')
        self.assertEqual(result['code'], 0, result)
        self.assertIn('hello-project', result['stdout'])

    def test_run_missing(self):
        result = run_file(self.root, 'nope.py')
        self.assertNotEqual(result['code'], 0)

    def test_run_rejects_escape(self):
        result = run_file(self.root, '../hi.py')
        self.assertEqual(result['code'], 127)


class ProjectTagTest(unittest.TestCase):
    """Own-line RUN/READ, talk is ignored."""

    def test_strips_run(self):
        vis, action, name = take_project_tag('done.\n<RUN:src/hello.py>\n')
        self.assertEqual(action, 'run')
        self.assertEqual(name, 'src/hello.py')
        self.assertEqual(vis, 'done.')

    def test_inline_is_talk(self):
        text = 'emit tags like `<RUN:src/hello.py>` and keep talking'
        vis, action, name = take_project_tag(text)
        self.assertIsNone(action)
        self.assertIsNone(name)
        self.assertEqual(vis, text)

    def test_placeholder_ignored(self):
        vis, action, name = take_project_tag('<RUN:filename.py>')
        self.assertIsNone(name)
        self.assertIsNone(action)
        self.assertIn('filename.py', vis)

    def test_feed_split_tag(self):
        feed = ProjectNeedFeed()
        a, hit = feed.feed('Lead.\n<RU')
        self.assertFalse(hit)
        self.assertEqual(a, 'Lead.\n')
        b, hit = feed.feed('N:app.js>\n')
        self.assertTrue(hit)
        self.assertEqual(b, '')
        self.assertEqual(feed.action, 'run')
        self.assertEqual(feed.path, 'app.js')

    def test_read_tag(self):
        _, action, name = take_project_tag('<READ:notes.md>')
        self.assertEqual(action, 'read')
        self.assertEqual(name, 'notes.md')


if __name__ == '__main__':
    unittest.main()
