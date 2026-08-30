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
    add_project,
    delete_file,
    extract_named_fences,
    list_files,
    list_projects,
    persist_named_fences,
    project_root,
    read_file,
    remove_project,
    run_file,
    safe_relpath,
    scratch_root,
    select_project,
    snapshot,
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
        self.assertIn('project: workspace', listing)

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

    def test_skips_vendor_and_git(self):
        write_file(self.root, 'app.py', 'x\n')
        root = scratch_root(self.root)
        (root / 'node_modules').mkdir()
        (root / 'node_modules' / 'left-pad.js').write_text('nope\n', encoding='utf-8')
        git_obj = root / '.git' / 'objects' / 'ab'
        git_obj.mkdir(parents=True)
        (git_obj / 'cdef').write_text('blob\n', encoding='utf-8')
        paths = {row['path'] for row in list_files(self.root)}
        self.assertEqual(paths, {'app.py'})


class ImportProjectTest(unittest.TestCase):
    """Register an existing directory in place."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = self.tmp.name
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.root) / 'repos' / 'demo-app'
        self.repo.mkdir(parents=True)
        (self.repo / '.git').mkdir()
        (self.repo / 'main.py').write_text('print("from-repo")\n', encoding='utf-8')

    def test_add_selects_and_lists(self):
        result = add_project(self.root, str(self.repo))
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['active'], 'demo-app')
        self.assertTrue(result['project']['git'])
        self.assertEqual(result['project']['kind'], 'imported')
        self.assertEqual(project_root(self.root).resolve(), self.repo.resolve())
        paths = {row['path'] for row in result['files']}
        self.assertIn('main.py', paths)
        self.assertIn('from-repo', read_file(self.root, 'main.py') or '')

    def test_write_stays_in_imported_dir(self):
        add_project(self.root, str(self.repo))
        stored = write_file(self.root, 'src/new.py', 'print(2)\n')
        self.assertEqual(stored, 'src/new.py')
        self.assertEqual((self.repo / 'src' / 'new.py').read_text(encoding='utf-8'), 'print(2)\n')
        self.assertFalse((scratch_root(self.root) / 'src' / 'new.py').exists())

    def test_jail_stays_inside_import(self):
        add_project(self.root, str(self.repo))
        self.assertIsNone(write_file(self.root, '../escape.py', 'nope\n'))
        self.assertFalse((self.repo.parent / 'escape.py').exists())
        outside = run_file(self.root, '../main.py')
        self.assertEqual(outside['code'], 127)

    def test_run_uses_imported_cwd(self):
        add_project(self.root, str(self.repo))
        result = run_file(self.root, 'main.py')
        self.assertEqual(result['code'], 0, result)
        self.assertIn('from-repo', result['stdout'])

    def test_duplicate_path_selects(self):
        first = add_project(self.root, str(self.repo))
        write_file(self.root, 'keep.py', '1\n')
        select_project(self.root, 'workspace')
        second = add_project(self.root, str(self.repo))
        self.assertTrue(second['ok'])
        self.assertEqual(second['active'], first['active'])
        ids = [p['id'] for p in list_projects(self.root) if p['kind'] == 'imported']
        self.assertEqual(ids, [first['active']])
        self.assertIsNotNone(read_file(self.root, 'keep.py'))

    def test_switch_and_remove(self):
        add_project(self.root, str(self.repo))
        write_file(self.root, 'only-repo.py', '1\n')
        switched = select_project(self.root, 'workspace')
        self.assertTrue(switched['ok'])
        self.assertEqual(switched['active'], 'workspace')
        self.assertIsNone(read_file(self.root, 'only-repo.py'))
        removed = remove_project(self.root, 'demo-app')
        self.assertTrue(removed['ok'])
        self.assertEqual(removed['active'], 'workspace')
        self.assertTrue((self.repo / 'only-repo.py').is_file())
        ids = [p['id'] for p in list_projects(self.root)]
        self.assertEqual(ids, ['workspace'])

    def test_cannot_remove_scratch(self):
        result = remove_project(self.root, 'workspace')
        self.assertFalse(result['ok'])

    def test_missing_and_root_rejected(self):
        missing = add_project(self.root, str(Path(self.root) / 'no-such-dir'))
        self.assertFalse(missing['ok'])
        self.assertIn('Not a directory', missing['error'])
        root = add_project(self.root, '/')
        self.assertFalse(root['ok'])
        empty = add_project(self.root, '  ')
        self.assertFalse(empty['ok'])

    def test_id_collision(self):
        other = Path(self.root) / 'other' / 'demo-app'
        other.mkdir(parents=True)
        add_project(self.root, str(self.repo))
        second = add_project(self.root, str(other))
        self.assertTrue(second['ok'], second)
        self.assertEqual(second['active'], 'demo-app-2')
        snap = snapshot(self.root)
        self.assertEqual(
            [p['id'] for p in snap['projects']],
            ['workspace', 'demo-app', 'demo-app-2'],
        )


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
