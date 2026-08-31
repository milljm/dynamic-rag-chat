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
    apply_tag,
    create_project,
    delete_file,
    extract_named_fences,
    list_files,
    list_projects,
    list_tools,
    persist_named_fences,
    project_root,
    read_file,
    read_tool,
    remove_project,
    run_file,
    run_git,
    run_tool,
    safe_relpath,
    scratch_root,
    select_project,
    snapshot,
    take_project_tag,
    tools_listing,
    tools_root,
    tree_listing,
    write_file,
    write_tool,
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
        self.assertIn('kind: scratch', listing)
        self.assertIn('git: no', listing)

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

    def test_run_args(self):
        write_file(self.root, 'echo.py', 'import sys\nprint(sys.argv[1])\n')
        result = run_file(self.root, 'echo.py', ['hello world'])
        self.assertEqual(result['code'], 0, result)
        self.assertEqual(result['stdout'].strip(), 'hello world')

    def test_run_args_are_literal(self):
        write_file(self.root, 'echo.py', 'import sys\nprint(sys.argv[1])\n')
        result = run_file(self.root, 'echo.py', ['hello; rm -rf /'])
        self.assertEqual(result['code'], 0, result)
        self.assertEqual(result['stdout'].strip(), 'hello; rm -rf /')

    def test_worker_writes_files(self):
        write_file(
            self.root,
            'agents/make.py',
            'from pathlib import Path\n'
            'Path("src").mkdir(exist_ok=True)\n'
            'Path("src/out.py").write_text("print(1)\\n")\n'
            'print("ok")\n',
        )
        result = run_file(self.root, 'agents/make.py')
        self.assertEqual(result['code'], 0, result)
        self.assertIn('ok', result['stdout'])
        self.assertEqual(read_file(self.root, 'src/out.py'), 'print(1)\n')

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
        vis, action, name, args = take_project_tag('done.\n<RUN:src/hello.py>\n')
        self.assertEqual(action, 'run')
        self.assertEqual(name, 'src/hello.py')
        self.assertEqual(args, [])
        self.assertEqual(vis, 'done.')

    def test_strips_run_args(self):
        vis, action, name, args = take_project_tag(
            'ok.\n<RUN:agents/do.py --name "my app">\n',
        )
        self.assertEqual(action, 'run')
        self.assertEqual(name, 'agents/do.py')
        self.assertEqual(args, ['--name', 'my app'])
        self.assertEqual(vis, 'ok.')

    def test_args_after_closing_bracket(self):
        vis, action, name, args = take_project_tag(
            '<TOOL:uv_setup.py> init -n matplotlib-env python=3.11 '
            'matplotlib numpy pandas\n',
        )
        self.assertEqual(action, 'tool')
        self.assertEqual(name, 'uv_setup.py')
        self.assertEqual(
            args,
            [
                'init', '-n', 'matplotlib-env', 'python=3.11',
                'matplotlib', 'numpy', 'pandas',
            ],
        )
        self.assertEqual(vis, '')
        _, action, name, args = take_project_tag('<RUN:app.py> --port 8\n')
        self.assertEqual(action, 'run')
        self.assertEqual(name, 'app.py')
        self.assertEqual(args, ['--port', '8'])
        _, action, name, args = take_project_tag('<GIT:commit> -m "hi"\n')
        self.assertEqual(action, 'git')
        self.assertEqual(name, 'commit')
        self.assertEqual(args, ['-m', 'hi'])

    def test_inline_is_talk(self):
        text = 'emit tags like `<RUN:src/hello.py>` and keep talking'
        vis, action, name, args = take_project_tag(text)
        self.assertIsNone(action)
        self.assertIsNone(name)
        self.assertEqual(args, [])
        self.assertEqual(vis, text)

    def test_placeholder_ignored(self):
        vis, action, name, args = take_project_tag('<RUN:filename.py>')
        self.assertIsNone(name)
        self.assertIsNone(action)
        self.assertEqual(args, [])
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
        self.assertEqual(feed.args, [])

    def test_feed_run_args(self):
        feed = ProjectNeedFeed()
        _, hit = feed.feed('<RUN:agents/do.py --x foo>\n')
        self.assertTrue(hit)
        self.assertEqual(feed.path, 'agents/do.py')
        self.assertEqual(feed.args, ['--x', 'foo'])

    def test_feed_args_after_gt(self):
        feed = ProjectNeedFeed()
        a, hit = feed.feed('<TOOL:uv_setup.py> init')
        self.assertFalse(hit)
        self.assertEqual(a, '')
        b, hit = feed.feed(' -n env\n')
        self.assertTrue(hit)
        self.assertEqual(b, '')
        self.assertEqual(feed.action, 'tool')
        self.assertEqual(feed.path, 'uv_setup.py')
        self.assertEqual(feed.args, ['init', '-n', 'env'])

    def test_read_tag(self):
        _, action, name, args = take_project_tag('<READ:notes.md>')
        self.assertEqual(action, 'read')
        self.assertEqual(name, 'notes.md')
        self.assertEqual(args, [])

    def test_git_tag(self):
        vis, action, name, args = take_project_tag(
            'ok.\n<GIT:commit -m "start here">\n',
        )
        self.assertEqual(action, 'git')
        self.assertEqual(name, 'commit')
        self.assertEqual(args, ['-m', 'start here'])
        self.assertEqual(vis, 'ok.')

    def test_git_placeholder_ignored(self):
        vis, action, name, args = take_project_tag('<GIT:example>')
        self.assertIsNone(action)
        self.assertIsNone(name)
        self.assertEqual(args, [])
        self.assertIn('example', vis)

    def test_feed_git_tag(self):
        feed = ProjectNeedFeed()
        _, hit = feed.feed('<GIT:status>\n')
        self.assertTrue(hit)
        self.assertEqual(feed.action, 'git')
        self.assertEqual(feed.path, 'status')
        self.assertEqual(feed.args, [])


class GitAgentTest(unittest.TestCase):
    """Local git only; init after add of a plain directory."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = self.tmp.name
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.root) / 'repos' / 'plain-app'
        self.repo.mkdir(parents=True)
        (self.repo / 'readme.md').write_text('hi\n', encoding='utf-8')
        add_project(self.root, str(self.repo))

    def test_tree_git_no_until_init(self):
        listing = tree_listing(self.root)
        self.assertIn('git: no', listing)
        result = run_git(self.root, ['init'])
        self.assertEqual(result['code'], 0, result)
        listing = tree_listing(self.root)
        self.assertIn('git: yes', listing)
        rec = next(p for p in list_projects(self.root) if p['id'] == 'plain-app')
        self.assertTrue(rec['git'])

    def test_push_denied(self):
        result = run_git(self.root, ['push'])
        self.assertEqual(result['code'], 127)
        self.assertIn('not allowed', result['stderr'])

    def test_escape_denied(self):
        result = run_git(self.root, ['status', '-C', '/tmp'])
        self.assertEqual(result['code'], 127)

    def test_global_config_denied(self):
        result = run_git(self.root, ['config', '--global', 'user.name', 'nope'])
        self.assertEqual(result['code'], 127)

    def test_status_add_commit(self):
        self.assertEqual(run_git(self.root, ['init'])['code'], 0)
        self.assertEqual(run_git(self.root, ['config', 'user.email', 'a@b.c'])['code'], 0)
        self.assertEqual(run_git(self.root, ['config', 'user.name', 'Test'])['code'], 0)
        self.assertEqual(run_git(self.root, ['add', 'readme.md'])['code'], 0)
        result = run_git(self.root, ['commit', '-m', 'start'])
        self.assertEqual(result['code'], 0, result)
        st = run_git(self.root, ['status'])
        self.assertEqual(st['code'], 0, st)

    def test_apply_tag_git(self):
        body, status = apply_tag(self.root, 'git', 'init', [])
        self.assertTrue(status.startswith('Git init'))
        self.assertIn('PROJECT_GIT', body)
        self.assertIn('git: yes', tree_listing(self.root))


class ToolNamespaceTest(unittest.TestCase):
    """Tools live outside the project and persist across switches."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

    def test_tool_fence_not_in_project(self):
        text = (
            '```python tool:uv_setup.py\n'
            'from pathlib import Path\n'
            'Path("marker.txt").write_text("ok\\n")\n'
            'print("installed")\n'
            '```\n'
        )
        arts = extract_named_fences(text)
        self.assertEqual(arts[0]['ns'], 'tool')
        self.assertEqual(arts[0]['file'], 'uv_setup.py')
        written = persist_named_fences(self.root, text)
        self.assertEqual(written, ['tool:uv_setup.py'])
        self.assertIsNone(read_file(self.root, 'uv_setup.py'))
        self.assertIn('Path("marker.txt")', read_tool(self.root, 'uv_setup.py') or '')
        self.assertTrue((tools_root(self.root) / 'uv_setup.py').is_file())
        paths = {row['path'] for row in list_files(self.root)}
        self.assertNotIn('uv_setup.py', paths)

    def test_tool_runs_in_project_cwd(self):
        write_tool(
            self.root,
            'uv_setup.py',
            'from pathlib import Path\n'
            'import os\n'
            'Path("marker.txt").write_text("ok\\n")\n'
            'print(os.getcwd())\n',
        )
        result = run_tool(self.root, 'uv_setup.py')
        self.assertEqual(result['code'], 0, result)
        self.assertEqual(read_file(self.root, 'marker.txt'), 'ok\n')
        project = scratch_root(self.root).resolve()
        self.assertIn(str(project), result['stdout'])
        self.assertFalse((tools_root(self.root) / 'marker.txt').exists())

    def test_tools_survive_project_switch(self):
        write_tool(self.root, 'shared.py', 'print(1)\n')
        repo = Path(self.root) / 'repos' / 'other'
        repo.mkdir(parents=True)
        add_project(self.root, str(repo))
        names = {row['path'] for row in list_tools(self.root)}
        self.assertEqual(names, {'shared.py'})
        self.assertIn('shared.py', tools_listing(self.root))
        snap = snapshot(self.root)
        self.assertEqual([t['path'] for t in snap['tools']], ['shared.py'])

    def test_tool_jail(self):
        self.assertIsNone(write_tool(self.root, '../escape.py', 'nope\n'))
        result = run_tool(self.root, '../escape.py')
        self.assertEqual(result['code'], 127)

    def test_tool_tag(self):
        vis, action, name, args = take_project_tag(
            'ok.\n<TOOL:uv_setup.py --quiet>\n',
        )
        self.assertEqual(action, 'tool')
        self.assertEqual(name, 'uv_setup.py')
        self.assertEqual(args, ['--quiet'])
        self.assertEqual(vis, 'ok.')

    def test_apply_tag_tool(self):
        write_tool(self.root, 'hi.py', 'print("tool-hi")\n')
        body, status = apply_tag(self.root, 'tool', 'hi.py', [])
        self.assertTrue(status.startswith('Tool hi.py'))
        self.assertIn('PROJECT_TOOL', body)
        self.assertIn('tool-hi', body)

    def test_missing_tool_tells_write_first(self):
        result = run_tool(self.root, 'uv_setup.py')
        self.assertEqual(result['code'], 127)
        self.assertIn('Write it first', result['stderr'])
        self.assertIn('tool:uv_setup.py', result['stderr'])
        self.assertIn('nothing is built-in', result['stderr'].lower())

    def test_empty_tools_listing_says_write(self):
        listing = tools_listing(self.root)
        self.assertIn('no tools yet', listing)
        self.assertIn('nothing is built-in', listing)


class CreateProjectTest(unittest.TestCase):
    """<NEW:hello_world> is its own git repo, not workspace."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

    def test_new_is_named_and_git(self):
        result = create_project(self.root, 'hello_world')
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['active'], 'hello_world')
        self.assertEqual(result['project']['kind'], 'managed')
        dest = Path(self.root) / 'projects' / 'hello_world'
        self.assertEqual(project_root(self.root).resolve(), dest.resolve())
        self.assertTrue((dest / '.git').exists())
        self.assertFalse((scratch_root(self.root) / '.git').exists())
        listing = tree_listing(self.root)
        self.assertIn('project: hello_world', listing)
        self.assertIn('git: yes', listing)
        self.assertIn('kind: created', listing)

    def test_writes_land_in_named_project(self):
        create_project(self.root, 'hello_world')
        stored = write_file(self.root, 'hello_world.py', 'print(1)\n')
        self.assertEqual(stored, 'hello_world.py')
        dest = Path(self.root) / 'projects' / 'hello_world' / 'hello_world.py'
        self.assertTrue(dest.is_file())
        self.assertFalse((scratch_root(self.root) / 'hello_world.py').exists())

    def test_new_tag(self):
        vis, action, name, args = take_project_tag('go.\n<NEW:hello_world>\n')
        self.assertEqual(action, 'new')
        self.assertEqual(name, 'hello_world')
        self.assertEqual(args, [])
        self.assertEqual(vis, 'go.')
        body, status = apply_tag(self.root, 'new', 'hello_world', [])
        self.assertTrue(status.startswith('Project hello_world'))
        self.assertIn('PROJECT_NEW', body)
        self.assertIn('git=yes', body)

    def test_fences_above_new_land_in_project(self):
        fence = '```python hi.py\nprint(1)\n```\n<NEW:hello_world>\n'
        body, _ = apply_tag(self.root, 'new', 'hello_world', [], fence)
        self.assertIn('git=yes', body)
        dest = Path(self.root) / 'projects' / 'hello_world' / 'hi.py'
        self.assertTrue(dest.is_file())
        self.assertFalse((scratch_root(self.root) / 'hi.py').exists())
        self.assertFalse((scratch_root(self.root) / '.git').exists())

    def test_reject_workspace_name(self):
        result = create_project(self.root, 'workspace')
        self.assertFalse(result['ok'])

    def test_second_new_selects(self):
        first = create_project(self.root, 'hello_world')
        write_file(self.root, 'keep.py', '1\n')
        second = create_project(self.root, 'hello_world')
        self.assertEqual(second['active'], first['active'])
        self.assertIsNotNone(read_file(self.root, 'keep.py'))


if __name__ == '__main__':
    unittest.main()
