"""Run with: python src/attachment_store_test.py"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

try:
    from .attachment_store import (
        delete_attachment,
        get_attachment,
        list_attachments,
        put_attachment,
        safe_filename,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from attachment_store import (
        delete_attachment,
        get_attachment,
        list_attachments,
        put_attachment,
        safe_filename,
    )


class AttachmentStoreTest(unittest.TestCase):
    """Filesystem cabinet for whole gold files."""

    def test_safe_filename(self):
        self.assertEqual(safe_filename('/tmp/spur-server.py'), 'spur-server.py')
        self.assertEqual(safe_filename('../etc/passwd'), 'passwd')
        self.assertIn('_', safe_filename('foo bar.py'))

    def test_roundtrip_list_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            put_attachment(tmp, 'README.md', '# hi')
            put_attachment(tmp, 'spur-server.py', 'print(1)')
            names = [r['name'] for r in list_attachments(tmp)]
            self.assertEqual(names, ['README.md', 'spur-server.py'])
            self.assertEqual(get_attachment(tmp, 'readme.md'), '# hi')
            self.assertTrue(delete_attachment(tmp, 'README.md'))
            self.assertIsNone(get_attachment(tmp, 'README.md'))
            self.assertEqual(
                [r['name'] for r in list_attachments(tmp)],
                ['spur-server.py'],
            )


if __name__ == '__main__':
    unittest.main()
