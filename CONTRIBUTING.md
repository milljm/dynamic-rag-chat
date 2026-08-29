# Contributing

Install is in the [README](README.md). This file is the test loop.

## Run the tests

From the repo root, with the project env active:

```bash
python test_harness.py
```

That discovers every `src/*_test.py` and runs them as unittest. A green run ends with `OK`.

The harness loads each file as a **top-level** module (`think_tags_test`, not `src.think_tags_test`). That is on purpose: `src/__init__.py` imports langchain and the rest of the app. `python -m src.think_tags_test` will fight you.

## One file

```bash
python src/think_tags_test.py
python src/gold_fetch_test.py
```

Same rule: path on the command line, not `-m src.…`.

## Adding a test

1. Put it next to the code: `src/<module>_test.py`.
2. Use `unittest`. The harness only picks up `src/*_test.py`.
3. Import the module under test with the try / `sys.path` fallback the existing files use, so both `python src/foo_test.py` and the harness work.
4. Run `python test_harness.py` before you open a PR.

PRs that change behavior in `src/` should come with a matching `*_test.py` (or an update to one).
