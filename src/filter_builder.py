"""Build filter-schema for Chroma, then match comma-joined metadata in Python."""
from typing import Dict, List, Union
try:
    from .chat_utils import RAGTag
except ImportError:
    from chat_utils import RAGTag

_EMPTY = {'', 'null', 'none', 'unspecified', 'unknown', 'n/a'}


def _as_values(content) -> list[str]:
    """Normalize a RAGTag content (str or list) into stripped lowercase values."""
    if content is None:
        return []
    if isinstance(content, (list, tuple, set)):
        raw = list(content)
    else:
        raw = str(content).split(',')
    out = []
    seen = set()
    for item in raw:
        val = str(item).strip().lower()
        if not val or val in _EMPTY or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def metadata_matches(metadata: dict, field: str, values: list[str]) -> bool:
    """True if stored metadata[field] shares any token with values.

    Chroma only stores scalars, so lists are joined with ', ' at write time.
    `$in` therefore never matches a single name inside that string. This
    does the membership check Python-side.
    """
    if not values:
        return True
    raw = metadata.get(field, '') if metadata else ''
    hay = set(_as_values(raw))
    needles = set(_as_values(values))
    return bool(hay & needles)


class FilterBuilder:
    """
    ### FilterBuilder

    Turn pre-processor ``RAGTag`` values into ``{field, values}`` for
    Python-side membership. Chroma stores lists as comma-joined strings,
    so ``$in`` never matches a single name.

    *Class init args:*
        .. code-block:: python
            (none)

    *Usage:*
        - build a spec the retriever can honor:
            .. code-block:: python
                spec = FilterBuilder().build(tags, field='entity')
                # spec is {'field': 'entity', 'values': [...]} or None

        - test one document:
            .. code-block:: python
                ok = metadata_matches(doc.metadata, spec['field'], spec['values'])
    """

    def values_for(self, tags: List[RAGTag], field: str) -> list[str]:
        """Return normalized values for the must-field only."""
        values = []
        for tag in tags:
            if tag.tag != field:
                continue
            values.extend(_as_values(tag.content))
        # stable unique
        seen = set()
        out = []
        for val in values:
            if val in seen:
                continue
            seen.add(val)
            out.append(val)
        return out

    def build(self, tags: List[RAGTag], field: str) -> Union[Dict, None]:
        """Return `{field, values}` for Python-side matching, or None."""
        values = self.values_for(tags, field)
        if not values:
            return None
        return {'field': field, 'values': values}

    @staticmethod
    def tags_are_nsfw(tags: List[RAGTag]) -> bool:
        """True if content_rating or scene_mode says nsfw."""
        keys = {'content_rating', 'scene_mode'}
        for tag in tags:
            if tag.tag not in keys:
                continue
            blob = tag.content
            if isinstance(blob, (list, tuple)):
                blob = ' '.join(str(x) for x in blob)
            if 'nsfw' in str(blob).lower():
                return True
        return False
