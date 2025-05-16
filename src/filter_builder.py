""" Build filter objects suitable for Chroma use """
from .ragtag_manager import RAGTag # for type hinting
class FilterBuilder:
    """
    Initialize FilterBuilder then call .build(RAGTag) to receive a proper weighted
    filter designed to focus on retrieving important information.
    """
    def __init__(
        self,
        skip_fields=None,
        composite_fields=None,
        multi_delimiters=",/",
        field_overrides=None,
        strict_fields=('focus',)
    ):
        self.skip_fields = skip_fields or {"time", "date"}
        self.composite_fields = composite_fields or {"focus", "tone"}
        self.multi_delimiters = multi_delimiters
        self.field_overrides = field_overrides or {}
        self.strict_fields = strict_fields

    def build_flexible_filter(self, tags: list[tuple[str, str]])->dict:
        """Create a Chroma filter that uses AND for important fields, OR for others"""
        must_conditions = []
        soft_conditions = []

        for tag in tags:
            if tag.content.lower() in {'null', 'none', '', 'unspecified', 'unknown'}:
                continue
            condition = {tag.tag: {"$in": tag.content.split(',') if ',' in
                                          tag.content else [tag.content]}}
            if tag.tag in self.strict_fields:
                must_conditions.append(condition)
            else:
                soft_conditions.append(condition)

        if not must_conditions and not soft_conditions:
            return None
        if not must_conditions:
            return {"$or": soft_conditions}
        if not soft_conditions:
            return {"$and": must_conditions}
        return {
            "$and": must_conditions + [{"$or": soft_conditions}]
        }

    def parse_value(self, tag, value)->dict:
        """ clean incoming values (remove null, none, etc) """
        if not value or value.lower() in {"null", "none", ""}:
            return None

        override = self.field_overrides.get(tag)
        if override == "skip":
            return None

        if override == "composite" or tag in self.composite_fields:
            parts = [p.strip() for p in value.split("/")]
            return {tag: {"$in": ["/".join(parts[:i]) for i in range(1, len(parts)+1)]}}

        if override == "multi" or any(d in value for d in self.multi_delimiters):
            fragments = [frag.strip() for d in self.multi_delimiters for frag in value.split(d)]
            return {tag: {"$in": list(set(fragments))}}

        # default: exact match
        return {tag: value}

    def build(self, tags: list[RAGTag[str, str]])->dict|None:
        """
        Call with a list of RAGTag compatible objects (see ragtag_manager for details).
        Basically the RAGTag is a NamedTuple(tag: str, content: str) object.
        """
        return self.build_flexible_filter(tags)
