""" Build filter objects suitable for Chroma use """
import ast
from .ragtag_manager import RAGTag # for type hinting
class FilterBuilder:
    """
    Initialize FilterBuilder then call .build(RAGTag) to receive a proper weighted
    filter designed to focus on retrieving important information.
    """

    def build_flexible_filter(self, tags: list[tuple[str, str]], field: str)->dict:
        """Create a Chroma filter that uses AND for important fields, OR for others"""
        must_conditions = []
        soft_conditions = []

        for tag in tags:
            if tag.content.lower() in {'null', 'none', '', 'unspecified', 'unknown'}:
                continue
            condition = {tag.tag: {"$in": tag.content.split(',') if ',' in
                                          tag.content else [tag.content]}}
            if tag.tag in [field]:
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

    def build(self, tags: list[RAGTag[str, str]], field: str)->dict|None:
        """
        Call with a list of RAGTag compatible objects (see ragtag_manager for details).
        Basically the RAGTag is a NamedTuple(tag: str, content: str) object.
        """
        return self.build_flexible_filter(tags, field)
