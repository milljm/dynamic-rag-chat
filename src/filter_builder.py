""" Build filter-schema for use with Chroma """
from typing import List, Dict, Union
from .chat_utils import RAGTag  # for type hinting

class FilterBuilder:
    """
    A class to construct robust Chroma filters based on tag conditions.
    """
    def build_flexible_filter(self,
                              tags: List[RAGTag[str,str]],
                              field: str) -> Union[Dict, None]:
        """
        Build a Chroma filter that uses 'AND' for important fields (e.g., specified `field`),
        and 'OR' for less critical fields.

        Args:
            tags (List[RAGTag]): A list of RAGTag objects.
            field (str): A specific field that should be treated as a 'must' condition.

        Returns:
            dict: A valid Chroma filter as a dictionary or None if no valid filter is found.
        """
        must_conditions = []
        soft_conditions = []
        for tag in tags:
            # Skip lists, null, unspecified, and empty content
            if isinstance(tag.content, list) or isinstance(tag.content, bool):
                continue
            if not tag.content or tag.content.lower() in {'null',
                                                          'none',
                                                          '',
                                                          'unspecified',
                                                          'unknown'}:
                continue
            # Handle the content, splitting it if there are multiple values
            condition_values = tag.content.split(',') if ',' in tag.content else [tag.content]
            condition = {tag.tag: {"$in": condition_values}}
            if tag.tag == field:
                must_conditions.append(condition)
            else:
                soft_conditions.append(condition)
        # If no conditions were added, return None
        if not must_conditions and not soft_conditions:
            return None
        # Build final filter object
        if not must_conditions:
            return {"$or": soft_conditions}
        if not soft_conditions:
            return {"$and": must_conditions}
        return {
            "$and": must_conditions + soft_conditions
        }

    def build(self, tags: List[RAGTag[str,str]], field: str) -> Union[Dict, None]:
        """
        Main method to call with a list of RAGTag objects.

        Args:
            tags (List[RAGTag]): A list of RAGTag objects.
            field (str): The field to prioritize as an 'AND' condition.

        Returns:
            dict: The final filter for Chroma, or None if no valid filter is constructed.
        """
        return self.build_flexible_filter(tags, field)
