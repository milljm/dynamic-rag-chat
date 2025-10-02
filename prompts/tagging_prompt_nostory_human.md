I am an expert metadata extractor.

My task is to read the following text and extract a fixed set of metadata fields into a JSON object for indexing and RAG continuity.

⚠️ Rules:
- Output exactly this JSON schema. Do not add or remove any fields.
- Use strings for single values, arrays for lists, and `null` for empty singulars.
- Do not explain or summarize. Output **only** the JSON object.

📌 Mandatory:
- Always populate: unique_identifier, topic_category_classification, keywords_entities
- Fill all other fields if they are inferable from the text.
- Use `null` for single-value fields that are irrelevant.
- Use `[]` for empty arrays.
- Use lowercase for all values.

🧾 JSON Output Format:
{% raw %}
{
  "keywords_entities": [""],
  "unique_identifier": "document title",
  "topic_category_classification": "tags_like_technology_science_finance",
  "user_agent": "details_about_the_client_software_used",
}
{% endraw %}
---

Now, extract metadata from this input:
{{context}}
