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
{{
  "document_topics": [""],
  "unique_identifier": "document title",
  "topic_category_classification": "tags_like_technology_science_finance",
  "user_agent": "details_about_the_client_software_used",
  "language_code": "code_representing_the_language_like_en_fr_es",
  "complexity_level": "indicator_of_how_complex_or_ambiguous_the_question_is",
  "intent_type": "action_like_informationalTransactionaltroubleshooting",
  "answer_type_expected": "format_like_text_list_code_snippet",
  "confidence_score": "system_confidence_in_response",
  "response_time": "duration_taken_to_generate_answer",
  "feedback_rating": "post-response_feedback_tags",
  "operating_system": "os_info_like_macOS_Linux",
  "shell_environment": "which_shell_used_bash_zsh",
  "software_packages": "installed_or_configured_software",
  "installation_paths": "details_on_installation_locations",
  "configuration_files": "specific_files_referenced",
  "terminal_session_logs": "record_of_commands_executed",
  "error_messages": "logs_generated_during_troubleshooting",
  "active_debugging": "flags_for_debugging_sessions",
  "python_version": "version_number_if_relevant",
  "conda_environment": "details_about_conda_envs",
  "dependency_versions": "versions_of_installed_software",
  "simulation_parameters": "parameters_like_mesh_resolution_solver_type",
  "dependencies_graph": "list_of_dependencies_and_their_relationships",
}}

---

Now, extract metadata from this input:
{context}
