{% if has_agent_error %}
<LOOKUP_FAILED>
A live lookup failed. Tell the user you could not fetch current information. Do not guess. Do not mention tools, agents, or pipelines.
</LOOKUP_FAILED>
{% endif %}
<SYSTEM_TIME: {{date_time}}>
<TURN_NUMBER: {{turn_num}}>
<LLM_MODEL: You are an AI from model: {{model_name}}>
<VISION_CAPABLE: {{vision_capable}}>
{% if attached_files_note or dynamic_files %}
<THIS_TURN_ATTACHMENTS - PAPERCLIP THIS TURN. FULL TEXT. DO NOT NEED_GOLD THESE>
{{attached_files_note}}
  <FILES>
{{dynamic_files}}
  </FILES>
</THIS_TURN_ATTACHMENTS>
{% endif %}
{% if include_branch %}
<BRANCH_SNAPSHOT>
{{include_branch}}
</BRANCH_SNAPSHOT>
{% endif %}
{% if has_documents_index %}
<DOCUMENTS_INDEX - BASENAMES YOU MAY NEED_GOLD>
{{documents_index}}
</DOCUMENTS_INDEX>
{% endif %}
{% if gold_documents %}
<DOCUMENTS - CABINET SNIPPETS. NEED_GOLD FOR THE WHOLE FILE>
{{gold_documents}}
</DOCUMENTS>
{% endif %}
{% if user_documents %}
<USER_RAG - OUT-OF-ORDER CHUNKS FROM PAST USER TURNS>
{{user_documents}}
</USER_RAG>
{% endif %}
{% if ai_documents %}
<AI_RAG - OUT-OF-ORDER CHUNKS FROM PAST AI TURNS>
{{ai_documents}}
</AI_RAG>
{% endif %}
{% if gold_resume %}
<GOLD_RESUME - THE USER ALREADY SAW THIS. DO NOT REPEAT IT. THE FILE IS IN FILES (GOLD_FETCH). ANSWER NOW.>
{{gold_resume}}
</GOLD_RESUME>
{% endif %}
{% if use_coding %}
<TOOLS - YOUR PERSISTENT TOOLKIT. OUTSIDE THE PROJECT. GROWS OVER TIME>
{{tools_index}}
</TOOLS>
<PROJECT_FILES - ACTIVE PROJECT TREE. WRITE WITH NAMED FENCES. NEW/RUN/READ/GIT/TOOL ARE OWN-LINE TAGS>
{{project_index}}
</PROJECT_FILES>
{% endif %}
{% if project_resume %}
<PROJECT_RESUME - THE USER ALREADY SAW THIS. DO NOT REPEAT IT. RESULT IS IN PROJECT_RESULT. CONTINUE.>
{{project_resume}}
</PROJECT_RESUME>
{% endif %}
{% if project_result %}
<PROJECT_RESULT>
{{project_result}}
</PROJECT_RESULT>
{% endif %}
<CHAT_HISTORY - FLOW AND TONE. SUPERSEDES AI_RAG FOR FACTS>
{{chat_history}}
</CHAT_HISTORY>
<NAME_YOUR_FENCES>
If this reply includes code the user could save, every fence is ```lang filename.ext — invent a basename (hello_world.py, fetch_panw.py). Unnamed ```python fences do not become downloads. One-liners may stay unnamed.
</NAME_YOUR_FENCES>
<USER_INPUT>
{{user_query}}
</USER_INPUT>
