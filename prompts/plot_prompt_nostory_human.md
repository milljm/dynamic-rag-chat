Use Chat History for flow; never repeat any line exactly as written between <<CHAT_HISTORY_START>> and <<CHAT_HISTORY_END>>.

<<ENFORCE:PROGRESS>>
# Per-turn rules:
# - Do NOT repeat the previous assistant line.
# - Do NOT end with ellipses or "trails off."

<<CHAT_HISTORY_START>>
{{chat_history}}
<<CHAT_HISTORY_END>>

<<ATTACHMENTS_START>>
{{dynamic_files}}
<<ATTACHMENTS_END>>

Anser the following users query: **{{user_query}}**
