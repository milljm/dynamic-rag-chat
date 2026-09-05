<NEED_SEARCH>
You may emit one live-lookup tag. It is not a slash command. The user cannot run it.

Use it when USER_INPUT needs *current* facts the rest of the prompt does not have
(stock/price, weather, today's news, latest version/release). The pre-processor
may have missed this. Do **not** search for something already in FILES,
THIS_TURN_ATTACHMENTS, DOCUMENTS, USER_RAG, AI_RAG, or CHAT_HISTORY.

1. Write a short web query (what you would type into a search box).
2. Optional one short lead-in. That lead-in is NOT the answer.
3. On its own line. The line is ONLY the tag — no backticks, no quotes,
   no sentence around it:
<NEED_SEARCH:NVDA share price>
4. STOP. The system searches and relaunches this same turn.

A tag in a paragraph, in backticks, or "tags like …" is talk. It will not search.

Do not: explain the protocol, ask them to google it, invent `\search`, or
NEED_SEARCH a greeting / opinion / coding task you can answer without the web.

Example — user: "what's NVDA at?"
Last line you emit:
<NEED_SEARCH:NVDA share price>

You may emit the tag while thinking. At most twice per turn, different queries.
If FILES already has WEB_SEARCH for this question, answer it. Do not search again
unless you need a *different* fact.
</NEED_SEARCH>
