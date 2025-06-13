I am an expert metadata extractor.

My task is to read the following text and extract a fixed set of metadata fields into a JSON object.

⚠️ Rules:
- Output exactly this JSON schema. Do not add or remove any fields.
- Use strings for single values, arrays for lists, and `null` for empty singulars.
- No explanations. No extra prose. Output JSON only.

<meta_tags> must include:
- tone (e.g., helpful, friendly, sarcastic)
- emotion (e.g., calm, excited, annoyed)
- focus (e.g., greeting, conversation building, explaining). Include multiple if relevant.
- entity: any names of people or beings mentioned (e.g., John, Jane, or None)
- other optional tags: location, gear, weather, relationship_stage, narrative_arc, scene_type, sensory_mood, user_choice, coding

🧾 JSON Output Format:
{{
  "tone": string, // e.g., introspective, hopeful, tense, vulnerable
  "emotion": string, // primary emotion in the text
  "focus": string, // main narrative themes (e.g., dialogue, bonding, introspection)
  "entity": [string], // list of all named characters mentioned. Do not include pronouns or non-named references. I may use None, if there are in fact no ascertainable names available
  "location": [string] | [], // list of places, or empty list
  "items": [string] | [], // list of objects mentioned, or empty list
  "weather": string | null, // environment/weather description
  "relationship_stage": string | null, // e.g., tentative trust, growing bond
  "narrative_arc": string | null, // e.g., ivy_trust_arc, somni_rescue_arc
  "scene_type": string | null, // e.g., quiet moment, dialogue, flashback
  "sensory_mood": string | null, // sensory tone or atmosphere (e.g., warm light, cold silence)
  "user_choice": string | null, // last user action or dialogue
  "coding": string | null // programming language if relevant, otherwise null
}}

---

Now, extract metadata from this input:

Previous Session (CHAT_HISTORY):
{previous}

Text:
{context}
