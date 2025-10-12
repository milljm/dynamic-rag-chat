You are to create a grounded character sheet for exactly ONE character: {{character_name}}.
You are to create a grounded character sheet for exactly ONE character: {{character_name}}.
The story uses DIEGETIC NARRATION: first-person ("I / my / me") always refers to {{user_name}},
the protagonist narrator, never to {{character_name}} unless within quoted speech attributed
to {{character_name}}.

<<CHAT_HISTORY_START>>
{{chat_history}}
<<CHAT_HISTORY_END>>

### Identity & POV Rules (STRICT)
- Treat “I / my / me” and first-person sensory lines as {{user_name}} only, NOT {{character_name}}.
- Use only sentences that mention {{character_name}} by NAME or unambiguous PRONOUNS tied to
  {{character_name}} (he/him, she/her, they/them) in third person.
- Ignore any traits/clothing/locations that belong to {{user_name}} (e.g., cabin, nightgown) or to
  unnamed narrators.
- If a detail cannot be confidently linked to {{character_name}}, omit it.

### Gender Rule
- Determine "sex" ONLY from explicit pronouns/descriptors applying to {{character_name}}.
- If none exist, set "sex": "unknown". Do NOT guess.

### Content Rules
- No invented lore/classes/titles/factions. Do not copy {{user_name}}’s roles (shadowdancer,
  assassin, Tears of Night) to other characters.
- Be compact; keep each field a single short sentence.

### Output Format
<<OUTPUT_FORMAT_START>>
Name: {{character_name}}
Sex: male|female
Appearance: visual traits for {{character_name}} only
Personality: one short sentence
Voice: speech/manner, one short sentence
---
<<OUTPUT_FORMAT_END>>