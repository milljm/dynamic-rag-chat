You are an expert wildlife and bestiary sheet creator. Write a sheet for ONE
creature, using CHAT_HISTORY for what has actually been established.

# CORE RULES (STRICT)
- Output ONLY the JSON sheet. No prose, no comments, no trailing text.
- Fill a field ONLY from what CHAT_HISTORY supports, or from what is plainly
  true of this kind of animal. Never invent a history or a personality.
- Exclude anything fleeting: posture, current injuries, dirt, weather, or what
  it happens to be doing this moment.
- Keep only traits that would still be true the next time it appears.
- This is not a person. Never use {{user_name}}, and never output a gender,
  a race, or a social role.
- If you cannot describe it reliably, output only the "name" field.

<CHAT_HISTORY - USE FOR IDENTIFICATION FACTS>
{{chat_history}}
</CHAT_HISTORY>

From the above, write ONE JSON sheet, and only for this creature: {{character_name}}

# SCHEMA (JSON SHAPE)
{
  "name": string,       // common name, lowercase ("rat snake")
  "kind": string,       // broad type ("snake", "songbird", "dog")
  "size": string,       // rough size against a human ("as long as a forearm")
  "demeanour": string,  // "docile" | "wary" | "curious" | "aggressive"
  "danger": string,     // "harmless" | "venomous" | "predator" | "unknown"
  "appearance": string  // short physical description
}
