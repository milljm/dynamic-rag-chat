# Entities

Place any information that will match characters in your stories that which should contain immutable information about said character. Example:

john.txt:
### John's Character Sheet
Name: John, male
Class: Fighter (level 12)
Relationships: Has a daughter named Jane.
Favorite Food: etc...

Place `john.txt` in this folder. Now, whenever the pre-processor encounters the name 'John', the system will discover this file and load it as part of the system prompt as immutable information about said character.

This is useful to ground key facts about characters, keeping your LLM experience grounded.