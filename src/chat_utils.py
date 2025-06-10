""" common utils used by multiple class modules """
import os
import re
import sys
import pickle
import json
from typing import NamedTuple

class RAGTag(NamedTuple):
    """
    namedtuple class constructor
      RAGTag(tag: str, content: str)
    """
    tag: str
    content: str

class CommonUtils():
    """ method holder for command methods used throughout the project """
    def __init__(self, console, **kwargs):
        self.history_dir = kwargs['vector_dir']
        if not os.path.exists(self.history_dir):
            try:
                os.makedirs(self.history_dir)
            except OSError:
                print(f'Unable to create directory: {self.history_dir}')
                sys.exit(1)

        self.chat_max = kwargs['chat_max']
        self.light_mode = kwargs['light_mode']
        self.debug = kwargs['debug']
        self.color = 245 if self.light_mode else 233

        # Heat Map
        self.console = console
        self.heat_map = 0
        self.prompt_map = self.create_heatmap(8000)
        self.cleaned_map = self.create_heatmap(1000)

        # Class variables
        self.chat_history_session = self.load_chat(self.history_dir)
        self.llm_prompt = self.load_prompt(self.history_dir)

        # Regular expression in use throughout the project
        self.find_prompt  = re.compile(r'(?<=[<m]eta_prompt: ).*?(?=[>)])', re.DOTALL)
        self.meta_data = re.compile(r'(?<=[<m]eta_tags: ).*?(?=[>)])', re.DOTALL)
        self.meta_iter = re.compile(r'(\w+):\s*([^;]*)')
        self.json_style = re.compile(r'```json(.*)```', re.DOTALL)
        self.json_template = re.compile(r'\{\{\s*(.*?)\s*\}\}', re.DOTALL)


        # Ephemeral scene tracking
        self._scene_meta = {
            # Core Spatial-Temporal Anchors
            'location': 'unknown',                  # e.g., 'Beast cockpit'
            'time': 'unknown',                      # 'day', 'night', 'dusk', 'morning'
            'status': 'unknown',                    # 'in motion', 'camped', 'combat'
            'entity_location': ['unknown'],         # john backseat, jane frontseat, bo navigation

            # Character Presence & Perspective
            'audience': [],                         # Characters being spoken to directly

            # Narrative Flow / Contextual Arc
            'narrative_arc': 'unspecified',         # e.g., 'john_trust_arc'
            'scene_type': 'unspecified',            # e.g., 'banter', 'flashback', 'combat',
            'tone': 'neutral',                      # e.g., 'playful', 'tense'
            'emotion': 'neutral',                   # emotion felt or scene tone
            'focus': ['conversation'],              # narrative function (dialogue, strategy, etc.)
            'narrative_arcs': set(['']),            # e.g., 'head_to_sector8', 'save_world'
            'completed_narrative_arcs': set(['']),  # e.g., 'talk_to_john', 'help_jane'

            # Environmental & Sensory Cues
            'weather': None,                        # e.g., 'storm approaching'
            'sensory_mood': None,                   # e.g., 'dim light and engine hum'

            # Mechanical/Narrative Nudges
            'user_choice': None,                    # User's last clear decision or action
            'last_object_interacted': None,         # e.g., 'radio', 'rifle', 'memory shard'

            # System Controls / Optional
            'time_jump_allowed': False,             # Can the LLM skip forward?
            'scene_locked': False,                  # Prevents new characters from entering scene
            'narrator_mode': False,                 # If True, LLM may use omniscient 3rd-person
        }
        # Load from file. If file does not exist then self.scene_meta == self._scene_meta above
        self.scene_meta = self.load_scene(self.history_dir)

    @staticmethod
    def parse_tags(tag_input: dict | list[tuple[str, str]]) -> list[RAGTag]:
        """Normalize any kind of tag input into RAGTag list."""
        tags = []
        for key, val in dict(tag_input).items():
            if val is None or (isinstance(val, str) and val.lower() in {'null', 'none', ''}):
                continue
            if isinstance(val, list):
                val = ",".join(str(v).strip() for v in val if v)
            tags.append(RAGTag(key.lower(), str(val).strip().lower()))
        return tags

    def scene_tracker_from_tags(self, tags: list[RAGTag]) -> str:
        """ Build a formatted scene state string based on incoming RAGTags and internal memory """
        tag_dict = {tag.tag: tag.content for tag in tags}
        # Allow for an update to self._scene_meta
        for key, value in self._scene_meta.items():
            if key not in self.scene_meta:
                self.scene_meta[key] = value
        scene = self.scene_meta.copy()
        for key, value in scene.items():
            if key not in self._scene_meta:
                self.scene_meta.pop(key)

        scene = self.scene_meta.copy()
        for key in scene:
            incoming = tag_dict.get(key)
            # Skip if incoming is invalid or empty
            if incoming in (None, 'none', 'unknown', [], '', {}, 'null'):
                continue
            # Handle list fields safely
            if isinstance(scene[key], list) and not isinstance(incoming, list):
                incoming = [incoming]
            # Handle unique scene tracking features (like missions and completed_missiones)
            if key == 'narrative_arcs':
                if isinstance(incoming, list):
                    scene[key].update(i for i in incoming if i and i.strip())
                elif isinstance(incoming, str) and incoming.strip():
                    scene[key].add(incoming.strip())
            elif key == 'completed_narrative_arcs':
                if isinstance(incoming, list):
                    scene[key].update(i for i in incoming if i and i.strip())
                elif isinstance(incoming, str) and incoming.strip():
                    scene[key].add(incoming.strip())
                try:
                    for _mission in scene[key]:
                        scene['completed_narrative_arcs'].remove(_mission)
                        scene['narrative_arcs'].remove(_mission)
                except KeyError:
                    pass
            else:
                scene[key] = incoming
        # Update internal memory with merged scene state
        self.scene_meta = scene  # No `.copy()` needed
        # Convert to string for LLM injection
        def stringify(k, v):
            if isinstance(v, (list, set)):
                clean = sorted(i for i in v if i and str(i).strip())
                return f'{k}=' + (', '.join(clean) if clean else 'none')
            if v in (None, '', [], {}, 'none', 'unknown'):
                return f'{k}=none'
            return f'{k}={v}'
        scene_str = '#SCENE_STATE: ' + '; '.join(stringify(k, v) for k, v in sorted(scene.items()))
        self.save_scene(self.history_dir, self.scene_meta)
        return scene_str

    def get_tags(self, response: str, debug=False) -> list[RAGTag]:
        """Extract tags from either JSON or meta_tag format in the LLM response."""
        try:
            # Check for JSON-style block
            json_match = self.json_style.search(response)
            if json_match:
                data = json.loads(json_match.group(1))
                return self.parse_tags(data)

            # Fallback to meta_tag format
            meta_matches = self.meta_data.findall(response)
            if meta_matches:
                flat_pairs = []
                for match in meta_matches:
                    flat_pairs.extend(self.meta_iter.findall(match))
                return self.parse_tags(flat_pairs)

            return []
        # pylint: disable=broad-exception-caught  # too many ways for this to go wrong
        except Exception as e:
            if debug:
                print(f'[get_tags error] {e}')
            return []

    @staticmethod
    def normalize_for_dedup(text: str) -> str:
        """ remove emojis and other markdown """
        text = re.sub(r'[\U0001F600-\U0001F64F\u2600-\u26FF\u2700-\u27BF]', '', text)
        #text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.lower().split())

    @staticmethod
    def stringify_lists(nested_list)->str:
        """ return a flat string """
        def process(item):
            result = []
            if isinstance(item, list):
                for subitem in item:
                    result.extend(process(subitem))
            else:
                result.append(str(item))
            return result
        flat_strings = process(nested_list)
        return '\n\n'.join(flat_strings)

    def create_heatmap(self, hot_max: int = 0, reverse: bool =False)->dict[int:int]:
        """
        Return a dictionary of ten color ascii codes (values) with the keys representing
        the maximum integer for said color code:
        ./heat_map(10) --> {0: 123, 1: 51, 2: 46, 3: 42, 4: 82, 5: 154,
                            6: 178, 7: 208, 8: 166, 9: 203, 10: 196}
        Options: reverse = True for oppisite effect
        """
        heat = {0: 123} # declare a zero
        colors = [51, 46, 42, 82, 154, 178, 208, 166, 203, 196]
        if self.light_mode:
            heat = {0: 21} # declare a zero
            colors = [19, 26, 30, 28, 65, 58, 94, 130, 124, 196]
        if reverse:
            colors = colors[::-1]
            heat = {0: 196} # declare a zero
        for i in range(10):
            x = int(((i+1)/10) * hot_max)
            heat[x] = colors[i]
        return heat

    @staticmethod
    def save_scene(history_path, scene):
        """ persist scenes """
        scene_file = os.path.join(history_path, 'ephemeral_scene.pkl')
        try:
            with open(scene_file, "wb") as f:
                pickle.dump(scene, f)
        except FileNotFoundError as e:
            print(f'Error saving chat. Check --history-dir\n{e}')

    def load_scene(self, history_path: str)->dict:
        """ Persist chat history (load) """
        scene_file = os.path.join(history_path, 'ephemeral_scene.pkl')
        try:
            with open(scene_file, "rb") as f:
                loaded_scene = pickle.load(f)
        except FileNotFoundError:
            with open(scene_file, "wb") as f:
                pickle.dump(self._scene_meta, f)
            return self._scene_meta
        except pickle.UnpicklingError as e:
            print(f'Scene file {scene_file} not a pickle file:\n{e}')
            sys.exit(1)
        # pylint: disable=broad-exception-caught  # so many ways to fail, catch them all
        except Exception as e:
            print(f'Warning: Error loading scene file: {e}')
        return loaded_scene

    @staticmethod
    def save_chat(history_path, chat_history) ->None:
        """ Persist chat history (save) """
        history_file = os.path.join(history_path, 'chat_history.pkl')
        try:
            with open(history_file, "wb") as f:
                pickle.dump(chat_history, f)
        except FileNotFoundError as e:
            print(f'Error saving chat. Check --history-dir\n{e}')

    def load_chat(self, history_path: str)->list:
        """ Persist chat history (load) """
        loaded_list = []
        history_file = os.path.join(history_path, 'chat_history.pkl')
        try:
            with open(history_file, "rb") as f:
                loaded_list = pickle.load(f)
                # trunacate to max ammount
                loaded_list = loaded_list[-self.chat_max:]
        except FileNotFoundError:
            pass
        except pickle.UnpicklingError as e:
            print(f'Chat history file {history_file} not a pickle file:\n{e}')
            sys.exit(1)
        # pylint: disable=broad-exception-caught  # so many ways to fail, catch them all
        except Exception as e:
            print(f'Warning: Error loading chat: {e}')
        return loaded_list

    def save_prompt(self, prompt)->str:
        """ Save the LLMs prompt, overwriting the previous one """
        prompt_file = os.path.join(self.history_dir, 'llm_prompt.pkl')
        try:
            with open(prompt_file, "wb") as f:
                pickle.dump(prompt, f)
        except FileNotFoundError as e:
            print(f'Error saving LLM prompt. Check --history-dir\n{e}')
        return prompt

    @staticmethod
    def load_prompt(history_path)->str:
        """ Persist LLM dynamic prompt (load) """
        prompt_file = os.path.join(history_path, 'llm_prompt.pkl')
        try:
            with open(prompt_file, "rb") as f:
                prompt_str = pickle.load(f)
        except FileNotFoundError:
            return ''
        except pickle.UnpicklingError as e:
            print(f'Chat history file {prompt_file} not a pickle file:\n{e}')
            sys.exit(1)
        # pylint: disable=broad-exception-caught  # so many ways to fail, catch them all
        except Exception as e:
            print(f'Warning: Error loading chat: {e}')
        return prompt_str

    def check_prompt(self, last_message):
        """ allow the LLM to add to its own system prompt """
        prompt = self.find_prompt.findall(last_message)[-1:]
        if prompt:
            prompt = self.stringify_lists(prompt)
            self.llm_prompt = self.save_prompt(prompt)
            if self.debug:
                self.console.print(f'PROMPT CHANGE: {self.llm_prompt}',
                                   style=f'color({self.color})', highlight=True)
