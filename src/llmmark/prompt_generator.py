import yaml
import opik
from pathlib import Path
from .settings import Settings

class PromptGenerator:

    def __init__(self, settings: Settings, prompts_path: str = "../prompts.yaml"):
        self.settings = settings
        self.prompts_path = Path(prompts_path)
        self.prompts_data = self._load_prompts_from_yaml()
        
        # Opik prompts
        self.opik_prompts = {}
        self._register_prompts_in_opik()

    def _load_prompts_from_yaml(self) -> dict:
        """
        Loads prompts from a YAML file.

        Raises:
            FileNotFoundError: If the prompts file is not found.

        Returns:
            dict: The loaded prompts data.
        """
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompts_path}")
        
        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _register_prompts_in_opik(self):
        """
        Iterates over the loaded prompts and registers them in Opik,
        storing the resulting objects.
        Raises:
            ValueError: If a prompt has an unexpected format.
        """
        print("Registering prompts in the Opik library...")
        # Iterates over all languages and prompts defined in the YAML
        for lang, prompts in self.prompts_data.items():
            for key, template in prompts.items():
                opik_prompt_name = f"{key}_{lang}"

                prompt_text = ""
                if isinstance(template, dict):
                    system_prompt = template.get('system', '')
                    user_prompt = template.get('user', '')
                    # Unify system and user prompts into a single prompt text
                    prompt_text = f"System: {system_prompt}\nUser: {user_prompt}"
                elif isinstance(template, str):
                    prompt_text = template
                else:
                    print(f"Skipping prompt '{key}' in language '{lang}' due to unexpected format: {type(template)}")
                    continue

                # Opik prompt object
                opik_prompt_obj = opik.Prompt(
                    name=opik_prompt_name,
                    prompt=prompt_text
                )
                self.opik_prompts[opik_prompt_name] = opik_prompt_obj
        print("✅ Registration complete.")
        
        
    def get_system_prompt(self, prompt_key: str, question_type: str, **kwargs) -> str:
        """
        Gets a formatted system prompt based on the provided key and question type.
        It combines a base question prompt with a technique's system prompt,
        then formats it with the provided arguments.

        Args:
            prompt_key (str): The key of the technique prompt (e.g., "S1", "R2").
            question_type (str): The key of the base question type (e.g.,"open_answer" or "multiple_choice").
            **kwargs: Arguments to format the final prompt (e.g., question="...", example="...").

        Returns:
            str: The combined and formatted system prompt.
        """
        
        print(f"------ Generating system prompt for key: {prompt_key}, question type: {question_type} ------")
        
        lang = self.settings.language
        
        # System prompt
        try:
            technique_system_prompt = self.prompts_data[lang][prompt_key].get('system', '')
            if not isinstance(technique_system_prompt, str):
                raise TypeError("The 'system' part of the technique prompt is not a string.")
        except (KeyError, AttributeError):
            raise ValueError(f"Technique prompt with key '{prompt_key}' for language '{lang}' not found or is not a dictionary.")

        # Base prompt
        try:
            base_prompt_template = self.prompts_data[lang][question_type]
            
            # Handle if the base prompt is a dictionary or a simple string
            if isinstance(base_prompt_template, dict):
                # If it's a dict, get the 'system' part, or an empty string if it doesn't exist
                base_prompt_template = base_prompt_template.get("system", "")
            elif not isinstance(base_prompt_template, str):
                raise ValueError(f"Base prompt with key '{question_type}' for language '{lang}' has an invalid format.")
                
        except KeyError:
            raise ValueError(f"Base prompt with key '{question_type}' for language '{lang}' not found.")

        # Combine the base prompt and the technique's system prompt
        combined_template = f"{base_prompt_template} {technique_system_prompt}".strip()

        try:
            # Format the final combined prompt with the provided arguments
            final_prompt = combined_template.format(**kwargs)
        except KeyError as e:
            print(f"Formatting error in system prompt: Missing placeholder {e}")
            print(f"Available placeholders: {kwargs.keys()}")
            raise

        return final_prompt


    def get_user_prompt(self, prompt_key: str, **kwargs) -> str:
        """
        Gets a formatted user prompt based on the provided key.
        The user prompt is primarily defined by the 'user' part of the technique prompt.

        Args:
            prompt_key (str): The key of the technique prompt (e.g., "S1", "R2").
            **kwargs: Arguments to format the final prompt (e.g., question="...").

        Returns:
            str: The formatted user prompt
        """
        
        print(f"------ Generating user prompt for key: {prompt_key} ------")
        
        lang = self.settings.language
        
        # The user prompt is determined by the 'user' part of the technique prompt (e.g., S1, R2)
        try:
            user_prompt_template = self.prompts_data[lang][prompt_key]['user']
            if not isinstance(user_prompt_template, str):
                raise TypeError("The 'user' part of the technique prompt is not a string.")
        except (KeyError, AttributeError):
            raise ValueError(f"Technique prompt with key '{prompt_key}' for language '{lang}' not found or does not contain a 'user' key.")

        try:
            # Format the final prompt with the provided arguments
            final_prompt = user_prompt_template.format(**kwargs)
        except KeyError as e:
            print(f"Formatting error in user prompt: Missing placeholder {e}")
            print(f"Available placeholders: {kwargs.keys()}")
            raise

        return final_prompt

    def get_opik_prompt_object(self, prompt_key: str) -> opik.Prompt:
        """
        Returns the registered opik.Prompt object, useful for linking it to
        evaluation experiments.

        Args:
            prompt_key (str): The key of the prompt (e.g. "S1", "R2").

        Returns:
            opik.Prompt: The corresponding Prompt object.
        """
        lang = self.settings.language
        opik_prompt_name = f"{prompt_key}_{lang}"
        
        prompt_obj = self.opik_prompts.get(opik_prompt_name)
        if not prompt_obj:
            print(f"------------- {opik_prompt_name} not found in Opik prompts -------------")
            
            raise ValueError(f"Opik.Prompt object with name '{opik_prompt_name}' was not registered.")

        return prompt_obj