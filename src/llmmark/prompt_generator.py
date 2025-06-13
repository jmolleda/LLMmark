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
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompts_path}")
        
        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _register_prompts_in_opik(self):
        """
        Iterates over the loaded prompts and registers them in Opik,
        storing the resulting objects.
        """
        print("Registering prompts in the Opik library...")
        # Iterates over all languages and prompts defined in the YAML
        for lang, prompts in self.prompts_data.items():
            for key, template in prompts.items():
                opik_prompt_name = f"{key}_{lang}"

                # Opik prompt object
                opik_prompt_obj = opik.Prompt(
                    name=opik_prompt_name,
                    prompt=template
                )
                self.opik_prompts[opik_prompt_name] = opik_prompt_obj
        print("✅ Registration complete.")

    # Dentro de tu clase PromptGenerator en prompt_generator.py

    def get_prompt(self, prompt_key: str, question_type: str, **kwargs) -> str:
        """
        Gets a formatted prompt based on the provided key and question type.
        Combines a technique prompt with a base question prompt, formatting it
        with the provided arguments.

        Args:
            prompt_key (str): The key of the technique prompt (e.g. "S1", "R2").
            question_type (str): The key of the base question type (e.g. "open_answer").
            **kwargs: Arguments to format the final prompt (e.g. question="...", example="...").

        Returns:
            str: The combined and formatted prompt, ready for the LLM.
        """
        
        print(f"------ Generating prompt for key: {prompt_key}, question type: {question_type}, with args: {kwargs}")
        
        lang = self.settings.language
        
        try:
            technique_template = self.prompts_data[lang][prompt_key]
            
            print(f"Technique template for {prompt_key} in {lang}: {technique_template}")
            
        except KeyError:
            raise ValueError(f"Technique prompt with key '{prompt_key}' for language '{lang}' not found.")

        try:
            base_key = question_type
            base_prompt_template = self.prompts_data[lang][base_key]
        except KeyError:
            raise ValueError(f"Base prompt with key '{base_key}' for language '{lang}' not found.")

        if "{question}" in base_prompt_template:
            combined_template = base_prompt_template.replace("{question}", technique_template)
        else:
            combined_template = f"{base_prompt_template} {technique_template}"

        try:
            final_prompt = combined_template.format(**kwargs)
        except KeyError as e:
            print(f"Format error: {e}")
            print(f"Make sure to pass all necessary placeholders: {kwargs.keys()}")
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