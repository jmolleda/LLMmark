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
        Obtiene y combina una plantilla base (por tipo de pregunta) con una 
        plantilla de técnica (por clave de prompt), y la formatea con los 
        argumentos proporcionados.

        Args:
            prompt_key (str): La clave de la técnica de prompting (ej. "S1", "R2").
            question_type (str): La clave del tipo de pregunta base (ej. "open_answer").
            **kwargs: Argumentos para formatear el prompt final (ej. question="...", example="...").

        Returns:
            str: El prompt combinado y formateado, listo para el LLM.
        """
        lang = self.settings.language
        
        # --- Paso 1: Obtener la plantilla de la técnica (ej. S1) ---
        try:
            technique_template = self.prompts_data[lang][prompt_key]
        except KeyError:
            raise ValueError(f"Prompt de técnica con clave '{prompt_key}' para el idioma '{lang}' no encontrado.")

        # --- Paso 2: Obtener la plantilla base (ej. open_answer) ---
        try:
            # El question_type que viene de main.py puede tener un formato como "open_answer_questions"
            # y en el yaml lo tienes como "open_answer". Lo normalizamos.
            base_key = question_type.replace('_questions', '')
            base_prompt_template = self.prompts_data[lang][base_key]
        except KeyError:
            raise ValueError(f"Prompt base con clave '{base_key}' para el idioma '{lang}' no encontrado.")

        # --- Paso 3: Combinar las plantillas ---
        # La plantilla de técnica (S1, R1, etc.) es la que define la estructura final.
        # La plantilla base (open_answer) actúa como una instrucción inicial que la envuelve.
        # Reemplazamos el {question} de la plantilla base con la plantilla de técnica completa.
        
        # Verificamos si la plantilla base tiene el placeholder {question} para evitar errores
        if "{question}" in base_prompt_template:
            combined_template = base_prompt_template.replace("{question}", technique_template)
        else:
            # Si no lo tiene, simplemente las unimos con un espacio.
            combined_template = f"{base_prompt_template} {technique_template}"

        # --- Paso 4: Formatear el prompt final con los datos reales (la pregunta, el ejemplo, etc.) ---
        # El método .format(**kwargs) reemplazará todos los placeholders que encuentre
        # en la plantilla combinada (como {question}, {example}, {info}) con los valores
        # que pases desde main.py.
        try:
            final_prompt = combined_template.format(**kwargs)
        except KeyError as e:
            print(f"Error de formato: Falta el argumento {e} para el prompt combinado.")
            print(f"Asegúrate de pasar todos los placeholders necesarios: {kwargs.keys()}")
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