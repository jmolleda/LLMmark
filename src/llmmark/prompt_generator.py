from typing import Dict
import yaml
import opik
from pathlib import Path
from .settings import Settings
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class PromptGenerator:
    def __init__(self, settings: Settings, prompts_path: str = "prompts.yaml"):
        self.settings = settings
        self.prompts_path = Path(__file__).resolve().parent.parent.parent / prompts_path
        self.prompts_data = self._load_prompts_from_yaml()
        self.opik_prompts = {}
        self._register_prompts_in_opik()
        self.retriever = None
        self.rag_chain = None
        self.loaded_document_path = None

    def _load_prompts_from_yaml(self) -> dict:
        if not self.prompts_path.exists():
            logger.error(f"Prompt file not found: {self.prompts_path}")
            raise FileNotFoundError(f"Prompt file not found: {self.prompts_path}")
        with open(self.prompts_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _register_prompts_in_opik(self):
        logger.info("Registering prompts in the Opik library...")
        for lang, prompts in self.prompts_data.items():
            for key, template in prompts.items():
                if key == "evaluation" and isinstance(template, dict):
                    for sub_key, sub_template in template.items():
                        opik_prompt_name = f"{key}_{sub_key}_{lang}"
                        if isinstance(sub_template, str):
                            opik_prompt_obj = opik.Prompt(
                                name=opik_prompt_name, prompt=sub_template
                            )
                            self.opik_prompts[opik_prompt_name] = opik_prompt_obj
                        else:
                            logger.warning(
                                f"Skipping prompt '{sub_key}' in '{key}' due to unexpected format."
                            )
                    continue

                opik_prompt_name = f"{key}_{lang}"
                if isinstance(template, dict):
                    prompt_text = f"System: {template.get('system', '')}\nUser: {template.get('user', '')}"
                elif isinstance(template, str):
                    prompt_text = template
                else:
                    logger.warning(f"Skipping prompt '{key}' due to unexpected format.")
                    continue
                self.opik_prompts[opik_prompt_name] = opik.Prompt(
                    name=opik_prompt_name, prompt=prompt_text
                )
        logger.info("✅ Prompt registration complete.")

    def _initialize_rag_chain(self, document_path: Path):
        """Initializes the RAG chain for a given document.

        Args:
            document_path (Path): The path to the document to use for RAG.
        """
        if document_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(document_path))
        elif document_path.suffix.lower() == ".tex":
            loader = TextLoader(str(document_path), encoding="cp1252")
        else:
            logger.error(f"Unsupported document type: {document_path.suffix}")
            return

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Gemini embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        #k = 3 as number of chunks to retrieve
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        self.loaded_document_path = document_path
        
        logger.info(f"Initialized RAG chain with {len(splits)} document splits from {document_path.name}")


    def get_system_prompt(self, prompt_key: str, question_type: str, **kwargs) -> str:
        lang = self.settings.language
        try:
            technique_system = self.prompts_data[lang][prompt_key].get("system", "")
            base_prompt = self.prompts_data[lang][question_type]
            base_system = (
                base_prompt.get("system", "")
                if isinstance(base_prompt, dict)
                else base_prompt
            )
            combined = f"{base_system} {technique_system}".strip()
            return combined.format(**kwargs)
        except KeyError as e:
            logger.error(
                f"Prompt key not found: {e}. Check prompts.yaml for lang='{lang}', key='{prompt_key}', type='{question_type}'"
            )
            raise

    def get_user_prompt(self, prompt_key: str, **kwargs) -> str:
        lang = self.settings.language
        try:
            template = self.prompts_data[lang][prompt_key]["user"]
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(
                f"User prompt key not found: {e}. Check prompts.yaml for lang='{lang}', key='{prompt_key}'"
            )
            raise

    def get_evaluation_prompts(self) -> Dict[str, str]:
        try:
            prompts = self.prompts_data["en"]["evaluation"]
            return {
                "task_introduction": prompts["task_introduction"],
                "evaluation_criteria": prompts["evaluation_criteria"],
            }
        except KeyError as e:
            logger.error(f"Evaluation prompt key not found in prompts.yaml: {e}")
            raise

    def get_few_shot_examples(self, question_folder: str) -> str:
        file_path = Path(question_folder) / self.settings.few_shot_examples_file
        if not file_path.exists():
            logger.warning(
                f"Few-shot file not found at '{file_path}'. Proceeding without examples."
            )
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f).get(self.settings.language, {})
            examples = "\n\n".join(
                f"Q: {ex.get('q')}\nA: {ex.get('a')}"
                for ex in data.values()
                if "q" in ex and "a" in ex
            )
            if examples:
                logger.info(f"Loaded few-shot examples from: {file_path}")
            return examples
        except Exception as e:
            logger.error(f"Error loading few-shot examples from {file_path}: {e}")
            return ""

    def get_reasoning_information(self):
        lang = self.settings.language
        try:
            template = self.prompts_data[lang]["reasoning_instructions"]
            return template
        except KeyError as e:
            logger.error(
                f"User prompt key not found: {e}. Check prompts.yaml for lang='{lang}', key='reasoning_info'"
            )
            raise
    
    def get_information(self, question_folder: str, question: str) -> str:
        """
        Retrieves relevant information for a given question using a RAG pipeline.
        Initializes the pipeline on the first call for a given document.
        """
        info_file = Path(question_folder) / self.settings.information_file
        if not info_file.exists():
            logger.warning(f"Information file not found at '{info_file}'. Proceeding without context.")
            return ""

        try:
            with open(info_file, "r", encoding="utf-8") as f:
                chapter_filename = yaml.safe_load(f).get("chapter_file")
            
            if not chapter_filename:
                return ""

            chapter_file = (
                Path(self.settings.context_chapters_path)
                / self.settings.language
                / chapter_filename
            )

            if not chapter_file.exists():
                logger.error(f"Chapter file '{chapter_file}' does not exist.")
                return ""

            # Initialize the RAG chain
            if not self.retriever:
                self._initialize_rag_chain(chapter_file)
            
            # Relevant documents related to the question
            logger.info(f"Retrieving context for question: {question} from {chapter_file}")
            if self.retriever:
                docs = self.retriever.invoke(question)
                # Combine the content of the retrieved documents
                context = "\n\n".join([doc.page_content for doc in docs])
                logger.info(f"Retrieved {len(docs)} relevant chunks for the question from {chapter_file}.")
                return context
            else:
                return ""

        except Exception as e:
            logger.error(f"Error loading or retrieving context information: {e}")
            return ""

    def get_opik_prompt_object(self, prompt_key: str) -> opik.Prompt:
        name = f"{prompt_key}_{self.settings.language}"
        prompt_obj = self.opik_prompts.get(name)
        if not prompt_obj:
            raise ValueError(f"Opik.Prompt object '{name}' was not registered.")
        return prompt_obj