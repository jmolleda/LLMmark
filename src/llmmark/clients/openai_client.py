from openai import OpenAI, APIError, InternalServerError
import time
import logging

logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        max_requests_per_minute=0,
        top_p=0.1,
        seed=27,
        max_retries=3,
    ):
        if not api_key:
            raise ValueError("API key must be provided for OpenAIClient.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_requests_per_minute = max_requests_per_minute
        self.top_p = top_p
        self.seed = seed
        self.max_retries = max_retries
        self.request_delay = (
            (60 / max_requests_per_minute) if max_requests_per_minute > 0 else 0
        )
        logger.info(f"OpenAI client initialized for base_url: {base_url}")

    def chat(self, model, messages, stream=False, temperature=0.7):
        for attempt in range(self.max_retries):
            try:
                if self.request_delay > 0:
                    logger.debug(
                        f"Delaying for {self.request_delay:.2f}s due to rate limit."
                    )
                    time.sleep(self.request_delay)

                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stream=stream,
                    seed=self.seed,
                    top_p=self.top_p,
                )

                if response.choices and response.choices[0].message:
                    return response.choices[0].message.content
                logger.warning("Received an empty or malformed response from the API.")
                return ""

            except InternalServerError as e:
                wait_time = 15 * (attempt + 1)
                logger.warning(
                    f"[{e.status_code} Server Error] Retrying in {wait_time}s... (Attempt {attempt + 1}/{self.max_retries})"
                )
                time.sleep(wait_time)
            except APIError as e:
                logger.error(f"OpenAI API Error on attempt {attempt + 1}: {e}")
                if attempt + 1 == self.max_retries:
                    break
                time.sleep(5 * (attempt + 1))
            except Exception as e:
                logger.error(f"An unexpected error occurred in OpenAI client: {e}")
                return ""

        logger.error("Max retries reached. Request failed.")
        return ""
