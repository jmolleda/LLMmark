import json
import logging

logger = logging.getLogger(__name__)


class Statistics:
    def __init__(self):
        self.num_experiments = 0
        self.total_response_time = 0.0
        self.run_params = {}

    @property
    def average_response_time(self):
        return (
            (self.total_response_time / self.num_experiments)
            if self.num_experiments > 0
            else 0.0
        )

    def record_experiment(self, response_time: float):
        self.num_experiments += 1
        self.total_response_time += response_time

    def set_run_parameters(self, params: dict):
        self.run_params = params

    def log_statistics(self):
        logger.info("--- Experiment Summary (Generation Phase) ---")
        logger.info(f"Total LLM Calls: {self.num_experiments}")
        logger.info(f"Average Response Time: {self.average_response_time:.3f}s")
        logger.info(f"Total Response Time: {self.total_response_time:.3f}s")
        if self.run_params:
            logger.info("Run Parameters:")
            for key, value in self.run_params.items():
                logger.info(f"  {key}: {value}")

    def save_statistics(self, filename):
        data = {
            "num_llm_calls": self.num_experiments,
            "average_response_time_s": round(self.average_response_time, 3),
            "total_response_time_s": round(self.total_response_time, 3),
            "run_parameters": self.run_params
        }
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Statistics saved to {filename}")
        except IOError as e:
            logger.error(f"Failed to save statistics to {filename}: {e}")