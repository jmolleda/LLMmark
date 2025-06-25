import json
import logging

logger = logging.getLogger(__name__)


class Statistics:
    def __init__(self):
        self.num_experiments = 0
        self.num_correct = 0
        self.num_incorrect = 0
        self.total_response_time = 0.0

    @property
    def accuracy(self):
        total = self.num_correct + self.num_incorrect
        return (self.num_correct / total) * 100 if total > 0 else 0.0

    @property
    def average_response_time(self):
        return (
            (self.total_response_time / self.num_experiments)
            if self.num_experiments > 0
            else 0.0
        )

    def record_experiment(self, correct: bool, response_time: float):
        self.num_experiments += 1
        if correct:
            self.num_correct += 1
        else:
            self.num_incorrect += 1
        self.total_response_time += response_time

    def log_statistics(self):
        logger.info("--- Experiment Summary ---")
        logger.info(f"Total Runs: {self.num_experiments}")
        # logger.info(f"Correct: {self.num_correct}")
        # logger.info(f"Incorrect: {self.num_incorrect}")
        # logger.info(f"Accuracy: {self.accuracy:.2f}%")
        logger.info(f"Average Response Time: {self.average_response_time:.3f}s")
        logger.info(f"Total Response Time: {self.total_response_time:.3f}s")

    def save_statistics(self, filename):
        data = {
            "num_runs": self.num_experiments,
            "num_correct": self.num_correct,
            "num_incorrect": self.num_incorrect,
            "accuracy_percent": round(self.accuracy, 2),
            "average_response_time_s": round(self.average_response_time, 3),
            "total_response_time_s": round(self.total_response_time, 3),
        }
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Statistics saved to {filename}")
        except IOError as e:
            logger.error(f"Failed to save statistics to {filename}: {e}")
