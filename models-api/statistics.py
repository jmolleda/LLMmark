import json

class Statistics:
    def __init__(self):
        self.num_experiments = 0
        self.num_correct = 0
        self.num_incorrect = 0
        self.total_response_time = 0.0

    @property
    def accuracy(self):
        total = self.num_correct + self.num_incorrect
        return (self.num_correct / total) if total > 0 else 0.0

    @property
    def average_response_time(self):
        return (self.total_response_time / self.num_experiments) if self.num_experiments > 0 else 0.0

    def record_experiment(self, correct: bool, response_time: float):
        self.num_experiments += 1
        if correct:
            self.num_correct += 1
        else:
            self.num_incorrect += 1
        self.total_response_time += response_time

    def print_statistics(self):
        print(f"Experiment summary:")
        print(f"  Experiments: {self.num_experiments}")
        print(f"  Correct: \033[92m{self.num_correct}\033[0m")
        print(f"  Incorrect: \033[91m{self.num_incorrect}\033[0m")
        print(f"  Accuracy: {self.accuracy:.2f}")
        print(f"  Average Response Time: {self.average_response_time:.2f}s")

    def save_statistics(self, filename):
        data = {
            "num_experiments": self.num_experiments,
            "num_correct": self.num_correct,
            "num_incorrect": self.num_incorrect,
            "accuracy": round(self.accuracy, 2),
            "average_response_time": round(self.average_response_time, 2),
            "total_response_time (s.)": round(self.total_response_time, 3),
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)