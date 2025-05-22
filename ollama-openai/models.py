class Models:
    def __init__(self):
        self.models = [
            # (display_name, model_id)
            ("Gemma3:1b", "gemma3:1b"),
            ("Moondream 2", "moondream"),
        ]

    def get_models(self):
        return self.models