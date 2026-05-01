"""
Stage 3 train/test setting for image classification.
"""

from local_code.base_class.setting import setting


class Setting_Image_Classification(setting):
    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()

        self.method.data = loaded_data
        learned_result = self.method.run()

        self.evaluate.data = learned_result
        metrics = self.evaluate.evaluate()
        learned_result["metrics"] = metrics

        self.result.data = learned_result
        self.result.fold_count = None
        self.result.save()

        return metrics, None
