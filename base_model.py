import tensorflow as tf
import os
import os.path

class BaseModel():
    def imitate(self, source, polyak=0.995):
        new_weights = [polyak * s + (1-polyak) * t for s, t in zip(self.model.get_weights(), source.model.get_weights())]
        self.model.set_weights(new_weights)

    def save(self):
        self.model.save_weights(self.checkpoint_file)

    def restore(self):
        if os.path.exists(self.checkpoint_file):
            self.model.load_weights(self.checkpoint_file)

    @property
    def get_checkpoint_file(self):
        return self.checkpoint_file
