
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import numpy as np

class NCreater :
    def __init__(self) :
        self.discriminator = load_model("model/discriminator.h5")
        self.generatoor = load_model("generator.h5")

    def __getGenImage(self, number) :
        noise = np.random.uniform(-1, 1, size=[1, 10])
        label = np.array([[0 for _ in range(9)]])
        label[0][number] = 1
        
        gen_image = self.generator.predict(np.concatenate([noise, label], axis=1))
        gen_image = gen_image.reshape(-1, 28, 28)

        return gen_image

    def __addImage(self, image1, image2) :
        pass

    def create(self, number) :
        pass