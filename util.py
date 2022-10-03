
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

class NCreater :
    def __init__(self) :
        self.discriminator = load_model("model/discriminator.h5")
        self.generator = load_model("model/generator.h5")

    def __getGenImage(self, number) :
        noise = np.random.uniform(-1, 1, size=[1, 10])
        label = np.array([[0 for _ in range(10)]])
        label[0][number] = 1
        
        gen_image = self.generator.predict(np.concatenate([noise, label], axis=1))
        gen_image = gen_image.reshape(28, 28)

        return gen_image

    def create(self, number) :
        result = self.__getGenImage(int(str(number)[0]))
        
        for c in str(number)[1:] :
            temp = self.__getGenImage(int(c))
            result = np.concatenate([result, temp], axis=1)

        return result
