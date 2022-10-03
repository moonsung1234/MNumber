
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Input, Concatenate
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import os

BATCH_SIZE = 128
EPOCHS = 50
NOISE = 10
LABEL = 10

# preprocessing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / (255 / 2) - 1
x_test = x_test / (255 / 2) - 1

y_train= to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

if os.path.isfile("discriminator.h5") and os.path.isfile("generator.h5") :
    discriminator = load_model("discriminator.h5")
    generator = load_model("generator.h5")

    # show
    plt.figure(figsize=(8, 4))

    noise = np.random.uniform(-1, 1, size=[10, NOISE])
    labels = to_categorical(np.array([9 for i in range(10)]))
    gen_images = generator.predict(np.concatenate([noise, labels], axis=1))
    gen_images = gen_images.reshape(-1, 28, 28)

    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(gen_images[i], interpolation="nearest", cmap="gray")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

else :
    adam = Adam(lr=0.0002, beta_1=0.5)

    generator = Sequential([
        Dense(256, input_shape=(NOISE + LABEL,)),
        LeakyReLU(0.2),
        Dense(512),
        LeakyReLU(0.2),
        Dense(1024),
        LeakyReLU(0.2),
        Dense(28 * 28, activation="tanh")
    ])

    discriminator = Sequential([
        Dense(1024, input_shape=(28 * 28 + LABEL,)),
        LeakyReLU(0.2),
        Dropout(0.3),
        Dense(512),
        LeakyReLU(0.2),
        Dropout(0.3),
        Dense(256),
        LeakyReLU(0.2),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    # discriminator
    discriminator.compile(loss="binary_crossentropy", optimizer=adam)

    # gan
    discriminator.trainable = False
    gen_input = Input(shape=(NOISE + LABEL,))
    x = generator(gen_input)
    label_input = Input(shape=(LABEL,))
    dis_input = Concatenate(axis=1)([x, label_input])
    output = discriminator(dis_input)

    gan = Model([gen_input, label_input], output)
    gan.compile(loss="binary_crossentropy", optimizer=adam)

    # batch
    batches_x = []
    batches_y = []

    for i in range(int(x_train.shape[0] // BATCH_SIZE)):
        batch_x = x_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        batch_y = y_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        
        batches_x.append(batch_x)
        batches_y.append(batch_y)

    batches_x = np.asarray(batches_x)
    batches_y = np.asarray(batches_y)

    for i in range(EPOCHS) :
        loss_dis, loss_gen = None, None

        for j in range(len(batches_x)) :
            real_images = batches_x[j]
            labels = batches_y[j]

            # train discriminator
            noise = np.random.uniform(-1, 1, size=[BATCH_SIZE, NOISE])
            gen_images = generator.predict(np.concatenate([noise, labels], axis=1))
            x_dis = np.concatenate([
                np.concatenate([real_images, labels], axis=1), 
                np.concatenate([gen_images, labels], axis=1)
            ], axis=0)
            y_dis = np.concatenate([
                np.ones(BATCH_SIZE), 
                np.zeros(BATCH_SIZE)
            ], axis=0)

            discriminator.trainable = True
            loss_dis = discriminator.train_on_batch(x_dis, y_dis)

            # train generator
            noise = np.random.uniform(-1, 1, size=[BATCH_SIZE, NOISE])
            y_gen = np.ones(BATCH_SIZE)

            discriminator.trainable = False
            loss_gen = gan.train_on_batch([
                np.concatenate([noise, labels], axis=1), 
                labels
            ], y_gen)

        print("epoch : {}, discriminator loss : {}, generator loss : {}".format(i, loss_dis, loss_gen))

        # show
        # plt.figure(figsize=(8, 4))

        # noise = np.random.uniform(-1, 1, size=[10, NOISE])
        # labels = to_categorical(np.array([9 for i in range(10)]))
        # gen_images = generator.predict(np.concatenate([noise, labels], axis=1))
        # gen_images = gen_images.reshape(-1, 28, 28)

        # for i in range(10):
        #     plt.subplot(2, 5, i+1)
        #     plt.imshow(gen_images[i], interpolation="nearest", cmap="gray")
        #     plt.axis('off')
        
        # plt.tight_layout()
        # plt.show()

    generator.save("generator.h5")
    discriminator.save("discriminator.h5")
