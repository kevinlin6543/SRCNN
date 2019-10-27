import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
import cv2
import prepare_data as pd
import numpy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def psnr(target, ref):
    # assume RGB image
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(numpy.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


class Model(tf.Module):
    def __init__(self, input_shape=(32, 32, 1)):
        self.SRCNN = Sequential()
        self.SRCNN.add(Conv2D(filters=64, kernel_size=[9, 9], strides=[1, 1], input_shape=input_shape))
        self.SRCNN.add(tf.keras.layers.ReLU())
        self.SRCNN.add(Conv2D(filters=32, kernel_size=[1, 1], strides=[1, 1]))
        self.SRCNN.add(tf.keras.layers.ReLU())
        self.SRCNN.add(Conv2D(filters=1, kernel_size=[5, 5], strides=[1, 1]))
        self.SRCNN.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mean_squared_error'])

    def __str__(self):
        return self.SRCNN.summary();


def model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(filters=64, kernel_size=[9, 9], strides=[1, 1], input_shape=(32, 32, 1)))
    SRCNN.add(tf.keras.layers.ReLU())
    SRCNN.add(Conv2D(filters=32, kernel_size=[1, 1], strides=[1, 1]))
    SRCNN.add(tf.keras.layers.ReLU())
    SRCNN.add(Conv2D(filters=1, kernel_size=[5, 5], strides=[1, 1]))
    SRCNN.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def predict_model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(filters=64, kernel_size=[9,9], strides=[1,1], input_shape=(None, None, 1)))
    SRCNN.add(tf.keras.layers.ReLU())
    SRCNN.add(Conv2D(filters=32, kernel_size=[1,1], strides=[1,1]))
    SRCNN.add(tf.keras.layers.ReLU())
    SRCNN.add(Conv2D(filters=1, kernel_size=[5,5], strides=[1,1]))
    SRCNN.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def train():
    srcnn_model = Model()
    print(srcnn_model)

    data, label = pd.read_training_data("./train.h5")

    filepath = "./checkpoint/saved-model-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath, save_best_only=False,
                                 save_weights_only=False, mode='min', period=50)
    callbacks_list = [checkpoint]

    srcnn_model.fit(data, label, batch_size=128,
                    callbacks=callbacks_list, shuffle=True, epochs=200)


def predict():
    srcnn_model = predict_model(input_shape=(None, None, 1))
    srcnn_model.load_weights("./checkpoint/saved-model-200.h5")

    IMG_NAME = "./Test/Temp/butterfly_GT.bmp"
    INPUT_NAME = "./result/bicubic.png"
    OUTPUT_NAME = "./result/SRCNN.png"

    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    shape = img.shape
    Y_img = cv2.resize(img[:, :, 0], (shape[1] // 2, shape[0] // 2), cv2.INTER_CUBIC)
    Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    img[:, :, 0] = Y_img
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(INPUT_NAME, img)

    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.
    pre = srcnn_model.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(numpy.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)

    # psnr calculation:
    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]

    original = mpimg.imread(IMG_NAME)
    bicubic = mpimg.imread(INPUT_NAME)
    bicubic_snr = cv2.PSNR(im1, im2)
    srcnn = mpimg.imread(OUTPUT_NAME)
    srcnn_snr = cv2.PSNR(im1, im3)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(original)
    axs[0, 0].set_title('Original / PSNR')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(bicubic)
    axs[0, 1].set_title('Bicubic / %.2f dB' % (bicubic_snr))
    axs[0, 1].axis('off')

    axs[1, 1].imshow(srcnn)
    axs[1, 1].set_title('SRCNN / %.2f dB' % (srcnn_snr))
    axs[1, 1].axis('off')
    plt.show()


if __name__ == "__main__":
    train()
    predict()
