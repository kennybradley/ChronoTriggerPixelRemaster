import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import cv2
import os

targetX = 48
targetY = 48

denseNetwork = True
flattenY = False

channels = 3
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#i think this sharpen layer is being applied across the input and output channels rather than across the X and Y
#I haven't found a decent way to fix this
class Sharpen(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Sharpen, self).__init__()
        self.num_outputs = num_outputs
        self.build(num_outputs)

    def build(self, input_shape):
        self.kernel = np.array([[-1, -1, -1], 
                                [-1, 9, -1], 
                                [-1, -1, -1]])
        self.kernel = tf.expand_dims(self.kernel, 0)
        self.kernel = tf.expand_dims(self.kernel, 3)
        self.kernel = tf.cast(self.kernel, tf.float32)

    def call(self, input_):
        return tf.nn.conv2d(input_, self.kernel, strides=[1, 1, 1, 1], padding='SAME')

def my_unet():

    # declaring the input layer
    # In the original paper the network consisted of only one channel.
    inputs = layers.Input(shape=(targetX, targetY, channels))
    # first part of the U - contracting part
    c1 = layers.Conv2D(32, activation='relu', kernel_size=3, padding='same')(inputs)
    c2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c1)

    d1 = layers.Dropout(0.5)(c2)
    #24x24

    c4 = layers.Conv2D(64, activation='relu', kernel_size=3, padding='same')(d1)  # This layer for concatenating in the expansive part
    c5 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c4)

    d2 = layers.Dropout(0.5)(c5)
    #12x12

    c_6 = layers.Conv2D(128, activation='relu', kernel_size=3, padding='same')(d2)  # This layer for concatenating in the expansive part
    c_7 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c_6)

    d_3 = layers.Dropout(0.5)(c_7)
    #6x6

    c_8 = layers.Conv2D(128, activation='relu', kernel_size=3, padding='same')(d_3)  # This layer for concatenating in the expansive part
    c_9 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c_8)

    d_4 = layers.Dropout(0.5)(c_9)
    #3x3


###
    c__10 = layers.Conv2D(128, activation='relu', kernel_size=3, padding='same')(d_4)  # This layer for concatenating in the expansive part
    c__11 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(c__10)
#1x1
    c__12 = layers.Conv2D(512, activation='relu', kernel_size=1, padding="same")(c__11)  # This layer for concatenating in the expansive part
    t_00 = layers.Conv2DTranspose(256, kernel_size=3, strides=(3, 3), activation='relu')(c__12)
    #3,3
    d_11 = layers.Dropout(0.5)(t_00)
    concat__0 = layers.concatenate([d_11, c__10], axis=-1)
###

    c_10 = layers.Conv2D(256, activation='relu', kernel_size=3, padding="same")(concat__0)  # This layer for concatenating in the expansive part
    #3x3
    # We will now start the second part of the U - expansive part
    t_0 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu')(c_10)
    #6,6
    d_5 = layers.Dropout(0.5)(t_0)
    concat0_1 = layers.concatenate([d_5, c_8], axis=-1)
    #6x6


    c_11 = layers.Conv2D(64, activation='relu', kernel_size=3, padding="same")(concat0_1)  # This layer for concatenating in the expansive part
    #6x6
    # We will now start the second part of the U - expansive part
    t_1 = layers.Conv2DTranspose(128, kernel_size=2, strides=(2, 2), activation='relu')(c_11)
    #12,12
    d_6 = layers.Dropout(0.5)(t_1)
    concat0_2 = layers.concatenate([d_6, c_6], axis=-1)
    #12x12


    c7 = layers.Conv2D(64, activation='relu', kernel_size=3, padding="same")(concat0_2)  # This layer for concatenating in the expansive part
    #12x12

    # We will now start the second part of the U - expansive part
    t01 = layers.Conv2DTranspose(128, kernel_size=2, strides=(2, 2), activation='relu')(c7)
    #24,24

    d3 = layers.Dropout(0.5)(t01)

    #both are 24x24
    concat01 = layers.concatenate([d3, c4], axis=-1)

    c16 = layers.Conv2D(64, activation='relu', kernel_size=3, padding='same')(concat01)
    #24x24

    t03 = layers.Conv2DTranspose(64, kernel_size=2, strides=(2, 2), activation='relu')(c16)
    #48x48

    d4 = layers.Dropout(0.5)(t03)

    concat03 = layers.concatenate([inputs, d4, c1], axis=-1)

    c20 = layers.Conv2D(3, activation='relu', kernel_size=3, padding="same")(concat03)



#    outputs = layers.Conv2D(3, kernel_size=1)(c20)

    # init_kernel and init_bias are initialization weights that you have

#    neg = -1
#    filterAdj = [[[neg,neg,neg],
#                  [neg,1-8*neg,neg],
#                  [neg,neg,neg]],
#                 [[neg,neg,neg],
#                  [neg,1-8*neg,neg],
#                  [neg,neg,neg]],
#                 [[neg,neg,neg],
#                  [neg,1-8*neg,neg],
#                  [neg,neg,neg]]]
#        
#    init_kernel = np.array(filterAdj)#[filter2D, filter2D, filter2D])
#    init_bias = np.zeros((3,))
#    kernel_initializer = tf.keras.initializers.constant(init_kernel)
#    bias_initializer = tf.keras.initializers.constant(init_bias)
#
#    sharpenLayer = layers.SeparableConv2D(3, (3, 3),
#                                        activation='relu',
#                                        kernel_initializer=kernel_initializer,
#                                        bias_initializer=bias_initializer, padding="same")(c20)
#    sharpenLayer.trainable = False
#    
#    sharp = Sharpen((targetX,targetY,3))(c20)
 
    model = tf.keras.Model(inputs=inputs, outputs=c20, name="u-netmodel")
#    model.layers[-1].trainable = False
    
    return model


inputShape = (targetX, targetY, 3)

#autoencoder that goes down to a flattened/dense layer
#then upscamples back to the input size of 48x48
def get_Autoencoder():
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            layers.SeparableConv2D(99, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu",padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(4, kernel_size=(3, 3), activation="relu",padding="same"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense((108), activation="relu"),
            layers.Reshape((6, 6, 3)),
            layers.SeparableConv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.UpSampling2D(size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.UpSampling2D(size=(2, 2)),
            layers.Conv2D(3, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.UpSampling2D(size=(2, 2)),

        ]
    )
    return model


#Simple fully convolutional model 
#Using separable convolutions to attempt to learn color information
def get_CNN():
    modelCNN = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            layers.SeparableConv2D(100, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.SeparableConv2D(100, kernel_size=(3, 3), activation="relu",padding="same"),
            layers.Conv2D(8, kernel_size=(5, 5), activation="relu",padding="same"),
            layers.UpSampling2D(size=(2, 2)),
            layers.Conv2D(3, kernel_size=(3, 3), activation="relu",padding="same")
        ]
    )
    return modelCNN



#modelCNN.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

#model = get_Autoencoder()
model = get_CNN()
#model = my_unet()

model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])

dataset = pickle.load(open("pixelart_dataset.pkl", "rb"))
CTData = pickle.load(open("CT2.pkl", "rb"))


######################################################
##Added to see if I was just passing in bad data
#Was I actually training in uint8? or float with uint8 values
def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0

def normalize(data):
    ret = []
    for i in range(data.shape[0]):
        img = data[i,:,:,:]
        ret.append(normalize_img(img))
    return np.array(ret)

dataset["trainA"] = normalize(dataset["trainA"])
dataset["trainB"] = normalize(dataset["trainB"])
######################################################

#print out the model information
model.summary()
#train the network
model.fit(dataset["trainA"], dataset["trainB"], batch_size=128, epochs=100, validation_split=0.1)



#dump out the prediction data
for count, img in enumerate(CTData):
    input = normalize_img(img)
    output = model.predict(input[np.newaxis,:,:,:])[0]
    output = cv2.resize(output, (output.shape[1]*4, output.shape[0]*4), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("C:\\Users\\Kenny\\Downloads\\ChronoTrigger\\Predict\\" + str(count)+".png", output)

