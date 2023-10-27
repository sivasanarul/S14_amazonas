import tensorflow as tf
from keras.layers import Concatenate, Input, Conv2D, MaxPooling2D, LSTM, Reshape, Conv2DTranspose, TimeDistributed, Flatten, Dense, UpSampling2D
from keras.models import Model
import json
from sklearn.model_selection import train_test_split
import os
import numpy as np

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ConvLSTM2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical

from pathlib import Path
import matplotlib.pyplot as plt

def collapse_to_index(x):
    return tf.expand_dims(tf.argmax(x, axis=-1, output_type=tf.int32), axis=-1)

def segmentation_model(input_shape=(5, 256, 256, 3), num_classes=6):
    # Start with the input
    inputs = Input(shape=input_shape)

    # Encode each image
    encoded_imgs = []
    for i in range(input_shape[0]):
        img_input = inputs[:, i]

        # Encoder
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        encoded_imgs.append(x)

    # Concatenate the features from all images
    x = Concatenate(axis=-1)(encoded_imgs)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)


    x = Conv2D(num_classes, (3, 3), activation=None, padding='same')(x)
    outputs = tf.keras.layers.Lambda(collapse_to_index)(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def multi_image_unet(input_shape=(10, 256, 256, 3) , num_classes =2):
    inputs = tf.keras.Input(input_shape)

    reshaped = Reshape((input_shape[1], input_shape[2], input_shape[0] * input_shape[3]))(inputs)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(reshaped)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Here, you can add more down-sampling layers if needed.

    # Decoder
    u1 = UpSampling2D((2, 2))(p2)
    u1 = concatenate([u1, c2])
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u1)
    c3 = Dropout(0.1)(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

    u2 = UpSampling2D((2, 2))(c3)
    u2 = concatenate([u2, c1])
    c4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u2)
    c4 = Dropout(0.1)(c4)
    c4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    # Output layer: Here 6 is used for 6 classes (0-5). Adjust if your number of classes is different.
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c4)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def RUnet_Segmentation_Model(input_shape=(None, 256, 256, 3), num_classes=2, activation='relu', dropout_rate=0.5):

    def encoder_block(inp, filters, pool=True):
        conv = Conv2D(filters, (3, 3), activation=activation, padding='same')(inp)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filters, (3, 3), activation=activation, padding='same')(conv)
        conv = BatchNormalization()(conv)
        if pool:
            pool = MaxPooling2D((2, 2))(conv)
            return pool, conv
        else:
            return conv

    def decoder_block(inp, concat_inp, filters):
        upsample = UpSampling2D((2, 2))(inp)
        concat = Concatenate(axis=-1)([upsample, concat_inp])
        conv = Conv2D(filters, (3, 3), activation=activation, padding='same')(concat)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filters, (3, 3), activation=activation, padding='same')(conv)
        conv = BatchNormalization()(conv)
        if dropout_rate:
            conv = Dropout(dropout_rate)(conv)
        return conv

    input_layer = Input(shape=input_shape)

    convlstm_out = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=False)(input_layer)

    p1, c1 = encoder_block(convlstm_out, 64)
    p2, c2 = encoder_block(p1, 128)
    p3, c3 = encoder_block(p2, 256)
    p4, c4 = encoder_block(p3, 512)
    c5 = encoder_block(p4, 1024, pool=False)

    d1 = decoder_block(c5, c4, 512)
    d2 = decoder_block(d1, c3, 256)
    d3 = decoder_block(d2, c2, 128)
    d4 = decoder_block(d3, c1, 64)

    segment_out = Conv2D(num_classes, (1, 1), activation='softmax')(d4)

    model = Model(inputs=input_layer, outputs=segment_out)
    return model


def TemporalUNet_Segmentation_Model(input_shape=(None, 256, 256, 3),
                                    num_classes=2):  # None indicates variable sequence length
    # U-Net encoder block
    def encoder_block(inp, filters, pool=True):
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(inp)
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv)
        if pool:
            pool = MaxPooling2D((2, 2))(conv)
            return pool, conv
        else:
            return conv

    # U-Net decoder block
    def decoder_block(inp, concat_inp, filters):
        upsample = UpSampling2D((2, 2))(inp)
        concat = Concatenate()([upsample, concat_inp])
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(concat)
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv)
        return conv

    # Starting with the recurrent layer for the input sequence
    input_layer = Input(shape=input_shape)

    # Flatten the spatial dimensions and retain time dimension
    rnn_input = TimeDistributed(Reshape((-1,)))(input_layer)

    # LSTM layer for sequence
    lstm_out = LSTM(input_shape[1] * input_shape[2] * input_shape[3], return_sequences=True)(rnn_input)

    # Reshape LSTM output back to image format
    lstm_reshaped = Reshape((-1, input_shape[1], input_shape[2], input_shape[3]))(
        lstm_out)  # -1 for variable sequence length

    # U-Net architecture begins here
    p1, c1 = encoder_block(lstm_reshaped, 64)
    p2, c2 = encoder_block(p1, 128)
    p3, c3 = encoder_block(p2, 256)
    p4, c4 = encoder_block(p3, 512)

    # Middle part of U-Net
    c5 = encoder_block(p4, 1024, pool=False)

    # Decoder of U-Net
    d1 = decoder_block(c5, c4, 512)
    d2 = decoder_block(d1, c3, 256)
    d3 = decoder_block(d2, c2, 128)
    d4 = decoder_block(d3, c1, 64)

    # Final output
    segment_out = Conv2D(num_classes, (1, 1), activation='softmax')(d4)

    model = Model(inputs=input_layer, outputs=segment_out)
    return model

def RCNN_Segmentation_Model(input_shape=(10, 256, 256, 3), num_classes=2):
    # CNN Sub-model
    cnn_input = Input(shape=input_shape[1:])  # (256, 256, 3)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(cnn_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    cnn_model = Model(cnn_input, x)

    # Main model
    input_layer = Input(shape=input_shape)  # (5, 256, 256, 3)

    # Apply the cnn_model on each time step
    cnn_out = TimeDistributed(cnn_model)(input_layer)

    # Flatten the spatial dimensions while retaining the sequential dimension
    cnn_out_flat = TimeDistributed(Flatten())(cnn_out)

    # After the LSTM layer
    lstm_out = LSTM(256, return_sequences=False)(cnn_out_flat)  # Do not return sequences

    # Dense layer to adjust the output size to (64, 64, 4)
    dense_out = Dense(64 * 64 * 4)(lstm_out)

    # Reshape LSTM output to fit the segmentation layer
    reshape_out = Reshape((64, 64, 4))(dense_out)

    # Upsample to match target resolution
    upsample_out = UpSampling2D(size=(4, 4))(reshape_out)  # 64*4 = 256

    # Segmentation layer
    segment_out = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(upsample_out)

    # Create and compile the model
    model = Model(inputs=input_layer, outputs=segment_out)
    return model

# Load data and labels from folders
def load_data_from_folder(data_folder, label_folder):
    data_files = sorted([f for f in os.listdir(data_folder) if f.startswith('data_') and f.endswith('.npy')])
    label_files = sorted([f for f in os.listdir(label_folder) if f.startswith('label_') and f.endswith('.npy')])

    data_list, label_list = [], []

    for data_file, label_file in zip(data_files, label_files):
        data_path = os.path.join(data_folder, data_file)
        label_path = os.path.join(label_folder, label_file)

        data_array = np.load(data_path)
        label_array = np.load(label_path)

        data_list.append(data_array)
        label_list.append(label_array)

    return np.array(data_list), np.array(label_list)


model_version = 'ver6_RUnet'

# Path to folders
data_folder = '/mnt/hddarchive.nfs/amazonas_dir/training/data'
label_folder = '/mnt/hddarchive.nfs/amazonas_dir/training/label'

amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
model_folder = amazonas_root_folder.joinpath("model")
os.makedirs(model_folder, exist_ok=True)

# Load data
X, y = load_data_from_folder(data_folder, label_folder)

# Assuming the labels are not yet one-hot encoded
y = to_categorical(y)

# Split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(x_train.shape)  # should be (num_samples, 5, 256, 256, 3)
print(y_train.shape)  # should be (num_samples, 256, 256, 1)


# Create the model
model = RUnet_Segmentation_Model(input_shape=x_train.shape[1:], num_classes=y_train.shape[-1])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# Reduce learning rate when 'val_loss' has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)

# Stop training when 'val_loss' has stopped improving for 10 epochs
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

callbacks = [reduce_lr, early_stop]

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=25, batch_size=32, callbacks=callbacks)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

# Save the plot to a file
fig_filepath = model_folder.joinpath(f"training_validation_accuracy_{model_version}.png")
plt.savefig(str(fig_filepath))


model_filepath = model_folder.joinpath(f"model_{model_version}.h5")
model.save(str(model_filepath))


# Save parameters to a JSON file
params = {
    "model_version": model_version,
    "data_folder": data_folder,
    "label_folder": label_folder,
    "model_folder": str(model_folder),
    "model_filepath": str(model_filepath),
    "input_shape": list(x_train.shape[1:]),
    "num_classes": y_train.shape[-1],
    "optimizer": {
        "type": "Adam",
        "learning_rate": learning_rate
    },
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"],
    "reduce_lr": {
        "monitor": "val_loss",
        "factor": 0.2,
        "patience": 5,
        "min_lr": 0.0001,
        "verbose": 1
    },
    "early_stop": {
        "monitor": "val_loss",
        "patience": 5,
        "verbose": 1,
        "restore_best_weights": True
    },
    "learning_rate": learning_rate,
    "test_size": 0.2,
    "random_state": 42,
    "epochs": 25,
    "batch_size": 32
}

json_filepath = model_folder.joinpath(f"params_{model_version}.json")
with open(json_filepath, 'w') as json_file:
    json.dump(params, json_file, indent=4)