import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Input, Conv2D, MaxPooling2D, LSTM, Reshape, Conv2DTranspose, TimeDistributed, Flatten, Dense, UpSampling2D
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
import os
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, Input

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


# Create and compile the model
model = segmentation_model(input_shape=(10, 256, 256, 3), num_classes=2)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_version = 'ver3'

# Path to folders
data_folder = '/mnt/ssdarchive.nfs/amazonas_dir/training/data'
label_folder = '/mnt/ssdarchive.nfs/amazonas_dir/training/label'

amazonas_root_folder = Path("/mnt/ssdarchive.nfs/amazonas_dir")
model_folder = amazonas_root_folder.joinpath("model")
os.makedirs(model_folder, exist_ok=True)

# Load data
X, y = load_data_from_folder(data_folder, label_folder)
y = y[..., np.newaxis]



# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)  # should be (num_samples, 5, 256, 256, 3)
print(y_train.shape)  # should be (num_samples, 256, 256, 1)

history = model.fit(x=X_train, y=y_train, batch_size=8, epochs=2, validation_data=(X_val, y_val))


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

