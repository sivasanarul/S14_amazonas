import onnx
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import load_model
import tf2onnx
from cnn_architectures import build_vgg16_segmentation_bn

# Load the HDF5 model
h5_file_path = '/mnt/hddarchive.nfs/amazonas_dir/model/model_best_build_vgg16_segmentation_batchingestion_labelmorethan120dataset_weighted_f1score.h5'  # Replace with your HDF5 model path

# Step 2: Rebuild the model architecture
model = build_vgg16_segmentation_bn((256, 256, 15))
model.load_weights(h5_file_path)

# Convert the Keras model to ONNX
spec = (tf.TensorSpec((None,) + model.input.shape[1:], tf.float32),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save the ONNX model
onnx_model_path = '/mnt/hddarchive.nfs/amazonas_dir/training/hdf5_folder/amazonas_ai_cnn.onnx'  # Replace with your desired ONNX file path
# Save the ONNX model
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Model successfully converted to ONNX and saved at {onnx_model_path}")
