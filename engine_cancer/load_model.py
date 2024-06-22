import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TFLite model and allocate tensors
# interpreter = tf.lite.Interpreter(model_path="model4.tflite")
# interpreter.allocate_tensors()

# # Get input and output tensors
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# Load and preprocess the image
def load_and_preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_shape[1], input_shape[2]))
    image = np.array(image, dtype=np.float32)
    # Normalize the image to the range [0, 1] or the range the model expects
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# image_path = 'data.png'
# input_shape = input_details[0]['shape']
# input_data = load_and_preprocess_image(image_path, input_shape)

# # Set the input tensor
# interpreter.set_tensor(input_details[0]['index'], input_data)

# # Run the inference
# interpreter.invoke()

# # Get the output tensor
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print("Output:", output_data)

# # Process the output (example for classification)
# predicted_class = np.argmax(output_data)
# if predicted_class == 1:
#     print("EarlyPreB")
# elif predicted_class == 2:
#     print('PreB')
# elif predicted_class == 3:
#     print("ProB")
# else:
#     print("Begin")