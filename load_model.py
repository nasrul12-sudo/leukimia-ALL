import numpy as np
from PIL import Image

def load_and_preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_shape[1], input_shape[2]))
    image = np.array(image, dtype=np.float32)
    # Normalize the image to the range [0, 1] or the range the model expects
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image
