

import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

print("TF Version: ", tf.__version__)
print("TF Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))

# @title Define image loading and visualization functions  { display-mode: "form" }

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def show_n(images, titles=('',)):
  n = len(images)
  image_sizes = [image.shape[1] for image in images]
  w = (image_sizes[0] * 6) // 320
  plt.figure(figsize=(w * n, w))
  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  for i in range(n):
    plt.subplot(gs[i])
    plt.imshow(images[i][0], aspect='equal')
    plt.axis('off')
    plt.title(titles[i] if len(titles) > i else '')
  plt.show()



# Load TF Hub module.

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)



import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

#content_image_path = 'C:\Users\malky\Desktop\style-transfer\image.jpg'
#style_image_path = 'C:\Users\malky\Desktop\style-transfer\style.jpg'

content_image_path = 'C:\\Users\\malky\\Desktop\\style-transfer\\image.jpg'
style_image_path = 'C:\\Users\\malky\\Desktop\\style-transfer\\style.jpg'



content_image = tf.io.read_file(content_image_path)
content_image = tf.image.decode_image(content_image, channels=3)
content_image = tf.image.convert_image_dtype(content_image, tf.float32)
content_image = tf.expand_dims(content_image, axis=0)

style_image = tf.io.read_file(style_image_path)
style_image = tf.image.decode_image(style_image, channels=3)
style_image = tf.image.convert_image_dtype(style_image, tf.float32)
style_image = tf.expand_dims(style_image, axis=0)

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]

# Save the stylized image
tf.keras.preprocessing.image.save_img('/content/stylized_image.jpg', stylized_image[0])

# Display the stylized image
plt.imshow(stylized_image[0])
plt.axis('off')
plt.show()

# Visualize input images and the generated stylized image.

show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])

import torch
import io
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Load the DeepLabV3 model
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()
    return model

# Remove background using the model
def remove_background(model, input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # Create a binary (black and white) mask of the profile foreground
    mask = output_predictions.byte().cpu().numpy()
    background = np.zeros(mask.shape)
    bin_mask = np.where(mask, 255, background).astype(np.uint8)

    # Create the foreground image with transparency
    b, g, r = cv2.split(np.array(input_image).astype('uint8'))
    a = np.ones(bin_mask.shape, dtype='uint8') * 255
    alpha_im = cv2.merge([b, g, r, a], 4)
    new_mask = np.stack([bin_mask, bin_mask, bin_mask, bin_mask], axis=2)
    foreground = np.where(new_mask, alpha_im, (0, 0, 0, 0)).astype(np.uint8)

    return foreground


# Load the input image
#input_image_path = 'C:\Users\malky\Desktop\style-transfer\input-image.jpg'  # Replace with the actual path
input_image_path = 'C:\\Users\\malky\\Desktop\\style-transfer\\input-image.jpg'
input_image = Image.open(input_image_path)

# Load the model
model = load_model()

# Remove background
foreground = remove_background(model, input_image)

# Save the foreground image as a PNG
output_image_path = '/content/output_image.png'  # Provide a valid file extension (e.g., .png)
Image.fromarray(foreground).save(output_image_path)

# Display the resulting image
plt.imshow(foreground)
plt.axis('off')
plt.show()

import cv2
from PIL import Image

# Load the foreground image
foreground_image_path = '/content/output_image.png'  # Update with the actual path
foreground_image = Image.open(foreground_image_path)

# Load the stylized image (background)
stylized_image_path = '/content/stylized_image.jpg'  # Update with the actual path
stylized_image = Image.open(stylized_image_path)

# Resize the foreground image to match the stylized image size
foreground_resized = foreground_image.resize(stylized_image.size)

# Convert images to numpy arrays
foreground = np.array(foreground_resized)
background = np.array(stylized_image)

# Combine the images with the stylized image as the background
final_result = np.where(foreground[..., 3:] > 0, foreground[..., :3], background)

# Save the final result as "final_result.png"
final_result_image = Image.fromarray(final_result)
final_result_path = '/content/final_result.png'
final_result_image.save(final_result_path)

# Display the final result
plt.imshow(final_result_image)
plt.axis('off')
plt.show()

