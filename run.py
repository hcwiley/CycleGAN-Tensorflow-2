import pylib as py
import tensorflow as tf
import tf2lib as tl
import os


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--model_dir')
py.arg('--input_dir')
py.arg('--output_dir')
py.arg('-v', '--verbose', type=bool, default=False)
args = py.args()

# print all the args we are running with
print('test.py args:')
for k, v in vars(args).items():
  print(k, v)

# look for the model
model_path = args.model_dir
# check if they gave us the model file or the directory
if os.path.isdir(model_path):
  pass
elif os.path.isfile(model_path):
  # get the directory from the model file
  model_path = os.path.dirname(model_path)

print('model_path: {}'.format(model_path))
if not os.path.exists(model_path):
  raise Exception('model not found: {}'.format(model_path))

new_model = tf.keras.models.load_model(model_path)

# if we are in verbose mode, print the model summary
if args.verbose:
  new_model.summary()

# get all the input images
input_images = py.glob(args.input_dir, '*.jpg')

# check if we have any images
if len(input_images) == 0:
  raise Exception('no images found in: {}'.format(args.input_dir))

input_shape = new_model.input.shape
# delete the first entry from input_shape to drop the null
input_shape = input_shape[1:]
input_width = input_shape[0]
input_height = input_shape[1]
input_channels = input_shape[2]

def coPilotHallucinate(input_image):
  # load the image
  image = tf.io.read_file(input_image)
  image = tf.image.decode_jpeg(image, channels=input_channels)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [input_width, input_height])
  image = tf.expand_dims(image, 0)

  # run the model
  output_image = new_model(image)
  output_image = tf.squeeze(output_image, 0)

  # save the image
  output_image = tf.image.convert_image_dtype(output_image, tf.uint8)
  output_image = tf.image.encode_jpeg(output_image)
  output_image_path = os.path.join(args.output_dir, os.path.basename(input_image))
  tf.io.write_file(output_image_path, output_image)

  print('wrote: {}'.format(output_image_path))

# iterate through all the images
for input_image in input_images:
  coPilotHallucinate(input_image)