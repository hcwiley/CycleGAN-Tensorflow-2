import imlib as im
import pylib as py
import tensorflow as tf
import tf2lib as tl
import numpy as np
import os
import datetime
from printUtils import printProgressBar

import data

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--models_dir')
py.arg('--input_dir')
py.arg('--output_dir')
py.arg('--max_images', type=int, default=0)
py.arg('-v', '--verbose', type=bool, default=False)
py.arg('--use_G_A2B', type=bool, default=True)
args = py.args()

# print all the args we are running with
print('test.py args:')
for k, v in vars(args).items():
  print(k, v)

# look for the model
model_path = args.models_dir
# ensure they gave us a directory
if not os.path.isdir(model_path):
  print('error: models_dir must be a directory')
  sys.exit(1)

print('model_path: {}'.format(model_path))
if not os.path.exists(model_path):
  raise Exception('model not found: {}'.format(model_path))

G_A2B = tf.keras.models.load_model(os.path.join(model_path, 'A2B'))
G_B2A = tf.keras.models.load_model(os.path.join(model_path, 'B2A'))

# if we are in verbose mode, print the model summary
if args.verbose:
  G_A2B.summary()
  G_B2A.summary()

# get all the input images
input_images = py.glob(args.input_dir, '*.jpg')
output_dir = args.output_dir

if output_dir is None:
  # if the output dir is none use the ./tmp/DATE_TIME/ directory
  output_dir = os.path.join(
      './tmp', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
  print('output_dir: {}'.format(output_dir))

# check if we have any images
if len(input_images) == 0:
  raise Exception('no images found in: {}'.format(args.input_dir))

input_shape = G_A2B.input.shape
# delete the first entry from input_shape to drop the null
input_shape = input_shape[1:]
input_width = input_shape[0]
input_height = input_shape[1]
input_channels = input_shape[2]


def coPilotHallucinate(input_image):
  # github copilot hallucinated this whole function. it apparently works the same as the G_A2B so that's neat.

  # load the image
  image = tf.io.read_file(input_image)
  image = tf.image.decode_jpeg(image, channels=input_channels)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [input_width, input_height])
  image = tf.expand_dims(image, 0)

  # run the model
  output_image = G_A2B(image)
  output_image = tf.squeeze(output_image, 0)

  # save the image
  output_image = tf.image.convert_image_dtype(output_image, tf.uint8)
  output_image = tf.image.encode_jpeg(output_image)
  output_image_path = os.path.join(output_dir, os.path.basename(input_image))
  tf.io.write_file(output_image_path, output_image)

  print('wrote: {}'.format(output_image_path))


@tf.function
def sample_A2B(A):
  A2B = G_A2B(A, training=False)
  A2B2A = G_B2A(A2B, training=False)
  return A2B, A2B2A


@tf.function
def sample_B2A(B):
  B2A = G_B2A(B, training=False)
  B2A2B = G_A2B(B2A, training=False)
  return B2A, B2A2B


batch_size = 32
size = input_width
if input_height > input_width:
  size = input_height

i = 0
max_images = args.max_images
if max_images == 0:
  max_images = len(input_images)

total_predict_time = 0
predict_time_avg = 0
# iterate through all the images
for input_image in input_images:
  if args.use_G_A2B:
    # benchmark time for
    start_time = datetime.datetime.now()
    A_dataset = data.make_dataset([input_image], batch_size, size, size,
                                  training=False, drop_remainder=False, shuffle=False, repeat=1)
    A = A_dataset.get_single_element(0)
    A2B, A2B2A = sample_A2B(A)
    zipped = zip(A, A2B, A2B2A)
    A_i, A2B_i, A2B2A_i = next(zipped)
    img = np.concatenate([A_i.numpy(), A2B_i.numpy(), A2B2A_i.numpy()], axis=1)
    # clip the range -1 to 1
    img = np.clip(img, -1, 1)
    output_image_path = os.path.join(output_dir, os.path.basename(input_image))
    # make sure the output directory exists
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    im.imwrite(img, output_image_path)

    end_time = datetime.datetime.now()

    i += 1

    predict_time = end_time - start_time

    total_predict_time += predict_time.total_seconds()
    predict_time_avg = total_predict_time / i

    if args.verbose:
      print('{}/{}: wrote: {}'.format(i, max_images, output_image_path))
      # print the time it took to run the model
      print('sample_A2B time: {}'.format(predict_time))

    # make a progress bar with the current % done and estimated time remaining
    progress = float(i) / float(max_images)
    # remaining = predict_time_avg * (1.0 - progress) / progress
    remaining = (max_images * total_predict_time) / (i - total_predict_time)
    remaining = datetime.timedelta(seconds=remaining)
    # print('progress: {:.2f}%, remaining: {}'.format(
    # progress * 100.0, remaining))

    printProgressBar(i, max_images, prefix='Progress:',
                     suffix='Est Time: {} (avg: {:.2f})'.format(remaining, predict_time_avg), length=50)

  else:
    # benchmark time for
    start_time = datetime.datetime.now()
    coPilotHallucinate(input_image)
    end_time = datetime.datetime.now()
    predict_time = end_time - start_time

    print('coPilotHallucinate time: {}'.format(predict_time))

  if i >= max_images:
    break
