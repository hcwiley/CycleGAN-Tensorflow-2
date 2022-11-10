import glob
import os
import shutil
import subprocess

# get the directory from the environment variable
dir = os.environ['input_dir']
processing_dir = '/Users/hcwiley/cws-git/internal/CycleGAN-Tensorflow-2/datasets/landscape2cole_3/testA'

test_cmd = 'python test.py --experiment_dir ./output/landscape2cole_3'
# test_cmd = test_cmd.split(" ")

max_images = 30

# get all the files in the directory
files = glob.glob(os.path.join(dir, '*'))

# make a copy of the original list
copied_files = []

# check to see if there's more files to copy
while len(copied_files) < len(files):
  # copy the next max_images into the processing directory
  start = len(copied_files)
  batch_of_files = files[start:start+max_images]

  if os.path.exists(processing_dir):
    shutil.rmtree(processing_dir)
  os.mkdir(processing_dir)

  # copy all the files in the batch
  for file in batch_of_files:
    shutil.copy(file, processing_dir)

  # run the test_cmd as a subprocess
  subprocess.run(test_cmd, shell=True)
  # print('run test_cmd: {}'.format(test_cmd))

  # update the copied_files list
  copied_files.extend(batch_of_files)

  # print the copied files vs the total files
  print('copied {}/{} files'.format(len(copied_files), len(files)))
