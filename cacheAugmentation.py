import SimpleITK as sitk
import numpy as np
import NiftiDataset
import os, sys
import math
import datetime
import shutil
from glob import glob

nAug=100
input_data_dir='/data/deasy/DylanHsu/N200/training'
output_data_dir='/data/deasy/DylanHsu/N200/augcache/training'
image_filename='img.nii.gz'
label_filename='label_smoothed.nii.gz'

case = sys.argv[1]

augTransforms = [
  NiftiDataset.RandomHistoMatch(input_data_dir, image_filename, 1.0),
  NiftiDataset.StatisticalNormalization(5.0, 5.0, nonzero_only=True, zero_floor=True),
  NiftiDataset.BSplineDeformation(),
  NiftiDataset.RandomRotation(maxRot = 20*0.01745),  # 20 degrees
  NiftiDataset.ThresholdCrop(),
  ]
augDataset = NiftiDataset.NiftiDataset(
  data_dir = input_data_dir,
  image_filename = image_filename,
  label_filename = label_filename,
  transforms = augTransforms,
  train=True
)

image_path = os.path.join(input_data_dir, case, image_filename)
label_path = os.path.join(input_data_dir, case, label_filename)
image = augDataset.read_image(image_path)
label = augDataset.read_image(label_path)

writer = sitk.ImageFileWriter()
writer.UseCompressionOn()
for iAug in range(1, nAug+1):
  slug = "{0:s}_aug{1:03d}".format(case,iAug)
  
  #[image_stack, label_stack] = augDataset.input_parser(image_path.encode("utf-8"), label_path.encode("utf-8"))
  #image_np = image_stack[0,:]
  #label_np = label_stack[0,:]
  sample_tfm = {'image': [image], 'label':label}  
  for transform in augDataset.transforms:
    sample_tfm = transform(sample_tfm)

  output_case_dir = os.path.join(output_data_dir, slug)
  if os.path.isdir(output_case_dir):
    shutil.rmtree(output_case_dir)
  os.mkdir(output_case_dir)
  writer.SetFileName(os.path.join(output_case_dir, image_filename))
  writer.Execute(sample_tfm['image'][0])
  writer.SetFileName(os.path.join(output_case_dir, label_filename))
  writer.Execute(sample_tfm['label'])

  




