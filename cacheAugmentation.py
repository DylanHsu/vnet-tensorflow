import SimpleITK as sitk
import numpy as np
import NiftiDataset
import os, sys
import math
import datetime
import shutil
from glob import glob

nAug=50
input_data_dir='/data/deasy/DylanHsu/SRS_N401/nifti'
output_data_dir='/data/deasy/DylanHsu/SRS_N401/augcache'
image_filenames=('mr1.nii.gz', 'ct.nii.gz')
label_filename='label_smoothed.nii.gz'

case = sys.argv[1]

augTransforms = [
  NiftiDataset.RandomHistoMatch(0, input_data_dir, image_filenames[0], 1.0),
  NiftiDataset.StatisticalNormalization(0, 5.0, 5.0, nonzero_only=True, zero_floor=True), # MR [0,999999] -> [0,255]
  NiftiDataset.ManualNormalization(1, 0, 100.), # CT [0,100] H.U. -> [0,255] Arbitrary units
  NiftiDataset.BSplineDeformation(),
  NiftiDataset.RandomRotation(maxRot = 20*0.01745),  # 20 degrees
  NiftiDataset.ThresholdCrop(),
  ]
augDataset = NiftiDataset.NiftiDataset(
  data_dir = input_data_dir,
  image_filenames = image_filenames,
  label_filename = label_filename,
  transforms = augTransforms,
  train=True
)

images = []
for image_filename in list(image_filenames):
  image_path = os.path.join(input_data_dir, case, image_filename)
  image = augDataset.read_image(image_path)
  images.append(image)
label_path = os.path.join(input_data_dir, case, label_filename)
label = augDataset.read_image(label_path)

writer = sitk.ImageFileWriter()
writer.UseCompressionOn()
statFilter = sitk.StatisticsImageFilter()
distanceMapFilter = sitk.SignedDanielssonDistanceMapImageFilter()
for iAug in range(1, nAug+1):
  slug = "{0:s}_aug{1:03d}".format(case,iAug)
  
  hasTrueVoxels=False
  while not hasTrueVoxels:
    sample_tfm = {'image': images, 'label':label}  
    for transform in augDataset.transforms:
      sample_tfm = transform(sample_tfm)
      statFilter.Execute(sample_tfm['label'])
      #print('after', transform, statFilter.GetSum(), 'true voxels')
    statFilter.Execute(sample_tfm['label'])
    if statFilter.GetSum() > 0:
      hasTrueVoxels=True

  output_case_dir = os.path.join(output_data_dir, slug)
  if os.path.isdir(output_case_dir):
    shutil.rmtree(output_case_dir)
  os.mkdir(output_case_dir)
  for iImage in range(len(image_filenames)):
    writer.SetFileName(os.path.join(output_case_dir, image_filenames[iImage]))
    writer.Execute(sample_tfm['image'][iImage])
  writer.SetFileName(os.path.join(output_case_dir, label_filename))
  writer.Execute(sample_tfm['label'])

  # Recalculate distance map from the transformed label,
  # since it needs to be defined everywhere
  distanceMap = distanceMapFilter.Execute( sample_tfm['label'], False, False, False)
  writer.SetFileName( os.path.join(output_case_dir, "distance_map.nii.gz") )
  writer.Execute(distanceMap)


