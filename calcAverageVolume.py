from glob import glob
import SimpleITK as sitk
import os, sys
import random

data_dir = '/data/deasy/DylanHsu/SRS_N401/augcache'

reader = sitk.ImageFileReader()
N=0
Vtotal=0
cases = os.listdir(data_dir)
random.shuffle(cases)
for case in cases:
  image_path = os.path.join(data_dir,case,'mr1.nii.gz')
  reader.SetFileName(image_path)
  image = reader.Execute()
  N += 1
  Vtotal += image.GetSize()[0]*image.GetSize()[1]*image.GetSize()[2]
  if N % 100 == 0:
    print('average V is %.3e after %d images'%(Vtotal/float(N),N))



