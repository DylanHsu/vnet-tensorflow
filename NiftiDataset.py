import SimpleITK as sitk
import tensorflow as tf
import os, sys
import numpy as np
import math
import random
import time

class NiftiDataset(object):
  """
  load image-label pair for training, testing and inference.
  Currently only support linear interpolation method
  Args:
	data_dir (string): Path to data directory.
    image_filenames (string|tuple): Filenames of image data, separated by commas or in a tuple.
    label_filename (string): Filename of label data.
    transforms (list): List of SimpleITK image transformations.
    num_crops: Number of random crops to serve per image per epoch
    train (bool): Determine whether the dataset class run in training/inference mode. When set to false, an empty label with same metadata as image is generated.

  """
# update above section descriptions

  def __init__(self,
    data_dir = '',
    image_filenames = '',
    label_filename = '',
    transforms=None,
    num_crops=1,
    train=False,
    labels=[0,1],
    small_bias=1,
    small_bias_diameter=10.,
    bounding_boxes=False,
    bb_slice_axis=2,
    bb_slices=1,
    cpu_threads=4):
    
    assert isinstance(image_filenames, (str,tuple))
    if isinstance(image_filenames,str):
      self.image_filenames = tuple(image_filenames.split(','))
    else:
      assert isinstance(image_filenames[0], str)
      self.image_filenames = image_filenames

    # Init membership variables
    self.data_dir = data_dir
    self.label_filename = label_filename
    self.labels = labels
    self.transforms = transforms
    self.train = train
    self.num_crops = num_crops
    self.small_bias = small_bias
    self.small_bias_diameter = small_bias_diameter
    self.bounding_boxes = bounding_boxes
    self.bb_slice_axis = 2
    self.bb_slices = 1
    self.cpu_threads = cpu_threads

    self.profile = {}
    if self.transforms:
      for transform in self.transforms:
        self.profile[type(transform)] = 0.0
  
  def get_dataset(self):
    case_list = os.listdir(self.data_dir)

    dataset = tf.data.Dataset.from_tensor_slices(case_list)
    dataset = dataset.shuffle(buffer_size=len(case_list))
    
    if self.bounding_boxes is True: # I believe this is optimal since get_dataset only called once
      dataset = dataset.map(lambda case: tuple(tf.py_func(
        self.bb_input_parser, [case], [tf.float32,tf.float32])), num_parallel_calls=self.cpu_threads)
    else:
      dataset = dataset.map(lambda case: tuple(tf.py_func(
        self.input_parser, [case], [tf.float32,tf.int32])), num_parallel_calls=self.cpu_threads)

    self.dataset = dataset
    self.data_size = len(case_list)
    return self.dataset

  def read_image(self,path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute()
    return image

  def input_parser(self, case):
    try:
      case_decoded = case.decode("utf-8")
      case = case_decoded
    except:
      pass
    # Input parser python function which gets wrapped
    # read image and label
    image_paths = []
    for channel in range(len(self.image_filenames)):
      image_paths.append(os.path.join(self.data_dir,case,self.image_filenames[channel]))  
    
    # read image and label
    images = []
    for channel in range(len(image_paths)):
      image = self.read_image(image_paths[channel])
      images.append(image)

    for channel in range(len(images)):
      # check header same
      sameSize = images[channel].GetSize() == images[0].GetSize()
      sameSpacing = images[channel].GetSpacing() == images[0].GetSpacing()
      sameDirection = images[channel].GetDirection() == images[0].GetDirection()
      if sameSize and sameSpacing and sameDirection:
        continue
      else:
        raise Exception('Header info inconsistent: {}'.format(image_paths[channel]))
        exit()
      
    label = sitk.Image(images[0].GetSize(),sitk.sitkUInt8)
    label.SetOrigin(images[0].GetOrigin())
    label.SetSpacing(images[0].GetSpacing())
    label.SetDirection(images[0].GetDirection())

    if self.train:
      label_ = self.read_image(os.path.join(self.data_dir,case, self.label_filename))

      # check header same
      sameSize = label_.GetSize() == images[0].GetSize()
      sameSpacing = label_.GetSpacing() == images[0].GetSpacing()
      sameDirection = label_.GetDirection() == images[0].GetDirection()
      if not (sameSize and sameSpacing and sameDirection):
        raise Exception('Header info inconsistent: {}'.format(os.path.join(self.data_dir,case, self.label_filename)))
        exit()

      thresholdFilter = sitk.BinaryThresholdImageFilter()
      thresholdFilter.SetOutsideValue(0)
      thresholdFilter.SetInsideValue(1)

      castImageFilter = sitk.CastImageFilter()
      castImageFilter.SetOutputPixelType(sitk.sitkUInt8)
      for channel in range(len(self.labels)):
        thresholdFilter.SetLowerThreshold(self.labels[channel])
        thresholdFilter.SetUpperThreshold(self.labels[channel])
        one_hot_label_image = thresholdFilter.Execute(label_)
        multiFilter = sitk.MultiplyImageFilter()
        one_hot_label_image = multiFilter.Execute(one_hot_label_image, channel)
        # cast one_hot_label to sitkUInt8
        one_hot_label_image = castImageFilter.Execute(one_hot_label_image)
        one_hot_label_image.SetSpacing(images[0].GetSpacing())
        one_hot_label_image.SetDirection(images[0].GetDirection())
        one_hot_label_image.SetOrigin(images[0].GetOrigin())
        addFilter = sitk.AddImageFilter()
        label = addFilter.Execute(label,one_hot_label_image)
    if self.train:
      label_ = self.read_image(os.path.join(self.data_dir, case, self.label_filename))
      # check header same
      sameSize = label_.GetSize() == images[0].GetSize()
      sameSpacing = label_.GetSpacing() == images[0].GetSpacing()
      sameDirection = label_.GetDirection() == images[0].GetDirection()
      if not (sameSize and sameSpacing and sameDirection):
        raise Exception('Header info inconsistent: {}'.format(os.path.join(self.data_dir,case, self.label_filename)))
        exit()
   
      thresholdFilter = sitk.BinaryThresholdImageFilter()
      thresholdFilter.SetOutsideValue(0)
      thresholdFilter.SetInsideValue(1)

      castImageFilter = sitk.CastImageFilter()
      castImageFilter.SetOutputPixelType(sitk.sitkUInt8)
      for channel in range(len(self.labels)):
        thresholdFilter.SetLowerThreshold(self.labels[channel])
        thresholdFilter.SetUpperThreshold(self.labels[channel])
        one_hot_label_image = thresholdFilter.Execute(label_)
        multiFilter = sitk.MultiplyImageFilter()
        one_hot_label_image = multiFilter.Execute(one_hot_label_image, channel)
        # cast one_hot_label to sitkUInt8
        one_hot_label_image = castImageFilter.Execute(one_hot_label_image)
        one_hot_label_image.SetSpacing(images[0].GetSpacing())
        one_hot_label_image.SetDirection(images[0].GetDirection())
        one_hot_label_image.SetOrigin(images[0].GetOrigin())
        addFilter = sitk.AddImageFilter()
        label = addFilter.Execute(label,one_hot_label_image)
    else:
      # Not sure this handles the case where "image" is 4D
      label = sitk.Image(images[0].GetSize(),sitk.sitkUInt8)
      label.SetOrigin(images[0].GetOrigin())
      label.SetSpacing(images[0].GetSpacing())
      label.SetDirection(images[0].GetDirection())
    # Form the sample dict with the list of 3D ITK Images and the label
    sample = {'image': images, 'label':label}
    
    images_np=[]
    labels_np=[]
    cubicMmPerVoxel = label.GetSpacing()[0] * label.GetSpacing()[1] * label.GetSpacing()[2];
    num_crops = self.num_crops
    if self.small_bias is not 1:
      # Do the connected component analysis to find out if this is small
      ccFilter = sitk.ConnectedComponentImageFilter()
      labelCC = ccFilter.Execute(label)
      labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
      labelShapeFilter.SetComputeFeretDiameter(True)
      labelShapeFilter.Execute(labelCC)
      avgD = 0.0
      minD = 999.0
      for iLabel in range(1,labelShapeFilter.GetNumberOfLabels()+1):
        avgD += labelShapeFilter.GetFeretDiameter(iLabel)
        minD = min(minD, labelShapeFilter.GetFeretDiameter(iLabel))
      if labelShapeFilter.GetNumberOfLabels() > 0:
        avgD /= float(labelShapeFilter.GetNumberOfLabels())
        #if avgD < self.small_bias_diameter:
        if minD < self.small_bias_diameter:
          num_crops *= self.small_bias
    for multiple_crops in range(0,num_crops):
      sample_tfm = {'image': images.copy(), 'label':label}
      if self.transforms:
        for transform in self.transforms:
          t1=time.process_time()
          sample_tfm = transform(sample_tfm)
          t2=time.process_time()
          self.profile[type(transform)] += (t2-t1)

      # convert sample to tf tensors
      image_np = [] # New size of image_np, inferred from the transformed shape
      for volume in sample_tfm['image']:
        image_np += [sitk.GetArrayFromImage(volume)]
      image_np = np.asarray(image_np,np.float32)
      
      label_np = sitk.GetArrayFromImage(sample_tfm['label'])
      label_np = np.asarray(label_np,np.int32)

      # to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])
      image_np = np.transpose(image_np,(3,2,1,0)) # (T,Z,Y,X) -> (X,Y,Z,T) 
      label_np = np.transpose(label_np,(2,1,0))
      #return image_np, label_np

      images_np.append(image_np)
      labels_np.append(label_np)
      #print('DEBUG: serving crop %d, with %d true voxels'%(multiple_crops,np.sum(label_np)))
    image_stack = np.stack(images_np)
    label_stack = np.stack(labels_np)
    return image_stack, label_stack
 
  def bb_input_parser(self, case):
    # Input parser python function which gets wrapped
    # This version of the function differs greatly and returns a bounding box
    try:
      case_decoded = case.decode("utf-8")
      case = case_decoded
    except:
      pass
    # read image and label
    image_paths = []
    for channel in range(len(self.image_filenames)):
      image_paths.append(os.path.join(self.data_dir,case,self.image_filenames[channel]))  
    
    # read image and label
    images = []
    for channel in range(len(image_paths)):
      image = self.read_image(image_paths[channel])
      images.append(image)

    for channel in range(len(images)):
      # check header same
      sameSize = images[channel].GetSize() == images[0].GetSize()
      sameSpacing = images[channel].GetSpacing() == images[0].GetSpacing()
      sameDirection = images[channel].GetDirection() == images[0].GetDirection()
      if sameSize and sameSpacing and sameDirection:
        continue
      else:
        raise Exception('Header info inconsistent: {}'.format(image_paths[channel]))
        exit()
      
    label = sitk.Image(images[0].GetSize(),sitk.sitkUInt8)
    label.SetOrigin(images[0].GetOrigin())
    label.SetSpacing(images[0].GetSpacing())
    label.SetDirection(images[0].GetDirection())

    if self.train:
      label_ = self.read_image(os.path.join(self.data_dir,case, self.label_filename))

      # check header same
      sameSize = label_.GetSize() == images[0].GetSize()
      sameSpacing = label_.GetSpacing() == images[0].GetSpacing()
      sameDirection = label_.GetDirection() == images[0].GetDirection()
      if not (sameSize and sameSpacing and sameDirection):
        raise Exception('Header info inconsistent: {}'.format(os.path.join(self.data_dir,case, self.label_filename)))
        exit()

      thresholdFilter = sitk.BinaryThresholdImageFilter()
      thresholdFilter.SetOutsideValue(0)
      thresholdFilter.SetInsideValue(1)

      castImageFilter = sitk.CastImageFilter()
      castImageFilter.SetOutputPixelType(sitk.sitkUInt8)
      for channel in range(len(self.labels)):
        thresholdFilter.SetLowerThreshold(self.labels[channel])
        thresholdFilter.SetUpperThreshold(self.labels[channel])
        one_hot_label_image = thresholdFilter.Execute(label_)
        multiFilter = sitk.MultiplyImageFilter()
        one_hot_label_image = multiFilter.Execute(one_hot_label_image, channel)
        # cast one_hot_label to sitkUInt8
        one_hot_label_image = castImageFilter.Execute(one_hot_label_image)
        one_hot_label_image.SetSpacing(images[0].GetSpacing())
        one_hot_label_image.SetDirection(images[0].GetDirection())
        one_hot_label_image.SetOrigin(images[0].GetOrigin())
        addFilter = sitk.AddImageFilter()
        label = addFilter.Execute(label,one_hot_label_image)
    if self.train:
      label_ = self.read_image(os.path.join(self.data_dir, case, self.label_filename))
      # check header same
      sameSize = label_.GetSize() == images[0].GetSize()
      sameSpacing = label_.GetSpacing() == images[0].GetSpacing()
      sameDirection = label_.GetDirection() == images[0].GetDirection()
      if not (sameSize and sameSpacing and sameDirection):
        raise Exception('Header info inconsistent: {}'.format(os.path.join(self.data_dir,case, self.label_filename)))
        exit()
   
      thresholdFilter = sitk.BinaryThresholdImageFilter()
      thresholdFilter.SetOutsideValue(0)
      thresholdFilter.SetInsideValue(1)

      castImageFilter = sitk.CastImageFilter()
      castImageFilter.SetOutputPixelType(sitk.sitkUInt8)
      for channel in range(len(self.labels)):
        thresholdFilter.SetLowerThreshold(self.labels[channel])
        thresholdFilter.SetUpperThreshold(self.labels[channel])
        one_hot_label_image = thresholdFilter.Execute(label_)
        multiFilter = sitk.MultiplyImageFilter()
        one_hot_label_image = multiFilter.Execute(one_hot_label_image, channel)
        # cast one_hot_label to sitkUInt8
        one_hot_label_image = castImageFilter.Execute(one_hot_label_image)
        one_hot_label_image.SetSpacing(images[0].GetSpacing())
        one_hot_label_image.SetDirection(images[0].GetDirection())
        one_hot_label_image.SetOrigin(images[0].GetOrigin())
        addFilter = sitk.AddImageFilter()
        label = addFilter.Execute(label,one_hot_label_image)
    else:
      # Not sure this handles the case where "image" is 4D
      label = sitk.Image(images[0].GetSize(),sitk.sitkUInt8)
      label.SetOrigin(images[0].GetOrigin())
      label.SetSpacing(images[0].GetSpacing())
      label.SetDirection(images[0].GetDirection())

    # Form the sample dict with the list of 3D ITK Images and the label
    sample = {'image': images, 'label':label}
    
    sample_tfm = {'image': images.copy(), 'label':label}
    if self.transforms:
      for transform in self.transforms:
        sample_tfm = transform(sample_tfm)

    # Now we have to take 3D simple ITK images and make a stack of 2D images with bounding boxes
    # Warning: This only makes single-class bounding boxes right now.
    # For multiclass, we would need to find the connected components for each possible nonzero
    # label value and make bounding boxes for those connected components with the corresponding class number.
    ccFilter = sitk.ConnectedComponentImageFilter()
    labelCC = ccFilter.Execute(sample_tfm['label'])
    labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    labelShapeFilter.Execute(labelCC)
    statisticsFilter = sitk.StatisticsImageFilter()
    statisticsFilter.Execute(sample_tfm['label'])
    
    # For a bounding box network, we can't have images with no ground truth
    assert labelShapeFilter.GetNumberOfLabels() > 0, "Label for case %s has dimensions [%d,%d,%d], %d true voxels, and no label CCs"%(case, label.GetSize()[0],label.GetSize()[1],label.GetSize()[2],statisticsFilter.GetSum())
    
    images_np=[]
    boxlists_np=[]

    boxes = []
    # centernet wants [y,x,height,width,class] where y,x are the center of the bounding box
    for i in range(1,labelShapeFilter.GetNumberOfLabels()+1):
      # Assume that we have already transformed to the correct 2.5D input size using RandomCrop or ConfidenceCrop.
      
      box_3d = labelShapeFilter.GetBoundingBox(i)
      # "The GetBoundingBox and GetRegion of the LabelShapeStatisticsImageFilter returns a bounding box is as [xstart, ystart, zstart, xsize, ysize, zsize]."
      # following line is wrong - do not give the corner of the bounding box
      # box_2d = [box_3d[1], box_3d[0], box_3d[4], box_3d[3],0]
      # this is right - give the center of the bounding box
      box_2d = [box_3d[1] + float(box_3d[4])/2.,
                box_3d[0] + float(box_3d[3])/2.,
                box_3d[4],
                box_3d[3],
                0 # single class detection -> everything is class 0
               ] 
      boxes += [box_2d]
      # need to modify this to account for bb_slice_axis
    # pad boxes with empty bounding boxes so we can make a sensible square numpy array
    
    for j in range(len(boxes),64):
      boxes.append([-1,-1,-1,-1,0])
    boxlist_np = np.asarray(boxes, np.float32)
    
    image_np = [] # list of (Z,Y,X) arrays 
    for volume in sample_tfm['image']:
      image_np += [sitk.GetArrayFromImage(volume)]
    image_3d = np.asarray(image_np, np.float32) # (T,Z,Y,X) array
    # need to modify this to account for choice of bb_slice_axis
    image_3d = np.transpose(image_np,(2,3,1,0)) # (T,Z,Y,X) -> (Y,X,Z,T) 
    # Make an arbitrary choice on how to stack the Z values and sequences
    # Last dimension will be (z1c1,z2c1,...,z1c2,z2c2)
    image_2p5d = image_3d.reshape( (image_3d.shape[0], image_3d.shape[1], image_3d.shape[2]*image_3d.shape[3]) , order='F') 
    images_np.append(image_2p5d)
    boxlists_np.append(boxlist_np)
    images_stack = np.stack(images_np)
    boxlists_stack = np.stack(boxlists_np)
    return images_stack, boxlists_stack

class Normalization(object):
  """
  Normalize an image to 0 - 255
  """

  def __init__(self):
    self.name = 'Normalization'

  def __call__(self, sample):
    # normalizeFilter = sitk.NormalizeImageFilter()
    # image, label = sample['image'], sample['label']
    # image = normalizeFilter.Execute(image)
    rescaleFilter = sitk.RescaleIntensityImageFilter()
    rescaleFilter.SetOutputMaximum(255)
    rescaleFilter.SetOutputMinimum(0)
    image, label = sample['image'], sample['label']
    image[:] = [rescaleFilter.Execute(volume) for volume in image]
    return {'image': image, 'label': label}

class StatisticalNormalization(object):
  """
  Normalize an image by mapping intensity with intensity distribution
  """

  def __init__(self, channel, sigmaUp, sigmaDown, nonzero_only=False, zero_floor=False):
    self.name = 'StatisticalNormalization'
    #assert isinstance(nSigma, float)
    #self.nSigma = nSigma
    assert isinstance(channel, int)
    assert isinstance(sigmaUp, float)
    assert isinstance(sigmaDown, float)
    assert isinstance(nonzero_only, bool)
    self.channel = channel
    self.sigmaUp = sigmaUp
    self.sigmaDown = sigmaDown
    self.nonzero_only = nonzero_only
    self.zero_floor = zero_floor
    self.threshold = 0.01

  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    statisticsFilter = sitk.StatisticsImageFilter()
    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    intensityWindowingFilter.SetOutputMaximum(255)
    intensityWindowingFilter.SetOutputMinimum(0)
    normalizedImage = image
    for i,volume in enumerate(image):
      if i != self.channel:
        continue
      if self.nonzero_only:
        volume_np = sitk.GetArrayFromImage(volume)
        nonzero_voxels = volume_np[volume_np > self.threshold]
        # Catastrophic cancellation:
        #mean = np.dot(nonzero_voxels, np.ones(nonzero_voxels.shape)) / nonzero_voxels.size
        #x2bar = np.dot(nonzero_voxels, nonzero_voxels) / nonzero_voxels.size
        #sigma = math.sqrt(x2bar - mean**2)

        # Hack to use SITK multithreaded computation
        # This speeds up the calculation by over a factor of 50
        # Create a 1xN image of the nonzero voxels then calculate the stats
        # Well, this is commented out because it's causing segfaults
        #nv_strand = sitk.GetImageFromArray(nonzero_voxels[:,np.newaxis])
        #statisticsFilter.Execute(nv_strand)
        #sigma = statisticsFilter.GetSigma()
        #mean = statisticsFilter.GetMean()
        
        #Do it the stupid way for now
        mean = np.mean(nonzero_voxels)
        sigma = np.std(nonzero_voxels)

      else:
        statisticsFilter.Execute(volume)
        sigma = statisticsFilter.GetSigma()
        mean = statisticsFilter.GetMean()

      intensityWindowingFilter.SetWindowMaximum(mean + self.sigmaUp * sigma)
      if self.zero_floor:
        intensityWindowingFilter.SetWindowMinimum(max(0, mean - self.sigmaDown * sigma))
      else:
        intensityWindowingFilter.SetWindowMinimum(mean - self.sigmaDown * sigma)
      
      normalizedImage[i] = intensityWindowingFilter.Execute(volume)
    return {'image': normalizedImage, 'label': label}

class ManualNormalization(object):
  """
  Normalize an image by mapping intensity with given max and min window level
  """

  def __init__(self,channel,windowMin, windowMax):
    self.name = 'ManualNormalization'
    assert isinstance(windowMax, (int,float))
    assert isinstance(windowMin, (int,float))
    self.windowMax = windowMax
    self.windowMin = windowMin
    self.channel = channel

  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    intensityWindowingFilter.SetOutputMaximum(255)
    intensityWindowingFilter.SetOutputMinimum(0)
    intensityWindowingFilter.SetWindowMaximum(self.windowMax);
    intensityWindowingFilter.SetWindowMinimum(self.windowMin);
    normalizedImage = image
    for i,volume in enumerate(image):
      if i != self.channel:
        continue
      normalizedImage[i] = intensityWindowingFilter.Execute(volume)
    return {'image': normalizedImage, 'label': label}

class Reorient(object):
  """
  (Beta) Function to orient image in specific axes order
  The elements of the order array must be an permutation of the numbers from 0 to 2.
  """

  def __init__(self, order):
    self.name = 'Reoreient'
    assert isinstance(order, (int, tuple))
    assert len(order) == 3
    self.order = order

  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    reorientFilter = sitk.PermuteAxesImageFilter()
    reorientFilter.SetOrder(self.order)
    image[:] = [reorientFilter.Execute(volume) for volume in image]
    label = reorientFilter.Execute(label)

    return {'image': image, 'label': label}

class Invert(object):
  """
  Invert the image intensity from 0-255 
  """

  def __init__(self):
    self.name = 'Invert'

  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    invertFilter = sitk.InvertIntensityImageFilter()
    image[:] = [invertFilter.Execute(volume,255) for volume in image]
    label = label

    return {'image': image, 'label': label}

class Resample(object):
  """
  Resample the volume in a sample to a given voxel size

	Args:
		voxel_size (float or tuple): Desired output size.
		If float, output volume is isotropic.
		If tuple, output voxel size is matched with voxel size
		Currently only support linear interpolation method
  """

  def __init__(self, voxel_size):
    self.name = 'Resample'

    assert isinstance(voxel_size, (float, tuple))
    if isinstance(voxel_size, float):
      self.voxel_size = (voxel_size, voxel_size, voxel_size)
    else:
      assert len(voxel_size) == 3
      self.voxel_size = voxel_size

  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    
    old_spacing = label.GetSpacing()
    old_size = label.GetSize()
    
    new_spacing = self.voxel_size

    new_size = []
    for i in range(3):
      new_size.append(int(math.ceil(old_spacing[i]*old_size[i]/new_spacing[i])))
    new_size = tuple(new_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(2)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)

    # resample on image
    resampler.SetOutputOrigin(label.GetOrigin())
    resampler.SetOutputDirection(label.GetDirection())
    image[:] = [resampler.Execute(volume) for volume in image]

    # resample on segmentation
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(label.GetOrigin())
    resampler.SetOutputDirection(label.GetDirection())
    label = resampler.Execute(label)

    return {'image': image, 'label': label}

class Padding(object):
  """
  Add padding to the image if size is smaller than patch size

	Args:
		output_size (tuple or int): Desired output size. If int, a cubic volume is formed
	"""

  def __init__(self, output_size):
    self.name = 'Padding'

    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size, output_size)
    else:
      assert len(output_size) == 3
      self.output_size = output_size

    assert all(i > 0 for i in list(self.output_size))

  def __call__(self,sample):
    image, label = sample['image'], sample['label']
    size_old = label.GetSize()
    
    if (size_old[0] >= self.output_size[0]) and (size_old[1] >= self.output_size[1]) and (size_old[2] >= self.output_size[2]):
      return sample
    else:
      output_size = self.output_size
      output_size = list(output_size)
      if size_old[0] > self.output_size[0]:
        output_size[0] = size_old[0]
      if size_old[1] > self.output_size[1]:
        output_size[1] = size_old[1]
      if size_old[2] > self.output_size[2]:
        output_size[2] = size_old[2]
 
      output_size = tuple(output_size)

      resampler = sitk.ResampleImageFilter()
      resampler.SetOutputSpacing(label.GetSpacing())
      resampler.SetSize(output_size)

      # resample on image
      resampler.SetInterpolator(2)
      resampler.SetOutputOrigin(label.GetOrigin())
      resampler.SetOutputDirection(label.GetDirection())
      image[:] = [resampler.Execute(volume) for volume in image]

      # resample on label
      resampler.SetInterpolator(sitk.sitkNearestNeighbor)
      resampler.SetOutputOrigin(label.GetOrigin())
      resampler.SetOutputDirection(label.GetDirection())

      label = resampler.Execute(label)

      return {'image': image, 'label': label}

class RandomCrop(object):
  """
  Crop randomly the image in a sample. This is usually used for data augmentation.
	Drop ratio is implemented for randomly dropout crops with empty label. (Default to be 0.2)
	This transformation only applicable in train mode

  Args:
    output_size (tuple or int): Desired output size. If int, cubic crop is made.
  """

  def __init__(self, output_size, drop_ratio=0.1, min_pixel=1, num_crops=1):
    self.name = 'Random Crop'

    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size, output_size)
    else:
      assert len(output_size) == 3
      self.output_size = output_size

    assert isinstance(drop_ratio, (int,float))
    if drop_ratio >=0 and drop_ratio<=1:
      self.drop_ratio = drop_ratio
    else:
      raise RuntimeError('Drop ratio should be between 0 and 1')

    assert isinstance(min_pixel, int)
    if min_pixel >=0 :
      self.min_pixel = min_pixel
    else:
      raise RuntimeError('Min label pixel count should be integer larger than 0')

  def __call__(self,sample):
    image, label = sample['image'], sample['label']
    size_old = label.GetSize()
    size_new = self.output_size

    statFilter = sitk.StatisticsImageFilter()
    statFilter.Execute(label)
    
    contain_label = False

    roiFilter = sitk.RegionOfInterestImageFilter()
    roiFilter.SetSize([size_new[0],size_new[1],size_new[2]])

    # statFilter = sitk.StatisticsImageFilter()
    # statFilter.Execute(label)
    
    # Calculate this boolean for the image, not for the crop
    # That way, drop_ratio is the fraction of images for which we get an empty crop,
    # not the fraction of crops
    if statFilter.GetSum() >= self.min_pixel:
      keep_empty = self.drop(self.drop_ratio)
    else:
      keep_empty = True
    samples=0
    while not contain_label: 
      # get the start crop coordinate in ijk
      if size_old[0] <= size_new[0]:
        start_i = 0
      else:
        start_i = np.random.randint(0, size_old[0]-size_new[0])

      if size_old[1] <= size_new[1]:
        start_j = 0
      else:
        start_j = np.random.randint(0, size_old[1]-size_new[1])

      if size_old[2] <= size_new[2]:
        start_k = 0
      else:
        start_k = np.random.randint(0, size_old[2]-size_new[2])

      roiFilter.SetIndex([start_i,start_j,start_k])

      label_crop = roiFilter.Execute(label)
      statFilter.Execute(label_crop)

      # will iterate until a sub volume containing label is extracted
      if statFilter.GetSum()>=self.min_pixel or keep_empty is True:
        contain_label = True
      samples+=1
      if samples>100000:
        sys.exit("Too many attempts to randomly crop, reconsider min_pixel and drop_ratio values")

    image_crop = [roiFilter.Execute(volume) for volume in image]

    return {'image': image_crop, 'label': label_crop}

  def drop(self,probability):
    return random.random() <= probability

class RandomNoise(object):
  """
  Randomly noise to the image in a sample. This is usually used for data augmentation.
  """
  def __init__(self, channel, std=0.1):
    self.name = 'Random Noise'
    assert isinstance(channel, int)
    self.channel = channel
    self.std = std

  def __call__(self, sample):
    self.noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
    self.noiseFilter.SetMean(0)
    self.noiseFilter.SetStandardDeviation(self.std)

    image,label = sample['image'],sample['label']
    noisyImage = image
    for i,volume in enumerate(image):
      if i != self.channel:
        continue
      noisyImage[i] = self.noiseFilter.Execute(volume)
    return {'image': noisyImage, 'label': label}

class ConfidenceCrop(object):
  """
  Crop the image in a sample that is certain distance from individual labels center. 
  This is usually used for data augmentation with very small label volumes.
  The distance offset from connected label centroid is model by Gaussian distribution with mean zero and user input sigma (default to be 2.5)
  i.e. If n isolated labels are found, one of the label's centroid will be randomly selected, and the cropping zone will be offset by following scheme:
  s_i = np.random.normal(mu, sigma*crop_size/2), 1000)
  offset_i = random.choice(s_i)
  where i represents axis direction
  A higher sigma value will provide a higher offset

  Args:
    output_size (tuple or int): Desired output size. If int, cubic crop is made.
    sigma (float): Normalized standard deviation value.
  """

  def __init__(self, output_size, sigma=2.5):
    self.name = 'Confidence Crop'

    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size, output_size)
    else:
      assert len(output_size) == 3
      self.output_size = output_size

    assert isinstance(sigma, (float, tuple))
    if isinstance(sigma, float) and sigma >= 0:
      self.sigma = (sigma,sigma,sigma)
    else:
      assert len(sigma) == 3
      self.sigma = sigma

  def __call__(self,sample):
    image, label = sample['image'], sample['label']
    size_new = self.output_size

    # guarantee label type to be integer
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkInt8)
    label = castFilter.Execute(label)

    ccFilter = sitk.ConnectedComponentImageFilter()
    labelCC = ccFilter.Execute(label)

    labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    labelShapeFilter.Execute(labelCC)

    if labelShapeFilter.GetNumberOfLabels() == 0:
      # handle image without label
      selectedLabel = 0
      centroid = (int(self.output_size[0]/2), int(self.output_size[1]/2), int(self.output_size[2]/2))
    else:
      # randomly select of the label's centroid
      selectedLabel = random.randint(1,labelShapeFilter.GetNumberOfLabels())
      centroid = label.TransformPhysicalPointToIndex(labelShapeFilter.GetCentroid(selectedLabel))

    centroid = list(centroid)

    start = [-1,-1,-1] #placeholder for start point array
    end = [self.output_size[0]-1, self.output_size[1]-1,self.output_size[2]-1] #placeholder for start point array
    offset = [-1,-1,-1] #placeholder for start point array
    # Resize the crop region of interest...
    # Assuming that all the volumes in image[:] are the same!
    for i in range(3):
      # edge case
      #if centroid[i] < (self.output_size[i]/2):
      if centroid[i] < int(self.output_size[i]/2):
        centroid[i] = int(self.output_size[i]/2)
      elif (label.GetSize()[i]-centroid[i]) < int(self.output_size[i]/2):
        #centroid[i] = label.GetSize()[i] - int(self.output_size[i]/2) -1
        centroid[i] = label.GetSize()[i] - int(self.output_size[i]/2)

      # get start point
      while ((start[i]<0) or (end[i]>(label.GetSize()[i]-1))):
        offset[i] = self.NormalOffset(self.output_size[i],self.sigma[i])
        start[i] = centroid[i] + offset[i] - int(self.output_size[i]/2)
        end[i] = start[i] + self.output_size[i] - 1
        # print(i, start[i], end[i], label.GetSize()[i]-1) # debug infinite while loop

    roiFilter = sitk.RegionOfInterestImageFilter()
    roiFilter.SetSize(self.output_size)
    roiFilter.SetIndex(start)
    croppedImage = [roiFilter.Execute(volume) for volume in image]
    croppedLabel = roiFilter.Execute(label)

    return {'image': croppedImage, 'label': croppedLabel}

  def NormalOffset(self,size, sigma):
    #s = np.random.normal(0, size*sigma/2, 100) # 100 sample is good enough
    s = np.random.normal(0, size*sigma/2, 1000) 
    return int(round(random.choice(s)))

class BSplineDeformation(object):
  """
  Image deformation with a sparse set of control points to control a free form deformation.
  Details can be found here: 
  https://simpleitk.github.io/SPIE2018_COURSE/spatial_transformations.pdf
  https://itk.org/Doxygen/html/classitk_1_1BSplineTransform.html

  Args:
    randomness (int,float): BSpline deformation scaling factor, default is 10.
  """

  def __init__(self, randomness=10):
    self.name = 'BSpline Deformation'

    assert isinstance(randomness, (int,float))
    if randomness > 0:
      self.randomness = randomness
    else:
      raise RuntimeError('Randomness should be non zero values')

  def __call__(self,sample):
    image, label = sample['image'], sample['label']
    spline_order = 3
    # Assuming that all the volumes in image[:] are the same!
    domain_physical_dimensions = [label.GetSize()[0]*label.GetSpacing()[0],label.GetSize()[1]*label.GetSpacing()[1],label.GetSize()[2]*label.GetSpacing()[2]]

    bspline = sitk.BSplineTransform(3, spline_order)
    bspline.SetTransformDomainOrigin(label.GetOrigin())
    bspline.SetTransformDomainDirection(label.GetDirection())
    bspline.SetTransformDomainPhysicalDimensions(domain_physical_dimensions)
    bspline.SetTransformDomainMeshSize((10,10,10))

    # Random displacement of the control points.
    originalControlPointDisplacements = np.random.random(len(bspline.GetParameters()))*self.randomness
    bspline.SetParameters(originalControlPointDisplacements)

    image[:] = [sitk.Resample(volume, bspline) for volume in image]
    label = sitk.Resample(label, bspline)
    return {'image': image, 'label': label}

  def NormalOffset(self,size, sigma):
    s = np.random.normal(0, size*sigma/2, 100) # 100 sample is good enough
    return int(round(random.choice(s)))

class RandomRotation(object):
  """
  Rotate an image by a random amount
  """

  def __init__(self, interpolator=sitk.sitkBSpline, maxRot=2*np.pi):
    self.name = 'RandomRotation'
    self.interpolator = interpolator
    self.maxRot = maxRot

  def __call__(self, sample):
    image, label = sample['image'], sample['label']
   
    ## choose label as centroid
    #ccFilter = sitk.ConnectedComponentImageFilter()
    #labelCC = ccFilter.Execute(label)
    #labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    #labelShapeFilter.Execute(labelCC)
    #
    #if labelShapeFilter.GetNumberOfLabels() == 0:
    #  # handle image without label
    #  selectedLabel = 0
    #  centroid = (int(size_new[0]/2), int(size_new[1]/2), int(size_new[2]/2))
    #else:
    #  # randomly select of the label's centroid
    #  selectedLabel = random.randint(1,labelShapeFilter.GetNumberOfLabels())
    #  centroidPP = labelShapeFilter.GetCentroid(selectedLabel)
    #  centroid = label.TransformPhysicalPointToIndex(centroidPP)
    rot_center = label.TransformIndexToPhysicalPoint((label.GetSize()[0]//2, label.GetSize()[1]//2,label.GetSize()[2]//2))
    
    # Generate a random rotation direction on the unit sphere
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)
    theta = np.arccos( costheta )
    x = np.sin( theta) * np.cos( phi )
    y = np.sin( theta) * np.sin( phi )
    z = np.cos( theta )
    alpha = np.random.uniform(0, self.maxRot) # amount to rotate by
    rotation = sitk.VersorTransform((x,y,z), alpha)
    rotation.SetCenter(rot_center)
    
    # Define reference image which is larger than the input image
    # This is necessary so we don't rotate the true label out of bounds!
    # An SxSxS image with a random rotation will always fit in a box of size S*sqrt(2)
    # Assume there is some uninteresting stuff at the edge of the image.
    # We could rotate around a chosen label centroid, and throw out some information,
    # This is more conservative in terms of retaining information but is maybe more memory intensive.
    resampler = sitk.ResampleImageFilter()
    # Perform the interpolation
    image_size = label.GetSize()
    reference_size = tuple([math.ceil(i*1.5) for i in list(image_size)])
    resampler.SetOutputSpacing(label.GetSpacing())
    resampler.SetSize(reference_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(label.GetOrigin())
    resampler.SetOutputDirection(label.GetDirection())
    #reference = resampler.Execute(label)
    reference = label

    image[:] = [sitk.Resample(volume, reference, rotation, self.interpolator, 0) for volume in image]
    # Label interpolation must be nearest neighbor
    label = sitk.Resample(label, reference, rotation, sitk.sitkNearestNeighbor, 0)
    return {'image': image, 'label': label}

class ThresholdCrop(object):
  """
  Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
  usually air. Then crop the image using the foreground's axis aligned bounding box.
  Assume the anatomy and background intensities form a bi-modal distribution (Otsu's method)
  """
  def __init__(self, inside_value=0, outside_value=255):
    self.name = 'ThresholdCrop'
    self.inside_value = inside_value
    self.outside_value = outside_value

  def __call__(self, sample):
    if len(image) == 0:
      return sample
    image, label = sample['image'], sample['label']
    # Build a composite image from the slices and the label
    # The bounding box must contain the average of the image slices and the label
    # alternative: could require it contain the sum of the slices
    composite_image = image[0]
    aif = sitk.AddImageFilter()
    if len(image) > 1:
      for i in range(1,len(image)):
        composite_image = aif.Execute(composite_image, image[i])
      dif = sitk.DivideImageFilter()
      # Here we are taking the average, comment out to take the sum
      composite_image = dif.Execute(composite_image, len(image))

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    composite_image = aif.Execute(composite_image, castImageFilter.Execute(label))
        
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute( sitk.OtsuThreshold(composite_image, self.inside_value, self.outside_value) )
    bounding_box = label_shape_filter.GetBoundingBox(self.outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
   
    croppedImage = [sitk.RegionOfInterest(volume, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)]) for volume in image]
    croppedLabel = sitk.RegionOfInterest(label, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
    return {'image': croppedImage, 'label': croppedLabel}


class RandomFlip(object):
  """
  Flip an image about randomly chosen axes.
  flip_prob: chance to flip about a given axis
  flip_axes: booleans for the axes about which you want a chance to flip

  Note: Not sure right now if [0,1,2]<=>[x,y,z] or [z,y,x]
  """
  def __init__(self, flip_prob=0.5, flip_axes=[True,True,True]):
    self.name = 'RandomFlip'
    assert isinstance(flip_prob, (float))
    self.flip_prob = flip_prob
    assert isinstance(flip_axes, (list))
    self.flip_axes = flip_axes

  def __call__(self, sample):
    
    # Choose axes to flip
    chosen_flip_axes = []
    for axis in self.flip_axes:
      if axis is True and (random.random() > self.flip_prob):
        chosen_flip_axes.append(True)
      else:
        chosen_flip_axes.append(False)
    
    image, label = sample['image'], sample['label']
    flippedImage = image
    fif = sitk.FlipImageFilter()
    fif.SetFlipAxes(chosen_flip_axes)
    for i in range(0,len(image)):
      flippedImage[i] = fif.Execute(image[i])
    flippedLabel = fif.Execute(label)
    return {'image': flippedImage, 'label': flippedLabel}

class RandomHistoMatch(object):
  """
  Randomly match this image's histogram to that of another.
  This must be performed *before* any intensity standardization!

  data_dir: Directory containing the cases used for target distributions
  image_filename: filename of the target image, e.g. 'img.nii.gz'
  channel: Which channel to transform this way
  match_prob: Chance to perform a matching
  """
  def __init__(self, channel, data_dir, image_filename, match_prob=1.0):
    self.name = 'RandomHistoMatch'
    assert isinstance(data_dir, str)
    self.data_dir = data_dir
    assert isinstance(image_filename, str)
    assert isinstance(channel, int)
    self.channel = channel
    self.image_filename = image_filename
    assert isinstance(match_prob, float)
    self.match_prob = match_prob
    self.cases = os.listdir(self.data_dir)

  def __call__(self, sample):
    if self.match_prob<1.0 and (random.random() > self.match_prob):
      return sample
    
    image, label = sample['image'], sample['label']
    
    # Choose a random target image file
    target_case = random.choice(self.cases)
    target_path = os.path.join(self.data_dir, target_case, self.image_filename)

    # Get the array of target images to match to
    reader = sitk.ImageFileReader()
    # no utf decoding, since we're not wrapping as a py_func in tf
    reader.SetFileName(target_path) 
    target_image = reader.Execute()
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    target_image = castImageFilter.Execute(target_image)

    histoMatchFilter = sitk.HistogramMatchingImageFilter()
    histoMatchFilter.SetNumberOfHistogramLevels(500)
    histoMatchFilter.SetNumberOfMatchPoints(500)
    
    matchedImage = []
    
    btif = sitk.BinaryThresholdImageFilter()
    mif = sitk.MaskImageFilter()
    for i in range(len(image)):
      if i != self.channel:
        matchedImage.append(image[i])
      else:
        volume = image[i]
        volume_matched = histoMatchFilter.Execute( volume, target_image )
        matchedImage.append(volume_matched)

    return {'image': matchedImage, 'label': label}

class DifficultyIndex(object):
  """
  Compute a map of the "difficulty" of finding the label's connected
  components, based on its dimensions and the image intensities
  in the CC versus its neighborhood.
  The label will no longer be integers.
  This should be called prior to any cropping so the full geometric
  information of the labels and their surroundings is available.
  For multi-volume images, this is best done after statistical normalization,
  to aid the intensity balancing calculation.

  nhd_size: int or tuple representing the neighborhood size for the
    intensity calculation
  kD: weight of the volumetric diameter term
  kI: weight of the intensity balancing term
  """
  def __init__(self, nhd_size, kD=5.0, kI=1.0):
    self.name = 'DifficultyIndex'
    assert isinstance(nhd_size, (int, tuple))
    if isinstance(nhd_size, int):
      self.nhd_size = (nhd_size, nhd_size, nhd_size)
    else:
      assert len(nhd_size) == 3
      self.nhd_size = nhd_size
    
    assert isinstance(kD, float)
    self.kD = kD
    assert isinstance(kI, float)
    self.kI = kI
    
    self.threshold = 0.001 # threshold for the nonzero voxels
  def __call__(self, sample):
    image, label = sample['image'], sample['label']

    # instantiate filters
    addImageFilter        = sitk.AddImageFilter()
    castImageFilter       = sitk.CastImageFilter()
    ccFilter              = sitk.ConnectedComponentImageFilter()
    labelShapeFilter      = sitk.LabelShapeStatisticsImageFilter()
    mapFilter             = sitk.LabelImageToLabelMapFilter()
    maskImageFilter       = sitk.LabelMapMaskImageFilter()
    roiFilter             = sitk.RegionOfInterestImageFilter()
    statisticsImageFilter = sitk.StatisticsImageFilter()
    multiplyImageFilter   = sitk.MultiplyImageFilter()

    roiFilter.SetSize(self.nhd_size)
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    
    # Compute the mean intensity value of the image
    # This is computed over all volumes of the image, which
    # might be problematic for multiple volumes with different intensities.
    mean = 0.
    for volume in image:
      image_np = np.transpose(sitk.GetArrayFromImage(volume),(1,2,0))
      nonzero_voxels = (image_np > self.threshold)
      arr_nonzero = image_np[nonzero_voxels]
      mean += np.mean(arr_nonzero)
      # No stdev calculation for right now
    mean = mean / float(len(image))

    # compute connected components of the label
    labelCC = ccFilter.Execute(label)
    labelShapeFilter.Execute(labelCC)
    # create a SimpleITK label map from the connected components
    # this is needed to mask specific CC's
    labelMap=mapFilter.Execute(labelCC,0)

    # Create an image with floating point values in the
    # geometry of the original label. The values will be the
    # difficulty indices of the connected components
    weightedLabel = sitk.Image(label.GetSize(), sitk.sitkFloat32)
    weightedLabel.SetOrigin(label.GetOrigin())
    weightedLabel.SetSpacing(label.GetSpacing())
    weightedLabel.SetDirection(label.GetDirection())
    
    # Loop over all the lesions in this label
    for i in range(1,labelShapeFilter.GetNumberOfLabels()+1):
      # Get a binary mask for only this lesion
      maskImageFilter.SetLabel(i)
      lesion = maskImageFilter.Execute(labelMap, label)
      
      # From this binary mask, compute the mean 
      lesion_np = np.transpose(sitk.GetArrayFromImage(lesion),(1,2,0))
      
      # neighborhood calculation
      centroid = list(label.TransformPhysicalPointToIndex(labelShapeFilter.GetCentroid(i)))
      start=[-1,-1,-1]
      end=[-1,-1,-1] 
      for dim in range(3):
        if centroid[dim] < self.nhd_size[dim]/2:
          centroid[dim] = int(self.nhd_size[dim]/2)
        elif label.GetSize()[dim] - centroid[dim] < self.nhd_size[dim]/2:
          centroid[dim] = label.GetSize()[dim] - int(self.nhd_size[dim]/2) - 1
        
        start[dim] = centroid[dim] - int(self.nhd_size[dim]/2)
        end[dim]   = start[dim] + self.nhd_size[dim] - 1
      roiFilter.SetIndex(start)
      croppedLabel = roiFilter.Execute(label)
      croppedLabel_np = np.transpose(sitk.GetArrayFromImage(croppedLabel),(1,2,0))
      nhd_mean = 0.
      lesion_mean = 0.
      lesion_volume = -1
      for volume in image:
        croppedImage = roiFilter.Execute(volume)
        croppedImage_np = np.transpose(sitk.GetArrayFromImage(croppedImage),(1,2,0))
        nhd_voxels = croppedImage_np[croppedLabel_np <= self.threshold]
        
        image_np = np.transpose(sitk.GetArrayFromImage(volume),(1,2,0))
        lesion_voxels = image_np[lesion_np > self.threshold] 
        # compute lesion volume only once
        if lesion_volume == -1:
          lesion_volume = len(lesion_voxels)
          q90 = np.quantile(lesion_voxels,0.9)
          q10 = np.quantile(lesion_voxels,0.1)
          nhd_mean = np.mean(nhd_voxels)
          lesion_mean = np.mean(lesion_voxels)
      #nhd_mean = nhd_mean / float(len(image))
      #lesion_mean = lesion_mean / float(len(image))
      if nhd_mean==0: # handle the case of an empty image (crop)
        q90_balance = 0.
        q10_balance = 0.
        intensity_balance = 0. 
      else:
        q90_balance = q90/nhd_mean - 1.
        q10_balance = q10/nhd_mean - 1.
        intensity_balance = abs(lesion_mean/nhd_mean - 1.)
      cubicMmPerVoxel = 1.
      for dimSpacing in list(label.GetSpacing()):
        cubicMmPerVoxel = cubicMmPerVoxel * dimSpacing

      diameter = (1.909859 * float(lesion_volume) * cubicMmPerVoxel)  ** (1./3.) # (6V/pi)^(1/3)
      
      difficulty = self.kD * (1./diameter) + self.kI * (1. - q10_balance/2.)
      
      weightedLesion = castImageFilter.Execute(lesion)
      weightedLesion = multiplyImageFilter.Execute(weightedLesion, difficulty)
      weightedLabel = addImageFilter.Execute(weightedLabel, weightedLesion)

    return {'image': image, 'label': weightedLabel}



