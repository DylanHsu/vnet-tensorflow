import SimpleITK as sitk
import tensorflow as tf
import os
import numpy as np
import math
import random

class NiftiDataset(object):
  """
  load image-label pair for training, testing and inference.
  Currently only support linear interpolation method
  Args:
		data_dir (string): Path to data directory.
    image_filename (string): Filename of image data.
    label_filename (string): Filename of label data.
    transforms (list): List of SimpleITK image transformations.
    num_crops: Number of random crops to serve per image per epoch
    train (bool): Determine whether the dataset class run in training/inference mode. When set to false, an empty label with same metadata as image is generated.
    peek_dir (string): Place to write images after the transforms for debugging.

  """

  def __init__(self,
    data_dir = '',
    image_filename = '',
    label_filename = '',
    transforms=None,
    num_crops=1,
    train=False,
    peek_dir = ''):

    # Init membership variables
    self.data_dir = data_dir
    self.image_filename = image_filename
    self.label_filename = label_filename
    self.transforms = transforms
    self.train = train
    self.num_crops = num_crops
    self.peek_dir = peek_dir
  def get_dataset(self):
    image_paths = []
    label_paths = []
    cases = os.listdir(self.data_dir)
    for case in cases:
      image_paths.append(os.path.join(self.data_dir,case,self.image_filename))
      label_paths.append(os.path.join(self.data_dir,case,self.label_filename))

    dataset = tf.data.Dataset.from_tensor_slices((image_paths,label_paths))

    dataset = dataset.shuffle(buffer_size=len(cases))
    dataset = dataset.map(lambda image_path, label_path: tuple(tf.py_func(
      self.input_parser, [image_path, label_path], [tf.float32,tf.int32])), num_parallel_calls=4)

    self.dataset = dataset
    self.data_size = len(image_paths)
    return self.dataset

  def read_image(self,path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute()
    return image

  def input_parser(self,image_path, label_path):
    # read image and label
    image = self.read_image(image_path.decode("utf-8"))
    castImageFilter = sitk.CastImageFilter()

    #castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    castImageFilter.SetOutputPixelType(sitk.sitkInt16)
    # Workaround: convert to 4D Numpy then back to 3D SimpleITK images
    # This will have length N, where N is the number of (MRI) series
    image_np = sitk.GetArrayFromImage(image)
    itkImages3d = []
    if len(image_np.shape) == 3:
      itkImage3d = sitk.GetImageFromArray(image_np)
      itkImage3d.SetOrigin(image.GetOrigin())
      itkImage3d.SetSpacing(image.GetSpacing())
      itkImage3d.SetDirection(image.GetDirection())
      itkImage3d = castImageFilter.Execute(itkImage3d)
      itkImages3d += [itkImage3d]
      #itkImages3d = [image_np]
    else:
      # Weird NIFTI convention: Dimension 3 is the index!
      for i in range(image_np.shape[3]): 
        volume = image_np[:,:,:,i]
        itkImage3d = sitk.GetImageFromArray(volume)
        itkImage3d.SetOrigin(image.GetOrigin())
        itkImage3d.SetSpacing(image.GetSpacing())
        itkImage3d.SetDirection(image.GetDirection())
        itkImage3d = castImageFilter.Execute(itkImage3d)
        itkImages3d += [itkImage3d]
    
    if self.train:
      label = self.read_image(label_path.decode("utf-8"))
      castImageFilter.SetOutputPixelType(sitk.sitkInt8)
      label = castImageFilter.Execute(label)
    else:
      # Not sure this handles the case where "image" is 4D
      label = sitk.Image(itkImages3d[0].GetSize(),sitk.sitkInt8)
      label.SetOrigin(itkImages3d[0].GetOrigin())
      label.SetSpacing(itkImages3d[0].GetSpacing())
      label.SetDirection(itkImages3d[0].GetDirection())
    # Form the sample dict with the list of 3D ITK Images and the label
    sample = {'image': itkImages3d, 'label':label}
    
    if self.peek_dir != '':
      case_name = os.path.basename(os.path.dirname(image_path)) # for peeking
      peek_case_folder = os.path.join(os.fsencode(self.peek_dir), case_name)
      try:
        os.mkdir(peek_case_folder)
      except:
        pass
      writer = sitk.ImageFileWriter()
      writer.UseCompressionOn()
   
    images_np=[]
    labels_np=[]
    for multiple_crops in range(0,self.num_crops):
      sample_tfm = sample
      if self.transforms:
        for transform in self.transforms:
          sample_tfm = transform(sample_tfm)
      
      # If we are peeking, write the images to the directory
      if self.peek_dir != '':
        i=0
        for volume in sample_tfm['image']:
          
          writer.SetFileName(os.fsdecode(os.path.join(peek_case_folder,os.fsencode('img_%d_crop%d.nii.gz'%(i,multiple_crops)))))
          writer.Execute(volume)
          i+=1
        writer.SetFileName(os.fsdecode(os.path.join(peek_case_folder,os.fsencode('label_crop%d.nii.gz'%(multiple_crops)))))
        castImageFilter.SetOutputPixelType(sitk.sitkInt16)
        labelInt16 = castImageFilter.Execute(sample_tfm['label'])
        labelInt16.CopyInformation(sample_tfm['image'][0])
        writer.Execute(labelInt16)

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

  def __init__(self, sigmaUp, sigmaDown, nonzero_only=False):
    self.name = 'StatisticalNormalization'
    #assert isinstance(nSigma, float)
    #self.nSigma = nSigma
    assert isinstance(sigmaUp, float)
    assert isinstance(sigmaDown, float)
    assert isinstance(nonzero_only, bool)
    self.sigmaUp = sigmaUp
    self.sigmaDown = sigmaDown
    self.nonzero_only = nonzero_only
    self.threshold = 0.01

  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    statisticsFilter = sitk.StatisticsImageFilter()
    normalizedImage = image
    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    intensityWindowingFilter.SetOutputMaximum(255)
    intensityWindowingFilter.SetOutputMinimum(0)
    for i,volume in enumerate(image):
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
      #intensityWindowingFilter.SetWindowMinimum(max(self.threshold, mean - self.sigmaDown * sigma))
      intensityWindowingFilter.SetWindowMinimum(mean - self.sigmaDown * sigma)
      
      normalizedImage[i] = intensityWindowingFilter.Execute(volume)
    return {'image': normalizedImage, 'label': label}

class ManualNormalization(object):
  """
  Normalize an image by mapping intensity with given max and min window level
  """

  def __init__(self,windowMin, windowMax):
    self.name = 'ManualNormalization'
    assert isinstance(windowMax, (int,float))
    assert isinstance(windowMin, (int,float))
    self.windowMax = windowMax
    self.windowMin = windowMin

  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    intensityWindowingFilter.SetOutputMaximum(255)
    intensityWindowingFilter.SetOutputMinimum(0)
    intensityWindowingFilter.SetWindowMaximum(self.windowMax);
    intensityWindowingFilter.SetWindowMinimum(self.windowMin);

    image[:] = [intensityWindowingFilter.Execute(volume) for volume in image]

    return {'image': image, 'label': label}

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
    
    old_spacing = image[0].GetSpacing()
    old_size = image[0].GetSize()
    
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
    resampler.SetOutputOrigin(image[0].GetOrigin())
    resampler.SetOutputDirection(image[0].GetDirection())
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
    size_old = image[0].GetSize()
    
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
      resampler.SetOutputSpacing(image[0].GetSpacing())
      resampler.SetSize(output_size)

      # resample on image
      resampler.SetInterpolator(2)
      resampler.SetOutputOrigin(image[0].GetOrigin())
      resampler.SetOutputDirection(image[0].GetDirection())
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
    size_old = image[0].GetSize()
    size_new = self.output_size

    contain_label = False

    roiFilter = sitk.RegionOfInterestImageFilter()
    roiFilter.SetSize([size_new[0],size_new[1],size_new[2]])

    # statFilter = sitk.StatisticsImageFilter()
    # statFilter.Execute(label)
    
    # Calculate this boolean for the image, not for the crop
    # That way, drop_ratio is the fraction of images for which we get an empty crop,
    # not the fraction of crops
    keep_empty = self.drop(self.drop_ratio)
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
      statFilter = sitk.StatisticsImageFilter()
      statFilter.Execute(label_crop)

      # will iterate until a sub volume containing label is extracted
      # pixel_count = seg_crop.GetHeight()*seg_crop.GetWidth()*seg_crop.GetDepth()
      # if statFilter.GetSum()/pixel_count<self.min_ratio:
      #if statFilter.GetSum()>self.min_pixel:
      #  contain_label = keep_self.drop(self.drop_ratio) # has some probabilty to contain patch with empty label
      #else:
      #  contain_label = True
      if statFilter.GetSum()>self.min_pixel or keep_empty is True:
        contain_label = True

    image_crop = [roiFilter.Execute(volume) for volume in image]

    return {'image': image_crop, 'label': label_crop}

  def drop(self,probability):
    return random.random() <= probability

class RandomNoise(object):
  """
  Randomly noise to the image in a sample. This is usually used for data augmentation.
  """
  def __init__(self):
    self.name = 'Random Noise'

  def __call__(self, sample):
    self.noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
    self.noiseFilter.SetMean(0)
    self.noiseFilter.SetStandardDeviation(0.1)

    # print("Normalizing image...")
    image, label = sample['image'], sample['label']
    image[:] = [self.noiseFilter.Execute(volume) for volume in image]

    return {'image': image, 'label': label}

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
      if centroid[i] < (self.output_size[i]/2):
        centroid[i] = int(self.output_size[i]/2)
      elif (image[0].GetSize()[i]-centroid[i]) < (self.output_size[i]/2):
        centroid[i] = image[0].GetSize()[i] - int(self.output_size[i]/2) -1

      # get start point
      while ((start[i]<0) or (end[i]>(image[0].GetSize()[i]-1))):
        offset[i] = self.NormalOffset(self.output_size[i],self.sigma[i])
        start[i] = centroid[i] + offset[i] - int(self.output_size[i]/2)
        end[i] = start[i] + self.output_size[i] - 1

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
    domain_physical_dimensions = [image[0].GetSize()[0]*image[0].GetSpacing()[0],image[0].GetSize()[1]*image[0].GetSpacing()[1],image[0].GetSize()[2]*image[0].GetSpacing()[2]]

    bspline = sitk.BSplineTransform(3, spline_order)
    bspline.SetTransformDomainOrigin(image[0].GetOrigin())
    bspline.SetTransformDomainDirection(image[0].GetDirection())
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
    rot_center = image[0].TransformIndexToPhysicalPoint((image[0].GetSize()[0]//2, image[0].GetSize()[1]//2,image[0].GetSize()[2]//2))
    
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
    image_size = image[0].GetSize()
    reference_size = tuple([math.ceil(i*1.2) for i in list(image_size)])
    resampler.SetOutputSpacing(image[0].GetSpacing())
    resampler.SetSize(reference_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(image[0].GetOrigin())
    resampler.SetOutputDirection(image[0].GetDirection())
    reference = resampler.Execute(image[0])
    reference = image[0]

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
    castImageFilter.SetOutputPixelType(sitk.sitkInt16)
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

