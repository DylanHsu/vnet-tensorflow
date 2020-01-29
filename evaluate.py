from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import NiftiDataset
import os
import datetime
import SimpleITK as sitk
import math
import numpy as np
from tqdm import tqdm

# select gpu devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # e.g. "0,1,2", "0,2" 

# tensorflow app flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir','./data/evaluate',
    """Directory of evaluation data""")
tf.app.flags.DEFINE_string('image_filename','img.nii.gz',
    """Image filename""")
tf.app.flags.DEFINE_string('label_filename','label.nii.gz',
    """Label filename""")
tf.app.flags.DEFINE_string('model_path','./tmp/ckpt/checkpoint-2240.meta',
    """Path to saved models""")
tf.app.flags.DEFINE_string('checkpoint_path','./tmp/ckpt/checkpoint-2240',
    """Directory of saved checkpoints""")
tf.app.flags.DEFINE_integer('patch_size',128,
    """Size of a data patch""")
tf.app.flags.DEFINE_integer('patch_layer',128,
    """Number of layers in data patch""")
tf.app.flags.DEFINE_integer('stride_inplane', 32, 
    """Stride size in 2D plane""")
tf.app.flags.DEFINE_integer('stride_layer',32, 
    """Stride size in layer direction""")
tf.app.flags.DEFINE_integer('batch_size',1,
    """Setting batch size (currently only accept 1)""")
tf.app.flags.DEFINE_string('suffix','',
    """Suffix for saving""")
tf.app.flags.DEFINE_string('case','',
    """Specific case to evaluate""")
tf.app.flags.DEFINE_boolean('is_batch',False,
    """Disable progress bar if this is a batch job""")
#python evaluate.py --data_dir /data/deasy/DylanHsu/N120/testing --model_path tmp/ckpt/bak/checkpoint_n120-4Layers-945.meta  --checkpoint_path tmp/ckpt/bak/checkpoint_n120-4Layers-945 --patch_size 32 --patch_layer 16 --stride_inplane 16 --stride_layer 8
def np_dice_coe(output, target, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
    output = output[:,:,:,1]
    target = target[:,:,:,1]
    #axis = tuple(axis)
    axis = (0,1,2)
    inse = np.sum(output*target, axis=axis)

    if loss_type == 'jaccard':
        l = np.sum(output*output, axis=axis)
        r = np.sum(target*target, axis=axis)
    elif loss_type == 'sorensen':
        l = np.sum(output, axis=axis)
        r = np.sum(target, axis=axis)
    else:
        raise Exception("Unknown loss_type")
    ## old axis=[0,1,2,3]
    dice = 2 * (inse) / (l + r)
    epsilon = 1e-5
    dice = np.clip(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    #dice = (tf.constant(2.0) * tf.cast(inse,dtype=tf.float32) + tf.constant(smooth)) / (tf.cast(l + r, dtype=tf.float32) + tf.constant(smooth))
    ##
    #dice = tf.reduce_mean(dice)
    return dice

def prepare_batch(image,ijk_patch_indices):
    # not using this function anymore - requires too much memory!
    image_batches = []
    for batch in ijk_patch_indices:
        image_batch = []
        for patch in batch:
            image_patch = image[patch[0]:patch[1],patch[2]:patch[3],patch[4]:patch[5]]
            image_batch.append(image_patch)

        image_batch = np.asarray(image_batch)
        #image_batch = image_batch[:,:,:,:,np.newaxis]
        image_batch = image_batch[:,:,:,:]
        image_batches.append(image_batch)
        
    return image_batches

def get_one_batch(image, batch):
    image_batch = []
    for patch in batch:
        image_patch = image[patch[0]:patch[1],patch[2]:patch[3],patch[4]:patch[5]]
        image_batch.append(image_patch)
    image_batch = np.asarray(image_batch)
    return image_batch

def evaluate():
    """evaluate the vnet model by stepwise moving along the 3D image"""
    # restore model grpah
    tf.reset_default_graph()
    imported_meta = tf.train.import_meta_graph(FLAGS.model_path)

    input_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, 1) #FLAGS.num_channels) 
    output_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, 1) # 1 for binary classification
    #images_placeholder_eval = tf.placeholder(tf.float32, shape=input_batch_shape, name="images_placeholder")
    #labels_placeholder_eval = tf.placeholder(tf.int32, shape=output_batch_shape, name="labels_placeholder")   
    # create transformations to image and labels
    transforms = [
        # NiftiDataset.Normalization(),
        NiftiDataset.StatisticalNormalization(5.0,5.0,nonzero_only=True,zero_floor=True),
        #NiftiDataset.Resample(0.75),
        #NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer))      
        ]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:  
        print("{}: Start evaluation...".format(datetime.datetime.now()))

        imported_meta.restore(sess, FLAGS.checkpoint_path)
        print("{}: Restore checkpoint success".format(datetime.datetime.now()))
        
        for case in os.listdir(FLAGS.data_dir):
            if FLAGS.case != '' and FLAGS.case != case:
                continue
            # ops to load data
            # support multiple image input, but here only use single channel, label file should be a single file with different classes

            # check image data exists
            image_path = os.path.join(FLAGS.data_dir,case,FLAGS.image_filename)
            true_label_path = os.path.join(FLAGS.data_dir,case,FLAGS.label_filename) #hack

            if not os.path.exists(image_path):
                print("{}: Image file not found at {}".format(datetime.datetime.now(),image_path))
                break
            else:
                print("{}: Evaluating image at {}".format(datetime.datetime.now(),image_path))

                # read image file
                reader = sitk.ImageFileReader()
                reader.SetFileName(image_path)
                image = reader.Execute()

                # preprocess the image and label before inference
                image_np = sitk.GetArrayFromImage(image)
                itkImages3d = []
                # Weird NIFTI convention: Dimension 3 is the index!
                if len(image_np.shape) == 3:
                  itkImage3d = sitk.GetImageFromArray(image_np)
                  itkImages3d += [itkImage3d]
                else:
                  for i in range(image_np.shape[3]): 
                    volume = image_np[:,:,:,i]
                    itkImage3d = sitk.GetImageFromArray(volume)
                    itkImages3d += [itkImage3d]
               
                # create empty label in pair with transformed image
                label_dims = image.GetSize()
                print('label_dims are ', label_dims)
                label_tfm = sitk.Image(label_dims,sitk.sitkUInt32)
                label_tfm.SetOrigin(image.GetOrigin())
                label_tfm.SetDirection(image.GetDirection())
                label_tfm.SetSpacing(image.GetSpacing())

                sample = {'image': itkImages3d, 'label': label_tfm}

                #hack
                reader.SetFileName(true_label_path)
                true_label = reader.Execute()
                statFilter = sitk.StatisticsImageFilter()
                statFilter.Execute(true_label)
                true_label_volume = statFilter.GetSum()
                true_sample = {'image': [], 'label': true_label}
                for transform in transforms:
                    sample = transform(sample)
                    true_sample = transform(true_sample)

                tfmItkImages3d, label_tfm = sample['image'], sample['label']
                true_label_tfm = true_sample['label']

                # create empty softmax image in pair with transformed image
                #softmax_tfm = sitk.Image(label_dims,sitk.sitkFloat32)
                softmax_tfm = sitk.Image(label_dims,sitk.sitkFloat32)
                softmax_tfm.SetOrigin(tfmItkImages3d[0].GetOrigin())
                softmax_tfm.SetDirection(image.GetDirection())
                softmax_tfm.SetSpacing(tfmItkImages3d[0].GetSpacing())

                # convert image to numpy array
                image_np = [] # New size of image_np, inferred from the transformed shape
                for volume in sample['image']:
                  image_np += [sitk.GetArrayFromImage(volume)]
                image_np = np.asarray(image_np,np.float32)

                label_np = sitk.GetArrayFromImage(label_tfm)
                label_np = np.asarray(label_np,np.int32)
                true_label_np = sitk.GetArrayFromImage(true_label_tfm)
                true_label_np = np.asarray(true_label_np,np.int32)

                softmax_np = sitk.GetArrayFromImage(softmax_tfm)
                softmax_np = np.asarray(softmax_np,np.float32)

                # unify numpy and sitk orientation
                image_np = np.transpose(image_np,(3,2,1,0)) # (T,Z,Y,X) -> (X,Y,Z,T) 
                label_np = np.transpose(label_np,(2,1,0))
                true_label_np = np.transpose(true_label_np,(2,1,0))
                true_label_np = np.clip(true_label_np, 0, 1)
                softmax_np = np.transpose(softmax_np,(2,1,0))
                #print('after transpose, image_np has shape ', image_np.shape)
                #print('after transpose, label_np has shape ', label_np.shape)
                #print('after transpose, softmax_np has shape ', softmax_np.shape)

                # a weighting matrix will be used for averaging the overlapped region
                weight_np = np.zeros(label_np.shape)

                # prepare image batch indices
                inum = int(math.ceil((image_np.shape[0]-FLAGS.patch_size)/float(FLAGS.stride_inplane))) + 1 
                jnum = int(math.ceil((image_np.shape[1]-FLAGS.patch_size)/float(FLAGS.stride_inplane))) + 1
                knum = int(math.ceil((image_np.shape[2]-FLAGS.patch_layer)/float(FLAGS.stride_layer))) + 1

                patch_total = 0
                ijk_patch_indices = []
                ijk_patch_indicies_tmp = []

                for i in range(inum):
                    for j in range(jnum):
                        for k in range (knum):
                            if patch_total % FLAGS.batch_size == 0:
                                ijk_patch_indicies_tmp = []

                            istart = i * FLAGS.stride_inplane
                            if istart + FLAGS.patch_size > image_np.shape[0]: #for last patch
                                istart = image_np.shape[0] - FLAGS.patch_size 
                            iend = istart + FLAGS.patch_size

                            jstart = j * FLAGS.stride_inplane
                            if jstart + FLAGS.patch_size > image_np.shape[1]: #for last patch
                                jstart = image_np.shape[1] - FLAGS.patch_size 
                            jend = jstart + FLAGS.patch_size

                            kstart = k * FLAGS.stride_layer
                            if kstart + FLAGS.patch_layer > image_np.shape[2]: #for last patch
                                kstart = image_np.shape[2] - FLAGS.patch_layer 
                            kend = kstart + FLAGS.patch_layer

                            ijk_patch_indicies_tmp.append([istart, iend, jstart, jend, kstart, kend])

                            if patch_total % FLAGS.batch_size == 0:
                                ijk_patch_indices.append(ijk_patch_indicies_tmp)

                            patch_total += 1
                            #print('Patch %d will encapsulate (%d,%d,%d) to (%d,%d,%d)' % (patch_total, istart, jstart, kstart, iend, jend, kend))
                
                #batches = prepare_batch(image_np,ijk_patch_indices)

                # actual segmentation
                #for i in tqdm(range(len(batches))):
                if not FLAGS.is_batch:
                    the_iterations = tqdm(range(len(ijk_patch_indices)))
                else:
                    the_iterations = range(len(ijk_patch_indices))

                for i in the_iterations:
                    batch = ijk_patch_indices[i]
                    #batch_image = batches[i]
                    batch_image = get_one_batch(image_np, batch)
                    [pred, softmax] = sess.run(['predicted_label/prediction:0','softmax/softmax:0'], feed_dict={'images_placeholder:0': batch_image})
                    istart = ijk_patch_indices[i][0][0]
                    iend = ijk_patch_indices[i][0][1]
                    jstart = ijk_patch_indices[i][0][2]
                    jend = ijk_patch_indices[i][0][3]
                    kstart = ijk_patch_indices[i][0][4]
                    kend = ijk_patch_indices[i][0][5]
                    label_np[istart:iend,jstart:jend,kstart:kend] += pred[0,:,:,:]
                    softmax_np[istart:iend,jstart:jend,kstart:kend] += softmax[0,:,:,:,1]
                    weight_np[istart:iend,jstart:jend,kstart:kend] += 1.0

                print("{}: Evaluation complete".format(datetime.datetime.now()))
                # eliminate overlapping region using the weighted value
                label_np = np.rint(np.float32(label_np)/np.float32(weight_np) + 0.0001)
                #softmax_np = softmax_np/np.float32(weight_np)
                softmax_np = softmax_np/np.float16(weight_np)
                
                softmax_onehot = np.transpose(np.asarray([1-softmax_np,softmax_np],np.float32),(1,2,3,0))
                true_label_onehot = np.eye(2)[true_label_np]
                label_onehot = np.eye(2)[label_np.astype(np.int16)]
                the_dice = np_dice_coe(label_onehot, true_label_onehot,loss_type='sorensen', axis=[0,1,2,3])
                true_dice = np_dice_coe(true_label_onehot, true_label_onehot,loss_type='sorensen', axis=[0,1,2,3])
                print('Dice score is %.3f, true label dice score is %.3f' % (the_dice,true_dice))


                # convert back to sitk space
                label_np = np.transpose(label_np,(2,1,0))
                softmax_np = np.transpose(softmax_np,(2,1,0))

                # convert label numpy back to sitk image
                label_tfm = sitk.GetImageFromArray(label_np)
                #label_tfm.SetOrigin(tfmItkImages3d[0].GetOrigin())
                #label_tfm.SetDirection(image.GetDirection())
                #label_tfm.SetSpacing(tfmItkImages3d[0].GetSpacing())
                label_tfm.SetOrigin(image.GetOrigin())
                label_tfm.SetDirection(image.GetDirection())
                label_tfm.SetSpacing(image.GetSpacing())

                softmax_tfm = sitk.GetImageFromArray(softmax_np)
                #softmax_tfm.SetOrigin(tfmItkImages3d[0].GetOrigin())
                #softmax_tfm.SetDirection(image.GetDirection())
                #softmax_tfm.SetSpacing(tfmItkImages3d[0].GetSpacing())
                softmax_tfm.SetOrigin(image.GetOrigin())
                softmax_tfm.SetDirection(image.GetDirection())
                softmax_tfm.SetSpacing(image.GetSpacing())

                # resample the label back to original space
                resampler = sitk.ResampleImageFilter()
                # save segmented label
                writer = sitk.ImageFileWriter()
                writer.UseCompressionOn()

                resampler.SetInterpolator(1)
                resampler.SetOutputSpacing(image.GetSpacing())
                resampler.SetSize(image.GetSize())
                resampler.SetOutputOrigin(image.GetOrigin())
                resampler.SetOutputDirection(image.GetDirection())
                
                print("{}: Resampling label back to original image space...".format(datetime.datetime.now()))
                #label = resampler.Execute(label_tfm)
                castFilter = sitk.CastImageFilter()
                castFilter.SetOutputPixelType(sitk.sitkInt16)
                label = castFilter.Execute(label_tfm)
                true_label = castFilter.Execute(true_label)


                label_path = os.path.join(FLAGS.data_dir,case,'label_vnet%s.nii.gz'%FLAGS.suffix)
                writer.SetFileName(label_path)
                writer.Execute(label)
                print("{}: Save evaluate label at {} success".format(datetime.datetime.now(),label_path))

                print("{}: Resampling probability map back to original image space...".format(datetime.datetime.now()))
                #prob = resampler.Execute(softmax_tfm)
                prob = softmax_tfm
                prob_path = os.path.join(FLAGS.data_dir,case,'probability_vnet%s.nii.gz'%FLAGS.suffix)
                writer.SetFileName(prob_path)
                writer.Execute(prob)
                print("{}: Save evaluate probability map at {} success".format(datetime.datetime.now(),prob_path))

                ccFilter = sitk.ConnectedComponentImageFilter()
                labelCC = ccFilter.Execute(label)
                trueLabelCC = ccFilter.Execute(true_label)
                
                labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
                #labelShapeFilter.SetComputeFeretDiameter(True)
                labelShapeFilter.Execute(labelCC)
                trueLabelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
                #labelShapeFilter.SetComputeFeretDiameter(True)
                trueLabelShapeFilter.Execute(trueLabelCC)

                biFilter = sitk.LabelMapToBinaryImageFilter()
                biFilter.SetBackgroundValue(0)
                biFilter.SetForegroundValue(1)
                mapFilter = sitk.LabelImageToLabelMapFilter()
                overlapFilter = sitk.LabelOverlapMeasuresImageFilter()
                statFilter=sitk.StatisticsImageFilter()
                
                trueLabelMap=mapFilter.Execute(trueLabelCC,0)
                labelMap=mapFilter.Execute(labelCC,0)
                
                cubicMmPerVoxel = image.GetSpacing()[0] * image.GetSpacing()[1] * image.GetSpacing()[2];
                nonartifacts=list(range(1,labelShapeFilter.GetNumberOfLabels()+1))
                for j in range(1,labelShapeFilter.GetNumberOfLabels()+1):
                  if (labelShapeFilter.GetNumberOfPixels(j) * cubicMmPerVoxel) < 25.:
                    nonartifacts.remove(j)
                
                #falsePositives=list(range(1,trueLabelShapeFilter.GetNumberOfLabels()+1))
                #falseNegatives = nonartifacts 
                falseNegatives=list(range(1,trueLabelShapeFilter.GetNumberOfLabels()+1))
                falsePositives = nonartifacts 
                mif=sitk.LabelMapMaskImageFilter()
                if labelShapeFilter.GetNumberOfLabels()<=5000:
                  f_specificDice = open(os.path.join(FLAGS.data_dir,case,'specificDice%s.txt'%FLAGS.suffix), 'w')
                  for i in range(1,trueLabelShapeFilter.GetNumberOfLabels()+1):
                    mif.SetLabel(i)
                    true_lesion = mif.Execute(trueLabelMap, true_label)
                    found_pred = False
                    specific_dice = 0.0
                    for j in nonartifacts:
                      mif.SetLabel(j)
                      predicted_lesion = mif.Execute(labelMap, label)
                      overlapFilter.Execute(predicted_lesion, true_lesion)
                      if overlapFilter.GetDiceCoefficient() > 0.0 and not found_pred:
                        found_pred = True
                        specific_dice = overlapFilter.GetDiceCoefficient()
                        if j in falsePositives:
                          falsePositives.remove(j)
                    if found_pred and i in falseNegatives:
                      falseNegatives.remove(i)
                    statFilter.Execute(true_lesion)
                    volumetric_d = ((cubicMmPerVoxel * 6.*statFilter.GetSum()/3.14159) ** (1./3.))
                    # difficulty calculation using TRUE label
                    # instantiate filters
                    castImageFilter       = sitk.CastImageFilter()
                    roiFilter             = sitk.RegionOfInterestImageFilter()

                    true_lesion_np = np.transpose(sitk.GetArrayFromImage(true_lesion),(1,2,0))
                    nhd_size=(64,64,32)
                    roiFilter.SetSize(nhd_size)

                    centroid = list(true_label.TransformPhysicalPointToIndex(trueLabelShapeFilter.GetCentroid(i)))

                    start=[-1,-1,-1]
                    end=[-1,-1,-1] 
                    for dim in range(3):
                      if centroid[dim] < nhd_size[dim]/2:
                        centroid[dim] = int(nhd_size[dim]/2)
                      elif true_label.GetSize()[dim] - centroid[dim] < nhd_size[dim]/2:
                        centroid[dim] = true_label.GetSize()[dim] - int(nhd_size[dim]/2) - 1
                      
                      start[dim] = centroid[dim] - int(nhd_size[dim]/2)
                      end[dim]   = start[dim] + nhd_size[dim] - 1
                    roiFilter.SetIndex(start)
                    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
                    
                    roiFilter.SetIndex(start)
                    croppedLabel = roiFilter.Execute(true_label)
                    croppedLabel_np = np.transpose(sitk.GetArrayFromImage(croppedLabel),(1,2,0))
                    nhd_mean = 0.
                    lesion_mean = 0.
                    lesion_volume = -1
                    threshold = 0.001
                    for volume in sample['image']:
                      croppedImage = roiFilter.Execute(volume)
                      croppedImage_np = np.transpose(sitk.GetArrayFromImage(croppedImage),(1,2,0))
                      nhd_voxels = croppedImage_np[croppedLabel_np <= threshold] # non lesion voxels
                      nhd_mean += np.mean(nhd_voxels)
                      
                      image_np = np.transpose(sitk.GetArrayFromImage(volume),(1,2,0))
                      lesion_voxels = croppedImage_np[croppedLabel_np > threshold] 
                      lesion_mean += np.mean(lesion_voxels)
                      # compute lesion volume only once
                      if lesion_volume == -1:
                        lesion_volume = len(lesion_voxels)
                        q90 = np.quantile(lesion_voxels,0.9)
                        q10 = np.quantile(lesion_voxels,0.1)
                    nhd_mean = nhd_mean / float(len(sample['image']))
                    lesion_mean = lesion_mean / float(len(sample['image']))
                    #intensity_balance = abs(lesion_mean/nhd_mean - 1.)
                    #inverse_balance = 1./intensity_balance
                    intensity_balance = lesion_mean/nhd_mean - 1.
                    q90_balance = q90/nhd_mean - 1.
                    q10_balance = q10/nhd_mean - 1.
                    #print("nhd_mean = %.3f, q10 = %.3f, q90 = %.3f"%(nhd_mean,q10,q90))

                    #f_specificDice.write("%s %d %.2f %.4f %.3f\n"%(case, i, volumetric_d, inverse_balance, specific_dice))
                    f_specificDice.write("%s %d %.2f %.4f %.4f %.4f %.3f\n"%(case, i, volumetric_d, intensity_balance, q10_balance, q90_balance, specific_dice))
                  f_specificDice.close()
                
                f_dice = open(os.path.join(FLAGS.data_dir,case,'dice%s.txt'%FLAGS.suffix), 'w')
                f_dice.write("%s %d %.3f %d %d\n"%(case, true_label_volume, the_dice, len(falsePositives), len(falseNegatives)))
                f_dice.close()






def main(argv=None):
    evaluate()

if __name__=='__main__':
    tf.app.run()
