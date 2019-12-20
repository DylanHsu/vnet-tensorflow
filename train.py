from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import NiftiDataset
import os, sys
import VNet
import math
import datetime
from tensorflow.python import debug as tf_debug
import logging
import resource

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s','%m-%d %H:%M:%S'))
logger = logging.getLogger('vnet_trainer')
logger.addHandler(console)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# select gpu devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # e.g. "0,1,2", "0,2" 

# tensorflow app flags
FLAGS = tf.app.flags.FLAGS
# Hack for Tensorflow 1.14, log_dir already defined in Abseil dependency
for name in list(FLAGS):
    if name=='log_dir':
      delattr(FLAGS,name)
tf.app.flags.DEFINE_string('data_dir', './data',
    """Directory of stored data.""")
tf.app.flags.DEFINE_string('image_filename','img.nii',
    """Image filename""")
tf.app.flags.DEFINE_string('label_filename','label.nii',
    """Label filename""")
tf.app.flags.DEFINE_integer('batch_size',1,
    """Size of batch""")               
tf.app.flags.DEFINE_integer('accum_batches',1,
    """Accumulate the gradient over this many batches before updating the gradient (1 = no accumulation)""")               
tf.app.flags.DEFINE_integer('num_crops',1,
    """Take this many crops from each image, per epoch""")               
tf.app.flags.DEFINE_float('ccrop_sigma',2.5,
    """Value of sigma to use for confidence crops""")               
tf.app.flags.DEFINE_integer('num_channels',1,
    """Number of channels (MRI series)""")               
tf.app.flags.DEFINE_integer('patch_size',128,
    """Size of a data patch""")
tf.app.flags.DEFINE_integer('patch_layer',128,
    """Number of layers in data patch""")
tf.app.flags.DEFINE_integer('epochs',999999999,
    """Number of epochs for training""")
tf.app.flags.DEFINE_string('log_dir', './tmp/log',
    """Directory where to write training and testing event logs """)
tf.app.flags.DEFINE_float('init_learning_rate',1e-2,
    """Initial learning rate""")
tf.app.flags.DEFINE_float('decay_factor',0.01,
    """Exponential decay learning rate factor""")
tf.app.flags.DEFINE_integer('decay_steps',100,
    """Number of epoch before applying one learning rate decay""")
tf.app.flags.DEFINE_integer('display_step',10,
    """Display and logging interval (train steps)""")
tf.app.flags.DEFINE_integer('save_interval',1,
    """Checkpoint save interval (epochs)""")
tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/ckpt',
    """Directory where to write checkpoint""")
tf.app.flags.DEFINE_string('model_dir','./tmp/model',
    """Directory to save model""")
tf.app.flags.DEFINE_bool('restore_training',True,
    """Restore training from last checkpoint""")
tf.app.flags.DEFINE_float('drop_ratio',0.5,
    """Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1""")
tf.app.flags.DEFINE_integer('min_pixel',10,
    """Minimum non-zero pixels in the cropped label""")
tf.app.flags.DEFINE_integer('shuffle_buffer_size',50,
    """Number of elements used in shuffle buffer""")
tf.app.flags.DEFINE_string('loss_function','dice',
    """Loss function used in optimization (ce, wce, dwce, dice, jaccard)""")
tf.app.flags.DEFINE_string('optimizer','sgd',
    """Optimization method (sgd, adam, momentum, nesterov_momentum)""")
tf.app.flags.DEFINE_float('momentum',0.5,
    """Momentum used in optimization""")
tf.app.flags.DEFINE_boolean('is_batch_job',False,
    """Disable some features if this is a batch job""")
tf.app.flags.DEFINE_string('batch_job_name','',
    """Name the batch job so the checkpoints and tensorboard output are identifiable.""")
tf.app.flags.DEFINE_float('max_ram',15.5,
    """Maximum amount of RAM usable by the CPU in GB.""")
tf.app.flags.DEFINE_float('wce_weight','10',
    """Weight to use for the True labels to fight class imbalance in the Weighted Cross Entropy.""")

tf.app.flags.DEFINE_string('vnet_convs','1,2,3,3,3',
    """Convolutions in each VNET layer.""")
tf.app.flags.DEFINE_integer('vnet_channels',16,
    """Channels after initial VNET convolution.""")
tf.app.flags.DEFINE_float('dropout_keepprob',1.0,
    """probability to randomly keep a parameter for dropout (default 1 = no dropout)""")
tf.app.flags.DEFINE_float('l2_weight',0.0,
    """Weight for L2 regularization (should be order of 0.0001)""")

# tf.app.flags.DEFINE_float('class_weight',0.15,
#     """The weight used for imbalanced classes data. Currently only apply on binary segmentation class (weight for 0th class, (1-weight) for 1st class)""")

def placeholder_inputs(input_batch_shape, output_batch_shape):
    """Generate placeholder variables to represent the the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded ckpt in the .run() loop, below.
    Args:
        patch_shape: The patch_shape will be baked into both placeholders.
    Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test ckpt sets.
    # batch_size = -1

    images_placeholder = tf.placeholder(tf.float32, shape=input_batch_shape, name="images_placeholder")
    labels_placeholder = tf.placeholder(tf.int32, shape=output_batch_shape, name="labels_placeholder")   
   
    return images_placeholder, labels_placeholder

def dice_coe(output, target, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5, compute='quotient'):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``dice``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """

    inse = tf.reduce_sum(tf.multiply(output,target), axis=axis)

    if loss_type == 'jaccard':
        l = tf.reduce_sum(tf.multiply(output,output), axis=axis)
        r = tf.reduce_sum(tf.multiply(target,target), axis=axis)
    elif loss_type == 'dice':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknown loss_type")
    
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    tf_smooth = tf.constant(smooth)
    numerator = tf.constant(2.0) * tf.cast(inse,dtype=tf.float32)
    denominator = tf.cast(l + r, dtype=tf.float32)
    if compute == 'quotient':
      dice = (numerator + tf_smooth) / (denominator + tf_smooth)
      #dice = (tf.constant(2.0) * tf.cast(inse,dtype=tf.float32) + tf.constant(smooth)) / (tf.cast(l + r, dtype=tf.float32) + tf.constant(smooth))
      dice = tf.reduce_mean(dice)
      return dice
    elif compute == 'numerator':
      numerator = tf.reduce_sum(numerator)
      return numerator
    elif compute == 'denominator':
      denominator = tf.reduce_sum(denominator)
      return denominator
    else:
      raise Exception("Unknown compute")

def train():
    """Train the Vnet model"""
    resource.setrlimit(resource.RLIMIT_DATA,(math.ceil((FLAGS.max_ram-1.)*(1024**2)*1000),math.ceil(FLAGS.max_ram*(1024**2)*1000))) # 1000 MB ~ 1 GB
    latest_filename = "checkpoint"
    if FLAGS.is_batch_job and FLAGS.batch_job_name is not '':
        latest_filename = latest_filename + "_" + FLAGS.batch_job_name
        resource.setrlimit(resource.RLIMIT_CORE,(524288,-1))
    latest_filename += "_latest"
    
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # patch_shape(batch_size, height, width, depth, channels)
        input_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, FLAGS.num_channels) 
        output_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, 1) # 1 for binary classification

        images_placeholder, labels_placeholder = placeholder_inputs(input_batch_shape,output_batch_shape)

        for batch in range(FLAGS.batch_size):
            images_log = tf.cast(images_placeholder[batch:batch+1,:,:,:,0], dtype=tf.uint8)
            labels_log = tf.cast(tf.scalar_mul(255,labels_placeholder[batch:batch+1,:,:,:,0]), dtype=tf.uint8)

            # needs attention for 4D support
            # DGH: disable images
            #tf.summary.image("image", tf.transpose(images_log,[3,1,2,0]),max_outputs=FLAGS.patch_layer)
            #tf.summary.image("label", tf.transpose(labels_log,[3,1,2,0]),max_outputs=FLAGS.patch_layer)

        # Get images and labels
        train_data_dir = os.path.join(FLAGS.data_dir,'training')
        test_data_dir = os.path.join(FLAGS.data_dir,'testing')
        # support multiple image input, but here only use single channel, label file should be a single file with different classes
        

        # Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
        with tf.device('/cpu:0'):
            # create transformations to image and labels
            trainTransforms = [
                NiftiDataset.RandomHistoMatch(train_data_dir, FLAGS.image_filename, 1.0),
                NiftiDataset.StatisticalNormalization(5.0, 5.0, nonzero_only=True),
                NiftiDataset.BSplineDeformation(),
                NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),FLAGS.drop_ratio,FLAGS.min_pixel),
                NiftiDataset.RandomNoise(),
                NiftiDataset.RandomFlip(0.5, [True,True,True]),
                
                #NiftiDataset.ThresholdCrop(),
                #NiftiDataset.RandomRotation(maxRot = 0.08727), # 5 degrees maximum
                #NiftiDataset.RandomRotation(maxRot = 10*0.01745), 
                #NiftiDataset.ThresholdCrop(),
                #NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
                #NiftiDataset.ConfidenceCrop((FLAGS.patch_size,FLAGS.patch_size,FLAGS.patch_layer), FLAGS.ccrop_sigma),
                # NiftiDataset.Normalization(),
                #NiftiDataset.Resample((0.45,0.45,0.45)),
                ]
            testTransforms = [
                NiftiDataset.StatisticalNormalization(5.0, 5.0, nonzero_only=True),
                NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),0,20),
                ]
            
            TrainDataset = NiftiDataset.NiftiDataset(
                data_dir=train_data_dir,
                image_filename=FLAGS.image_filename,
                label_filename=FLAGS.label_filename,
                transforms=trainTransforms,
                num_crops=FLAGS.num_crops,
                train=True
                #peek_dir='data/peek_train'
                )
            
            trainDataset = TrainDataset.get_dataset()
            # Here there are batches of size num_crops, unbatch and shuffle
            trainDataset = trainDataset.apply(tf.contrib.data.unbatch())
            trainDataset = trainDataset.repeat(3) 
            trainDataset = trainDataset.batch(FLAGS.batch_size)
            trainDataset = trainDataset.prefetch(5)
            #trainDataset = trainDataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))


            TestDataset = NiftiDataset.NiftiDataset(
                data_dir=test_data_dir,
                image_filename=FLAGS.image_filename,
                label_filename=FLAGS.label_filename,
                transforms=testTransforms,
                num_crops=FLAGS.num_crops, #10
                train=True
            )

            testDataset = TestDataset.get_dataset()
            # Here there are batches of size num_crops, unbatch and shuffle
            testDataset = testDataset.apply(tf.contrib.data.unbatch())
            testDataset = testDataset.repeat(3)
            testDataset = testDataset.batch(FLAGS.batch_size)
            testDataset = testDataset.prefetch(5)
            #testDataset = testDataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))
            
        train_iterator = trainDataset.make_initializable_iterator()
        next_element_train = train_iterator.get_next()

        test_iterator = testDataset.make_initializable_iterator()
        next_element_test = test_iterator.get_next()

        # Initialize the model
        with tf.name_scope("vnet"):
            convs=[int(i) for i in (FLAGS.vnet_convs).split(',')]
            model = VNet.VNet(
                num_classes=2,   
                keep_prob=FLAGS.dropout_keepprob,   
                num_channels=FLAGS.vnet_channels, 
                num_levels=len(convs)-1,    
                num_convolutions= tuple(convs[0:-1]),
                bottom_convolutions= convs[-1], 
                activation_fn="prelu") 

            logits = model.network_fn(images_placeholder)

        for batch in range(FLAGS.batch_size):
            logits_max = tf.reduce_max(logits[batch:batch+1,:,:,:,:])
            logits_min = tf.reduce_min(logits[batch:batch+1,:,:,:,:])

            logits_log_0 = logits[batch:batch+1,:,:,:,0]
            logits_log_1 = logits[batch:batch+1,:,:,:,1]

            # normalize to 0-255 range
            logits_log_0 = tf.cast((logits_log_0-logits_min)*255./(logits_max-logits_min), dtype=tf.uint8)
            logits_log_1 = tf.cast((logits_log_1-logits_min)*255./(logits_max-logits_min), dtype=tf.uint8)

            # DGH: disable images
            #tf.summary.image("logits_0", tf.transpose(logits_log_0,[3,1,2,0]),max_outputs=FLAGS.patch_layer)
            #tf.summary.image("logits_1", tf.transpose(logits_log_1,[3,1,2,0]),max_outputs=FLAGS.patch_layer)

        # Exponential decay learning rate
        # This number is wrong, fix
        train_batches_per_epoch = math.ceil(TrainDataset.data_size/(FLAGS.batch_size * FLAGS.accum_batches))
        decay_steps = train_batches_per_epoch*FLAGS.decay_steps

        with tf.name_scope("learning_rate"):
            learning_rate = FLAGS.init_learning_rate
            #learning_rate = tf.train.exponential_decay(FLAGS.init_learning_rate,
            #    global_step,
            #    decay_steps,
            #    FLAGS.decay_factor,
            #    staircase=False)
        #tf.summary.scalar('learning_rate', learning_rate)

        # softmax op for probability layer
        with tf.name_scope("softmax"):
            softmax_op = tf.nn.softmax(logits,name="softmax")

        for batch in range(FLAGS.batch_size):
            # grayscale to rainbow colormap, convert to HSV (H = reversed grayscale from 0:2/3, S and V are all 1)
            # then convert to RGB
            softmax_log_0H = (1. - tf.transpose(softmax_op[batch:batch+1,:,:,:,0],[3,1,2,0]))*2./3.
            softmax_log_1H = (1. - tf.transpose(softmax_op[batch:batch+1,:,:,:,1],[3,1,2,0]))*2./3.

            softmax_log_0H = tf.squeeze(softmax_log_0H,axis=-1)
            softmax_log_1H = tf.squeeze(softmax_log_1H,axis=-1)
            softmax_log_SV = tf.ones(softmax_log_0H.get_shape())

            softmax_log_0 = tf.stack([softmax_log_0H,softmax_log_SV,softmax_log_SV], axis=3)
            softmax_log_1 = tf.stack([softmax_log_1H,softmax_log_SV,softmax_log_SV], axis=3)

            softmax_log_0 = tf.image.hsv_to_rgb(softmax_log_0)
            softmax_log_1 = tf.image.hsv_to_rgb(softmax_log_1)

            softmax_log_0 = tf.cast(tf.scalar_mul(255,softmax_log_0), dtype=tf.uint8)
            softmax_log_1 = tf.cast(tf.scalar_mul(255,softmax_log_1), dtype=tf.uint8)
           
            # DGH: disable images
            # tf.summary.image("softmax_0", softmax_log_0,max_outputs=FLAGS.patch_layer)
            # tf.summary.image("softmax_1", softmax_log_1,max_outputs=FLAGS.patch_layer)

            # # this is grayscale one
            # softmax_log_0 = tf.cast(tf.scalar_mul(255,softmax_op[batch:batch+1,:,:,:,0]), dtype=tf.uint8)
            # softmax_log_1 = tf.cast(tf.scalar_mul(255,softmax_op[batch:batch+1,:,:,:,1]), dtype=tf.uint8)
            # tf.summary.image("softmax_0", tf.transpose(softmax_log_0,[3,1,2,0]),max_outputs=FLAGS.patch_layer)
            # tf.summary.image("softmax_1", tf.transpose(softmax_log_1,[3,1,2,0]),max_outputs=FLAGS.patch_layer)

        # Number of accumulated batches
        n_ab = tf.get_variable("n_ab", dtype=tf.float32, trainable=False, use_resource=True, initializer=tf.constant(0.))
        
        # Op for calculating loss
        with tf.name_scope("cross_entropy"):
            #ce_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            #    logits=logits,
            #    labels=tf.squeeze(labels_placeholder, 
            #    squeeze_dims=[4])))
            onehot_labels = tf.one_hot(tf.squeeze(labels_placeholder,squeeze_dims=[4]),depth = 2)
            ce_op = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(onehot_labels,logits,1.))
            ce_sum  = tf.get_variable("ce_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            ce2_sum = tf.get_variable("ce2_sum", dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            ce_avg = tf.cond(n_ab > 1., lambda: ce_sum/n_ab, lambda: tf.constant(0.)) 
            ce_stdev = tf.cond(n_ab > 1., lambda: tf.math.sqrt( (n_ab*ce2_sum - ce_sum*ce_sum) / (n_ab * (n_ab-1.))), lambda: tf.constant(0.))
            tf.summary.scalar('ce_batch',ce_avg)

        #with tf.name_scope("weighted_cross_entropy"):
            true_weight=FLAGS.wce_weight
            
            #onehot_labels = tf.one_hot(tf.squeeze(labels_placeholder,squeeze_dims=[4]),depth = 2)
            wce_op = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(onehot_labels,logits,true_weight))
                
            wce_sum  = tf.get_variable("wce_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            wce2_sum = tf.get_variable("wce2_sum", dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            wce_avg = tf.cond(n_ab > 1., lambda: wce_sum/n_ab, lambda: tf.constant(0.)) 
            wce_stdev = tf.cond(n_ab > 1., lambda: tf.math.sqrt( (n_ab*wce2_sum - wce_sum*wce_sum) / (n_ab * (n_ab-1.))), lambda: tf.constant(0.))
        
            tf.summary.scalar('wce_batch', wce_avg)
        #tf.summary.scalar('wce_stdev', wce_stdev)
        
        #with tf.name_scope("dynamic_cross_entropy"):
            # Dynamically weighted cross entropy
            # The true voxels in this crop receive a weight equal to (crop volume) / sum(true) / drop_ratio
            # This eliminates class imbalance but could cause numerical instability with extremely large crop volumes

            total_volume = tf.constant(FLAGS.patch_size*FLAGS.patch_size*FLAGS.patch_layer, tf.float32)
            label_volume = tf.cast(tf.reduce_sum(labels_placeholder),dtype=tf.float32);
            true_weight = tf.cond(label_volume>0, lambda: tf.divide(tf.divide(total_volume,label_volume), tf.constant(FLAGS.drop_ratio,tf.float32)), lambda: tf.constant(1.0, dtype=tf.float32))
            dynamic_class_weights = tf.stack([tf.constant(1.0, dtype=tf.float32), true_weight],0)
            
            # deduce weights for batch samples based on their true label
            weights = tf.reduce_sum(dynamic_class_weights * onehot_labels, axis=-1)
            # compute your (unweighted) softmax cross entropy loss
            unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf.squeeze(labels_placeholder, 
                squeeze_dims=[4]))
            # apply the weights, relying on broadcasting of the multiplication
            weighted_loss = unweighted_loss * weights
            # reduce the result to get your final loss
            dwce_op = tf.reduce_mean(weighted_loss)
            
            dwce_sum  = tf.get_variable("dwce_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            dwce2_sum = tf.get_variable("dwce2_sum", dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            dwce_avg = tf.cond(n_ab > 1., lambda: wce_sum/n_ab, lambda: tf.constant(0.)) 
            dwce_stdev = tf.cond(n_ab > 1., lambda: tf.math.sqrt( (n_ab*wce2_sum - wce_sum*wce_sum) / (n_ab * (n_ab-1.))), lambda: tf.constant(0.))

            tf.summary.scalar('dwce_batch', dwce_avg)
        #tf.summary.scalar('weighted_XE_loss',wce_op)

        # Argmax Op to generate label from logits
        with tf.name_scope("predicted_label"):
            pred = tf.argmax(logits, axis=4 , name="prediction")

        # DGH: disable images
        #for batch in range(FLAGS.batch_size):
        #    pred_log = tf.cast(tf.scalar_mul(255,pred[batch:batch+1,:,:,:]), dtype=tf.uint8)
        #    tf.summary.image("pred", tf.transpose(pred_log,[3,1,2,0]),max_outputs=FLAGS.patch_layer)

        # Accuracy of model
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.expand_dims(pred,-1), tf.cast(labels_placeholder,dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #tf.summary.scalar('accuracy', accuracy)

        
        # Dice Similarity, currently only for binary segmentation
        with tf.name_scope("dice"):
            # Define operations to compute Dice quantities
            
            # dice = dice_coe(tf.expand_dims(softmax_op[:,:,:,:,1],-1),tf.cast(labels_placeholder,dtype=tf.float32), loss_type='dice')
            # jaccard = dice_coe(tf.expand_dims(softmax_op[:,:,:,:,1],-1),tf.cast(labels_placeholder,dtype=tf.float32), loss_type='jaccard')
            
            # This is commented out because using all of the True-Negative pixels gives a very deceptive dice score
            # dice = dice_coe(softmax_op,tf.cast(tf.one_hot(labels_placeholder[:,:,:,:,0],depth=2),dtype=tf.float32), loss_type='dice', axis=[1,2,3,4])
            # jaccard = dice_coe(softmax_op,tf.cast(tf.one_hot(labels_placeholder[:,:,:,:,0],depth=2),dtype=tf.float32), loss_type='jaccard', axis=[1,2,3,4])
            
            # Computing the dice using only the second row of the 2-entry softmax vector seems more useful
            specific_dice_op     = dice_coe(softmax_op[:,:,:,:,1],tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='dice', axis=[1,2,3])
            specific_jaccard_op  = dice_coe(softmax_op[:,:,:,:,1],tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='jaccard', axis=[1,2,3])
            
            soft_dice_numerator_op      = dice_coe(softmax_op[:,:,:,:,1], tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='dice',axis=[1,2,3],compute='numerator')
            soft_dice_denominator_op    = dice_coe(softmax_op[:,:,:,:,1], tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='dice',axis=[1,2,3],compute='denominator')
            soft_jaccard_numerator_op   = dice_coe(softmax_op[:,:,:,:,1], tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='jaccard',axis=[1,2,3],compute='numerator')
            soft_jaccard_denominator_op = dice_coe(softmax_op[:,:,:,:,1], tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='jaccard',axis=[1,2,3],compute='denominator')
            
            hard_dice_numerator_op      = dice_coe(tf.round(softmax_op[:,:,:,:,1]), tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='dice',axis=[1,2,3],compute='numerator')
            hard_dice_denominator_op    = dice_coe(tf.round(softmax_op[:,:,:,:,1]), tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='dice',axis=[1,2,3],compute='denominator')
            hard_jaccard_numerator_op   = dice_coe(tf.round(softmax_op[:,:,:,:,1]), tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='jaccard',axis=[1,2,3],compute='numerator')
            hard_jaccard_denominator_op = dice_coe(tf.round(softmax_op[:,:,:,:,1]), tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='jaccard',axis=[1,2,3],compute='denominator')
            
            # Define variables for accumulating Dice quantities and their gradients
            soft_dice_numerator_sum    = tf.get_variable("soft_dice_numerator_sum"   , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            soft_dice_denominator_sum  = tf.get_variable("soft_dice_denominator_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            soft_jaccard_numerator_sum    = tf.get_variable("soft_jaccard_numerator_sum"   , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            soft_jaccard_denominator_sum  = tf.get_variable("soft_jaccard_denominator_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            
            specific_dice_sum  = tf.get_variable("specific_dice_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            specific_jaccard_sum  = tf.get_variable("specific_jaccard_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))

            # Sum the Gradient of these terms to calculate the Gradient of composite dice later
            # List of variables of length equal to the number of trainable variables
            t_vars = tf.trainable_variables()
            soft_numerator_gsum       = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in t_vars] 
            soft_denominator_gsum     = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in t_vars] 

            specific_dice_gsum = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in t_vars]
            
            hard_dice_numerator_sum    = tf.get_variable("hard_dice_numerator_sum"   , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            hard_dice_denominator_sum  = tf.get_variable("hard_dice_denominator_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            hard_jaccard_numerator_sum    = tf.get_variable("hard_jaccard_numerator_sum"   , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            hard_jaccard_denominator_sum  = tf.get_variable("hard_jaccard_denominator_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            
            smooth_batch = tf.constant(1e-5)
            #soft_dice_batch = tf.cond(n_ab > 0., lambda: (soft_dice_numerator_sum+smooth_batch)/(soft_dice_denominator_sum+smooth_batch), lambda: tf.constant(0.))
            #hard_dice_batch = tf.cond(n_ab > 0., lambda: (hard_dice_numerator_sum+smooth_batch)/(hard_dice_denominator_sum+smooth_batch), lambda: tf.constant(0.))
            #soft_jaccard_batch = tf.cond(n_ab > 0., lambda: (soft_jaccard_numerator_sum+smooth_batch)/(soft_jaccard_denominator_sum+smooth_batch), lambda: tf.constant(0.))
            #hard_jaccard_batch = tf.cond(n_ab > 0., lambda: (hard_jaccard_numerator_sum+smooth_batch)/(hard_jaccard_denominator_sum+smooth_batch), lambda: tf.constant(0.))
            
            # Operations to compute the metric for the batch
            
            soft_dice_batch    = (soft_dice_numerator_sum+smooth_batch)/(soft_dice_denominator_sum+smooth_batch)
            hard_dice_batch    = (hard_dice_numerator_sum+smooth_batch)/(hard_dice_denominator_sum+smooth_batch)
            soft_jaccard_batch = (soft_jaccard_numerator_sum+smooth_batch)/(soft_jaccard_denominator_sum+smooth_batch)
            hard_jaccard_batch = (hard_jaccard_numerator_sum+smooth_batch)/(hard_jaccard_denominator_sum+smooth_batch)
            
            specific_dice_batch = tf.cond(n_ab > 0., lambda: specific_dice_sum/n_ab, lambda: tf.constant(0.)) 
            specific_jaccard_batch = tf.cond(n_ab > 0., lambda: specific_jaccard_sum/n_ab, lambda: tf.constant(0.)) 

            dice_loss_op    = 1. - soft_dice_batch
            jaccard_loss_op = 1. - soft_jaccard_batch
            specific_dice_loss_op = 1. - specific_dice_batch
            specific_jaccard_loss_op = 1. - specific_jaccard_batch

            # Register these quantities in the Tensorboard output
            tf.summary.scalar('soft_dice_loss', dice_loss_op)
            tf.summary.scalar('soft_jaccard_loss', jaccard_loss_op)
            tf.summary.scalar('hard_dice_batch', hard_dice_batch)
            tf.summary.scalar('hard_jaccard_batch', hard_jaccard_batch)
            tf.summary.scalar('specific_dice_batch',specific_dice_batch)
            tf.summary.scalar('specific_dice_loss',specific_dice_loss_op)

        # Training Op
        with tf.name_scope("training"):
            # optimizer
            if FLAGS.optimizer == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.init_learning_rate)
            elif FLAGS.optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.init_learning_rate)
            elif FLAGS.optimizer == "momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.init_learning_rate, momentum=FLAGS.momentum)
            elif FLAGS.optimizer == "nesterov_momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.init_learning_rate, momentum=FLAGS.momentum, use_nesterov=True)
            else:
                sys.exit("Invalid optimizer");

            # loss function
            if (FLAGS.loss_function == "ce"):
                loss_fn = ce_op
                loss_avg = ce_avg
            elif(FLAGS.loss_function == "wce"):
                loss_fn = wce_op
                loss_avg = wce_avg
            elif(FLAGS.loss_function == "dwce"):
                loss_fn = dwce_op
                loss_avg = dwce_avg
            elif(FLAGS.loss_function == "dice"):
                loss_fn = dice_loss_op
                loss_avg = dice_loss_op
            elif(FLAGS.loss_function == "jaccard"):
                loss_fn = jaccard_loss_op
                loss_avg = jaccard_loss_op
            elif(FLAGS.loss_function == "specific_dice"):
                loss_fn = specific_dice_loss_op
                loss_avg = specific_dice_loss_op
            else:
                sys.exit("Invalid loss function");

            t_vars = tf.trainable_variables()
            
            if (FLAGS.l2_weight != 0.):
              l2_loss = FLAGS.l2_weight * tf.add_n( [tf.nn.l2_loss(var) for var in t_vars])
              l2_grad = optimizer.compute_gradients(l2_loss)
              loss_fn = loss_fn + l2_loss
              tf.summary.scalar('l2_loss', l2_loss)
            
            tf.summary.scalar('loss_fn',loss_fn)

            # train on single batch, unused
            #train_op = optimizer.minimize(
            #    loss = loss_fn,
            #    global_step=global_step)

            
            # create a copy of all trainable variables with `0` as initial values
            accum_tvars = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in t_vars]                                        
            # create a op to initialize all accums vars
            zero_op = [t_var.assign(tf.zeros_like(t_var)) for t_var in accum_tvars]
            

            # compute gradients for a batch
            if FLAGS.loss_function in ['ce','wce','dwce']:
              # The Cross-Entropies are just simple sums across voxels,
              # so we can sum the gradients across the cropped samples
              batch_grads_vars = optimizer.compute_gradients(loss_fn, t_vars)
              # Opperation to accumulate the gradients
              accum_op = [accum_tvars[i].assign_add(batch_grad_var[0]) for i, batch_grad_var in enumerate(batch_grads_vars)]
              compute_gradient_op = []
              # Operation to apply the accumulated gradients
            elif FLAGS.loss_function == 'specific_dice':
              # g_i: the gradient of the Dice metric for the i'th sample
              g_i = optimizer.compute_gradients(1. - specific_dice_op)
              accum_op = [specific_dice_gsum[j].assign_add(g_i[j][0]) for j in range(len(g_i))]
              zero_op += [specific_dice_gsum_j.assign(tf.zeros_like(specific_dice_gsum_j)) for specific_dice_gsum_j in specific_dice_gsum]

              compute_gradient_op = []
              for j in range(len(t_vars)):
                compute_gradient_op += [accum_tvars[j].assign(specific_dice_gsum[j]) ]
              
            elif FLAGS.loss_function == 'dice':
              # The Dice and Jaccard scores are evaluated across the voxels of all the samples in the batch.
              # To get the gradient of that thing, we need to sum the gradient of the numerator and denominator 
              # of the Dice (Jaccard) score over the batch samples.
              # num_g_i: the gradient of the numerator term for the i'th sample
              num_g_i   = optimizer.compute_gradients(soft_dice_numerator_op, t_vars)
              # den_g_i: the gradient of the denominator term for the i'th sample
              den_g_i = optimizer.compute_gradients(soft_dice_denominator_op, t_vars)
              # Operation to accumulate the numerator and denominators and their gradients
              accum_op =  [soft_numerator_gsum[j].assign_add(num_g_i[j][0])   for j in range(len(num_g_i))]
              accum_op += [soft_denominator_gsum[j].assign_add(den_g_i[j][0]) for j in range(len(den_g_i))]
              zero_op += [soft_numerator_gsum_j.assign(tf.zeros_like(soft_numerator_gsum_j)) for soft_numerator_gsum_j in soft_numerator_gsum]
              zero_op += [soft_denominator_gsum_j.assign(tf.zeros_like(soft_denominator_gsum_j)) for soft_denominator_gsum_j in soft_denominator_gsum]

              compute_gradient_op = []
              for j in range(len(t_vars)):
                compute_gradient_op += [accum_tvars[j].assign( ((soft_dice_numerator_sum+smooth_batch) * soft_denominator_gsum[j] - (soft_dice_denominator_sum+smooth_batch) * soft_numerator_gsum[j]) / ((soft_dice_denominator_sum+smooth_batch)*(soft_dice_denominator_sum+smooth_batch))) ]

            elif FLAGS.loss_function == 'jaccard':
              num_g_i   = optimizer.compute_gradients(soft_jaccard_numerator_op, t_vars)
              den_g_i = optimizer.compute_gradients(soft_jaccard_denominator_op, t_vars)
              # Operation to accumulate the numerator and denominators and their gradients
              accum_op =  [soft_numerator_gsum[j].assign_add(num_g_i[j][0])   for j in range(len(num_g_i))]
              accum_op += [soft_denominator_gsum[j].assign_add(den_g_i[j][0]) for j in range(len(den_g_i))]
              zero_op += [soft_numerator_gsum_j.assign(tf.zeros_like(soft_numerator_gsum_j)) for soft_numerator_gsum_j in soft_numerator_gsum]
              zero_op += [soft_denominator_gsum_j.assign(tf.zeros_like(soft_denominator_gsum_j)) for soft_denominator_gsum_j in soft_denominator_gsum]
              
              compute_gradient_op = []
              for j in range(len(t_vars)):
                compute_gradient_op += [accum_tvars[j].assign( (soft_jaccard_numerator_sum * soft_denominator_gsum[j] - soft_jaccard_denominator_sum * soft_numerator_gsum[j]) / (soft_jaccard_denominator_sum*soft_jaccard_denominator_sum)) ]

            if (FLAGS.l2_weight != 0.):
              for j in range(len(t_vars)):
                compute_gradient_op += [accum_tvars[j].assign_add( l2_grad[j][0]) ]
            
            #apply_gradients_op = optimizer.apply_gradients([(accum_tvars[i] / n_ab, batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars)], global_step=global_step)
            apply_gradients_op = optimizer.apply_gradients([(accum_tvars[i], t_vars[i]) for i in range(len(t_vars))], global_step=global_step)
        with tf.name_scope("avgloss"):
            # Here we accumulate the average loss and square of loss across the accum. batches
            sum_zero_op = [n_ab.assign(tf.zeros_like(n_ab))]

            sum_zero_op += [ce_sum.assign(tf.zeros_like(ce_sum))]
            sum_zero_op += [ce2_sum.assign(tf.zeros_like(ce2_sum))]
            sum_zero_op += [wce_sum.assign(tf.zeros_like(wce_sum))]
            sum_zero_op += [wce2_sum.assign(tf.zeros_like(wce2_sum))]
            sum_zero_op += [dwce_sum.assign(tf.zeros_like(dwce_sum))]
            sum_zero_op += [dwce2_sum.assign(tf.zeros_like(dwce2_sum))]

            sum_zero_op += [hard_dice_numerator_sum.assign(tf.zeros_like(hard_dice_numerator_sum))]
            sum_zero_op += [hard_dice_denominator_sum.assign(tf.zeros_like(hard_dice_denominator_sum))]
            sum_zero_op += [hard_jaccard_numerator_sum.assign(tf.zeros_like(hard_jaccard_numerator_sum))]
            sum_zero_op += [hard_jaccard_denominator_sum.assign(tf.zeros_like(hard_jaccard_denominator_sum))]
            
            sum_zero_op += [soft_dice_numerator_sum.assign(tf.zeros_like(soft_dice_numerator_sum))]
            sum_zero_op += [soft_dice_denominator_sum.assign(tf.zeros_like(soft_dice_denominator_sum))]
            sum_zero_op += [soft_jaccard_numerator_sum.assign(tf.zeros_like(soft_jaccard_numerator_sum))]
            sum_zero_op += [soft_jaccard_denominator_sum.assign(tf.zeros_like(soft_jaccard_denominator_sum))]
            
            sum_zero_op += [specific_dice_sum.assign(tf.zeros_like(specific_dice_sum))]
            sum_zero_op += [specific_jaccard_sum.assign(tf.zeros_like(specific_jaccard_sum))]

            sum_accum_op = [n_ab.assign_add(1.)]

            sum_accum_op += [ce_sum.assign_add(ce_op)]
            sum_accum_op += [ce2_sum.assign_add(ce_op*wce_op)]
            sum_accum_op += [wce_sum.assign_add(wce_op)]
            sum_accum_op += [wce2_sum.assign_add(wce_op*wce_op)]
            sum_accum_op += [dwce_sum.assign_add(dwce_op)]
            sum_accum_op += [dwce2_sum.assign_add(dwce_op*dwce_op)]
            sum_accum_op += [hard_dice_numerator_sum.assign_add(hard_dice_numerator_op)]
            sum_accum_op += [hard_dice_denominator_sum.assign_add(hard_dice_denominator_op)]
            sum_accum_op += [hard_jaccard_numerator_sum.assign_add(hard_jaccard_numerator_op)]
            sum_accum_op += [hard_jaccard_denominator_sum.assign_add(hard_jaccard_denominator_op)]
            sum_accum_op += [soft_dice_numerator_sum.assign_add(soft_dice_numerator_op)]
            sum_accum_op += [soft_dice_denominator_sum.assign_add(soft_dice_denominator_op)]
            sum_accum_op += [soft_jaccard_numerator_sum.assign_add(soft_jaccard_numerator_op)]
            sum_accum_op += [soft_jaccard_denominator_sum.assign_add(soft_jaccard_denominator_op)]
            sum_accum_op += [specific_dice_sum.assign_add(specific_dice_op)]
            sum_accum_op += [specific_jaccard_sum.assign_add(specific_jaccard_op)]

        # # epoch checkpoint manipulation
        start_epoch = tf.get_variable("start_epoch", shape=[1], initializer= tf.zeros_initializer,dtype=tf.int32)
        start_epoch_inc = start_epoch.assign(start_epoch+1)

        # saver
        summary_op = tf.summary.merge_all()
        checkpoint_slug = "checkpoint"
        if FLAGS.is_batch_job and FLAGS.batch_job_name is not '':
          checkpoint_slug = checkpoint_slug + "_" + FLAGS.batch_job_name
        checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, checkpoint_slug)
        print("Setting up Saver...")
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=10000,max_to_keep=1)

        #config = tf.ConfigProto(device_count={"CPU": 4})
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4

        # training cycle
        with tf.Session(config=config) as sess:
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            print("{}: Start training...".format(datetime.datetime.now()))

            # summary writer for tensorboard
            #train_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            #test_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)
            train_log_dir = FLAGS.log_dir + '/train'
            test_log_dir = FLAGS.log_dir + '/test'
            if FLAGS.is_batch_job and FLAGS.batch_job_name is not '':
              train_log_dir = train_log_dir + "_" + FLAGS.batch_job_name
              test_log_dir = test_log_dir + "_" + FLAGS.batch_job_name
            train_summary_writer = tf.summary.FileWriter(train_log_dir)
            test_summary_writer = tf.summary.FileWriter(test_log_dir)
            
            if not FLAGS.is_batch_job:
                train_summary_writer.add_graph(sess.graph)
                test_summary_writer.add_graph(sess.graph)

            # number of crops for gradient accumulation 
            n_accum_crops = int(round(FLAGS.num_crops * FLAGS.accum_batches / FLAGS.batch_size)) # number of crops for gradient accumulation 
            # restore from checkpoint
            if FLAGS.restore_training:
                # check if checkpoint exists
                if os.path.exists(checkpoint_prefix+"_latest"):
                    print("{}: Last checkpoint found at {}, loading...".format(datetime.datetime.now(),FLAGS.checkpoint_dir))
                    latest_checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir,latest_filename=latest_filename)
                    saver.restore(sess, latest_checkpoint_path)
            
            print("{}: Last checkpoint epoch: {}".format(datetime.datetime.now(),start_epoch.eval()[0]))
            print("{}: Last checkpoint global step: {}".format(datetime.datetime.now(),tf.train.global_step(sess, global_step)))
            sess.graph.finalize()
            # loop over epochs
            for epoch in np.arange(start_epoch.eval(), FLAGS.epochs):
              # initialize iterator in each new epoch
              sess.run(train_iterator.initializer)
              sess.run(test_iterator.initializer)
              logger.info("Epoch %d starts"%(epoch+1))

              # training phase
              #n_train = 0
              #logger.debug('Zeroing gradients and loss sums')
              #sess.run(zero_op) # reset gradient accumulation
              #sess.run(sum_zero_op) # reset loss-averaging
              model.is_training = True;
              logger.debug('Beginning accumulation batch')
              while True: # Beginning of Accumulation batch
                try:
                  logger.debug('Beginning loop over the accumulation crops')
                  n_train = int(sess.run(n_ab))
                  [image, label] = sess.run(next_element_train)
                  
                  logger.debug('Zeroing gradients and loss sums')
                  sess.run(zero_op) # reset gradient accumulation
                  sess.run(sum_zero_op) # reset loss-averaging

                  for i in range(n_accum_crops):
                    image = image[:,:,:,:,:] #image[:,:,:,:,np.newaxis]
                    label = label[:,:,:,:,np.newaxis]
                    train, train_loss, sum_accum = sess.run([accum_op, loss_fn, sum_accum_op], feed_dict={images_placeholder: image, labels_placeholder: label})
                    [image, label] = sess.run(next_element_train)
                  n_train = int(sess.run(n_ab))

                  logger.debug("Applying gradients after total %d accumulations"%n_train)
                  #if FLAGS.loss_function in ['dice','jaccard','specific_dice']:
                  if (compute_gradient_op != []):
                    sess.run(compute_gradient_op)
                  sess.run(apply_gradients_op)
                  
                  
                  logger.debug('Applying summary op')
                  summary = sess.run(summary_op)
                  train_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))
                except tf.errors.OutOfRangeError:
                  # We could compute the accumulated gradient even if we didn't run over a full set of n_batches
                  # This can explode the gradients, especially if i=0 at the time of this exception
                  # I don't think we should really do this?
                  if True:#n_train % (n_accum_crops) != 0:
                    logger.debug("Applying gradients after total %d accumulations"%n_train)
                    if FLAGS.loss_function in ['dice','jaccard','specific_dice']:
                      sess.run(compute_gradient_op)
                    sess.run(apply_gradients_op)
                    summary = sess.run(summary_op)
                    train_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))
                  
                  # Compute the average training loss across all batches in the epoch.
                  train_loss_avg = sess.run(loss_avg)
                  
                  print("{0}: Average training loss is {1:.3f} over {2:d}".format(datetime.datetime.now(), train_loss_avg, n_train))
                  start_epoch_inc.op.run()
                  # print(start_epoch.eval())
                  # save the model at end of each epoch training
                  print("{}: Saving checkpoint of epoch {} at {}...".format(datetime.datetime.now(),epoch+1,FLAGS.checkpoint_dir))
                  if not (os.path.exists(FLAGS.checkpoint_dir)):
                      os.makedirs(FLAGS.checkpoint_dir,exist_ok=True)

                  saver.save(sess, checkpoint_prefix, 
                      global_step=tf.train.global_step(sess, global_step),
                      latest_filename=latest_filename)
                  print("{}: Saving checkpoint succeed".format(datetime.datetime.now()))
                  break
                except MemoryError:
                  logger.error("Terminating due to exceeded memory limit. I am justly killed with mine own treachery!")
                  sys.exit(1)
              
              print('profile\n',TrainDataset.profile)

              # testing phase
              print("{}: Training of epoch {} finishes, testing start".format(datetime.datetime.now(),epoch+1))
              #test_loss_avg = 0.0
              #n_test = 0
              logger.debug('Zeroing gradients and loss sums')
              sess.run(zero_op) # reset gradient accumulation
              sess.run(sum_zero_op) # reset loss-averaging
              while True:
                try:
                  [image, label] = sess.run(next_element_test)

                  image = image[:,:,:,:,:] #image[:,:,:,:,np.newaxis]
                  label = label[:,:,:,:,np.newaxis]
                  
                  model.is_training = False;
                  test_loss, sum_accum = sess.run([loss_fn, sum_accum_op], feed_dict={images_placeholder: image, labels_placeholder: label})
                  # This is redundant, fix
                  #test_loss_avg += test_loss
                  #n_test += 1

                except tf.errors.OutOfRangeError:
                  #test_loss_avg = test_loss_avg / n_test
                  # Compute the average testing loss across the batch
                  if (FLAGS.loss_function == "ce"):
                    test_loss_avg = sess.run(ce_avg)
                  elif(FLAGS.loss_function == "wce"):
                    test_loss_avg = sess.run(wce_avg)
                  elif(FLAGS.loss_function == "dwce"):
                    test_loss_avg = sess.run(dwce_avg)
                  elif(FLAGS.loss_function == "dice"):
                    test_loss_avg = sess.run(dice_loss_op)
                  elif(FLAGS.loss_function == "jaccard"):
                    test_loss_avg = sess.run(jaccard_loss_op)
                  elif(FLAGS.loss_function == "specific_dice"):
                    test_loss_avg = sess.run(specific_dice_loss_op)

                  n_test = int(sess.run(n_ab))
                  print("{0}: Average testing loss is {1:.3f} over {2:d}".format(datetime.datetime.now(), test_loss_avg, n_test))
                  summary = sess.run(summary_op)
                  test_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))
                  break
                except MemoryError:
                  logger.error("Terminating due to exceeded memory limit. I am justly killed with mine own treachery!")
                  sys.exit(1)
              #n_test = 0
              logger.debug('Zeroing gradients and loss sums')
              sess.run(zero_op) # reset gradient accumulation
              sess.run(sum_zero_op) # reset loss-averaging

        # close tensorboard summary writer
        train_summary_writer.close()
        test_summary_writer.close()

def main(argv=None):
    #if not FLAGS.restore_training:
        # clear log directory
        #if tf.gfile.Exists(FLAGS.log_dir):
        #    tf.gfile.DeleteRecursively(FLAGS.log_dir)
        #tf.gfile.MakeDirs(FLAGS.log_dir)

        # clear checkpoint directory
        #if tf.gfile.Exists(FLAGS.checkpoint_dir):
        #    tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        #tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

        # # clear model directory
        # if tf.gfile.Exists(FLAGS.model_dir):
        #     tf.gfile.DeleteRecursively(FLGAS.model_dir)
        # tf.gfile.MakeDirs(FLAGS.model_dir)

    train()

if __name__=='__main__':
    tf.app.run()
