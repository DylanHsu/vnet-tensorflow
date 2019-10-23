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
    """Loss function used in optimization (xent, weight_xent, dice, jaccard)""")
tf.app.flags.DEFINE_string('optimizer','sgd',
    """Optimization method (sgd, adam, momentum, nesterov_momentum)""")
tf.app.flags.DEFINE_float('momentum',0.5,
    """Momentum used in optimization""")
tf.app.flags.DEFINE_boolean('is_batch_job',False,
    """Disable some features if this is a batch job""")
tf.app.flags.DEFINE_string('batch_job_name','',
    """Name the batch job so the checkpoints and tensorboard output are identifiable.""")
tf.app.flags.DEFINE_float('max_ram','15.5',
    """Maximum amount of RAM usable by the CPU in GB.""")

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

def dice_coe(output, target, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
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
    ## new haodong
    dice = (tf.constant(2.0) * tf.cast(inse,dtype=tf.float32) + tf.constant(smooth)) / (tf.cast(l + r, dtype=tf.float32) + tf.constant(smooth))
    ##
    dice = tf.reduce_mean(dice)
    return dice

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
                NiftiDataset.StatisticalNormalization(3.0, nonzero_only=True),
                NiftiDataset.ThresholdCrop(),
                #NiftiDataset.RandomRotation(),
                #NiftiDataset.ThresholdCrop(),
                #NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
                #NiftiDataset.ConfidenceCrop((FLAGS.patch_size,FLAGS.patch_size,FLAGS.patch_layer), FLAGS.ccrop_sigma),
                NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),FLAGS.drop_ratio,FLAGS.min_pixel)
                #NiftiDataset.RandomNoise(),
                # NiftiDataset.Normalization(),
                #NiftiDataset.Resample((0.45,0.45,0.45)),
                ]
            testTransforms = [
                NiftiDataset.StatisticalNormalization(3.0, nonzero_only=True),
                NiftiDataset.ThresholdCrop(),
                NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),0.5,FLAGS.min_pixel)
                #NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
                #NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),0.5,10),
                #NiftiDataset.ConfidenceCrop((FLAGS.patch_size,FLAGS.patch_size,FLAGS.patch_layer), FLAGS.ccrop_sigma),
                #NiftiDataset.ConfidenceCrop((FLAGS.patch_size,FLAGS.patch_size,FLAGS.patch_layer), 1.0),
                # NiftiDataset.Normalization(),
                #NiftiDataset.Resample((0.45,0.45,0.45)),
                #NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
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
            trainDataset = trainDataset.batch(FLAGS.batch_size)
            trainDataset = trainDataset.prefetch(1)
            #trainDataset = trainDataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))

            #testTransforms = [
            #    NiftiDataset.StatisticalNormalization(2.5),
            #    # NiftiDataset.Normalization(),
            #    NiftiDataset.Resample((0.45,0.45,0.45)),
            #    NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
            #    NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),FLAGS.drop_ratio,FLAGS.min_pixel)
            #    ]

            TestDataset = NiftiDataset.NiftiDataset(
                data_dir=test_data_dir,
                image_filename=FLAGS.image_filename,
                label_filename=FLAGS.label_filename,
                transforms=testTransforms,
                num_crops=10, #FLAGS.num_crops,
                train=True
            )

            testDataset = TestDataset.get_dataset()
            # Here there are batches of size num_crops, unbatch and shuffle
            testDataset = testDataset.apply(tf.contrib.data.unbatch())
            testDataset = testDataset.batch(FLAGS.batch_size)
            testDataset = testDataset.prefetch(1)
            #testDataset = testDataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))
            
        train_iterator = trainDataset.make_initializable_iterator()
        next_element_train = train_iterator.get_next()

        test_iterator = testDataset.make_initializable_iterator()
        next_element_test = test_iterator.get_next()

        # Initialize the model
        with tf.name_scope("vnet"):
            #model = VNet.VNet(
            #    num_classes=2, # binary for 2
            #    keep_prob=1.0, # default 1
            #    num_channels=16, # default 16 
            #    num_levels=4,  # default 4
            #    num_convolutions=(1,2,3,3), # default (1,2,3,3), size should equal to num_levels
            #    bottom_convolutions=3, # default 3
            #    activation_fn="prelu") # default relu
            model = VNet.VNet(
                num_classes=2,   
                keep_prob=1.0,   
                num_channels=16, 
                num_levels=4,    
                num_convolutions=(1,2,3,3),
                bottom_convolutions=3, 
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

        # Op for calculating loss
        with tf.name_scope("cross_entropy"):
            loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf.squeeze(labels_placeholder, 
                squeeze_dims=[4])))
        #tf.summary.scalar('loss',loss_op)

        with tf.name_scope("weighted_cross_entropy"):
            #class_weights = tf.constant([1.0, 1.0])
            #class_weights = tf.constant([1.0, 10.0])

            total_volume = tf.constant(FLAGS.patch_size*FLAGS.patch_size*FLAGS.patch_layer, tf.float32)
            label_volume = tf.cast(tf.reduce_sum(labels_placeholder),dtype=tf.float32);
            true_weight = tf.cond(label_volume>0, lambda: tf.divide(tf.divide(total_volume,label_volume), tf.constant(FLAGS.drop_ratio,tf.float32)), lambda: tf.constant(1.0, dtype=tf.float32))
            class_weights = tf.stack([tf.constant(1.0, dtype=tf.float32), true_weight],0)

            # deduce weights for batch samples based on their true label
            onehot_labels = tf.one_hot(tf.squeeze(labels_placeholder,squeeze_dims=[4]),depth = 2)

            weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
            # compute your (unweighted) softmax cross entropy loss
            unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf.squeeze(labels_placeholder, 
                squeeze_dims=[4]))
            # apply the weights, relying on broadcasting of the multiplication
            weighted_loss = unweighted_loss * weights
            # reduce the result to get your final loss
            weighted_loss_op = tf.reduce_mean(weighted_loss)
            
            weighted_loss_op = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(onehot_labels,logits,10000))
                
        #tf.summary.scalar('weighted_XE_loss',weighted_loss_op)

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
            # dice = dice_coe(tf.expand_dims(softmax_op[:,:,:,:,1],-1),tf.cast(labels_placeholder,dtype=tf.float32), loss_type='dice')
            # jaccard = dice_coe(tf.expand_dims(softmax_op[:,:,:,:,1],-1),tf.cast(labels_placeholder,dtype=tf.float32), loss_type='jaccard')
            
            # This is commented out because using all of the True-Negative pixels gives a very deceptive dice score
            # dice = dice_coe(softmax_op,tf.cast(tf.one_hot(labels_placeholder[:,:,:,:,0],depth=2),dtype=tf.float32), loss_type='dice', axis=[1,2,3,4])
            # jaccard = dice_coe(softmax_op,tf.cast(tf.one_hot(labels_placeholder[:,:,:,:,0],depth=2),dtype=tf.float32), loss_type='jaccard', axis=[1,2,3,4])
            
            # Computing the dice using only the second row of the 2-entry softmax vector seems more useful
            #dice = dice_coe(softmax_op[:,:,:,:,1],tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='dice', axis=[1,2,3])
            #jaccard  = dice_coe(softmax_op[:,:,:,:,1],tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='jaccard', axis=[1,2,3])
            
            # Hard versions
            dice = dice_coe(tf.round(softmax_op[:,:,:,:,1]), tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='dice',axis=[1,2,3])
            jaccard  = dice_coe(tf.round(softmax_op[:,:,:,:,1]),tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='jaccard', axis=[1,2,3])

            dice_loss = 1. - dice
            jaccard_loss = 1. - jaccard
            
            wce_sum  = tf.get_variable("wce_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            wce2_sum = tf.get_variable("wce2_sum", dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            dice_sum  = tf.get_variable("dice_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            dice2_sum = tf.get_variable("dice2_sum", dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            jaccard_sum   = tf.get_variable("jaccard_sum"  , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            jaccard2_sum  = tf.get_variable("jaccard2_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            # Number of accumulated batches
            n_ab = tf.get_variable("n_ab", dtype=tf.float32, trainable=False, use_resource=True, initializer=tf.constant(0.))
            wce_avg = tf.cond(n_ab > 1., lambda: wce_sum/n_ab, lambda: tf.constant(0.)) 
            wce_stdev = tf.cond(n_ab > 1., lambda: tf.math.sqrt( (n_ab*wce2_sum - wce_sum*wce_sum) / (n_ab * (n_ab-1.))), lambda: tf.constant(0.))
            dice_avg = tf.cond(n_ab > 1., lambda: dice_sum/n_ab, lambda: tf.constant(0.)) 
            dice_stdev = tf.cond(n_ab > 1., lambda: tf.math.sqrt( (n_ab*dice2_sum - dice_sum*dice_sum) / (n_ab * (n_ab-1.))), lambda: tf.constant(0.))
            dice_loss_avg = 1. - dice_avg
            jaccard_avg  = tf.cond(n_ab > 1., lambda: jaccard_sum /n_ab, lambda: tf.constant(0.)) 
            jaccard_stdev  = tf.cond(n_ab > 1., lambda: tf.math.sqrt( (n_ab*jaccard2_sum  - jaccard_sum*jaccard_sum  ) / (n_ab * (n_ab-1.))), lambda: tf.constant(0.))
            jaccard_loss_avg = 1. - jaccard_avg
        tf.summary.scalar('wce_avg', wce_avg)
        tf.summary.scalar('wce_stdev', wce_stdev)
        tf.summary.scalar('dice_avg', dice_avg)
        tf.summary.scalar('dice_stdev', dice_stdev)
        tf.summary.scalar('dice_loss_avg', dice_loss_avg)
        tf.summary.scalar('jaccard_avg', jaccard_avg)
        tf.summary.scalar('jaccard_stdev', jaccard_stdev)
        tf.summary.scalar('jaccard_loss_avg',jaccard_loss_avg)

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
            if (FLAGS.loss_function == "xent"):
                loss_fn = loss_op
            elif(FLAGS.loss_function == "weight_xent"):
                loss_fn = weighted_loss_op
            elif(FLAGS.loss_function == "dice"):
                loss_fn = dice_loss
            elif(FLAGS.loss_function == "jaccard"):
                loss_fn = jaccard_loss
            else:
                sys.exit("Invalid loss function");

            # train on single batch, unused
            #train_op = optimizer.minimize(
            #    loss = loss_fn,
            #    global_step=global_step)

            t_vars = tf.trainable_variables()
            # create a copy of all trainable variables with `0` as initial values
            accum_tvars = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in t_vars]                                        
            # create a op to initialize all accums vars
            zero_op = [t_var.assign(tf.zeros_like(t_var)) for t_var in accum_tvars]
            
            # compute gradients for a batch
            batch_grads_vars = optimizer.compute_gradients(loss_fn, t_vars)

            # collect the batch gradient into accumulated vars
            accum_op = [accum_tvars[i].assign_add(batch_grad_var[0]) for i, batch_grad_var in enumerate(batch_grads_vars)]
            
            # apply accums gradients 
            apply_gradients_op = optimizer.apply_gradients([(accum_tvars[i] / n_ab, batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars)], global_step=global_step)

        with tf.name_scope("avgloss"):
            # Here we accumulate the average loss and square of loss across the accum. batches
            sum_zero_op = []
            sum_zero_op += [n_ab.assign(tf.zeros_like(n_ab))]
            sum_zero_op += [wce_sum.assign(tf.zeros_like(wce_sum))]
            sum_zero_op += [wce2_sum.assign(tf.zeros_like(wce2_sum))]
            sum_zero_op += [dice_sum.assign(tf.zeros_like(dice_sum))]
            sum_zero_op += [dice2_sum.assign(tf.zeros_like(dice2_sum))]
            sum_zero_op += [jaccard_sum.assign(tf.zeros_like(jaccard_sum))]
            sum_zero_op += [jaccard2_sum.assign(tf.zeros_like(jaccard2_sum))]

            sum_accum_op = []
            sum_accum_op += [wce_sum.assign_add(weighted_loss_op)]
            sum_accum_op += [wce2_sum.assign_add(weighted_loss_op*weighted_loss_op)]
            sum_accum_op += [dice_sum.assign_add(dice)]
            sum_accum_op += [dice2_sum.assign_add(dice*dice)]
            sum_accum_op += [jaccard_sum.assign_add(jaccard)]
            sum_accum_op += [jaccard2_sum.assign_add(jaccard*jaccard)]
            sum_accum_op += [n_ab.assign_add(1.)]

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
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=10000,max_to_keep=3)

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
            n_accum_crops = FLAGS.num_crops * FLAGS.accum_batches # number of crops for gradient accumulation 
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
              train_loss_avg = 0.0
              n_train = 0
              model.is_training = True;
              logger.debug('Beginning accumulation batch')
              while True: # Beginning of Accumulation batch
                try:
                  logger.debug('Zeroing gradients and loss sums')
                  sess.run(zero_op) # reset gradient accumulation
                  sess.run(sum_zero_op) # reset loss-averaging
                  logger.debug('Beginning loop over the accumulation crops')
                  for i in range(n_accum_crops):
                    [image, label] = sess.run(next_element_train)
                    image = image[:,:,:,:,:] #image[:,:,:,:,np.newaxis]
                    label = label[:,:,:,:,np.newaxis]
                    train, train_loss, sum_accum = sess.run([accum_op, loss_fn, sum_accum_op], feed_dict={images_placeholder: image, labels_placeholder: label})
                    # This is redundant, remove this when we can get the average loss for all supported loss functions
                    train_loss_avg += train_loss
                    n_train += 1
                  logger.debug("Applying gradients after total %d accumulations"%n_train)
                  sess.run(apply_gradients_op)
                  logger.debug('Applying summary op')
                  summary = sess.run(summary_op)
                  train_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))
                except tf.errors.OutOfRangeError:
                  # We could compute the accumulated gradient even if we didn't run over a full set of n_batches
                  # This can explode the gradients, especially if i=0 at the time of this exception
                  # I don't think we should really do this?
                  if n_train % (n_accum_crops) != 0:
                    logger.debug("Applying gradients after total %d accumulations"%n_train)
                    sess.run(apply_gradients_op)
                    summary = sess.run(summary_op)
                    train_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))
                  
                  # Compute the average training loss across all batches in the epoch.
                  train_loss_avg = train_loss_avg / n_train
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
              
              # testing phase
              print("{}: Training of epoch {} finishes, testing start".format(datetime.datetime.now(),epoch+1))
              test_loss_avg = 0.0
              n_test = 0
              sess.run(sum_zero_op) # reset loss-averaging
              while True:
                try:
                  [image, label] = sess.run(next_element_test)

                  image = image[:,:,:,:,:] #image[:,:,:,:,np.newaxis]
                  label = label[:,:,:,:,np.newaxis]
                  
                  model.is_training = False;
                  test_loss, sum_accum = sess.run([loss_fn, sum_accum_op], feed_dict={images_placeholder: image, labels_placeholder: label})
                  # This is redundant, fix
                  test_loss_avg += test_loss
                  n_test += 1

                except tf.errors.OutOfRangeError:
                  test_loss_avg = test_loss_avg / n_test
                  print("{0}: Average testing loss is {1:.3f} over {2:d}".format(datetime.datetime.now(), test_loss_avg, n_test))
                  summary = sess.run(summary_op)
                  test_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))
                  break
                except MemoryError:
                  logger.error("Terminating due to exceeded memory limit. I am justly killed with mine own treachery!")
                  sys.exit(1)

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
