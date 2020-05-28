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
import faulthandler
from dice import dice_coe

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s','%m-%d %H:%M:%S'))
logger = logging.getLogger('vnet_trainer')
logger.addHandler(console)
logger.setLevel(logging.DEBUG)
logger.propagate = False
faulthandler.enable()

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
tf.app.flags.DEFINE_string('image_filenames','mr1.nii.gz,ct.nii.gz',
    """Image filename""")
tf.app.flags.DEFINE_string('label_filename','label.nii.gz',
    """Label filename""")
tf.app.flags.DEFINE_integer('batch_size',1,
    """Size of batch""")               
tf.app.flags.DEFINE_integer('accum_batches',1,
    """Accumulate the gradient over this many batches before updating the gradient (1 = no accumulation)""")               
tf.app.flags.DEFINE_integer('accum_batches_per_epoch',15,
    """Accumulated batches per epoch""")               
tf.app.flags.DEFINE_integer('num_crops',1,
    """Take this many crops from each image, per epoch""")               
tf.app.flags.DEFINE_integer('small_bias',1,
    """Bias factor by which to overrepresent cases with small CCs""")
tf.app.flags.DEFINE_float('small_bias_diameter',10.0,
    """Definition of small CCs [mm]""")
tf.app.flags.DEFINE_float('ccrop_sigma',2.5,
    """Value of sigma to use for confidence crops""")               
tf.app.flags.DEFINE_integer('num_channels',2,
    """Number of channels""")               
tf.app.flags.DEFINE_integer('patch_size',128,
    """Size of a data patch""")
tf.app.flags.DEFINE_integer('patch_layer',64,
    """Number of layers in data patch""")
tf.app.flags.DEFINE_integer('epochs',500,
    """Number of epochs for training""")
tf.app.flags.DEFINE_string('log_dir', './tmp/log',
    """Directory where to write training and testing event logs """)
tf.app.flags.DEFINE_float('init_learning_rate',1e-4,
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

tf.app.flags.DEFINE_float('boundary_loss_weight',0.0,
    """Coefficient for boundary loss term""")
tf.app.flags.DEFINE_integer('distance_map_index',-1,
    """Index of the distance map image in the images_filenames list (-1=no map)""")

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

tf.app.flags.DEFINE_string('vnet_convs','1,2,3,3,3',
    """Convolutions in each VNET layer.""")
tf.app.flags.DEFINE_integer('vnet_channels',8,
    """Channels after initial VNET convolution.""")


def train():
    """Train the Vnet model"""
    resource.setrlimit(resource.RLIMIT_DATA,(math.ceil((FLAGS.max_ram-1.)*(1024**2)*1000),math.ceil(FLAGS.max_ram*(1024**2)*1000))) # 1000 MB ~ 1 GB
    latest_filename = "checkpoint"
    if FLAGS.is_batch_job and FLAGS.batch_job_name is not '':
        latest_filename = latest_filename + "_" + FLAGS.batch_job_name
        #resource.setrlimit(resource.RLIMIT_CORE,(524288,-1))
    latest_filename += "_latest"
    
    image_filenames_list = FLAGS.image_filenames.split(',')
    auxiliary_indices_list = []
    anatomy_indices_list = []
    distance_map_aux_channel = -1 # Which channel of aux_placeholder the distance map ends up at
    if FLAGS.distance_map_index>=0:
      assert FLAGS.distance_map_index < len(image_filenames_list)
      distance_map_aux_channel = len(auxiliary_indices_list)
      auxiliary_indices_list.append(FLAGS.distance_map_index)
    if FLAGS.anatomy_indices != '':
      anatomy_indices_list_str  =  FLAGS.anatomy_indices_list.split(',')
      for string in anatomy_indices_list_str:
        index = int(string)
        assert (index >= 0 and index < len(image_filenames_list)) , "bad anatomy_indices flag"
        anatomy_indices_list.append(index)
    image_indices_list = []
    for i in range(len(image_filenames_list)):
      if i not in auxiliary_indices_list and i not in anatomy_indices_list:
        image_indices_list.append(i)
    assert len(image_indices_list)==FLAGS.num_channels


    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # patch_shape(batch_size, height, width, depth, channels)
        input_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, FLAGS.num_channels) 
        aux_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, len(auxiliary_indices_list)) 
        anatomy_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, len(anatomy_indices_list)) 
        output_batch_shape = (FLAGS.batch_size, FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer, 1+len(anatomy_indices_list)) # 1 for binary classification

        #images_placeholder, labels_placeholder = placeholder_inputs(input_batch_shape,output_batch_shape)
        images_placeholder  = tf.placeholder(tf.float32, shape=input_batch_shape, name="images_placeholder")
        aux_placeholder     = tf.placeholder(tf.float32, shape=aux_batch_shape, name="aux_placeholder")
        labels_placeholder  = tf.placeholder(tf.int32, shape=output_batch_shape, name="labels_placeholder")   
        anatomy_placeholder = tf.placeholder(tf.int32, shape=anatomy_batch_shape, name="anatomy_placeholder")

        #weighted_label = labels_placeholder
        #binary_label = tf.cast(tf.greater(labels_placeholder, 0.0), dtype=tf.int32)

        # Get images and labels
        train_data_dir = os.path.join(FLAGS.data_dir,'training')
        test_data_dir = os.path.join(FLAGS.data_dir,'testing')
        # support multiple image input, but here only use single channel, label file should be a single file with different classes
        

        # Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
        with tf.device('/cpu:0'):
            # create transformations to image and labels

            trainTransforms = [
                #NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
                NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),FLAGS.drop_ratio,FLAGS.min_pixel),
                NiftiDataset.RandomNoise(0,0.1),
                NiftiDataset.RandomFlip(0.5, [True,True,True]),
                ]
             
            testTransforms = [
                NiftiDataset.StatisticalNormalization(0, 5.0, 5.0, nonzero_only=True, zero_floor=True),
                NiftiDataset.ManualNormalization(1, 0, 100.),
                NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),0.5,FLAGS.min_pixel),
                ]
            
            TrainDataset = NiftiDataset.NiftiDataset(
                data_dir=train_data_dir,
                image_filenames=FLAGS.image_filenames,
                label_filename=FLAGS.label_filename,
                transforms=trainTransforms,
                num_crops=FLAGS.num_crops,
                train=True,
                small_bias=FLAGS.small_bias,
                small_bias_diameter=FLAGS.small_bias_diameter,
                cpu_threads=1
                )
            
            trainDataset = TrainDataset.get_dataset()
            # Here there are batches of size num_crops, unbatch and shuffle
            trainDataset = trainDataset.apply(tf.contrib.data.unbatch())
            if FLAGS.small_bias is not 1:
              trainDataset = trainDataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
            trainDataset = trainDataset.repeat() 
            trainDataset = trainDataset.batch(FLAGS.batch_size)
            trainDataset = trainDataset.prefetch(5)
            #trainDataset = trainDataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))


            TestDataset = NiftiDataset.NiftiDataset(
                data_dir=test_data_dir,
                image_filenames=FLAGS.image_filenames,
                label_filename=FLAGS.label_filename,
                transforms=testTransforms,
                num_crops=FLAGS.num_crops, #10
                train=True,
                cpu_threads=1
            )

            testDataset = TestDataset.get_dataset()
            # Here there are batches of size num_crops, unbatch and shuffle
            testDataset = testDataset.apply(tf.contrib.data.unbatch())
            testDataset = testDataset.repeat()
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
            model = WNet.WNet(
                num_classes         = 2 + len(anatomy_indices_list), # background + lesions + anatomy
                num_classes_aux1    = 1 + len(anatomy_indices_list), # background + anatomy
                keep_prob           = FLAGS.dropout_keepprob,   
                num_channels        = FLAGS.vnet_channels, 
                num_levels          = len(convs)-1,    
                num_aux_levels      = len(convs)-2,
                num_convolutions    = tuple(convs[0:-1]),
                bottom_convolutions = convs[-1], 
                activation_fn="prelu")

            final_logits, anatomy_logits = model.network_fn(images_placeholder)
        
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

        # Get a list of the indices of the final logits which are background or anatomy
        # i.e. drop index 1 which is lesions
        # [0, 2, 3, 4, ...]
        final_anatomy_indices = list(set(range(len(anatomy_indices_list))) - set([1]))
        # softmax op for probability layer
        with tf.name_scope("softmax"):
            final_label_softmax_op    = tf.nn.softmax(final_logits[:,:,:,:,[0,1]],name="final_label_softmax")
            final_anatomy_softmax_op  = tf.nn.softmax(final_logits[:,:,:,:,final_anatomy_indices],name="final_anatomy_softmax")
            branch_anatomy_softmax_op = tf.nn.softmax(anatomy_logits[:,:,:,:,:],name="branch_anatomy_softmax")
            # take a fair average of the branch and final softmax ops
            combined_anatomy_softmax_op = (final_anatomy_softmax_op + branch_anatomy_softmax_op) / 2.

            #onehot_label   = tf.one_hot(labels_placeholder[:,:,:,:,0], depth = 2)
            #onehot_anatomy = anatomy_placeholder[:,:,:,:,:] # is this right?


        # Number of accumulated batches
        n_ab = tf.get_variable("n_ab", dtype=tf.float32, trainable=False, use_resource=True, initializer=tf.constant(0.))
        
        # Argmax Op to generate label from logits
        with tf.name_scope("predicted_label"):
          pred = tf.argmax(final_logits[:,:,:,:,[0,1]], axis=4 , name="prediction")

        with tf.name_scope("distance"):
          # turn off calculation later
          if FLAGS.distance_map_index >= 0:
            #scalar_boundary_loss_weight = tf.constant(FLAGS.boundary_loss_weight, dtype=tf.float32)
            scalar_boundary_loss_weight = tf.get_variable("scalar_boundary_loss_weight", initializer=tf.constant(FLAGS.boundary_loss_weight), dtype=tf.float32, trainable=False)
            increase_boundary_loss_weight_op = scalar_boundary_loss_weight.assign(scalar_boundary_loss_weight + FLAGS.boundary_loss_weight)

            boundary_loss_sum  = tf.get_variable("boundary_loss_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            boundary_loss_op = tf.reduce_sum(tf.multiply(final_label_softmax_op[:,:,:,:,1], aux_placeholder[:,:,:,:,distance_map_aux_channel]))
            #print("shapes",softmax_op[:,:,:,:,1].get_shape(), aux_placeholder[:,:,:,:,distance_map_aux_channel].get_shape(), boundary_loss_op.get_shape())
            boundary_loss_batch = tf.cond(n_ab > 0., lambda: boundary_loss_sum/n_ab, lambda: tf.constant(0.)) 
            tf.summary.scalar('boundary_loss_batch',boundary_loss_batch)
        
        t_vars = tf.trainable_variables()
        smooth_batch = tf.constant(1e-5)
        # Dice Similarity, currently only for binary segmentation
        with tf.name_scope("dice"):
            # Define operations to compute Dice quantities
            
            # Computing the dice using only the second row of the 2-entry softmax vector seems more useful
            specific_dice_op     = dice_coe(final_label_softmax_op[:,:,:,:,1],tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='dice', axis=[1,2,3])
            
            soft_dice_numerator_op      = dice_coe(final_label_softmax_op[:,:,:,:,1], tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='dice',axis=[1,2,3],compute='numerator')
            soft_dice_denominator_op    = dice_coe(final_label_softmax_op[:,:,:,:,1], tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='dice',axis=[1,2,3],compute='denominator')
            
            hard_dice_numerator_op      = dice_coe(tf.round(final_label_softmax_op[:,:,:,:,1]), tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='dice',axis=[1,2,3],compute='numerator')
            hard_dice_denominator_op    = dice_coe(tf.round(final_label_softmax_op[:,:,:,:,1]), tf.cast(labels_placeholder[:,:,:,:,0],dtype=tf.float32), loss_type='dice',axis=[1,2,3],compute='denominator')
            
            # Define variables for accumulating Dice quantities and their gradients
            soft_dice_numerator_sum    = tf.get_variable("soft_dice_numerator_sum"   , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            soft_dice_denominator_sum  = tf.get_variable("soft_dice_denominator_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            
            specific_dice_sum  = tf.get_variable("specific_dice_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))

            # Sum the Gradient of these terms to calculate the Gradient of composite dice later
            # List of variables of length equal to the number of trainable variables
            soft_numerator_gsum       = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in t_vars] 
            soft_denominator_gsum     = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in t_vars] 

            specific_dice_gsum = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in t_vars]
            
            hard_dice_numerator_sum    = tf.get_variable("hard_dice_numerator_sum"   , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            hard_dice_denominator_sum  = tf.get_variable("hard_dice_denominator_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
            
            # Operations to compute the metric for the batch
            
            soft_dice_batch    = (soft_dice_numerator_sum+smooth_batch)/(soft_dice_denominator_sum+smooth_batch)
            hard_dice_batch    = (hard_dice_numerator_sum+smooth_batch)/(hard_dice_denominator_sum+smooth_batch)
            
            specific_dice_batch = tf.cond(n_ab > 0., lambda: specific_dice_sum/n_ab, lambda: tf.constant(0.)) 

            dice_loss_op    = 1. - soft_dice_batch
            specific_dice_loss_op = 1. - specific_dice_batch
            # make final loss function
            if FLAGS.distance_map_index >= 0 and FLAGS.boundary_loss_weight != 0 :
              specific_dice_loss_op += boundary_loss_batch * scalar_boundary_loss_weight

            # Register these quantities in the Tensorboard output
            tf.summary.scalar('soft_dice_loss', dice_loss_op)
            tf.summary.scalar('hard_dice_batch', hard_dice_batch)
            tf.summary.scalar('specific_dice_batch',specific_dice_batch)
            tf.summary.scalar('specific_dice_loss',specific_dice_loss_op)

        with tf.name_scope("anatomy"):
          # Construct a specific dice operation for each anatomy label.
          # bad = Branch Anatomy Dice (auxiliary branch from network)
          # fad = Final Anatomy Dice
          bad_numerator_ops     = []
          bad_denominator_ops   = []
          bad_ops               = []
          bad_sums              = []
          bad_batchavgs         = []
          fad_numerator_ops     = []
          fad_denominator_ops   = []
          fad_ops               = []
          fad_sums              = []
          fad_batchavgs         = []

          with tf.name_scope("branch1"):
            # sum of gradients across accumulation batches
            bad_gsum = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in t_vars]
            branch_anatomy_loss = 0
            for i, anatomy_index in enumerate(anatomy_indices_list):
              # i is the index in anatomy_placeholder
              # anatomy_index is the index in the image filenames

              # get the structure name e.g. "fsVentricles"
              structure = image_filenames_list[anatomy_index].replace('.nii.gz','')
              
              # note: branch_anatomy_softmax_op[:,:,:,:,0] is background
              bad_numerator_op   = dice_coe(branch_anatomy_softmax_op[:,:,:,:,i+1], tf.cast(anatomy_placeholder[:,:,:,:,i],dtype=tf.float32), loss_type='dice',axis=[1,2,3],compute='numerator')
              bad_denominator_op = dice_coe(branch_anatomy_softmax_op[:,:,:,:,i+1], tf.cast(anatomy_placeholder[:,:,:,:,i],dtype=tf.float32), loss_type='dice',axis=[1,2,3],compute='denominator')
              bad_op             = (bad_numerator_op + smooth_batch) / (bad_denominator_op + smooth_batch)
              bad_numerator_ops.append(bad_numerator_op)
              bad_denominator_ops.append(bad_denominator_op)
              bad_ops.append(bad_op)
              branch_anatomy_loss += bad_op / float(len(anatomy_indices_list))
              avg_branch_anatomy_loss += bad_batchavg/float(len(anatomy_indices_list))
              
              # Operations to compute the metric and the gradient for the batch
              bad_sum  = tf.get_variable(structure+"_dice_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
              #bad_numerator_sum    = tf.get_variable("bad_numerator_sum"   , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
              #bad_denominator_sum  = tf.get_variable("bad_denominator_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
              bad_sums.append(bad_sum)
              #anatomy_soft_dice_batch    = (bad_numerator_sum + smooth_batch) / (bad_denominator_sum + smooth_batch)
              bad_batchavg = tf.cond(n_ab > 0., lambda: bad_sum/n_ab, lambda: tf.constant(0.)) 
              bad_batchavgs.append(bad_batchavg)

              # Register these quantities in the Tensorboard output
              tf.summary.scalar(structure+'_dice',bad_batchavg)
            tf.summary.scalar('avg_dice', branch_anatomy_loss)
             
          with tf.name_scope("final"):
            # sum of gradients across accumulation batches
            fad_gsum = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in t_vars]
            final_anatomy_loss = 0
            avg_final_anatomy_loss = 0
            for i, anatomy_index in enumerate(anatomy_indices_list):
              # i is the index in anatomy_placeholder
              # anatomy_index is the index in the image filenames

              # get the structure name e.g. "fsVentricles"
              structure = image_filenames_list[anatomy_index].replace('.nii.gz','')
              
              # note: branch_anatomy_softmax_op[:,:,:,:,0] is background
              fad_numerator_op   = dice_coe(branch_anatomy_softmax_op[:,:,:,:,i+1], tf.cast(anatomy_placeholder[:,:,:,:,i],dtype=tf.float32), loss_type='dice',axis=[1,2,3],compute='numerator')
              fad_denominator_op = dice_coe(branch_anatomy_softmax_op[:,:,:,:,i+1], tf.cast(anatomy_placeholder[:,:,:,:,i],dtype=tf.float32), loss_type='dice',axis=[1,2,3],compute='denominator')
              fad_op             = (fad_numerator_op + smooth_batch) / (fad_denominator_op + smooth_batch)
              fad_numerator_ops.append(fad_numerator_op)
              fad_denominator_ops.append(fad_denominator_op)
              fad_ops.append(fad_op)
              final_anatomy_loss += fad_op / float(len(anatomy_indices_list))
              
              # Operations to compute the metric and the gradient for the batch
              fad_sum  = tf.get_variable(structure+"_dice_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
              #fad_numerator_sum    = tf.get_variable("fad_numerator_sum"   , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
              #fad_denominator_sum  = tf.get_variable("fad_denominator_sum" , dtype=tf.float32, trainable=False, initializer=tf.constant(0.))
              fad_sums.append(fad_sum)
              #anatomy_soft_dice_batch    = (fad_numerator_sum + smooth_batch) / (fad_denominator_sum + smooth_batch)
              fad_batchavg = tf.cond(n_ab > 0., lambda: fad_sum/n_ab, lambda: tf.constant(0.)) 
              fad_batchavgs.append(fad_batchavg)
              avg_final_anatomy_loss += fad_batchavg/float(len(anatomy_indices_list))

              # Register these quantities in the Tensorboard output
              tf.summary.scalar(structure+'_dice',fad_batchavg)
            tf.summary.scalar('avg_dice', final_anatomy_loss)

        # Training Op
        with tf.name_scope("training"):
            final_label_weight    = 0.6
            branch_anatomy_weight = 0.2
            final_anatomy_weight  = 0.2

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


            # create a copy of all trainable variables with `0` as initial values
            accum_tvars = [tf.Variable(tf.zeros_like(t_var.initialized_value()),trainable=False) for t_var in t_vars]                                        
            # create a op to initialize all accums vars
            zero_op = [t_var.assign(tf.zeros_like(t_var)) for t_var in accum_tvars]
            
            # finish defining the Final Label dice term
            specific_loss = (1. - specific_dice_op)
            if FLAGS.distance_map_index >= 0 and FLAGS.boundary_loss_weight!=0:
              specific_loss += boundary_loss_op * scalar_boundary_loss_weight
            
            # g_i: the gradient of the Dice metric for the i'th sample
            # g1_i: gradient vector of Final Label specific dice w.r.t. network parameters
            # g2_i: gradient vector of Final Anatomy dice w.r.t. network parameters
            # g3_i: gradient vector of Branch Anatomy dice w.r.t. network parameters
            g1_i = optimizer.compute_gradients(specific_loss)
            g2_i = optimizer.compute_gradients(final_anatomy_loss)
            g3_i = optimizer.compute_gradients(branch_anatomy_loss)
            
            accum_op = [specific_dice_gsum[j].assign_add(g1_i[j][0]) for j in range(len(g1_i))]
            accum_op += [fad_gsum[j].assign_add(g2_i[j][0]) for j in range(len(g2_i))]
            accum_op += [bad_gsum[j].assign_add(g2_i[j][0]) for j in range(len(g3_i))]

            zero_op += [specific_dice_gsum_j.assign(tf.zeros_like(specific_dice_gsum_j)) for specific_dice_gsum_j in specific_dice_gsum]
            zero_op += [fad_gsum_j.assign(tf.zeros_like(fad_gsum_j)) for fad_gsum_j in fad_gsum]
            zero_op += [bad_gsum_j.assign(tf.zeros_like(bad_gsum_j)) for bad_gsum_j in bad_gsum]

            compute_gradient_op = []
            for j in range(len(t_vars)):
              compute_gradient_op += [accum_tvars[j].assign((final_label_weight*specific_dice_gsum[j] + branch_anatomy_weight*bad_gsum[j] + final_anatomy_weight*fad_gsum[j])/ n_ab)]
              
            apply_gradients_op = optimizer.apply_gradients([(accum_tvars[i], t_vars[i]) for i in range(len(t_vars))], global_step=global_step)
            
            # average loss function for the whole accumulated batch, for monitoring
            loss_avg = final_label_weight*specific_dice_loss_op + branch_anatomy_weight*avg_branch_anatomy_loss + final_anatomy_weight*avg_final_anatomy_loss
        
        with tf.name_scope("avgloss"):
            # Here we accumulate the average loss and square of loss across the accum. batches
            sum_zero_op = [n_ab.assign(tf.zeros_like(n_ab))]
            sum_zero_op += [hard_dice_numerator_sum.assign(tf.zeros_like(hard_dice_numerator_sum))]
            sum_zero_op += [hard_dice_denominator_sum.assign(tf.zeros_like(hard_dice_denominator_sum))]
            sum_zero_op += [soft_dice_numerator_sum.assign(tf.zeros_like(soft_dice_numerator_sum))]
            sum_zero_op += [soft_dice_denominator_sum.assign(tf.zeros_like(soft_dice_denominator_sum))]
            sum_zero_op += [specific_dice_sum.assign(tf.zeros_like(specific_dice_sum))]
            # zero out branch anatomy dice and final anatomy dice stuff (bad/fad):
            for i in range(len(anatomy_indices_list)): 
              sum_zero_op += [bad_sums[i].assign(tf.zeros_like(bad_sums[i])), fad_sums[i].assign(tf.zeros_like(fad_sums[i]))]
            
            sum_accum_op = [n_ab.assign_add(1.)]
            sum_accum_op += [hard_dice_numerator_sum.assign_add(hard_dice_numerator_op)]
            sum_accum_op += [hard_dice_denominator_sum.assign_add(hard_dice_denominator_op)]
            sum_accum_op += [soft_dice_numerator_sum.assign_add(soft_dice_numerator_op)]
            sum_accum_op += [soft_dice_denominator_sum.assign_add(soft_dice_denominator_op)]
            sum_accum_op += [specific_dice_sum.assign_add(specific_dice_op)]
            # accumulate branch anatomy dice and final anatomy dice stuff (bad/fad): 
            for i in range(len(anatomy_indices_list)):
              sum_accum_op += [bad_sums[i].assign_add(bad_ops[i]), fad_sums[i].assign_add(fad_ops[i])]

            if FLAGS.distance_map_index >= 0:
              sum_zero_op += [boundary_loss_sum.assign(tf.zeros_like(boundary_loss_sum))]
              sum_accum_op += [boundary_loss_sum.assign_add(boundary_loss_op)]
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
        #saver = tf.train.Saver(keep_checkpoint_every_n_hours=8,max_to_keep=1)
        saver = tf.train.Saver(max_to_keep=3)

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
              
              # Boundary loss scheduler
              if FLAGS.distance_map_index >= 0:
                if epoch>0: #and (epoch%10)==0:
                  sess.run(increase_boundary_loss_weight_op)

              # training phase
              #n_train = 0
              #logger.debug('Zeroing gradients and loss sums')
              #sess.run(zero_op) # reset gradient accumulation
              #sess.run(sum_zero_op) # reset loss-averaging
              model.is_training = True;
              logger.debug('Beginning accumulation batch')
              #while True: # Beginning of Accumulation batch
              #  try:
              batch_train_loss_sum = 0
              for accum_batch in range(FLAGS.accum_batches_per_epoch):
                  logger.debug('Beginning loop over the accumulation crops')
                  #[image, label] = sess.run(next_element_train)
                  
                  # add rounding here
                  #difficulty_map = label
                  
                  logger.debug('Zeroing gradients and loss sums')
                  sess.run(zero_op) # reset gradient accumulation
                  sess.run(sum_zero_op) # reset loss-averaging

                  for i in range(n_accum_crops):
                    [inputs, label] = sess.run(next_element_train)
                    #image = image[:,:,:,:,:] 
                    image = inputs[:,:,:,:,image_indices_list] 
                    aux = inputs[:,:,:,:,auxiliary_indices_list]
                    label = label[:,:,:,:,np.newaxis]
                    feed_dict = {images_placeholder: image, labels_placeholder: label, aux_placeholder: aux}
                    train, train_loss, sum_accum = sess.run([accum_op, loss_avg, sum_accum_op], feed_dict=feed_dict)
                  n_train = int(sess.run(n_ab))

                  logger.debug("Applying gradients after total %d accumulations"%n_train)
                  #if FLAGS.loss_function in ['dice','jaccard','specific_dice']:
                  if (compute_gradient_op != []):
                    sess.run(compute_gradient_op)
                  sess.run(apply_gradients_op)
                  
                  batch_train_loss_avg = sess.run(loss_avg)
                  batch_train_loss_sum += batch_train_loss_avg
                  
                  logger.debug('Applying summary op')
                  summary = sess.run(summary_op)
                  train_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))
              '''
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
                '''
              # Compute the average training loss across all batches in the epoch.
              epoch_train_loss_avg = batch_train_loss_sum / float(FLAGS.accum_batches_per_epoch)
              
              print("{0}: Average training loss is {1:.3f} over {2:d}".format(datetime.datetime.now(), epoch_train_loss_avg, FLAGS.accum_batches_per_epoch*n_accum_crops))
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
              
              print('profile\n',TrainDataset.profile)

              # testing phase
              print("{}: Training of epoch {} finishes, testing start".format(datetime.datetime.now(),epoch+1))
              #test_loss_avg = 0.0
              #n_test = 0
              logger.debug('Zeroing gradients and loss sums')
              sess.run(zero_op) # reset gradient accumulation
              sess.run(sum_zero_op) # reset loss-averaging
              #while True:
              #  try:
              model.is_training = False;
              for i in range(n_accum_crops):
                  [inputs, label] = sess.run(next_element_test)
                  #image = image[:,:,:,:,:] 
                  image = inputs[:,:,:,:,image_indices_list] 
                  aux = inputs[:,:,:,:,auxiliary_indices_list]
                  label = label[:,:,:,:,np.newaxis]
                  feed_dict = {images_placeholder: image, labels_placeholder: label, aux_placeholder: aux}
                  #model.is_training = False;
                  test_loss, sum_accum = sess.run([loss_avg, sum_accum_op], feed_dict=feed_dict)
                  # This is redundant, fix
                  #test_loss_avg += test_loss
                  #n_test += 1
              '''
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
                '''
              # Compute the average testing loss across the batch
              test_loss_avg = sess.run(specific_dice_loss_op)
              n_test = int(sess.run(n_ab))
              print("{0}: Average testing loss is {1:.3f} over {2:d}".format(datetime.datetime.now(), test_loss_avg, n_test))
              summary = sess.run(summary_op)
              test_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))

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
