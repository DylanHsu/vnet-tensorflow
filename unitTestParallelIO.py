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
from dice import dice_coe

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s','%m-%d %H:%M:%S'))
logger = logging.getLogger('parallelIO')
logger.addHandler(console)
logger.setLevel(logging.DEBUG)
logger.propagate = False


with tf.Graph().as_default():
  with tf.device('/cpu:0'):
    trainTransforms = [
        NiftiDataset.RandomCrop((48,48,48), 0.5, 1),
        NiftiDataset.RandomNoise(0,0.1),
        NiftiDataset.RandomFlip(0.5, [True,True,True]),
        ]
    TrainDataset = NiftiDataset.NiftiDataset(
      data_dir='/data/deasy/DylanHsu/SRS_N401/subgroup1/training',
      image_filenames="mr1.nii.gz,ct.nii.gz,fsBasalGanglia.nii.gz,fsBrainstem.nii.gz,fsCerebellumCortex.nii.gz,fsCerebellumWM.nii.gz,fsCerebralCortex.nii.gz,fsCerebralWM.nii.gz,fsChoroidPlexus.nii.gz,fsCorpusCallosum.nii.gz,fsCSF.nii.gz,fsHippocampi.nii.gz,fsOpticChiasm.nii.gz,fsThalami.nii.gz,fsVentralDiencephalon.nii.gz,fsVentricles.nii.gz",
      label_filename="label_smoothed.nii.gz",
      transforms=trainTransforms,
      num_crops=1,
      train=True,
      cpu_threads=32
      )
    trainDataset = TrainDataset.get_dataset()
    trainDataset = trainDataset.apply(tf.contrib.data.unbatch())
    trainDataset = trainDataset.repeat() 
    trainDataset = trainDataset.batch(32)
    trainDataset = trainDataset.prefetch(5)
  
  train_iterator = trainDataset.make_initializable_iterator()
  next_element_train = train_iterator.get_next()
  
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()
    sess.run(train_iterator.initializer)
    while True:
      [inputs, label] = sess.run(next_element_train)
      print('opened batch')
