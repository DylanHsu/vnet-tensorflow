import os, sys

setupScript='/home/hsud3/.bash_profile'
workArea='/home/hsud3/vnet-tensorflow'
jobDir='/home/hsud3/vnet-tensorflow/jobs'
os.makedirs(jobDir,exist_ok=True)

bsubHeaderTemplate="""#!/bin/bash
#BSUB -J {0}-{1}
#BSUB -n 4
#BSUB -q gpuqueue
#BSUB -gpu num=1
#BSUB -R span[hosts=1]
#BSUB -R rusage[mem=2]
#BSUB -W 168:00
#BSUB -o  ./jobs/logs/{0}-{1}_%J.stdout
#BSUB -eo ./jobs/logs/{0}-{1}_%J.stderr
source {2}
cd {3}"""
#BSUB -W 5:59

class JobConfig(object):
  def __init__(self,
    jobName='',
    useMr1=True, useMr2=False,
    useCt=False, useFs=False,
    dx=48, dz=48,
    dropRatio=0.5,
    gamma=0,
    vnetConvs="3,3,4,4,4"
  ):
    self.job_name             = jobName
    self.data_dir             = '/data/deasy/DylanHsu/SRS_N511'
    self.vnet_convs           = vnetConvs
    self.epochs               = 30
    self.label_filename       = "label_smoothed.nii.gz"
    self.patch_size           = dx
    self.patch_layer          = dz
    self.boundary_loss_weight = gamma
    self.drop_ratio           = dropRatio
    self.num_channels         = 0
    self.image_filenames      = []
    if useMr1:
      self.image_filenames.append("mr1.nii.gz")
      self.num_channels += 1
    if useMr2:
      self.image_filenames.append("mr2.nii.gz")
      self.num_channels += 1
    if useCt:
      self.image_filenames.append("ct.nii.gz")
      self.num_channels += 1
    if self.boundary_loss_weight != 0:
      self.distance_map_index = len(self.image_filenames)
      self.image_filenames.append("distance_map.nii.gz")
    if useFs:
      self.image_filenames += ["fsBasalGanglia.nii.gz", "fsBrainstem.nii.gz", "fsCerebellumCortex.nii.gz", "fsCerebellumWM.nii.gz", "fsCerebralCortex.nii.gz", "fsCerebralWM.nii.gz", "fsChoroidPlexus.nii.gz", "fsCorpusCallosum.nii.gz", "fsCSF.nii.gz", "fsHippocampi.nii.gz", "fsOpticChiasm.nii.gz", "fsThalami.nii.gz", "fsVentralDiencephalon.nii.gz", "fsVentricles.nii.gz"]
      self.num_channels += 14
    assert len(self.image_filenames) > 0
    self.image_filenames = ','.join(self.image_filenames)
    self.max_ram = 8
  
  def produceFiles(self,jobDir):
    subgroups = ['subgroup%d'%(i) for i in [1,2,3,4,5]]
    subgroups += ['final']
    for subgroup in subgroups:
      bsubHeader=bsubHeaderTemplate.format(self.job_name, subgroup, setupScript, workArea)
      trainCmd = ("python train.py --is_batch_job"+
        " --epochs %d  --batch_size 1  --accum_batches 500  --num_crops 1"%(self.epochs) +
        " --batch_job_name %s-%s"%(self.job_name, subgroup) +
        " --vnet_convs %s"%(self.vnet_convs) +
        " --num_channels %d"%(self.num_channels) + 
        " --image_filenames \"%s\""%(self.image_filenames) +
        " --label_filename %s"%(self.label_filename) +
        "  --init_learning_rate 0.01" +
        " --min_pixel 1 --drop_ratio %.6f"%(self.drop_ratio) +
        " --patch_size %d  --patch_layer %d"%(self.patch_size,self.patch_layer) +
        " --loss_function specific_dice --optimizer adam" +
        " --data_dir %s"%(os.path.join(self.data_dir,subgroup)) +
        " --max_ram 8"
      )
      if self.boundary_loss_weight != 0 :
        trainCmd += " --boundary_loss_weight %.1e"%(self.boundary_loss_weight)
        trainCmd += " --distance_map_index %d"%(self.distance_map_index)
      output = bsubHeader + "\n" + trainCmd
      f=open(os.path.join(jobDir,"%s-%s.lsf"%(self.job_name,subgroup)),"w+")
      f.write(output)
      f.close()

configs=[]
configs.append(JobConfig('n511-48mm-mr'))
configs.append(JobConfig('n511-48mm-mr-LB1',useCt=False,gamma=0.01))
configs.append(JobConfig('n511-48mm-mr-LB2',useCt=False,gamma=0.02))
configs.append(JobConfig('n511-48mm-mrct',useCt=True))
configs.append(JobConfig('n511-48mm-mrctfs',useCt=True,useFs=True))
configs.append(JobConfig('n511-48mm-mrctfs',useCt=True,useFs=True))
configs.append(JobConfig('n511-48mm-mrctfs-LB1',useCt=True,useFs=True,gamma=0.01))
configs.append(JobConfig('n511-48mm-mrctfs-LB2',useCt=True,useFs=True,gamma=0.02))

configs.append(JobConfig('n511-48mm-mrct-LB1',useCt=True,gamma=0.01))
configs.append(JobConfig('n511-48mm-mrct-LB2',useCt=True,gamma=0.02))
configs.append(JobConfig('n511-48mm-mrct-LB1-deep',useCt=True,gamma=0.01,vnetConvs="5,5,4,4,4"))
configs.append(JobConfig('n511-64mm-mrct-LB1',useCt=True,gamma=0.01,dx=64,dz=64))
configs.append(JobConfig('n511-48mm-mrct-LB1-dr0p00',useCt=True,gamma=0.01,dropRatio=0.0))
configs.append(JobConfig('n511-48mm-mrct-LB1-dr0p75',useCt=True,gamma=0.01,dropRatio=0.75))
configs.append(JobConfig('n511-40mm-mrct-LB1',useCt=True,gamma=0.01,dx=40,dz=40))
configs.append(JobConfig('n511-56mm-mrct-LB1',useCt=True,gamma=0.01,dx=56,dz=56))

configs.append(JobConfig('n511-48mm-mr1mr2'      ,useMr1=True,useMr2=True,useCt=False,gamma=0.00))
configs.append(JobConfig('n511-48mm-mr1mr2-LB1'  ,useMr1=True,useMr2=True,useCt=False,gamma=0.01))
configs.append(JobConfig('n511-48mm-mr1mr2-LB2'  ,useMr1=True,useMr2=True,useCt=False,gamma=0.02))
configs.append(JobConfig('n511-48mm-mr1mr2-LB3p3',useMr1=True,useMr2=True,useCt=False,gamma=0.033))
configs.append(JobConfig('n511-48mm-mr2'      ,useMr1=False,useMr2=True,useCt=False,gamma=0.00))
configs.append(JobConfig('n511-48mm-mr2-LB1'  ,useMr1=False,useMr2=True,useCt=False,gamma=0.01))
configs.append(JobConfig('n511-48mm-mr2-LB2'  ,useMr1=False,useMr2=True,useCt=False,gamma=0.02))
configs.append(JobConfig('n511-48mm-mr2-LB3p3',useMr1=False,useMr2=True,useCt=False,gamma=0.033))
configs.append(JobConfig('n511-48mm-mr1mr2ct'      ,useMr1=True,useMr2=True,useCt=True,gamma=0.00))
configs.append(JobConfig('n511-48mm-mr1mr2ct-LB1'  ,useMr1=True,useMr2=True,useCt=True,gamma=0.01))
configs.append(JobConfig('n511-48mm-mr1mr2ct-LB2'  ,useMr1=True,useMr2=True,useCt=True,gamma=0.02))
configs.append(JobConfig('n511-48mm-mr1mr2ct-LB3p3',useMr1=True,useMr2=True,useCt=True,gamma=0.033))
for config in configs:
  config.produceFiles(jobDir)
