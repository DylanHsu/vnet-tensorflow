import os, sys

setupScript='/home/hsud3/.bash_profile'
workArea='/home/hsud3/vnet-tensorflow'
jobDir='/home/hsud3/vnet-tensorflow/jobs'
os.makedirs(jobDir,exist_ok=True)

bsubHeaderTemplate="""#!/bin/bash
#BSUB -J {0}-subgroup{1}
#BSUB -n 4
#BSUB -q gpuqueue
#BSUB -gpu num=1
#BSUB -R span[hosts=1]
#BSUB -R rusage[mem=2]
#BSUB -W 168:00
#BSUB -o  ./jobs/logs/{0}-subgroup{1}_%J.stdout
#BSUB -eo ./jobs/logs/{0}-subgroup{1}_%J.stderr
source {2}
cd {3}"""

class JobConfig(object):
  def __init__(self,
    jobName='',
    useCt=False, useFs=False,
    dx=48, dz=48,
    drop_ratio=0.5,
    dydt=0,
  ):
    self.job_name             = jobName
    self.data_dir             = '/data/deasy/DylanHsu/SRS_N501'
    self.vnet_convs           = "3,3,4,4,4"
    self.epochs               = 50
    self.label_filename       = "label_smoothed.nii.gz"
    self.patch_size           = dx
    self.patch_layer          = dz
    self.boundary_loss_weight = dydt
    self.drop_ratio           = drop_ratio
    self.num_channels         = 1
    self.image_filenames      = "mr1.nii.gz"
    if useCt:
      self.num_channels    += 1
      self.image_filenames += ",ct.nii.gz"
    if self.boundary_loss_weight != 0:
      self.distance_map_index = self.num_channels
      self.image_filenames += ",distance_map.nii.gz"
    if useFs:
      self.num_channels    += 14
      self.image_filenames += ",fsBasalGanglia.nii.gz,fsBrainstem.nii.gz,fsCerebellumCortex.nii.gz,fsCerebellumWM.nii.gz,fsCerebralCortex.nii.gz,fsCerebralWM.nii.gz,fsChoroidPlexus.nii.gz,fsCorpusCallosum.nii.gz,fsCSF.nii.gz,fsHippocampi.nii.gz,fsOpticChiasm.nii.gz,fsThalami.nii.gz,fsVentralDiencephalon.nii.gz,fsVentricles.nii.gz"
    self.max_ram = 8
  
  def produceFiles(self,jobDir):
    for subgroup in [1,2,3,4,5]:
      bsubHeader=bsubHeaderTemplate.format(self.job_name, subgroup, setupScript, workArea)
      trainCmd = ("python train.py --is_batch_job"+
        " --epochs 50  --batch_size 1  --accum_batches 1000  --num_crops 1" +
        " --batch_job_name %s-subgroup%d"%(self.job_name, subgroup) +
        " --vnet_convs %s"%(self.vnet_convs) +
        " --num_channels %d"%(self.num_channels) + 
        " --image_filenames \"%s\""%(self.image_filenames) +
        " --label_filename %s"%(self.label_filename) +
        "  --init_learning_rate 0.01" +
        " --min_pixel 1 --drop_ratio %.6f"%(self.drop_ratio) +
        " --patch_size %d  --patch_layer %d"%(self.patch_size,self.patch_layer) +
        " --loss_function specific_dice --optimizer adam" +
        " --data_dir %s"%(os.path.join(self.data_dir,"subgroup%s"%subgroup))
      )
      if self.boundary_loss_weight != 0 :
        trainCmd += " --boundary_loss_weight %.1e"%(self.boundary_loss_weight)
        trainCmd += " --distance_map_index %d"%(self.distance_map_index)
      output = bsubHeader + "\n" + trainCmd
      f=open(os.path.join(jobDir,"%s-subgroup%s.lsf"%(self.job_name,subgroup)),"w+")
      f.write(output)
      f.close()

configs=[]
configs.append(JobConfig('n501-48mm-mr'))
configs.append(JobConfig('n501-48mm-mrct',useCt=True))
configs.append(JobConfig('n501-48mm-mrctfs',useCt=True,useFs=True))
configs.append(JobConfig('n501-48mm-mrctfs',useCt=True,useFs=True))
configs.append(JobConfig('n501-48mm-mrctfs-LB1',useCt=True,useFs=True,dydt=2e-7))
for config in configs:
  config.produceFiles(jobDir)
