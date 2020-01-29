import os,sys
from glob import glob
import numpy as np
# Setup job configs

# DGH note to self: add option to split this into N/10 GPU jobs or something?

useGpu=True

jobfolder = "./jobs/"
data_dir = '/data/deasy/DylanHsu/N200_1mm3/nifti'
patch_size  = 64 
patch_layer = 64 
stride_inplane = 4
stride_layer   = 4

#suffix='-scanSizeV1mm3-bs1000-dr0p50-40mm'
#checkpoint_path='tmp/ckpt/bak/checkpoint_n200-scanSizeV1mm3-bs1000-dr0p50-40mm-1776'
#suffix='-scanSizeV1mm3-bs1000-dr0p50-48mm'
#checkpoint_path='tmp/ckpt/bak/checkpoint_n200-scanSizeV1mm3-bs1000-dr0p50-48mm-1200'
#suffix='-scanSizeV1mm3-bs1000-dr0p50-56mm'
#checkpoint_path='tmp/ckpt/bak/checkpoint_n200-scanSizeV1mm3-bs1000-dr0p50-56mm-1216'
suffix='-scanSizeV1mm3-bs1000-dr0p50-64mm'
checkpoint_path='tmp/ckpt/bak/checkpoint_n200-scanSizeV1mm3-bs1000-dr0p50-64mm-992'

model_path = checkpoint_path + '.meta'

if useGpu:
  cpu_cores=1
  cpu_ram=16
else:
  cpu_cores = 16
  cpu_ram = 16
for case in os.listdir(data_dir):
  jobname = 'evaluate%s-%s'%(suffix,case)
  #jobname = jobname.replace('.','p')
  jobfile = os.path.join(jobfolder,jobname+".lsf")
  # Setup job files
  f = open(jobfile,"w+")
  f.write("#!/bin/bash\n")
  f.write("#BSUB -J "+jobname+"\n")
  f.write("#BSUB -n %d\n" % cpu_cores)
  if useGpu:
    f.write("#BSUB -q gpuqueue -gpu \"num=1\" \n")
  else:
    f.write("#BSUB -q cpuqueue\n")
  f.write("#BSUB -R span[hosts=1]\n")
  f.write("#BSUB -R rusage[mem=%d]\n" % (cpu_ram//cpu_cores))
  f.write("#BSUB -W 24:00\n")
  f.write("#BSUB -o "+jobfolder+"%J.stdout\n")
  f.write("#BSUB -eo "+jobfolder+"%J.stderr\n")
  f.write("\n")
  f.write("source /home/hsud3/.bash_profile\n")
  f.write("cd /home/hsud3/vnet-tensorflow \n")
  f.write('python evaluate.py ')
  f.write(' --data_dir %s'%data_dir)
  f.write(' --label_filename label_smoothed.nii.gz')
  f.write(' --model_path %s'%model_path)
  f.write(' --checkpoint_path %s'%checkpoint_path)
  f.write(' --patch_size %d'%patch_size)
  f.write(' --patch_layer %d'%patch_layer)
  f.write(' --stride_inplane %d'%stride_inplane)
  f.write(' --stride_layer %d'%stride_layer)
  f.write(' --suffix %s'%suffix)
  f.write(' --case %s'%case)
  f.write(' --is_batch')
  f.write("\n")
  f.close()
# Submit jobs.
  the_command = "bsub < " + jobfile
  os.system(the_command)
