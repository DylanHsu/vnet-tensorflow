import os,sys
from glob import glob
import numpy as np
# Setup job configs

# DGH note to self: add option to split this into N/10 GPU jobs or something?

useGpu=True
#useGpu=True

jobfolder = "./jobs/"
#data_dir = '/data/deasy/DylanHsu/N200_1mm3/nifti'
patch_size  = 48 
patch_layer = 48 
stride_inplane = 8
stride_layer   = stride_inplane

#subgroup=5
#data_dir = '/data/deasy/DylanHsu/N401_unstripped/subgroup%d/testing'%subgroup
#suffix = '-48mm-noSmallBias-subgroup%d'%subgroup
#checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-noSmallBias-subgroup%d-1073'%subgroup

#sg=1; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-minSmall5x-mrct-subgroup1-645'
#sg=2; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-minSmall5x-mrct-subgroup2-510'
#sg=3; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-minSmall5x-mrct-subgroup3-810'
#sg=4; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-minSmall5x-mrct-subgroup4-810'
#sg=5; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-minSmall5x-mrct-subgroup5-750'

#sg=1; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-mrct-alpha0p8-subgroup1-780'
#sg=2; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-mrct-alpha0p8-subgroup2-885'
#sg=3; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-mrct-alpha0p8-subgroup3-660'
#sg=4; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-mrct-alpha0p8-subgroup4-885'
#sg=5; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-mrct-alpha0p8-subgroup5-765'
#sg=1; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-mrct-subgroup1-915'
#sg=2; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-mrct-subgroup2-780'
#sg=3; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-mrct-subgroup3-885'
#sg=4; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-mrct-subgroup4-765'
sg=5; checkpoint_path='tmp/ckpt/checkpoint_n401-48mm-mrct-subgroup5-855'

#suffix = '-mrct-alpha0p8-subgroup%d'%(sg)
suffix = '-mrct-subgroup%d'%(sg)
#suffix = '-48mm-minSmall5x-subgroup%d-stride%d'%(sg,stride_inplane)
#suffix = '-48mm-noSmallBias-subgroup%d'
#data_dir='/data/deasy/DylanHsu/N401_unstripped/subgroup%d/testing'%(sg)
data_dir='/data/deasy/DylanHsu/SRS_N401/subgroup%d/testing'%(sg)

model_path = checkpoint_path + '.meta'

try:
  os.makedirs(os.path.join(jobfolder,'logs'))
except:
  pass

if useGpu:
  cpu_cores=1
  cpu_ram=16
else:
  cpu_cores = 32
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
  f.write("#BSUB -R rusage[mem=%d]\n" % (cpu_ram/cpu_cores))
  f.write("#BSUB -W 24:00\n")
  f.write("#BSUB -o " +jobfolder+"/logs/"+jobname+"_%J.stdout\n")
  f.write("#BSUB -eo "+jobfolder+"/logs/"+jobname+"_%J.stderr\n")
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
  if not useGpu:
    f.write(' --use_cpu')
  f.write("\n")
  f.close()
  # Submit jobs.
  the_command = "bsub < " + jobfile
  os.system(the_command)
