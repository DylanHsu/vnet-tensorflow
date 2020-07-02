import os,sys
from glob import glob
import numpy as np
# Setup job configs

# DGH note to self: add option to split this into N/10 GPU jobs or something?

useGpu=False

jobfolder = "./jobs/"
subfolder = "training_noaug" # "testing"
case_folder = '/data/deasy/DylanHsu/SRS_N514'
patch_size  = 32 
patch_layer = 32 
stride_inplane = 8
stride_layer   = stride_inplane
useFreesurfer=False
useCt=True

#:21,55s/\(tmp.*n514\)\(.*-LB1\)\(-subgroup\)\(\d\)\(-.*\).meta/configs.append({'sg':\4, 'checkpoint_path':'\1\2\3\4\5', 'suffix':'\2\3\4'})/g

configs=[]
configs.append({'sg':1, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-32mm-mrct-LB1-subgroup1-540', 'suffix':'-32mm-mrct-LB1-subgroup1'})
configs.append({'sg':2, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-32mm-mrct-LB1-subgroup2-480', 'suffix':'-32mm-mrct-LB1-subgroup2'})
configs.append({'sg':3, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-32mm-mrct-LB1-subgroup3-750', 'suffix':'-32mm-mrct-LB1-subgroup3'})
configs.append({'sg':4, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-32mm-mrct-LB1-subgroup4-525', 'suffix':'-32mm-mrct-LB1-subgroup4'})
configs.append({'sg':5, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-32mm-mrct-LB1-subgroup5-750', 'suffix':'-32mm-mrct-LB1-subgroup5'})

#configs.append({'sg':1, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-64mm-mrct-LB1-subgroup1-750', 'suffix':'-64mm-mrct-LB1-subgroup1'})
#configs.append({'sg':2, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-64mm-mrct-LB1-subgroup2-750', 'suffix':'-64mm-mrct-LB1-subgroup2'})
#configs.append({'sg':3, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-64mm-mrct-LB1-subgroup3-750', 'suffix':'-64mm-mrct-LB1-subgroup3'})
#configs.append({'sg':4, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-64mm-mrct-LB1-subgroup4-750', 'suffix':'-64mm-mrct-LB1-subgroup4'})
#configs.append({'sg':5, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-64mm-mrct-LB1-subgroup5-750', 'suffix':'-64mm-mrct-LB1-subgroup5'})

#configs.append({'sg':1, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrctfs-LB1-subgroup1-750', 'suffix':'-48mm-mrctfs-LB1-subgroup1'})
#configs.append({'sg':2, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrctfs-LB1-subgroup2-750', 'suffix':'-48mm-mrctfs-LB1-subgroup2'})
#configs.append({'sg':3, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrctfs-LB1-subgroup3-750', 'suffix':'-48mm-mrctfs-LB1-subgroup3'})
#configs.append({'sg':4, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrctfs-LB1-subgroup4-750', 'suffix':'-48mm-mrctfs-LB1-subgroup4'})
#configs.append({'sg':5, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrctfs-LB1-subgroup5-750', 'suffix':'-48mm-mrctfs-LB1-subgroup5'})
#configs.append({'sg':1, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrctfs-subgroup1-750', 'suffix':'-48mm-mrctfs-subgroup1'})
#configs.append({'sg':2, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrctfs-subgroup2-750', 'suffix':'-48mm-mrctfs-subgroup2'})
#configs.append({'sg':3, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrctfs-subgroup3-750', 'suffix':'-48mm-mrctfs-subgroup3'})
#configs.append({'sg':4, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrctfs-subgroup4-750', 'suffix':'-48mm-mrctfs-subgroup4'})
#configs.append({'sg':5, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrctfs-subgroup5-750', 'suffix':'-48mm-mrctfs-subgroup5'})

#configs.append({'sg':1, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-deep-subgroup1-750', 'suffix':'-48mm-mrct-LB1-deep-subgroup1'})
#configs.append({'sg':2, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-deep-subgroup2-750', 'suffix':'-48mm-mrct-LB1-deep-subgroup2'})
#configs.append({'sg':3, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-deep-subgroup3-750', 'suffix':'-48mm-mrct-LB1-deep-subgroup3'})
#configs.append({'sg':4, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-deep-subgroup4-750', 'suffix':'-48mm-mrct-LB1-deep-subgroup4'})
#configs.append({'sg':5, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-deep-subgroup5-750', 'suffix':'-48mm-mrct-LB1-deep-subgroup5'})
#configs.append({'sg':1, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-dr0p00-subgroup1-750', 'suffix':'-48mm-mrct-LB1-dr0p00-subgroup1'})
#configs.append({'sg':2, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-dr0p00-subgroup2-750', 'suffix':'-48mm-mrct-LB1-dr0p00-subgroup2'})
#configs.append({'sg':3, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-dr0p00-subgroup3-750', 'suffix':'-48mm-mrct-LB1-dr0p00-subgroup3'})
#configs.append({'sg':4, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-dr0p00-subgroup4-750', 'suffix':'-48mm-mrct-LB1-dr0p00-subgroup4'})
#configs.append({'sg':5, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-dr0p00-subgroup5-750', 'suffix':'-48mm-mrct-LB1-dr0p00-subgroup5'})
#configs.append({'sg':1, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-dr0p75-subgroup1-750', 'suffix':'-48mm-mrct-LB1-dr0p75-subgroup1'})
#configs.append({'sg':2, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-dr0p75-subgroup2-750', 'suffix':'-48mm-mrct-LB1-dr0p75-subgroup2'})
#configs.append({'sg':3, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-dr0p75-subgroup3-750', 'suffix':'-48mm-mrct-LB1-dr0p75-subgroup3'})
#configs.append({'sg':4, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-dr0p75-subgroup4-750', 'suffix':'-48mm-mrct-LB1-dr0p75-subgroup4'})
#configs.append({'sg':5, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-dr0p75-subgroup5-750', 'suffix':'-48mm-mrct-LB1-dr0p75-subgroup5'})

#configs.append({'sg':1, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-subgroup1-750', 'suffix':'-48mm-mrct-LB1-subgroup1'})
#configs.append({'sg':2, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-subgroup2-750', 'suffix':'-48mm-mrct-LB1-subgroup2'})
#configs.append({'sg':3, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-subgroup3-750', 'suffix':'-48mm-mrct-LB1-subgroup3'})
#configs.append({'sg':4, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-subgroup4-750', 'suffix':'-48mm-mrct-LB1-subgroup4'})
#configs.append({'sg':5, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB1-subgroup5-750', 'suffix':'-48mm-mrct-LB1-subgroup5'})
#configs.append({'sg':1, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB2-subgroup1-750', 'suffix':'-48mm-mrct-LB2-subgroup1'})
#configs.append({'sg':2, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB2-subgroup2-750', 'suffix':'-48mm-mrct-LB2-subgroup2'})
#configs.append({'sg':3, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB2-subgroup3-750', 'suffix':'-48mm-mrct-LB2-subgroup3'})
#configs.append({'sg':4, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB2-subgroup4-750', 'suffix':'-48mm-mrct-LB2-subgroup4'})
#configs.append({'sg':5, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-LB2-subgroup5-750', 'suffix':'-48mm-mrct-LB2-subgroup5'})
#configs.append({'sg':1, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-subgroup1-750', 'suffix':'-48mm-mrct-subgroup1'})
#configs.append({'sg':2, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-subgroup2-750', 'suffix':'-48mm-mrct-subgroup2'})
#configs.append({'sg':3, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-subgroup3-750', 'suffix':'-48mm-mrct-subgroup3'})
#configs.append({'sg':4, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-subgroup4-750', 'suffix':'-48mm-mrct-subgroup4'})
#configs.append({'sg':5, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mrct-subgroup5-750', 'suffix':'-48mm-mrct-subgroup5'})

#configs.append({'sg':1, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mr-subgroup1-750', 'suffix':'-48mm-mr-subgroup1'})
#configs.append({'sg':2, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mr-subgroup2-750', 'suffix':'-48mm-mr-subgroup2'})
#configs.append({'sg':3, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mr-subgroup3-750', 'suffix':'-48mm-mr-subgroup3'})
#configs.append({'sg':4, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mr-subgroup4-750', 'suffix':'-48mm-mr-subgroup4'})
#configs.append({'sg':5, 'checkpoint_path':'tmp/ckpt/checkpoint_n514-48mm-mr-subgroup5-750', 'suffix':'-48mm-mr-subgroup5'})


#suffix = '-mrct-alpha0p8-subgroup%d'%(sg)
#suffix = '-mrct-subgroup%d'%(sg)
#suffix = '-48mm-minSmall5x-subgroup%d-stride%d'%(sg,stride_inplane)
#suffix = '-48mm-noSmallBias-subgroup%d'
#data_dir='/data/deasy/DylanHsu/N401_unstripped/subgroup%d/testing'%(sg)
#data_dir='/data/deasy/DylanHsu/SRS_N401/subgroup%d/testing'%(sg)

#model_path = checkpoint_path + '.meta'

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

for config in configs:
  sg, checkpoint_path, suffix = config['sg'], config['checkpoint_path'], config['suffix']
  model_path = checkpoint_path + '.meta'
  #data_dir='/data/deasy/DylanHsu/SRS_N514/subgroup%d/testing'%(sg)
  data_dir= os.path.join(case_folder, 'subgroup%d'%sg, subfolder)
  
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
    if useFreesurfer and useCt:
      f.write(' --image_filenames "mr1.nii.gz,ct.nii.gz,fsBasalGanglia.nii.gz,fsBrainstem.nii.gz,fsCerebellumCortex.nii.gz,fsCerebellumWM.nii.gz,fsCerebralCortex.nii.gz,fsCerebralWM.nii.gz,fsChoroidPlexus.nii.gz,fsCorpusCallosum.nii.gz,fsCSF.nii.gz,fsHippocampi.nii.gz,fsOpticChiasm.nii.gz,fsThalami.nii.gz,fsVentralDiencephalon.nii.gz,fsVentricles.nii.gz" ')
    elif useCt:
      f.write(' --image_filenames "mr1.nii.gz,ct.nii.gz" ')
    else:
      f.write(' --image_filenames "mr1.nii.gz" ')
    if not useGpu:
      f.write(' --use_cpu')
    f.write("\n")
    f.close()
    # Submit jobs.
    the_command = "bsub < " + jobfile
    os.system(the_command)
