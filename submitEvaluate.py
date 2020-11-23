import os,sys
from glob import glob
import numpy as np
# Setup job configs

# DGH note to self: add option to split this into N/10 GPU jobs or something?

useGpu=True

jobfolder = "./jobs/"
subfolder = "testing" # "testing"
case_folder = '/data/deasy/DylanHsu/SRS_N511'
#patch_size  = 48
#patch_layer = 48
stride_inplane = 8
stride_layer   = stride_inplane
#useFreesurfer=True
#useCt=True

#:24,83s/\(tmp.*n511\)\(.*\)\(-subgroup\)\(\d\)\(-.*\).meta/configs.append({'sg':\4, 'checkpoint_path':'\1\2\3\4\5', 'suffix':'\2\3\4'})/g
configs=[]
#configs.append({'sg':1,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-LB1-subgroup1-1110', 'suffix':'-48mm-mrctfs-LB1-subgroup1'})
#configs.append({'sg':2,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-LB1-subgroup2-1110', 'suffix':'-48mm-mrctfs-LB1-subgroup2'})
#configs.append({'sg':3,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-LB1-subgroup3-1110', 'suffix':'-48mm-mrctfs-LB1-subgroup3'})
#configs.append({'sg':4,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-LB1-subgroup4-1110', 'suffix':'-48mm-mrctfs-LB1-subgroup4'})
#configs.append({'sg':5,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-LB1-subgroup5-1110', 'suffix':'-48mm-mrctfs-LB1-subgroup5'})
#configs.append({'sg':1,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-LB2-subgroup1-1110', 'suffix':'-48mm-mrctfs-LB2-subgroup1'})
#configs.append({'sg':2,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-LB2-subgroup2-1110', 'suffix':'-48mm-mrctfs-LB2-subgroup2'})
#configs.append({'sg':3,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-LB2-subgroup3-1110', 'suffix':'-48mm-mrctfs-LB2-subgroup3'})
#configs.append({'sg':4,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-LB2-subgroup4-1110', 'suffix':'-48mm-mrctfs-LB2-subgroup4'})
#configs.append({'sg':5,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-LB2-subgroup5-1110', 'suffix':'-48mm-mrctfs-LB2-subgroup5'})
#configs.append({'sg':1,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-subgroup1-1110', 'suffix':'-48mm-mrctfs-subgroup1'})
#configs.append({'sg':2,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-subgroup2-1110', 'suffix':'-48mm-mrctfs-subgroup2'})
#configs.append({'sg':3,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-subgroup3-1110', 'suffix':'-48mm-mrctfs-subgroup3'})
#configs.append({'sg':4,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-subgroup4-1110', 'suffix':'-48mm-mrctfs-subgroup4'})
#configs.append({'sg':5,'dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-subgroup5-1110', 'suffix':'-48mm-mrctfs-subgroup5'})
#configs.append({'sg':1,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-deep-subgroup1-1110', 'suffix':'-48mm-mrct-LB1-deep-subgroup1'})
#configs.append({'sg':2,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-deep-subgroup2-1110', 'suffix':'-48mm-mrct-LB1-deep-subgroup2'})
#configs.append({'sg':3,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-deep-subgroup3-1110', 'suffix':'-48mm-mrct-LB1-deep-subgroup3'})
#configs.append({'sg':4,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-deep-subgroup4-1110', 'suffix':'-48mm-mrct-LB1-deep-subgroup4'})
#configs.append({'sg':5,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-deep-subgroup5-1110', 'suffix':'-48mm-mrct-LB1-deep-subgroup5'})
#configs.append({'sg':1,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-dr0p00-subgroup1-1110', 'suffix':'-48mm-mrct-LB1-dr0p00-subgroup1'})
#configs.append({'sg':2,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-dr0p00-subgroup2-1110', 'suffix':'-48mm-mrct-LB1-dr0p00-subgroup2'})
#configs.append({'sg':3,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-dr0p00-subgroup3-1110', 'suffix':'-48mm-mrct-LB1-dr0p00-subgroup3'})
#configs.append({'sg':4,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-dr0p00-subgroup4-1110', 'suffix':'-48mm-mrct-LB1-dr0p00-subgroup4'})
#configs.append({'sg':5,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-dr0p00-subgroup5-1110', 'suffix':'-48mm-mrct-LB1-dr0p00-subgroup5'})
#configs.append({'sg':1,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-dr0p75-subgroup1-1110', 'suffix':'-48mm-mrct-LB1-dr0p75-subgroup1'})
#configs.append({'sg':2,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-dr0p75-subgroup2-1110', 'suffix':'-48mm-mrct-LB1-dr0p75-subgroup2'})
#configs.append({'sg':3,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-dr0p75-subgroup3-1110', 'suffix':'-48mm-mrct-LB1-dr0p75-subgroup3'})
#configs.append({'sg':4,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-dr0p75-subgroup4-1110', 'suffix':'-48mm-mrct-LB1-dr0p75-subgroup4'})
#configs.append({'sg':5,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-dr0p75-subgroup5-1110', 'suffix':'-48mm-mrct-LB1-dr0p75-subgroup5'})
#configs.append({'sg':1,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-subgroup1-1110', 'suffix':'-48mm-mrct-LB1-subgroup1'})
#configs.append({'sg':2,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-subgroup2-1110', 'suffix':'-48mm-mrct-LB1-subgroup2'})
#configs.append({'sg':3,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-subgroup3-1110', 'suffix':'-48mm-mrct-LB1-subgroup3'})
#configs.append({'sg':4,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-subgroup4-1110', 'suffix':'-48mm-mrct-LB1-subgroup4'})
#configs.append({'sg':5,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-subgroup5-1110', 'suffix':'-48mm-mrct-LB1-subgroup5'})
#configs.append({'sg':1,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB2-subgroup1-1110', 'suffix':'-48mm-mrct-LB2-subgroup1'})
#configs.append({'sg':2,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB2-subgroup2-1110', 'suffix':'-48mm-mrct-LB2-subgroup2'})
#configs.append({'sg':3,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB2-subgroup3-1110', 'suffix':'-48mm-mrct-LB2-subgroup3'})
#configs.append({'sg':4,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB2-subgroup4-1110', 'suffix':'-48mm-mrct-LB2-subgroup4'})
#configs.append({'sg':5,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB2-subgroup5-1110', 'suffix':'-48mm-mrct-LB2-subgroup5'})
#configs.append({'sg':1,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-subgroup1-1110', 'suffix':'-48mm-mrct-subgroup1'})
#configs.append({'sg':2,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-subgroup2-1110', 'suffix':'-48mm-mrct-subgroup2'})
#configs.append({'sg':3,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-subgroup3-1110', 'suffix':'-48mm-mrct-subgroup3'})
#configs.append({'sg':4,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-subgroup4-1110', 'suffix':'-48mm-mrct-subgroup4'})
#configs.append({'sg':5,'dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-subgroup5-1110', 'suffix':'-48mm-mrct-subgroup5'})
#configs.append({'sg':1,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-subgroup1-1110', 'suffix':'-48mm-mr-subgroup1'})
#configs.append({'sg':2,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-subgroup2-1110', 'suffix':'-48mm-mr-subgroup2'})
#configs.append({'sg':3,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-subgroup3-1110', 'suffix':'-48mm-mr-subgroup3'})
#configs.append({'sg':4,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-subgroup4-1110', 'suffix':'-48mm-mr-subgroup4'})
#configs.append({'sg':5,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-subgroup5-1110', 'suffix':'-48mm-mr-subgroup5'})
#configs.append({'sg':1,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-LB1-subgroup1-1110', 'suffix':'-48mm-mr-LB1-subgroup1'})
#configs.append({'sg':2,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-LB1-subgroup2-1110', 'suffix':'-48mm-mr-LB1-subgroup2'})
#configs.append({'sg':3,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-LB1-subgroup3-1110', 'suffix':'-48mm-mr-LB1-subgroup3'})
#configs.append({'sg':4,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-LB1-subgroup4-1110', 'suffix':'-48mm-mr-LB1-subgroup4'})
#configs.append({'sg':5,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-LB1-subgroup5-1110', 'suffix':'-48mm-mr-LB1-subgroup5'})
#configs.append({'sg':1,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-LB2-subgroup1-1110', 'suffix':'-48mm-mr-LB2-subgroup1'})
#configs.append({'sg':2,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-LB2-subgroup2-1110', 'suffix':'-48mm-mr-LB2-subgroup2'})
#configs.append({'sg':3,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-LB2-subgroup3-1110', 'suffix':'-48mm-mr-LB2-subgroup3'})
#configs.append({'sg':4,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-LB2-subgroup4-1110', 'suffix':'-48mm-mr-LB2-subgroup4'})
#configs.append({'sg':5,'dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-LB2-subgroup5-1110', 'suffix':'-48mm-mr-LB2-subgroup5'})
#configs.append({'sg':1,'dx':40,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-40mm-mrct-LB1-subgroup1-1110', 'suffix':'-40mm-mrct-LB1-subgroup1'})
#configs.append({'sg':2,'dx':40,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-40mm-mrct-LB1-subgroup2-1110', 'suffix':'-40mm-mrct-LB1-subgroup2'})
#configs.append({'sg':3,'dx':40,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-40mm-mrct-LB1-subgroup3-1110', 'suffix':'-40mm-mrct-LB1-subgroup3'})
#configs.append({'sg':4,'dx':40,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-40mm-mrct-LB1-subgroup4-1110', 'suffix':'-40mm-mrct-LB1-subgroup4'})
#configs.append({'sg':5,'dx':40,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-40mm-mrct-LB1-subgroup5-1110', 'suffix':'-40mm-mrct-LB1-subgroup5'})
#configs.append({'sg':1,'dx':56,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-56mm-mrct-LB1-subgroup1-1110', 'suffix':'-56mm-mrct-LB1-subgroup1'})
#configs.append({'sg':2,'dx':56,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-56mm-mrct-LB1-subgroup2-1110', 'suffix':'-56mm-mrct-LB1-subgroup2'})
#configs.append({'sg':3,'dx':56,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-56mm-mrct-LB1-subgroup3-1110', 'suffix':'-56mm-mrct-LB1-subgroup3'})
#configs.append({'sg':4,'dx':56,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-56mm-mrct-LB1-subgroup4-1110', 'suffix':'-56mm-mrct-LB1-subgroup4'})
#configs.append({'sg':5,'dx':56,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-56mm-mrct-LB1-subgroup5-1110', 'suffix':'-56mm-mrct-LB1-subgroup5'})
#configs.append({'sg':1,'dx':64,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-64mm-mrct-LB1-subgroup1-1110', 'suffix':'-64mm-mrct-LB1-subgroup1'})
#configs.append({'sg':2,'dx':64,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-64mm-mrct-LB1-subgroup2-1110', 'suffix':'-64mm-mrct-LB1-subgroup2'})
#configs.append({'sg':3,'dx':64,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-64mm-mrct-LB1-subgroup3-1110', 'suffix':'-64mm-mrct-LB1-subgroup3'})
#configs.append({'sg':4,'dx':64,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-64mm-mrct-LB1-subgroup4-1110', 'suffix':'-64mm-mrct-LB1-subgroup4'})
#configs.append({'sg':5,'dx':64,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-64mm-mrct-LB1-subgroup5-1110', 'suffix':'-64mm-mrct-LB1-subgroup5'})

#configs.append({'sg':'','dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-LB1-final-1410', 'suffix':'-48mm-mrctfs-LB1-final'})
#configs.append({'sg':'','dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-LB2-final-1410', 'suffix':'-48mm-mrctfs-LB2-final'})
#configs.append({'sg':'','dx':48,'fs':1,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrctfs-final-1410', 'suffix':'-48mm-mrctfs-final'})
#configs.append({'sg':'','dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-deep-final-1410', 'suffix':'-48mm-mrct-LB1-deep-final'})
#configs.append({'sg':'','dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-dr0p00-final-1410', 'suffix':'-48mm-mrct-LB1-dr0p00-final'})
#configs.append({'sg':'','dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-dr0p75-final-1410', 'suffix':'-48mm-mrct-LB1-dr0p75-final'})
#configs.append({'sg':'','dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB1-final-1410', 'suffix':'-48mm-mrct-LB1-final'})
#configs.append({'sg':'','dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-LB2-final-1410', 'suffix':'-48mm-mrct-LB2-final'})
#configs.append({'sg':'','dx':48,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mrct-final-1410', 'suffix':'-48mm-mrct-final'})
#configs.append({'sg':'','dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-final-1410', 'suffix':'-48mm-mr-final'})
#configs.append({'sg':'','dx':40,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-40mm-mrct-LB1-final-1410', 'suffix':'-40mm-mrct-LB1-final'})
#configs.append({'sg':'','dx':56,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-56mm-mrct-LB1-final-1410', 'suffix':'-56mm-mrct-LB1-final'})
#configs.append({'sg':'','dx':64,'fs':0,'ct':1,'checkpoint_path':'tmp/ckpt/checkpoint_n511-64mm-mrct-LB1-final-1410', 'suffix':'-64mm-mrct-LB1-final'})
#configs.append({'sg':'','dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-LB1-final-1410', 'suffix':'-48mm-mr-LB1-final'})
#configs.append({'sg':'','dx':48,'fs':0,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr-LB2-final-1410', 'suffix':'-48mm-mr-LB2-final'})

configs.append({'sg':'','dx':48,'fs':0,'mr1':1,'mr2':1,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr1mr2-final-1410'      , 'suffix':'-48mm-mr1mr2-final'      })
configs.append({'sg':'','dx':48,'fs':0,'mr1':1,'mr2':1,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr1mr2-LB1-final-1410'  , 'suffix':'-48mm-mr1mr2-LB1-final'  })
configs.append({'sg':'','dx':48,'fs':0,'mr1':1,'mr2':1,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr1mr2-LB2-final-1410'  , 'suffix':'-48mm-mr1mr2-LB2-final'  })
configs.append({'sg':'','dx':48,'fs':0,'mr1':1,'mr2':1,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr1mr2-LB3p3-final-1410', 'suffix':'-48mm-mr1mr2-LB3p3-final'})
#configs.append({'sg':'','dx':48,'fs':0,'mr1':0,'mr2':1,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr2-final-1410'      , 'suffix':'-48mm-mr2-final'      })
#configs.append({'sg':'','dx':48,'fs':0,'mr1':0,'mr2':1,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr2-LB1-final-1410'  , 'suffix':'-48mm-mr2-LB1-final'  })
#configs.append({'sg':'','dx':48,'fs':0,'mr1':0,'mr2':1,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr2-LB2-final-1410'  , 'suffix':'-48mm-mr2-LB2-final'  })
#configs.append({'sg':'','dx':48,'fs':0,'mr1':0,'mr2':1,'ct':0,'checkpoint_path':'tmp/ckpt/checkpoint_n511-48mm-mr2-LB3p3-final-1410', 'suffix':'-48mm-mr2-LB3p3-final'})

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
  cpu_ram=8
else:
  cpu_cores = 32
  cpu_ram = 16

for config in configs:
  sg, checkpoint_path, suffix = config['sg'], config['checkpoint_path'], config['suffix']
  model_path = checkpoint_path + '.meta'
  useCt = bool(config['ct'])
  useMr1 = bool(config['mr1'])
  useMr2 = bool(config['mr2'])
  useFreesurfer = bool(config['fs'])
  patch_size = config['dx']
  patch_layer = config['dx']
  #data_dir='/data/deasy/DylanHsu/SRS_N514/subgroup%d/testing'%(sg)
  if sg != '':
    data_dir= os.path.join(case_folder, 'subgroup%d'%sg, subfolder)
  else:
    data_dir = os.path.join(case_folder, 'final', subfolder)
  
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
    f.write("#BSUB -W 5:59\n")
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
    image_filenames = []
    if useMr1:
      image_filenames.append("mr1.nii.gz")
    if useMr2:
      image_filenames.append("mr2.nii.gz")
    if useCt:
      image_filenames.append("ct.nii.gz")
    if useFreesurfer:
      image_filenames += ["fsBasalGanglia.nii.gz","fsBrainstem.nii.gz","fsCerebellumCortex.nii.gz","fsCerebellumWM.nii.gz","fsCerebralCortex.nii.gz","fsCerebralWM.nii.gz","fsChoroidPlexus.nii.gz","fsCorpusCallosum.nii.gz","fsCSF.nii.gz","fsHippocampi.nii.gz","fsOpticChiasm.nii.gz","fsThalami.nii.gz","fsVentralDiencephalon.nii.gz","fsVentricles.nii.gz"]
    assert len(image_filenames)>0
    f.write(' --image_filenames "%s"'%(','.join(image_filenames)))
    if not useGpu:
      f.write(' --use_cpu')
    f.write("\n")
    f.close()
    # Submit jobs.
    the_command = "bsub < " + jobfile
    os.system(the_command)
