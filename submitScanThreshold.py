import os,sys
from glob import glob
import numpy as np
# Setup job configs

jobfolder = "./jobs/"
cpu_cores = 4
cpu_ram = 8
#data_dir = '/data/deasy/DylanHsu/N401_unstripped/subgroup1'
#suffix='scanDrCachedAug-d0p50-dense'
#suffix='scanDrCachedAug-dr0p90-dense'
#suffix='scanSizeV1mm3-bs1000-dr0p50-32mm'
#suffix='scanSizeV1mm3-bs1000-dr0p50-64mm'
#suffix='scanSizeV1mm3-bs1000-dr0p50-56mm'
#suffix='scanSizeV1mm3-bs1000-dr0p50-48mm'
#suffix='scanSizeV1mm3-bs1000-dr0p50-40mm'
#suffix='48mm-noWeight-depth33444'
#suffix='48mm-small5x-depth33444'
#suffix='48mm-small10x-depth33444'
#suffix='48mm-noWeight-depth54444'
#suffix='48mm-noWeight-depth33444-inflate1mm'
#suffix='48mm-minSmall3x-depth33444'
#suffix='48mm-minSmall5x-depth33444'

for subgroup in [1,2,3,4,5]:
  #data_dir = '/data/deasy/DylanHsu/N401_unstripped/subgroup%d'%subgroup
  data_dir = '/data/deasy/DylanHsu/SRS_N401/subgroup%d'%subgroup
  #suffix = '48mm-noSmallBias-subgroup%d'%subgroup
  #suffix = '48mm-minSmall5x-subgroup%d-stride48'%subgroup
  #suffix = '48mm-minSmall5x-mrct-subgroup%d'%subgroup
  #suffix = 'testCenternet-subgroup%d'%subgroup
  #suffix='48mm-noWeight-depth33444-dilate0p2'
  #dilate_threshold = 0.2
  #suffix='48mm-noWeight-depth33444-dilate0p1'
  #dilate_threshold = 0.1
  #suffix='48mm-noWeight-depth33444-dilate0p05'
  #dilate_threshold = 0.05
  
  care_about_segmentation = True
  start = 0.2
  stop = 1.0
  step = 0.05
  seg_step = 0.05
  dilate_threshold = 0.00
  #suffix = 'mrct-subgroup%d'%(subgroup)
  suffix = 'mrct-alpha0p8-subgroup%d'%(subgroup)
  
  # centernet scans
  #suffix = '7mm-mrctACSAverage-subgroup%d'%subgroup
  #suffix = '7mm-mrctACSMinimal-subgroup%d'%subgroup
  #dilate_threshold = 0.00
  #start = 0.05
  #stop = 0.5
  #step = 0.025
  #seg_step = 0.025
  #care_about_segmentation = False

  for threshold in np.arange(start,stop,step):
    if care_about_segmentation:
      seg_thresholds = np.arange(seg_step,threshold+seg_step,seg_step)
    else:
      seg_thresholds = [threshold]
    for seg_threshold in seg_thresholds:
      jobname = 'scanThreshold-seed%.3f-seg%.3f-%s'%(threshold,seg_threshold,suffix)
      jobname = jobname.replace('.','p')
      jobfile = os.path.join(jobfolder,jobname+".lsf")
      # Setup job files
      f = open(jobfile,"w+")
      f.write("#!/bin/bash\n")
      f.write("#BSUB -J "+jobname+"\n")
      f.write("#BSUB -n %d\n" % cpu_cores)
      f.write("#BSUB -q cpuqueue\n")
      f.write("#BSUB -R span[hosts=1]\n")
      f.write("#BSUB -R rusage[mem=%d]\n" % (cpu_ram//cpu_cores))
      f.write("#BSUB -W 12:00\n")
      f.write("#BSUB -o " +jobfolder+"/logs/"+jobname+"_%J.stdout\n")
      f.write("#BSUB -eo "+jobfolder+"/logs/"+jobname+"_%J.stderr\n")
      f.write("\n")
      f.write("source /home/hsud3/.bash_profile\n")
      f.write("cd /home/hsud3/vnet-tensorflow \n")
      if dilate_threshold > 0.0:
        f.write('python scanThreshold.py --dilate_threshold %.3f %s %s %.3f %.3f\n' % (dilate_threshold, suffix, data_dir, threshold, seg_threshold))
      else:
        f.write('python scanThreshold.py %s %s %.3f %.3f\n' % (suffix, data_dir, threshold, seg_threshold))
      f.close()
      # Submit jobs.
      the_command = "bsub < " + jobfile
      os.system(the_command)
