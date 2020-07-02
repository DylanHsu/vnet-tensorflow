import os,sys
from glob import glob
import numpy as np
# Setup job configs

jobfolder = "./jobs/"
cpu_cores = 8
cpu_ram = 8
care_about_segmentation = True
start = 0.1
stop = 1.0
step = 0.05
seg_step = 0.05
dilate_threshold = 0.00
data_dir = '/data/deasy/DylanHsu/SRS_N514'
#suffix='48mm-minSmall5x-depth33444'
suffixes = []
suffixes += ['32mm-mrct-LB1-subgroup%d']
#suffixes += ['48mm-mrctfs-LB1-subgroup%d']
#suffixes += ['48mm-mrctfs-subgroup%d']
#suffixes += ['48mm-mrct-LB1-deep-subgroup%d']
#suffixes += ['48mm-mrct-LB1-dr0p00-subgroup%d']
#suffixes += ['48mm-mrct-LB1-dr0p75-subgroup%d']
#suffixes += ['48mm-mrct-LB1-subgroup%d']
#suffixes += ['48mm-mrct-LB2-subgroup%d']
#suffixes += ['48mm-mrct-subgroup%d']
#suffixes += ['48mm-mr-subgroup%d']
#suffixes += ['64mm-mrct-LB1-subgroup%d']

for suffix in suffixes:
  for subgroup in [1,2,3,4,5]:
    sg_suffix = suffix % (subgroup)
    sg_dir = os.path.join(data_dir,'subgroup%d'%(subgroup))
    
    for threshold in np.arange(start,stop,step):
      if care_about_segmentation:
        seg_thresholds = np.arange(seg_step,threshold+seg_step*0.99,seg_step) #0.99 is there cause otherwise weird float behavior includes the endpoint
      else:
        seg_thresholds = [threshold]
      for seg_threshold in seg_thresholds:
        jobname = 'scanThreshold-seed%.3f-seg%.3f-%s'%(threshold,seg_threshold,sg_suffix)
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
          f.write('python scanThreshold.py --dilate_threshold %.3f %s %s %.3f %.3f\n' % (dilate_threshold, sg_suffix, sg_dir, threshold, seg_threshold))
        else:
          f.write('python scanThreshold.py %s %s %.3f %.3f\n' % (sg_suffix, sg_dir, threshold, seg_threshold))
        f.close()
        # Submit jobs.
        the_command = "bsub < " + jobfile
        os.system(the_command)
