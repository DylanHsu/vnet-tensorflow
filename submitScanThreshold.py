import os,sys
from glob import glob
import numpy as np
# Setup job configs

jobfolder = "./jobs/"
cpu_cores = 8 
cpu_ram = 8
data_dir = '/data/deasy/DylanHsu/N200_1mm3'
#suffix='-scanDrCachedAug-d0p50-dense'
#suffix='-scanDrCachedAug-dr0p90-dense'
#suffix='-scanSizeV1mm3-bs1000-dr0p50-64mm'
suffix='-scanSizeV1mm3-bs1000-dr0p50-56mm'
#suffix='-scanSizeV1mm3-bs1000-dr0p50-48mm'
#suffix='-scanSizeV1mm3-bs1000-dr0p50-40mm'
start = 0.3
stop = 1.0
#step = 0.1
step = 0.05

for threshold in np.arange(start,stop,step):
  for seg_threshold in np.arange(step,threshold+step,step):
    jobname = 'scanThreshold-seed%.3f-seg%.3f%s'%(threshold,seg_threshold,suffix)
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
    f.write("#BSUB -W 24:00\n")
    f.write("#BSUB -o "+jobfolder+"%J.stdout\n")
    f.write("#BSUB -eo "+jobfolder+"%J.stderr\n")
    f.write("\n")
    f.write("source /home/hsud3/.bash_profile\n")
    f.write("cd /home/hsud3/vnet-tensorflow \n")
    f.write('python scanThreshold.py %s %s %.3f %.3f\n' % (suffix, data_dir, threshold, seg_threshold))
    f.close()
    # Submit jobs.
    the_command = "bsub < " + jobfile
    os.system(the_command)
