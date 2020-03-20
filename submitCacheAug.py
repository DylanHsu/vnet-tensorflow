import os
from glob import glob

# Setup job configs
prefix = 'cacheAug'
jobfolder = "./jobs/"
cpu_cores = 4 
cpu_ram = 4 
case_dir = "/data/deasy/DylanHsu/N401_unstripped/nifti/"
# Clear job folder
#old_jobs = glob(jobfolder + "*")
#for oj in old_jobs:
#  os.remove(oj)

configs = []

cases = os.listdir(case_dir)
for case in cases:
  config={}
  jobname = '%s-%s'%(prefix,case)
  config['case'] = case
  config['jobname'] = jobname 
  config['jobfile'] = jobfolder+jobname+".lsf"
  configs.append(config)
# Setup job files
for config in configs:
  jobname = config['jobname']
  f = open(config['jobfile'],"w+")
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
  f.write("python cacheAugmentation.py %s \n"%(config['case']))
  f.close()
# Submit jobs.
for config in configs:
  the_command = "bsub < " + config['jobfile']
  os.system(the_command)
