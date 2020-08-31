import os,sys
from glob import glob
import numpy as np
# Setup job configs

jobfolder = "./jobs/"
cpu_cores = 16
cpu_ram = 16

roc_mode=False

if roc_mode:
  start = 0
  stop = 1.0
  step = 0.001
  seg_step = 1
else:
  start = 0.1
  stop = 1.0
  step = 0.01 #0.05
  seg_step = 0.05
dilate_threshold = 0.00
data_dir = '/data/deasy/DylanHsu/SRS_N511'

suffixes = []
#suffixes += ['48mm-mr-%s']
#suffixes += ['48mm-mr-LB1-%s']
#suffixes += ['48mm-mr-LB2-%s']
suffixes += ['48mm-mrct-%s']
suffixes += ['48mm-mrctfs-%s']
suffixes += ['48mm-mrctfs-LB1-%s']
#suffixes += ['48mm-mrctfs-LB2-%s']
suffixes += ['48mm-mrct-LB1-%s']
suffixes += ['48mm-mrct-LB1-deep-%s']
suffixes += ['48mm-mrct-LB1-dr0p00-%s']
suffixes += ['48mm-mrct-LB1-dr0p75-%s']
suffixes += ['40mm-mrct-LB1-%s']
suffixes += ['48mm-mrct-LB2-%s']
suffixes += ['56mm-mrct-LB1-%s']
suffixes += ['64mm-mrct-LB1-%s']

#groups = ['subgroup1','subgroup2','subgroup3','subgroup4','subgroup5']
groups = ['subgroup1','subgroup2','subgroup3','subgroup4','subgroup5','final']
#groups = ['final']
if roc_mode:
  # Linear sampling between P=0.01 and P=0.99
  # Logarithmic sampling outside that window
  thresholds = np.concatenate((np.power( np.power(10,1/3.), np.arange(-36,-6,1)),  np.arange( 0.01, 0.99, 0.01 ), 1 - np.power( np.power(10,1/3.), np.arange(-7,-37,-1))))
else:
  thresholds = np.arange(start,stop,step)


jobfile = os.path.join(jobfolder,'scanThreshold-pack%03d.lsf')
jobpack=1
f = open(jobfile % jobpack,"w+")
packed=0
packsize=1000

for suffix in suffixes:
  for subgroup in groups:
    sg_suffix = suffix % subgroup
    os.makedirs(os.path.join(jobfolder,'logs',sg_suffix),exist_ok=True)
    sg_dir = os.path.join(data_dir,subgroup)
    if roc_mode:
      stats_dir = os.path.join(data_dir,'stats','roc-'+sg_suffix)
    else:
      stats_dir = os.path.join(data_dir,'stats','stats-'+sg_suffix)
    
    counter=0
    for threshold in thresholds:
      counter+=1
      if roc_mode:
        seg_thresholds = [threshold]
      else:
        seg_thresholds = [threshold]
        #seg_thresholds = np.arange(seg_step,threshold+seg_step*0.99,seg_step) #0.99 is there cause otherwise weird float behavior includes the endpoint
      for seg_threshold in seg_thresholds:
        if roc_mode:
          jobname = 'scanRocCurve-%03d-%s'%(counter,sg_suffix)
        else:
          jobname = 'scanThreshold-seed%.3f-seg%.3f-%s'%(threshold,seg_threshold,sg_suffix)
          jobname = jobname.replace('.','p')
        
        f.write(" -J "+jobname)
        f.write(" -n %d" % cpu_cores)
        f.write(" -q cpuqueue")
        f.write(" -R span[hosts=1]")
        f.write(" -R rusage[mem=%d]" % (cpu_ram//cpu_cores))
        f.write(" -W 12:00")
        f.write(" -o " +os.path.join(jobfolder,'logs',sg_suffix,jobname+"_%J.stdout"))
        f.write(" -eo "+os.path.join(jobfolder,'logs',sg_suffix,jobname+"_%J.stderr"))
        f.write(" source /home/hsud3/.bash_profile;")
        f.write(" cd /home/hsud3/vnet-tensorflow;")
        flags = ""
        if dilate_threshold > 0.0:
          flags += " --dilate_threshold %.3f"%dilate_threshold
        if roc_mode:
          flags += " --no_lesions"
        f.write(" python scanThreshold.py %s %s %s %s %.12f %.12f\n" % (flags, sg_suffix, sg_dir, stats_dir, threshold, seg_threshold))
        
        packed += 1
        if packed >= packsize:
          f.close()
          print("Submitting job pack: "+(jobfile % jobpack))
          os.system("bsub -pack " + (jobfile % jobpack) + " 2>&1 | grep -v \"is submitted to queue\"")
          jobpack += 1
          f = open(jobfile % jobpack,"w+")
          packed=0

f.close()
if packed > 0:
  os.system("bsub -pack " + (jobfile % jobpack))

