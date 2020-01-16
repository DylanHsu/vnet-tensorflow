import os
from glob import glob

# Setup job configs
prefix = 'N120bigfield'
jobfolder = "./jobs/"
cpu_cores = 16
cpu_ram = 32

# Hyperparameters
#learning_rates = [0.01, 0.1]
#box_sizes = [16, 24, 32, 48, 64]
learning_rates = [0.10]
box_sizes = [ [64,32] ]
drop_ratios = [0.5]
#ccrop_sigmas = [0.1, 0.2, 0.5, 0.8, 1.0]
wce_weights = [10,100,1000]

num_crops = [10]
####################################
configs = []
#for ccs in ccrop_sigmas:
for lr in learning_rates:
 for s in box_sizes:
  for dr in drop_ratios:
  #for ccs in ccrop_sigmas:
  #for w in wce_weights:
    for nc in num_crops:
      jobname = "%s-lr%.3f-s%d-dr%.2f-nc%d" % (prefix,lr,s,dr,nc)
      #jobname = "strip4-lr%.3f-s%d-ccs%.2f-nc%d" % (lr,s,ccs,nc)
      #jobname = "overfit2-lr%.3f-s%d-ccs%.2f-nc%d" % (lr,s,ccs,nc)
      #jobname = "overfit-lr%.3f-s%d-nc%d" % (lr,s,nc)
      jobname = jobname.replace(".","p")
      config = {}
      config['lr'] = lr
      config['dr'] = dr
      config['nc'] = nc
      #config['ccs'] = ccs
      config['dx'] = s[0]
      config['dz'] = s[1]
      #config['w'] = w
      config['jobname'] = jobname
      config['jobfile'] = jobfolder+jobname+".lsf"
      configs += [config]

# Clear job folder
#old_jobs = glob(jobfolder + "*")
#for oj in old_jobs:
#  os.remove(oj)

# Setup job files
for config in configs:
  jobname = config['jobname']
  f = open(config['jobfile'],"w+")
  f.write("#!/bin/bash\n")
  f.write("#BSUB -J "+jobname+"\n")
  f.write("#BSUB -n %d\n" % cpu_cores)
  f.write("#BSUB -q gpuqueue\n")
  f.write("#BSUB -gpu num=1\n")
  f.write("#BSUB -R span[hosts=1]\n")
  f.write("#BSUB -R rusage[mem=%d]\n" % (cpu_ram//cpu_cores))
  f.write("#BSUB -W 72:00\n")
  f.write("#BSUB -o "+jobfolder+"%J.stdout\n")
  f.write("#BSUB -eo "+jobfolder+"%J.stderr\n")
  f.write("\n")
  f.write("source /home/hsud3/.bash_profile\n")
  f.write("cd /home/hsud3/vnet-tensorflow \n")
  f.write("python train.py")
  f.write("  --is_batch_job")
  f.write("  --batch_job_name %s" % config['jobname'])
  f.write("  --num_channels 1")
  f.write("  --epochs 1000")
  f.write("  --batch_size 1")
  f.write("  --accum_batches 20")
  f.write("  --num_crops %d" % config['nc'])
  f.write("  --image_filename img.nii.gz")
  f.write("  --label_filename label.nii.gz")
  f.write("  --init_learning_rate %f" % config['lr'])
  f.write("  --patch_size %d" % config['dx'])
  f.write("  --patch_layer %d" % config['dz'])
  f.write("  --min_pixel 10")
  f.write("  --loss_function dice")
  f.write("  --optimizer adam")
  #f.write("  --wce_weight %d" % config['w'])
  f.write("  --drop_ratio %f\n" % config['dr'])
  #f.write("  --ccrop_sigma %.2f" % config['ccs'])
  #f.write("  --loss_function sorensen")
  f.close()
# Submit jobs.
for config in configs:
  the_command = "bsub < " + config['jobfile']
  os.system(the_command)
