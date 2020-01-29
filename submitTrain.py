import os
from glob import glob

# Setup job configs
prefix = 'n200-scanSizeV1mm3-bs1000'
jobfolder = "./jobs/"
cpu_cores = 8
cpu_ram = 16

# Hyperparameters
#learning_rates = [0.01, 0.1]
#box_sizes = [16, 24, 32, 48, 64]
learning_rates = [0.1]
#box_sizes = [ [64,32], [80,40], [96,48], [112,56], [128,64] ]
box_sizes = [ [32,32], [40,40], [48,48], [56,56], [64,64] ]
#drop_ratios = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
drop_ratios = [0.5]

#ccrop_sigmas = [0.1, 0.2, 0.5, 0.8, 1.0]
#wce_weights = [10,100,1000]

num_crops = [10]
####################################
configs = []
#for ccs in ccrop_sigmas:
for lr in learning_rates:
 for dr in drop_ratios:
  for s in box_sizes:
  #for ccs in ccrop_sigmas:
  #for w in wce_weights:
    #for nc in num_crops:
      #jobname = "%s-lr%.3f-s%d-dr%.2f-nc%d" % (prefix,lr,s,dr,nc)
      #jobname = "%s-dr%.2f" % (prefix,dr)
      jobname = "%s-dr%.2f-%dmm" % (prefix,dr,s[1])
      #jobname = "strip4-lr%.3f-s%d-ccs%.2f-nc%d" % (lr,s,ccs,nc)
      #jobname = "overfit2-lr%.3f-s%d-ccs%.2f-nc%d" % (lr,s,ccs,nc)
      #jobname = "overfit-lr%.3f-s%d-nc%d" % (lr,s,nc)
      jobname = jobname.replace(".","p")
      config = {}
      config['lr'] = lr
      config['dr'] = dr
      #config['nc'] = nc
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
  f.write("#BSUB -W 168:00\n")
  f.write("#BSUB -o "+jobfolder+"%J.stdout\n")
  f.write("#BSUB -eo "+jobfolder+"%J.stderr\n")
  f.write("\n")
  f.write("source /home/hsud3/.bash_profile\n")
  f.write("cd /home/hsud3/vnet-tensorflow \n")
  f.write("python train.py")
  f.write("  --is_batch_job")
  f.write("  --batch_job_name %s" % config['jobname'])
  f.write('  --vnet_convs "3,3,4,4,4"')
  f.write("  --num_channels 1")
  f.write("  --epochs 500")
  f.write("  --batch_size 1")
  f.write("  --accum_batches 1000")
  #f.write("  --num_crops %d" % config['nc'])
  f.write("  --num_crops 1")
  f.write("  --image_filename img.nii.gz")
  f.write("  --label_filename label_smoothed.nii.gz")
  f.write("  --init_learning_rate %f" % config['lr'])
  f.write("  --patch_size %d" % config['dx'])
  f.write("  --patch_layer %d" % config['dz'])
  f.write("  --min_pixel 1")
  f.write("  --loss_function specific_dice")
  #f.write("  --loss_function wce")
  f.write("  --optimizer adam")
  f.write("  --data_dir /data/deasy/DylanHsu/N200_1mm3/augcache")
  #f.write("  --wce_weight %d" % config['w'])
  f.write("  --drop_ratio %f\n" % config['dr'])
  #f.write("  --ccrop_sigma %.2f" % config['ccs'])
  #f.write("  --loss_function sorensen")
  f.close()
# Submit jobs.
for config in configs:
  the_command = "bsub < " + config['jobfile']
  os.system(the_command)
