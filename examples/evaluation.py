# coding: utf-8
import os
import re
import subprocess


GPUS = "1" #should be set
root = '/home/amin/Amin-SSD/caffe/'#should be set
os.chdir(root)
MODEL_DIR = 'examples/okutama-action-test/960x540/' #should be set
SNAP_DIR = 'models/VGGNet/okutama/action/SSD_960x540/' #should be set


SOLVER_FILE = MODEL_DIR+'solver.prototxt'


snapshots = os.listdir(SNAP_DIR)
snapshots = filter(lambda x: '.caffemodel' in x, snapshots) # only the .caffemodel files
snapshots.sort(key = lambda x: int(re.sub('\D','', x))) # sort by iteration

#snapshots = snapshots[1::2]
print snapshots
#snapshots = snapshots[4:]

for i, s in enumerate(snapshots):
    print(s)
    weights = os.path.join(SNAP_DIR, s)
    log_file = MODEL_DIR+ s.split('.')[0] + ".log"
    cmd = 'build/tools/caffe train --solver="{}" --weights="{}"'.format(SOLVER_FILE, weights)
    if len(GPUS)>0:
        cmd += ' --gpu {} 2>&1 | tee {}'.format(GPUS, log_file)
    else:
        cmd += ' 2>&1 | tee {}'.format(log_file)

    # Run the job.
    subprocess.call(cmd, shell=True)

