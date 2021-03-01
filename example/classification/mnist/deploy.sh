#!/bin/bash

export PYTHONUNBUFFERED="True"

python ../../../model/classification/mnist/pipe.py \
      --cfg cfg.yml \
      --mode deploy | tee results/deploy.log
