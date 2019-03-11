#!/usr/bin/env sh

TOOLS=/home/cunjian/code/caffe/build/tools

#$TOOLS/caffe train --solver=vanilla_adam_solver.prototxt>>log.txt 2>&1


#$TOOLS/caffe train -solver /home/cunjian/code/caffe/examples/caffe_landmark-master/solver_300w.prototxt -weights model_celebrity/_iter_200000.caffemodel 2>&1 | tee log.txt

$TOOLS/caffe train \
--solver=vanilla_adam_solver.prototxt>>log2.txt 2>&1 --snapshot=model/_iter_1348470.solverstate
