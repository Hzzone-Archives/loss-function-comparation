cd /home/hzzone/loss-function-comparation/softmax_loss/AlexNet\ Fine-tuning &&
sudo ~/caffe/build/tools/caffe train --solver solver.prototxt --weights ../../pretrained-models/bvlc_alexnet.caffemodel 2>&1 | tee train.log &&
cd /home/hzzone/loss-function-comparation/softmax_loss/CaffeNet\ Fine-tuning &&
sudo ~/caffe/build/tools/caffe train --solver solver.prototxt --weights ../../pretrained-models/bvlc_reference_caffenet.caffemodel 2>&1 | tee train.log &&
cd /home/hzzone/loss-function-comparation/softmax_loss/CaffeNet\ Off-the-shelf &&
sudo ~/caffe/build/tools/caffe train --solver solver.prototxt 2>&1 | tee train.log 
