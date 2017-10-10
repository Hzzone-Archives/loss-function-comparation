cd /home/hzzone/loss-function-comparation/2-Channels\ Network/AlexNet\ Off-the-shelf &&
sudo ~/caffe/build/tools/caffe train --solver solver.prototxt 2>&1 | tee train.log &&
cd /home/hzzone/loss-function-comparation/2-Channels\ Network/CaffeNet\ Off-the-shelf &&
sudo ~/caffe/build/tools/caffe train --solver solver.prototxt 2>&1 | tee train.log 
