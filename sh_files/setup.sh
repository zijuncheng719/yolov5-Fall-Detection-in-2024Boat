#!/bin/bash
### BEGIN INIT INFO
# Provides:          xxxx.com
# Required-Start:    $local_fs $network
# Required-Stop:     $local_fs
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: mylaunch service
# Description:       mylaunch service test
### END INIT INFO
# sleep 8
# cd catkin_ws/
# source devel/setup.bash
# cd ~/catkin_ws/src/CRAIC-vision-tasks/src/pytorch-cifar100
# while true
# do
# python CRAIC-vision-tasks.py
# done #新建终端启动节点

#sleep 10
#cd ~/catkin_ws/src/CRAIC-vision-tasks/src/pytorch-cifar100;OPENBLAS_CORETYPE=ARMV8 python CRAIC-vision-tasks.py
#gnome-terminal -- bash -c "cd catkin_ws/;source devel/setup.bash;rosrun realsense2_camera test.py"

#2024Boat
sleep 2
cd /home/cheng/Desktop/yolov5-Fall-Detection-in-2024Boat	#必须得进入到这个目录下面
/usr/bin/python3 /home/cheng/Desktop/yolov5-Fall-Detection-in-2024Boat/demo.py
sleep 10
done

