import cv2
import subprocess
import time

'''拉流url地址，指定 从哪拉流'''
# video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 自己摄像头
pull_url = 'rtsp://192.168.107.189/stream1' # "rtsp_address"
video_capture = cv2.VideoCapture(pull_url) # 调用摄像头的rtsp协议流
# pull_url = "rtmp_address"


'''推流url地址，指定 用opencv把各种处理后的流(视频帧) 推到 哪里'''
video_capture = cv2.VideoCapture(pull_url) # 调用摄像头的rtsp协议流
push_url = "rtsp://192.168.107.65:8554/room55"

width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS)) # Error setting option framerate to value 0. 
print("width", width, "height", height,  "fps：", fps) 


# command = [r'D:\Softwares\ffmpeg-5.1-full_build\bin\ffmpeg.exe', # windows要指定ffmpeg地址
command = ['ffmpeg', # linux不用指定
    '-y', '-an',
    '-f', 'rawvideo',
    '-vcodec','rawvideo',
    '-pix_fmt', 'bgr24', #像素格式
    '-s', "{}x{}".format(width, height),
    '-r', str(fps), # 自己的摄像头的fps是0，若用自己的notebook摄像头，设置为15、20、25都可。 
    '-i', '-',
    '-c:v', 'libx264',  # 视频编码方式
    '-pix_fmt', 'yuv420p',
    '-preset', 'ultrafast',
    '-f', 'rtsp', #  flv rtsp
    # '-rtsp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
    push_url] # rtsp rtmp  
pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

def frame_handler(frame):
    ...
    return frame


while True: # True or video_capture.isOpened():
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # handle the video capture frame
    start = time.time()
    frame = frame_handler(frame) 
    pipe.stdin.write(frame.tostring())
    # Display the resulting image. linux 需要注释该行代码
    # cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(delay=100) & 0xFF == ord('q'): #  delay=100ms为0.1s .若dealy时间太长，比如1000ms，则无法成功推流！
        break
    
    pipe.stdin.write(frame.tostring())
    # pipe.stdin.write(frame.tobytes())
    
video_capture.release()
cv2.destroyAllWindows()
pipe.terminate()


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/31 19:18
# @Author  : LXL
# @File    : tuiliu.py
import cv2
# subprocess 模块允许我们启动一个新进程，并连接到它们的输入/输出/错误管道，从而获取返回值。
import subprocess
# 视频读取对象
cap = cv2.VideoCapture(0)
# 读取一帧
ret, frame = cap.read()
# 推流地址
rtmp = "填写你自己的服务器地址"
# 推流参数
command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', '640*480',  # 根据输入视频尺寸填写
           '-r', '15',
           '-i', '-',
           '-c:v', 'h264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           rtmp]

# 创建、管理子进程
pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 循环读取
while cap.isOpened():
    # 读取一帧
    ret, frame = cap.read()
    if frame is None:
        print('read frame err!')
        continue

    # 显示一帧
    cv2.imshow("frame", frame)
    # 读取尺寸、推流
    img = cv2.resize(frame, size)
    pipe.stdin.write(img.tobytes())
    cv2.waitKey(30)

# 关闭窗口
cv2.destroyAllWindows()

# 停止读取
cap.release()


    # command = ['ffmpeg',
    #             '-y',
    #             '-f', 'rawvideo',
    #             '-vcodec', 'rawvideo',
    #            '-pix_fmt', 'bgr24',
    #            '-s', '640*480',  # 根据输入视频尺寸填写
    #            '-r', '25',
    #            '-i', '-',
    #            '-c:v', 'h264',
    #            '-pix_fmt', 'yuv420p',
    #            '-preset', 'ultrafast',
    #            '-f', 'flv',
    #            rtmp]