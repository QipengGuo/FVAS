Fudan Video Analysis System (FVAS)
==================================

School of Computer Science, Fudan University  

FudanVA is a C++ open-source software for detecting,recognizing and tracking certain people in the video.   
You give it a face/person dataset and it can find the people in the video and track him/her.  
  
FudanVA is designed to work on monitors' video.  

Source code is provided under a BSD style license. OpenCV with ffmpeg/gstreamer plugins and c++11 are required.  

Code, demo, and datasets are available at our website  
	http://omap.fudan.edu.cn/FVAS

Quick Start
===========

copy data/ into this directory  
$make   
$./fudanvideo_demo  
Or open our project in QT

Main Method
===========

Tracking:  
	Optical flow from Opencv  
Face Detection:  
	HOG+SVM from Opencv  
Body/Person Detection:  
	HOG+SVM from Opencv  
Face Recognition:  
	ASM+LBP from Stasm and Opencv  
Person Recognition/Re-Identification:  
	Mid-level filters coding by our group  

Output
======

Green windows represent body detection  
Red windows represent face detection, and number shows the label by face recognition  
White windows represent tracking, and number shows the label by person recognition  

Reference
=========

 * S. Milborrow and F. Nicolls
 * Active Shape Models with SIFT Descriptors and MARS
 * VISAPP,2014
 * www.milbo.users.sonic.net/stasm/

 * P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan.
 * Object Detection with Discriminatively Trained Part Based Models.
 * IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 32, No. 9, September 2010
 * Discriminatively Trained Deformable Part Models, Release 4
 * http://people.cs.uchicago.edu/~pff/latent-release4/


 * Rui Zhao Wanli Quyang Xiaogang Wang
 * Learning Mid-level Filters for Person-identification
 
 * Zhang W, Wang X, Zhao D, et al.
 * Graph degree linkage: Agglomerative clustering on a directed graph
 
 * LIBSVM:
 * Chih-Chung Chang and Chih-Jen Lin, LIBSVM :
 * a library for support vector machines.
 * ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011.
 * Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm

About Version 0.10 alpha
========================

Our project has just begun.There are countless works wait us to do.  
I am sorry to remain these bugs and consider less about human design.  
We hope open-source can bring us advices and companions. We have a long-time plan around video analysis.  
Finally, thanks for using.  

About Version 0.10 beta
=======================

We plan to release beta version this year.  
New functions:  
 * train model for face and person recognition
 * a simple UI
 * new demo video
 * may be windows version
 * friendly API for opencv

TO DO
=====

 * Body detection based on DPM

 * Real-time Face Detection based on CNN

 * Speed up linear SVM by optimizing matrix computation

 * Tracking based on TLD

Note
====
 * Contributors: Qipeng Guo, Jiajun Ou, Yanye Li, Zhedong Zheng, Guangzhen Zhou, Fengdong Zhang, Dequan Wang
 * Thanks for helping from BarclayII, David Gao (Github account)
 * School of Computer Science , Fudan university
 * Time: 2014
You can get more information from our website http://omap.fudan.edu.cn/FVAS

Schedule
========
http://10.141.208.19/schedule.html
