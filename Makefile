CXX = g++

# Require pkg-config to be installed
# Revert it back in case of errors
CFLAGS = -std=c++11 -fopenmp -O2 -Wno-long-long -Wno-unused-parameter \
	 -Wno-unknown-pragmas `pkg-config --cflags opencv`

#LIBS = /usr/local/lib/libopencv_calib3d.so \
/usr/local/lib/libopencv_nonfree.so \
/usr/local/lib/libopencv_contrib.so \
/usr/local/lib/libopencv_objdetect.so \
/usr/local/lib/libopencv_core.so \
/usr/local/lib/libopencv_ocl.so \
/usr/local/lib/libopencv_features2d.so \
/usr/local/lib/libopencv_photo.so \
/usr/local/lib/libopencv_flann.so \
/usr/local/lib/libopencv_stitching.so \
/usr/local/lib/libopencv_gpu.so \
/usr/local/lib/libopencv_superres.so \
/usr/local/lib/libopencv_highgui.so \
/usr/local/lib/libopencv_video.so \
/usr/local/lib/libopencv_imgproc.so \
/usr/local/lib/libopencv_videostab.so \
/usr/local/lib/libopencv_legacy.so
LIBS = `pkg-config --libs opencv`

Stasm_h = stasm/asm.h \
stasm/atface.h \
stasm/basedesc.h \
stasm/classicdesc.h \
stasm/convshape.h \
stasm/err.h \
stasm/eyedet.h \
stasm/eyedist.h \
stasm/faceroi.h \
stasm/hat.h \
stasm/hatdesc.h \
stasm/landmarks.h \
stasm/landtab_muct77.h \
stasm/misc.h \
stasm/pinstart.h \
stasm/print.h \
stasm/shape17.h \
stasm/shapehacks.h \
stasm/shapemod.h \
stasm/startshape.h \
stasm/stasm.h \
stasm/stasm_landmarks.h \
stasm/stasm_lib.h \
stasm/stasm_lib_ext.h \
stasm/MOD_1/facedet.h \
stasm/MOD_1/initasm.h

Stasm_src = stasm/asm.cpp \
stasm/classicdesc.cpp \
stasm/convshape.cpp \
stasm/err.cpp \
stasm/eyedet.cpp \
stasm/eyedist.cpp \
stasm/faceroi.cpp \
stasm/hat.cpp \
stasm/hatdesc.cpp \
stasm/landmarks.cpp \
stasm/misc.cpp \
stasm/pinstart.cpp \
stasm/print.cpp \
stasm/shape17.cpp \
stasm/shapehacks.cpp \
stasm/shapemod.cpp \
stasm/startshape.cpp \
stasm/stasm.cpp \
stasm/stasm_lib.cpp \
stasm/MOD_1/facedet.cpp \
stasm/MOD_1/initasm.cpp

Stasm_o = asm.o \
classicdesc.o \
convshape.o \
err.o \
eyedet.o \
eyedist.o \
faceroi.o \
hat.o \
hatdesc.o \
landmarks.o \
misc.o \
pinstart.o \
print.o \
shape17.o \
shapehacks.o \
shapemod.o \
startshape.o \
stasm.o \
stasm_lib.o \
facedet.o \
initasm.o
all: fduvideo_demo

face_predict.o: src/face_predict.cpp src/face_predict.hpp
	$(CXX) $(CFLAGS) -c src/face_predict.cpp $(LIBS)
svm.o: src/svm.cpp src/svm.h
	$(CXX) $(CFLAGS) -c src/svm.cpp
svm-predict.o: src/svm-predict.cpp src/svm.h svm.o
	$(CXX) $(CFLAGS) -c src/svm-predict.cpp svm.o $(LIBS) -lm
person_recognize.o: src/svm.cpp src/svm.h src/svm-predict.cpp src/person_recognize.cpp src/person_recognize.hpp
	$(CXX) $(CFLAGS) -c src/person_recognize.cpp $(LIBS)
asm.o: $(Stasm_src) $(Stasm_h)
	$(CXX) $(CFLAGS) -c $(Stasm_src) $(LIBS)
fduvideo_demo: src/main.cpp src/system_struct.hpp src/FduVideo.hpp svm.o face_predict.o person_recognize.o svm-predict.o $(Stasm_o)
	$(CXX) $(CFLAGS) src/main.cpp svm.o face_predict.o person_recognize.o \
		svm-predict.o $(Stasm_o) -o fduvideo_demo $(LIBS)

clean: 
	-rm -f *.o fduvideo_demo
