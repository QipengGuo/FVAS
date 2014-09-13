#-------------------------------------------------
#
# Project created by QtCreator 2014-09-01T14:24:09
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = fduvideo0.1.0alhpa
CONFIG   += console
CONFIG   -= app_bundle
CONFIG += c++11
QMAKE_CXXFLAGS+=-fopenmp
QMAKE_LFLAGS+=-fopenmp
TEMPLATE = app


SOURCES += \
    src/main.cpp \
    src/person_recognize.cpp \
    src/svm.cpp \
    src/svm-predict.cpp \
    stasm/asm.cpp \
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
    stasm/MOD_1/initasm.cpp \
    src/face_predict.cpp

HEADERS += \
    src/DebugTimer.hpp \
    src/distinct_boxes.hpp \
    src/FaceDetection.hpp \
    src/FduVideo.hpp \
    src/FduVideo_lib.hpp \
    src/FGSegmentation.hpp \
    src/people_detection_hog.hpp \
    src/person_recognize.hpp \
    src/stasm_lib.h \
    src/svm.h \
    src/system_struct.hpp \
    src/track_of.hpp \
    src/VideoProcessor.hpp \
    stasm/asm.h \
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
    stasm/MOD_1/initasm.h \
    src/face_predict.hpp


LIBS += /usr/local/lib/libopencv_calib3d.so \
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
