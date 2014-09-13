/*Copyright (c) 2014, Fudan Video Group*/
#ifndef FACE_PERDICT_H
#define FACE_PERDICT_H

#include "FduVideo_lib.hpp"
#include "stasm_lib.h"
#include "system_struct.hpp"

#define LEFT_EYE 31
#define RIGHT_EYE 36
#define LEFTEST 0
#define RIGHTEST 14

#define BOTTOM 7


class FaceRecognization
{
    private:
     const options &opt;
    public:
     cv::Ptr<cv::FaceRecognizer> model;
     void error(const char*, const char*);
     void facepredict(shared_data& );
     FaceRecognization(const options &_opt):
         opt(_opt)
     {
         stasm_init(opt.face_r.datadir,0);
         model = cv::createLBPHFaceRecognizer();
         model->set("threshold", opt.face_r.threshold);
         model->load(opt.face_r.model_address);
     }
};

#endif
