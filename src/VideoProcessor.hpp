/*Copyright (c) 2014, School of Computer Science, Fudan University*/
#ifndef VIDEOPROCESSOR_HPP
#define VIDEOPROCESSOR_HPP

#include "FduVideo_lib.hpp"

class VideoProcessor{
public:
    cv::VideoCapture capture;
    const std::string _input;

    int fps,frameId;

    // -------------- error classes -------------------
    class Error{
    public:
        std::string errorMsg;
        Error(const std::string &err):errorMsg(err){}
    };
    class InputError:public Error{
    public:
        InputError():Error("Input Error"){}
    };
    // -------------- error classes -------------------

    VideoProcessor(const std::string& openFileName=""):
        _input(openFileName),frameId(-1)
    {
        if(_input.length()==0){
            // read frames from the camera
            if(!capture.open(0)) throw InputError();
            fps=50;
        }else{
            // read frames from the file
            if(!capture.open(_input)) throw InputError();
            fps=capture.get(CV_CAP_PROP_FPS);
        }
    }

    cv::Mat read(){
        cv::Mat frame;
        frameId++;
        if(!capture.read(frame)) return cv::Mat();
        return frame;
    }

    ~VideoProcessor(){
        capture.release();
    }


};

#endif // VIDEOPROCESSOR_HPP
