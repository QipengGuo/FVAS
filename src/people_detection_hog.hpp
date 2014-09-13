/*Copyright (c) 2014, Fudan Video Group*/
#ifndef PEOPLE_DETECTION_HOG_H
#define PEOPLE_DETECTION_HOG_H

#include "FduVideo_lib.hpp"
#include "system_struct.hpp"

class people_detection_hog {
public:
    cv::HOGDescriptor hog;
    options::opt_person_detection opt;
    cv::Mat temp_resize;
    int scenter_x, scenter_y, center_x, center_y;
    people_detection_hog(const options& in_opt):
        opt(in_opt.person_d)
    {
        scenter_x=in_opt.width*opt.scale/2.0;
        scenter_y=in_opt.height*opt.scale/2.0;
        center_x=in_opt.width/2.0;
        center_y=in_opt.height/2.0;
        hog.setSVMDetector(hog.getDefaultPeopleDetector());
    }

    void box_safe(cv::Rect &t, int min_x, int min_y, int max_x, int max_y)
    {
        if (t.x<min_x) t.x=min_x;
        if (t.y<min_y) t.y=min_y;
        if (t.x+t.width>max_x) t.width-=t.x+t.width-max_x;
        if (t.y+t.height>max_y) t.height-=t.y+t.height-max_y;
    }

    void detect(shared_data& in_data)
    {
        std::vector<cv::Rect> temp_boxes;
        for (auto i: in_data.im_data.im_ROI) {
            temp_resize.create(in_data.im_data.image(i).rows*opt.scale, in_data.im_data.image(i).cols*opt.scale, CV_64FC3);
            cv::resize(in_data.im_data.image(i), temp_resize, temp_resize.size());
            //these magic numbers are from opencv
            if (temp_resize.rows > 128 && temp_resize.cols > 64) {
                temp_boxes.clear();
                hog.detectMultiScale(temp_resize,
                                     temp_boxes,
                                     0,
                                     cv::Size(8, 8),
                                     cv::Size(0, 0),
                                     1.05);
                for (auto j: temp_boxes) {
                    i.x -= center_x;
                    i.y -= center_y;
                    i.x = opt.scale*i.x+scenter_x;
                    i.y = opt.scale*i.y+scenter_y;
                    j.x += i.x;
                    j.y += i.y;
                    j.x -= scenter_x;
                    j.y -= scenter_y;
                    j.x = 1.0/opt.scale*j.x+center_x;
                    j.y = 1.0/opt.scale*j.y+center_y;
                    j.height*=2;
                    j.width*=2;
                    box_safe(j, 0, 0, in_data.im_data.image.cols-1, in_data.im_data.image.rows-1);
                    shared_data::bbox temp;
                    temp = j;
                    temp.type_label = TYPE_PERSON;
                    in_data.im_boxes.push_back(temp);
                }
            }
        }
    }
};

#endif // PEOPLE_DETECTION_HOG_H
