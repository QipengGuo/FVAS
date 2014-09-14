/*Copyright (c) 2014, School of Computer Science, Fudan University*/
#ifndef TRACK_OF_H
#define TRACK_OF_H

#include "FduVideo_lib.hpp"
#include "system_struct.hpp"

#define MAX_NUM (9999)
#define MIN_NUM (0)

class track_of
{
    public:
     class track_data
     {
         public:
         shared_data::frame im_data;
         shared_data::info im_info;
         std::vector<shared_data::bbox> im_boxes;
         track_data& operator =(const track_data& t)
         {
             cv::Mat temp=t.im_data.image.clone();
             std::swap(temp, im_data.image);
             im_data.im_ROI=t.im_data.im_ROI;
             im_boxes.clear();
             for (auto i:t.im_boxes)
                 im_boxes.push_back(i);
             im_info=t.im_info;
             return *this;
         }
         track_data& operator =(const shared_data& t)
         {
             cv::Mat temp=t.im_data.image.clone();
             std::swap(temp, im_data.image);
             im_boxes.clear();
             for (auto i:t.im_boxes)
                 im_boxes.push_back(i);
             im_info=t.im_info;
             return *this;
         }
     };

     track_data pre_data;
     bool have_pre{false};
     options::opt_track opt;
     track_of(options in_opt):
         opt(in_opt.track)
     {}
     void track_push(shared_data &in_data)
     {
         have_pre=true;
         pre_data=in_data;
     }


     void track(shared_data &in_data)
     {
         if (have_pre)
         {
             cv::Mat TR_status, TR_err, pre_im_gray, im_gray, pre_temp_ogz(in_data.im_data.image.rows,in_data.im_data.image.cols, CV_64FC3), temp_ogz(in_data.im_data.image.rows, in_data.im_data.image.cols, CV_64FC3);
             std::vector<cv::Point2f> temp_corner, next_corner;
             for (auto i:pre_data.im_boxes)
             {
                 if (i.type_label==TYPE_TRACK)
                 {
                     cv::cvtColor(pre_data.im_data.image, pre_im_gray, cv::COLOR_BGR2GRAY);
                     cv::cvtColor(in_data.im_data.image, im_gray, cv::COLOR_BGR2GRAY);
                     cv::goodFeaturesToTrack(pre_im_gray(i),temp_corner,opt.max_corners,opt.qlevel,opt.min_dist,cv::noArray(),3, false);
                     for (int j=0;j<temp_corner.size();j++)
                     {
                         temp_corner[j].x+=i.x;
                         temp_corner[j].y+=i.y;
                     }
                     if (temp_corner.size()<1)
                         continue;
                     cv::calcOpticalFlowPyrLK(pre_im_gray, im_gray, temp_corner, next_corner, TR_status, TR_err, cv::Size(opt.win_x, opt.win_y));
                     float temp, sum_x=0, sum_y=0;
                     int st;
                     int min_x=MAX_NUM, min_y=MAX_NUM, max_x=MIN_NUM, max_y=MIN_NUM, count=0;
                     for (int j=0;j<TR_err.rows;j++)
                     {
                         st=TR_status.at<uchar>(j, 0);
                         temp=TR_err.at<float>(j, 0);
                         if (st&&temp<opt.tol_err)
                         {
                             count++;
                             sum_x+=next_corner[j].x;
                             sum_y+=next_corner[j].y;
                         }
                     }
                     sum_x/=count;
                     sum_y/=count;
                     min_x=sum_x-i.width/2.0;
                     max_x=sum_x+i.width/2.0;
                     min_y=sum_y-i.height/2.0;
                     max_y=sum_y+i.height/2.0;
                     max_x=std::min(max_x, im_gray.cols);
                     max_y=std::min(max_y, im_gray.rows);
                     min_x=std::max(MIN_NUM, min_x);
                     min_y=std::max(MIN_NUM, min_y);
                     if (count/opt.max_corners>opt.limit_corners&&fabs(max_x-min_x-i.width)<opt.limit_scale*i.width&&fabs(max_y-min_y-i.height)<opt.limit_scale*i.height)
                     {
                         shared_data::bbox temp_box(cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y));
                         temp_box.type_label=TYPE_TRACK;
                         for (auto j:i.prob)
                             temp_box.prob.push_back(j);
                         for (auto j:i.result_label)
                             temp_box.result_label.push_back(j);
                         in_data.im_boxes.push_back(temp_box);
                     }
                 }
             }
         }
     }
};

#endif // TRACK_OF_H
