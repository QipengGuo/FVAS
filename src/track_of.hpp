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

     cv::Mat ogz_color(cv::Mat &rgb, cv::Mat &gray)
     {
         /*
         //rgb=cv::imread("/home/gqp/ogz.jpg", -1);
         //cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
         //cv::imshow("123",rgb);
         cv::Vec3b *rgb_ptr=rgb.ptr<cv::Vec3b>(0);
         uchar *gray_ptr=gray.ptr<uchar>(0);
         cv::Mat result(rgb.rows, rgb.cols, CV_8UC3);
         cv::Mat dst;
         double r, g, b;
         for (cv::MatIterator_<cv::Vec3b> tt=result.begin<cv::Vec3b>(); tt!=result.end<cv::Vec3b>(); tt++)
         {
             //*tt = *rgb_ptr-*gray_ptr;
             b=((*rgb_ptr)[0]-*gray_ptr)*4+40;
             g=((*rgb_ptr)[1]-*gray_ptr)*4+40;
             r=((*rgb_ptr)[2]-*gray_ptr)*4+40;

             b=cv::saturate_cast<uchar>(b);
             g=cv::saturate_cast<uchar>(g);
             r=cv::saturate_cast<uchar>(r);

             //b=(*rgb_ptr)[0]-b;
             //g=(*rgb_ptr)[0]-g;
             //r=(*rgb_ptr)[0]-r;
             (*tt)[0]=cv::saturate_cast<uchar>(b);
             (*tt)[1]=cv::saturate_cast<uchar>(g);
             (*tt)[2]=cv::saturate_cast<uchar>(r);
             rgb_ptr++;gray_ptr++;
         }

         cv::pyrMeanShiftFiltering(rgb, dst, 19, 22, 2);
              cv::RNG rng=cv::theRNG();
              cv::Mat mask(dst.rows+2,dst.cols+2,CV_8UC1,cv::Scalar::all(0));
              for(int i=0;i<dst.rows;i++)    //opencv图像等矩阵也是基于0索引的
                  for(int j=0;j<dst.cols;j++)
                      if(mask.at<uchar>(i+1,j+1)==0)
                     {
                          cv::Scalar newcolor(rng(256),rng(256),rng(256));
                        cv::floodFill(dst,mask,cv::Point(j,i),newcolor,0,cv::Scalar::all(1),cv::Scalar::all(1));
         };
         cv::imshow("321", dst);
         cv::imshow("orgin", rgb);
         cv::cvtColor(result, result, cv::COLOR_BGR2GRAY);

         //cv::waitKey();
         */
         return gray;
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
                     pre_temp_ogz=ogz_color(pre_data.im_data.image, pre_im_gray);
                     cv::cvtColor(in_data.im_data.image, im_gray, cv::COLOR_BGR2GRAY);
                     temp_ogz=ogz_color(in_data.im_data.image,im_gray);
                     cv::goodFeaturesToTrack(pre_temp_ogz(i),temp_corner,opt.max_corners,opt.qlevel,opt.min_dist,cv::noArray(),3, false);
                     for (int j=0;j<temp_corner.size();j++)
                     {
                         temp_corner[j].x+=i.x;
                         temp_corner[j].y+=i.y;
                     }
                     if (temp_corner.size()<1)
                         continue;
                     cv::calcOpticalFlowPyrLK(pre_temp_ogz, temp_ogz, temp_corner, next_corner, TR_status, TR_err, cv::Size(opt.win_x, opt.win_y));
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
                             //if (round(next_corner[j].x)>max_x)
                             //    max_x=round(next_corner[j].x);
                             //if (round(next_corner[j].x)<min_x)
                             //    min_x=round(next_corner[j].x);
                             //if (round(next_corner[j].y)>max_y)
                             //    max_y=round(next_corner[j].y);
                             //if (round(next_corner[j].y)<min_y)
                             //    min_y=round(next_corner[j].y);
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
