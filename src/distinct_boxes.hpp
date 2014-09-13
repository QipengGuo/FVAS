/*Copyright (c) 2014, School of Computer Science, Fudan University*/
#ifndef DISTINCT_BOXES_H
#define DISTINCT_BOXES_H

#include "system_struct.hpp"
class distinct_boxes
{
    private:
     bool flag[MAX_BOXES];
     options::opt_distinct opt;
    public:
     distinct_boxes(const options & in_opt):
        opt(in_opt.distinct)
     {}
     void distinct(shared_data & data)
     {

         memset(flag, 0, sizeof(bool)*MAX_BOXES);
         for (int i=0;i<data.im_boxes.size();i++)
         {
             for (int j=0;j<data.im_boxes.size();j++)
             {
                 if (data.im_boxes[i].type_label==TYPE_FACE||data.im_boxes[j].type_label==TYPE_FACE||i==j)
                     continue;
                 double temp=std::min(data.im_boxes[i].x+data.im_boxes[i].width, data.im_boxes[j].x+data.im_boxes[j].width)-std::max(data.im_boxes[i].x, data.im_boxes[j].x);
                 temp*=std::min(data.im_boxes[i].y+data.im_boxes[i].height, data.im_boxes[j].y+data.im_boxes[j].height)-std::max(data.im_boxes[i].y, data.im_boxes[j].y);
                 if (temp>opt.tol_square*data.im_boxes[i].height*data.im_boxes[i].width||data.im_boxes[i].height*data.im_boxes[i].width<opt.tol_scale)
                     flag[j]=true;
             }
         }
         std::vector<shared_data::bbox> temp_boxes;
         for (int i=0;i<data.im_boxes.size();i++)
         {
             if (!flag[i])
                 temp_boxes.push_back(data.im_boxes[i]);
         }
         data.im_boxes.clear();
         data.im_boxes=temp_boxes;
     }
};

#endif // DISTINCT_BOXES_H
