/*
*Copyright (c) 2014, Fudan Video Group
*This algorithm is conducted by following sequence
*	1.Enlarge the image and detect whether there is a face or not.
*	  If not, go to next loop. 
*	2.Use Stasam to get facial feature points of original image.
*	  Calculate the angle between the line connecting two eyes and horizon line 
*	  and rotate the image. 
*	3.Use Stasam to get facial feature points of rotated image.
*	  Cut the image and normalize its scale. 
*	4.Get LBP feature from the image.
*	  Use the trained model to predict the label. 
*Stasm in this part is used to find the facial feature points.
*And Stasm rights are all preserved by original author. 
*you can get more information about stasm at the following website.
*http://www.milbo.users.sonic.net/stasm/
*/

#include "face_predict.hpp"
using namespace cv;
using namespace std;

void FaceRecognization::error(const char* s1, const char* s2)
{
    printf("Stasm version %s: %s %s\n", stasm_VERSION, s1, s2);
    exit(1);
}

void FaceRecognization::facepredict(shared_data& data1)
{
    const int scale=1;
    Mat img;
    Mat change;
    Mat rotate;
    Size dsize=opt.face_r.dsize,dsize2=opt.face_r.dsize2;
    double threshold=opt.face_r.threshold;
    const char* path{""};
    double angle,tana,length;
    double eye_leftx , eye_lefty , eye_rightx,eye_righty;
    int foundface,facecount=0;
    int min_x,min_y,max_x,max_y;
    float landmarks[2 * stasm_NLANDMARKS]; 
    for (;facecount<data1.im_boxes.size();facecount++) {
            if(data1.im_boxes[facecount].type_label!=TYPE_FACE)
                    continue;
            img=data1.im_data.image(data1.im_boxes[facecount]).clone();
            //enlarge pic
            change=Mat(dsize,CV_32S);
            resize(img,change,dsize);
            cvtColor(change,img,CV_BGR2GRAY);
            //rotate pic
            length = sqrt(img.cols*img.cols + img.rows*img.rows) * scale;
            Mat tempImg(length, length, img.type());
            int ROI_x = length / 2 - img.cols / 2;
            int ROI_y = length / 2 - img.rows / 2;
            Rect ROIRect(ROI_x, ROI_y, img.cols, img.rows);
            Mat tempImgROI2(tempImg, ROIRect);
            img.copyTo(tempImgROI2);
            Point2f center(length / 2, length / 2);
            //the parameter "1" means one face in each image;
            //the parameter "10" means that one face should 10 pixels width at least
            if (!stasm_open_image((const char*)img.data, img.cols, img.rows, path, 1 , 10 ))
                 if(opt.debug_flag)
                     error("stasm_open_image failed: ", stasm_lasterr());
            //detect facial feature points
            if (!stasm_search_auto(&foundface, landmarks) )
                  if(opt.debug_flag)
                      error("stasm_search_auto failed: ", stasm_lasterr());
            if (!foundface)
            {
                continue;
            }
            stasm_convert_shape(landmarks, 68);
            stasm_force_points_into_image(landmarks, img.cols, img.rows);
            eye_lefty = landmarks[2* LEFT_EYE+1];
            eye_leftx = landmarks[2 * LEFT_EYE];
            eye_righty = landmarks[2 * RIGHT_EYE + 1];
            eye_rightx = landmarks[2 * RIGHT_EYE];
            tana = (eye_righty - eye_lefty) / (eye_rightx - eye_leftx);
            angle = atan(tana)*180/M_PI;
            Mat M = getRotationMatrix2D(center, angle, scale);
            warpAffine(tempImg, rotate , M, Size(length, length));//rotate
            //detect facial feature points of rotated face
            if (!stasm_open_image((const char*)rotate.data, rotate.cols, rotate.rows, path,1, 10))
                 if(opt.debug_flag)  error("stasm_open_image failed: ", stasm_lasterr());
            if (!stasm_search_auto(&foundface, landmarks))
                 if(opt.debug_flag)   error("stasm_search_auto failed: ", stasm_lasterr());
            if (!foundface)
            {
                continue;
            }
            stasm_convert_shape(landmarks, 68);
            stasm_force_points_into_image(landmarks, rotate.cols, rotate.rows);
            //cut face
            min_x = cvRound(landmarks[2 * LEFTEST]);
            max_x = cvRound(landmarks[2 * RIGHTEST]);
            eye_leftx = cvRound(landmarks[2 *  LEFT_EYE ]);
            eye_rightx = cvRound(landmarks[2 * RIGHT_EYE]);
            min_y=cvRound(landmarks[2 * 16 + 1])<cvRound(landmarks[2 * 17 + 1])?cvRound(landmarks[2 * 16 + 1]):cvRound(landmarks[2 * 17 + 1]);
            min_y=cvRound(landmarks[2 * 22 + 1])<min_y?cvRound(landmarks[2 * 22 + 1]):min_y;
            min_y=cvRound(landmarks[2 * 23 + 1])<min_y?cvRound(landmarks[2 * 23 + 1]):min_y;
            max_y = cvRound(landmarks[2 * BOTTOM + 1]);
            Mat img_roi (rotate, Rect(min_x, min_y, max_x-min_x, max_y-min_y) );
            if(opt.debug_flag)  cout<<img_roi.rows<<":"<<img_roi.cols<<endl;
            //resize img
            Mat change2=Mat(dsize2,CV_32S);
            resize(img_roi,change2,dsize);
            //predict label
            int predictedLabel = -1;
            double confidence=0.0,confidence_exp=0.0;
            model->predict(change2, predictedLabel, confidence);
            confidence_exp=exp(1-confidence/threshold)/M_E;
            if (predictedLabel!=-1&&confidence_exp>opt.face_r.rate_threshold) data1.im_boxes[facecount].prob.push_back(confidence_exp);
            else data1.im_boxes[facecount].prob.push_back(0.0);
            cout << "face found "<<predictedLabel<<endl;
            data1.im_boxes[facecount].result_label.push_back(predictedLabel);
            }
}




