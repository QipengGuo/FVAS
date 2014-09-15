/*Copyright (c) 2014, School of Computer Science, Fudan University*/
#include "system_struct.hpp"
#include "FduVideo.hpp"

using namespace std;
using namespace cv;

const std::string names[7]={"Zhou", "Li", "Guo", "Zheng", "Wang", "Zhang", "Ou"};
std::string face_name(int label)
{
    if (label==-1)
        return "Pedestrian";
    return names[label];
}

std::string person_name(int label)
{
    if (label<0||label>6)
        return "Pedestrian";
    return names[label];
}

int main(int argc, char *argv[])
{

    const string file="data/origin.avi";

    options ops;

    cv::Mat frame;
    VideoProcessor VP(file);
    FGSegmentation FGS(ops);
    FaceDetection FD(ops);
    FaceRecognization FR(ops);
    people_detection_hog PD(ops);
    person_recognize PR(ops);
    track_of TR(ops);
    distinct_boxes distinct(ops);

    PR.test_init();
    namedWindow("output");

    Timer t5;
    int count5=0;
    while(!(frame=VP.read()).empty()){
        if(VP.frameId%ops.input.jumpFrame==0){
            shared_data data(frame);
            resize(data.im_data.image,data.im_data.image,Size(ops.width,ops.height));

            Timer t;
            TR.track(data);
            t.out("TR");

            for(auto &bbox: data.im_boxes){
                if(bbox.result_label.size()!=0){
                    if (bbox.type_label==TYPE_TRACK)
                    {
                        rectangle(data.im_data.image,bbox,Scalar(255,255,255),1);
                        putText(data.im_data.image,person_name(bbox.result_label[0]),bbox.tl(),
                            CV_FONT_HERSHEY_PLAIN,1,Scalar(0,0,255));
                    }
                }
            }

            FGS.processFrame(data);
            t.out("FGS");

            FD.processFrame(data);
            t.out("FD");

            PD.detect(data);
            t.out("BD");

            distinct.distinct(data);

            FR.facepredict(data);
            t.out("FR");

            for(auto &bbox: data.im_boxes){
                if(bbox.result_label.size()!=0){
                    if (bbox.type_label==TYPE_FACE)
                    {
                        rectangle(data.im_data.image,bbox,Scalar(0,0,255),1);
                        putText(data.im_data.image,face_name(bbox.result_label[0]),bbox.tl(),
                            CV_FONT_HERSHEY_PLAIN,1,Scalar(0,0,255));
                    }
                }
                if (bbox.type_label==TYPE_PERSON)
                {
                    rectangle(data.im_data.image,Rect(bbox.x, bbox.y, bbox.width, bbox.height),Scalar(0,255,0),1);
                }
            }

            PR.Person_ReId(data);
            t.out("PR");
            TR.track_push(data);            
            count5++;
            if (count5==5)
            {
                t5.out("five");
                count5=0;
            }
            imshow("output",data.im_data.image);
            if(waitKey(1)==27){
                break;
            }
        }
    }

    printf("finish\n");
    destroyAllWindows();
    return 0;
}
