/*Copyright (c) 2014, School of Computer Science, Fudan University*/
#ifndef SYSTEM_STRUCT_H
#define SYSTEM_STRUCT_H

#include <opencv2/opencv.hpp>

#define REID_MAX_SCALE (10)  //Person Recognize
#define MAX_BOXES (500) //Max num of boxes in one frame

enum {
    TYPE_FACE=1,
    TYPE_PERSON,
    TYPE_TRACK
};
struct options
{
    struct opt_input
    {
        const int jumpFrame{5};
        const double mog_threshold{0.005};
        const int blurRadius{5};
        const int dilateRadius{20};
    };
    struct opt_track
    {
        const double sigma{0.2};
        const double lambda{0.01};
        const double learning_rate{0.075};
        const double compression_learning_rate{0.15};
        const int non_compressed_feature{1};
        const int compressed_feature{2};
        const int num_compressed_dim{2};

        const double tol_err{45};
        const double limit_scale{0.3};
        const double limit_corners{0.3};
        const int max_corners{40};
        const int win_x{40};
        const int win_y{40};
        const double qlevel {0.01};
        const double min_dist{0.1};
    };
    struct opt_face_detection
    {
        const std::string _hog_head{"data/hog_head.xml"};
        const std::string _body{"data/body.xml"};
    };
    struct opt_person_detection
    {
        const std::string person_model_path{"data/person_final.xml"};
        CvLatentSvmDetector* detector;
        const int tbbNumThreads{-1};
        const double score_threshold{-1.164949};
        const double score_upperbound{5.0};
        const int width_thresh{50};
        const int height_thresh{50};

        const double scale{0.5};
    };
    struct opt_face_recognition
    {
        std::string model_address{"data/ourface_v.xml"};
        const char * datadir{"data/"};
        cv::Size  dsize;
        cv::Size dsize2;
        double threshold{150.0};
        double rate_threshold{0.7};
        opt_face_recognition()
        {
            dsize = cv::Size(144, 200);
            dsize2= cv::Size(80,88);
        }
    };
    struct opt_person_recognition
    {
        const std::string current_path="data";
        int Patch_Size{10}, Nx{30}, Ny{10}, nScale{3}, sift_nBins{4}, color_nBins{32}, color{3}, nori{8}, norm{4}, alpha{9}, num_angles{8}, h{2}, nstripe{10}, rand_neg{30}, nRank_neg{3}, naux_neg{8}, ntop{10} ;
        double Epsi{1e-8}, Scale[REID_MAX_SCALE], clamp{0.2}, kstripe{0.3}, tol{0.1}, tol_confidence{0.7}, tol_threshold{0.4};
        int lower_bound{15}, upper_bound{300}, seleted_level{5}, dist_threshold_KNN_for_init_cluster{2}, hierachy_level{10}, split_into{4}, dist_threshold_KNN_for_affinity{30};
        double dist_threshold_KNN_rate_for_midselect{0.3}, scaling_a{1.0};
        cv::Mat Sigma, Sigma_edge;
        opt_person_recognition()
        {
            Scale[0]=0.5, Scale[1]=0.75, Scale[2]=1;

            Sigma.create(1, 1, CV_64FC1), Sigma.at<double>(0, 0)=0.6;

            Sigma_edge.create(1, 1, CV_64FC1), Sigma_edge.at<double>(0,0)=1;
        }
    };
    struct opt_distinct
    {
        const double tol_square{0.7};
        const double tol_scale{30};
    };

    opt_input input;
    opt_track track;
    opt_face_detection face_d;
    opt_face_recognition face_r;
    opt_person_detection person_d;
    opt_person_recognition person_r;
    opt_distinct distinct;
    int debug_flag{0}; //0 Release, 1 Debugging
    int height{480}, width{640};
};

class shared_data  //one frame multi bbox
{
public:
    struct info
    {
        time_t input_stamp; //time stamp
        int input_pos; //which probe
        info(time_t _stamp=0,int  _pos=-1):
            input_stamp(_stamp),
            input_pos(_pos)
        {}
    };
    struct bbox:public cv::Rect_<int>
    {
        using cv::Rect_<int>::Rect_;
        std::vector<double> prob; //confidence probability 0.0~1.0
        std::vector<int> result_label;//the label of this region, means which person in this region
        int type_label{0};//this bbox is for 1 face, 2 person or 3 tracking, 0 init
        bbox():Rect_<int>(){}
        bbox(const cv::Rect_<int> &t):Rect_(t){}
        bbox(cv::Rect_<int> &&t):Rect_(t){}
    };
    struct frame
    {
        cv::Mat image; // origin image
        std::vector<bbox> im_ROI; //valued region
        frame(const cv::Mat &_frame=cv::Mat()):
            image(_frame)
        {}
    };

    info im_info;
    frame im_data;
    std::vector<bbox> im_boxes;

    explicit shared_data(const cv::Mat &_frame):
        im_data(_frame)
    {}

    shared_data(shared_data &&t){
        im_info=t.im_info;
        im_data=t.im_data;
        im_boxes=move(t.im_boxes);
    }

    shared_data & operator = (shared_data &&t){
        im_info=t.im_info;
        im_data=t.im_data;
        im_boxes=move(t.im_boxes);
        return *this;
    }

    bool empty() const{
        return im_data.image.empty();
    }
};


#endif
