/*Copyright (c) 2014, School of Computer Science, Fudan University*/
#ifndef PERSON_RECOGNIZE_H
#define PERSON_RECOGNIZE_H

#include "FduVideo_lib.hpp"
#include "system_struct.hpp"
#include "svm.h"

#define DIM4(ptr,size,a,b,c,d) (ptr+(a)*(size[1]*size[2]*size1[3])+(b)*(size[2]*size1[3])+(c)*(size[3])+d)
#define DIM3(ptr,size,a,b,c) (ptr+(a)*(size[1]*size[2])+(b)*(size[2])+c)
#define CACHE_DIR "cache"
#define MAX_FILENAME (500)
#define LSVM_MODEL_PATTERN "LSVM_%d_%d"
#define MAX_NODES (300)
#define MAX_MODELS (500)
#define MAX_GALLERY_IMAGES (2000)
#define MAX_PROBE_IMAGES (100)
#define MAX_HEIGHT (10)
#define MAX_DISTANCE (9999)
#define MAX_ITEMS (60000)
#define MAX_TRAIN_ITEMS (190800)
#define MAX_TRAIN_IMAGES (800)
#define MAX_DIM (672)
#define NUM_BINS (32)
#define GAUSSP_COEF (130.0)
#define GAUSSF_COEF (1.0)
#define SIGMAF (0.1)
#define SIGMAP (2.8)
#define TRAIN_FLAG (1)
#define PAR_SIZE (8)

class person_recognize
{
    public :
    typedef options::opt_person_recognition option;
    typedef shared_data::frame frame;
    typedef shared_data::bbox bbox;

    struct Samples //storing train samples in this struct
    {
        int next{-1}, im_index{0};
    };

    struct sortsupport
    {
        double patch_dist, filter_dist;
        int index;

        bool operator <(const sortsupport & t) const
        {
            return filter_dist+patch_dist>t.filter_dist+t.patch_dist;
        }
    };

    struct sortAssistant
    {
        int index, im_index, pos_index;
        double dist;
        bool operator <(const sortAssistant & t) const
        {
            return dist<t.dist;
        }
    };

    class SVM_data //data in svm_style
    {
        public:
        int Max_items{0}, count{0};
        int *index{NULL};
        double *data{NULL}, *output{NULL};//like index(i):data(i), and output are predict labels
        int *L_dim{NULL}, L_items{0};
        SVM_data()
        {}

        SVM_data(int max_items)
        {
            index=new int[MAX_DIM*max_items];
            data=new double[MAX_DIM*max_items];
            output=new double[max_items];
            L_dim=new int [max_items];
            memset(L_dim, 0, sizeof(int)*max_items);
            Max_items=max_items;
        }

        void resize(int max_items)
        {
            if (max_items==Max_items)
                return;
            if (index!=NULL) delete []index;
            if (data!=NULL)  delete []data;
            if (output!=NULL) delete []output;
            if (L_dim!=NULL) delete []L_dim;
            L_items=0,count=0;
            index=new int[MAX_DIM*max_items];
            data=new double[MAX_DIM*max_items];
            output=new double[max_items];
            L_dim=new int [max_items];
            memset(L_dim, 0, sizeof(int)*max_items);
            Max_items=max_items;
        }

        void clear()
        {
            L_items=0,count=0;
            memset(L_dim, 0, sizeof(int)*Max_items);
        }

        ~SVM_data()
        {
            if (index!=NULL) delete []index;
            if (data!=NULL)  delete []data;
            if (output!=NULL) delete []output;
            if (L_dim!=NULL) delete []L_dim;
        }
    };
    struct feature //storing low-level features, and it contains sift and color hist
    {
        cv::Mat feat;
        int im_index, pos_index, height, index;//im_index=image_index, pos_index=position_in_image_index, index is using for clustering
        double sAUC;//same as sAUC in paper, represention of frequency
        feature()
        {
            im_index=-1;pos_index=-1;height=-1;sAUC=-1;
        }
        feature(cv::Mat &t, int im, int pos, int h, double s)
        {
            im_index=im;pos_index=pos;height=h;sAUC=s;feat=t;
        }
        feature(cv::Mat &t, int im, int pos)
        {
            im_index=im;pos_index=pos;feat=t;height=-1;sAUC=-1;
        }
        feature(cv::Mat &t, int im, int pos, int h)
        {
            im_index=im;pos_index=pos;height=h;feat=t;sAUC=-1;
        }
        feature(const feature & f)
        {
            feat=f.feat;height=f.height;im_index=f.im_index;
            pos_index=f.pos_index;sAUC=f.sAUC;index=f.index;
        }
        double operator-(const feature &f2) const
        {
            double sum=0, temp=0;
            const double *ptr1=feat.ptr<double>(0);
            const double *ptr2=f2.feat.ptr<double>(0);
            for (int i=0;i<this->feat.cols;i++)
            {
                temp=ptr1[i]-ptr2[i];
                sum+=temp*temp;
            }
            return sum;
        }
        double operator-(double t) const
        {
            double sum=0;
            double temp=0;
            const double *ptr1=feat.ptr<double>(0);
            for (int i=0;i<feat.cols;i++)
            {
                temp=ptr1[i]-t;
                sum+=temp*temp;
            }
            return sum;
        }
        bool operator ==( const feature &f2 ) const
        {
            return im_index == f2.im_index && pos_index == f2.pos_index && height == f2.height && index == f2.index;
        }
    };

    struct im_feature //storing mid-level features
    {
        cv::Mat feat;
        int im_index;
        im_feature()
        {
            im_index=-1;
        }
        im_feature(cv::Mat &t, int im)
        {
            im_index=im;feat=t;
        }
        im_feature(const im_feature & f)
        {
            feat=f.feat;im_index=f.im_index;
        }

    };

    struct cluster
    {
        std::list<feature> feats;
        int index;
        bool operator == ( const cluster& c ) const
        {
            return feats == c.feats;
        }
    };

    struct affinityNode
    {
    public:
        int a;
        int b;
        double affi;
        affinityNode( int aa , int bb, double af )
        {
            a = aa;
            b = bb;
            affi = af;
        }

        friend bool operator < ( const affinityNode& a, const affinityNode& b )
        {
            return a.affi < b.affi;
        }
    };

    int g_count{0}, p_count{0}, first_flag{1}, num_models{0};//g_count=gallery_count p_count=probe_count num_models=num_of_svm_models
    int last_node[MAX_TRAIN_IMAGES], train_label[MAX_TRAIN_IMAGES], num_samples{-1},  box_index[MAX_PROBE_IMAGES];
    Samples samples[MAX_TRAIN_IMAGES];//trainning samples
    svm_model *models[MAX_MODELS], *ranksvm_model;//svm_models
    SVM_data svm_data, ranksvm_data[PAR_SIZE];//svm_data
    const options &in_opt;
    std::vector<std::vector<std::string> >model_name;//name of svm_model
    std::string rank_model_name;
    std::vector<std::vector<feature> > gallery_features;//low-level features of gallery images
    std::vector<im_feature> gallery_im;//mid-level features of gallery images

    cv::Mat norm_image(cv::Mat & image, const option & opt);
    cv::Mat get_sift(cv::Mat & image, const option & opt);
    cv::Mat get_colorhist(cv::Mat & image, const option & opt);
    std::vector<feature> get_features(cv::Mat & image, const option & opt, int image_index);
    cv::Mat generate_gauss(const cv::Mat & sigma);
    cv::Mat generate_gauss(double sigma);
    void generate_dgauss(cv::Mat & Gaussian_X, cv::Mat & Gaussian_Y, const cv::Mat & sigma); //(out, out, in)
    void generate_dgauss(cv::Mat & Gaussian_X, cv::Mat & Gaussian_Y, double sigma);//(out, out, in)
    void colorHist(double * hist, cv::Mat & data, double max_bar,int dim);//(out, in, in, in)
    cv::Mat rgb2lab_1(const cv::Mat & image);
    cv::Mat rgb2lab_2(const cv::Mat & image);
    double patch_distance(double* a, double * b, int dim);
    std::vector<std::vector<feature> > Read_Extract(const char * filename, const option & opt, int& image_count);
    void pre_mid_filter(SVM_data & data, const std::vector<feature> & low_level_features_of_one_image, const option & opt);//(out, in, in)
    void exec_mid_filter(std::vector<im_feature> & mid_level_features,
                         const std::vector<std::vector<std::string> >& linear_svm_name,
                         const option & opt,
                         int start_image,
                         int end_image,
                         int* & temp_array,
                         int max_images); //(out, in..)
    void extract_im_feature(std::vector<im_feature> & mid_level_features,
                            const std::vector<std::vector<feature> > & all_low_level_features,
                            const std::vector<std::vector<std::string> > & linear_svm_names,
                            int * & temp_array,
                            const option & opt,
                            int image_count,
                            int max_images);//(out, in..)
    void norm_im_f(std::vector<im_feature> & mid_level_features, const option & opt);
    std::vector<std::vector<feature> > Read_Extract(frame & one_frame, std::vector<bbox> bounding_boxes,  const option & opt, int & image_count);
    void norm(cv::Mat & mat);
    double gauss_filter_dist(const im_feature & a, const im_feature & b);
    double gauss_patch_dist(const std::vector<feature> & A, const std::vector<feature> & B);
    bool cmp_sortsupport(sortsupport a, sortsupport b);
    std::vector<feature> select_all(std::vector<std::vector<feature> > & all_low_level_features, int given_height, const option & opt);
    std::vector<std::vector<feature> >collect_aux(const std::vector<std::vector<feature> >& culster_nodes,
                                                  int n_th_node,
                                                  const std::vector<std::vector<sortAssistant> >& distance_table,
                                                  const std::vector<std::vector<feature> > & all_low_level_features,
                                                  const option & opt);
    std::string train_lsvm(const std::vector<std::vector<feature> >& pos_and_neg, const option & opt, int height, int n_th_svm);
    void prepare_rank(const std::vector<im_feature> & mid_level_feat,
                      const std::vector<std::vector<feature> > & low_level_feat,
                      int n_th_image,
                      const option & opt);
    void add_node(int label, int index);
    std::string train_ranksvm(const option & opt);
    int b_search(const std::vector<std::vector<sortAssistant> > & distance_table, const feature & x);
    void train_models(const option & opt);
    std::vector<std::vector<feature> > Read_Extract_Train(const char * filename, const option & opt, int & image_count);
    void read_models(const option& opt);
    void test_init();
    void test_it(std::vector<bbox> & bounding_boxes,
                 const std::vector<im_feature > & mid_level_feat,
                 const std::vector<std::vector<feature> > & low_level_feat,
                 int n_th_image,
                 const option & opt);
    int Person_ReId(shared_data & one_frame_from_video);

    person_recognize(const options &t_opt):
        in_opt(t_opt), svm_data(MAX_ITEMS)
    {
            printf("version 1.0\n");
    }
    ~person_recognize()
    {
        for (int i=0;i<num_models;i++)
            svm_free_and_destroy_model(&models[i]);
    }

    //---------------------GDL clustering-------------------------//
    void clustering( std::vector<feature>& feats,
                     std::vector< std::vector<sortAssistant> >& KNN_array,
                     std::vector< std::vector<feature> >& cluster_result,
                     const option& opt );
    void hierachy( std::vector<feature>& allFeats, cv::Mat& allWeight, std::list< cluster >& result, const option& opt );
    void doCluster( std::list< cluster >& allCluster, int k, cv::Mat& w );
    void initialCluster( std::vector<feature>& feats, std::list< cluster >& allCluster, const option& opt );
    void BFS( int start, cluster& c, std::vector<feature>& feats, bool* visited, double* w );


    int* getCluster( cluster& c );
    void merge( cluster* a, cluster* b );
    double affinity( cluster* a, cluster* b, cv::Mat& w );

    void calcAllPairDist( double* w, std::vector<feature>& f, double* s, int k_KNN );
    void calcAllPairDist( double* w, std::vector<feature>& f, double* s, int k_KNN, std::vector< std::vector<sortAssistant> >& KNN_array );
    bool find( std::list<cluster>& allCluster, cluster& a, cluster& b );
    void calcWeight( double* w, int n, int KNN, double* s, double scaling_a );

    double getEuclidDistSquare( feature& a, feature& b );

    void midLevelSelect(std::vector<feature>& all_feats, std::vector<feature>& mid_level_feats, const option& opt );
    void debug();

};

#endif
