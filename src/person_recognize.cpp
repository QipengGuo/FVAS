/*
 *Copyright (c) 2014, Fudan Video Group
 * Coder: Qipeng Guo, Yanye Li
 * School of Computer Science , Fudan university
 * Time: 2014
 *
 * Origin Paper:
 * Rui Zhao Wanli Quyang Xiaogang Wang
 * Learning Mid-level Filters for Person-identification
 *
 * Zhang W, Wang X, Zhao D, et al.
 * Graph degree linkage: Agglomerative clustering on a directed graph
 *
 * LIBSVM:
 * Chih-Chung Chang and Chih-Jen Lin, LIBSVM :
 * a library for support vector machines.
 * ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011.
 * Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm
 *
 */


#include "person_recognize.hpp"
#include "DebugTimer.hpp" //record time , it's for debugging

typedef person_recognize::feature feature;
typedef person_recognize::im_feature im_feature;
typedef person_recognize::sortsupport sortsupport;
using namespace std;
using namespace cv;

//in comments, i means input, o means output

//read models and extract features from gallery
void person_recognize::test_init()
{
    if (!first_flag)
        return;

    //init
    char temp_str[MAX_FILENAME];
    const option& opt(in_opt.person_r);
    int *g_labels= new int[MAX_GALLERY_IMAGES*MAX_HEIGHT*MAX_NODES];

    //read linear SVM and Rank SVM
    read_models(opt);

    //read Gallery
    sprintf(temp_str, "%s/test_fdu", opt.current_path.c_str());
    gallery_features=Read_Extract(temp_str, opt, g_count);//low-level features
    extract_im_feature(gallery_im,   //mid-level features
                       gallery_features,
                       model_name,
                       g_labels,
                       opt,
                       g_count,
                       MAX_GALLERY_IMAGES);

    norm_im_f(gallery_im, opt);//normalize mid-level features

    delete(g_labels);
    first_flag=0;
}


//recognize one image with multi-regions
int person_recognize::Person_ReId(shared_data & in_data)//i:shared_data is a package of one frame and its' informations
{
    //Timer tt; // record run time
    const option& opt(in_opt.person_r);//params set

    //init
    srand(time(NULL));

    //ensure create new files, it's for debugging
    char temp_str[MAX_FILENAME];
    sprintf(temp_str, "%s/result", opt.current_path.c_str());
    FILE *file=fopen(temp_str, "w");
    fclose(file);
    sprintf(temp_str,"%s/debug", opt.current_path.c_str());
    file=fopen(temp_str, "w");
    fclose(file);

    //testing
    //extract low-level features
    vector<vector<feature> > probe_features=Read_Extract(in_data.im_data,
                                                         in_data.im_boxes,
                                                         opt,
                                                         p_count);
    if (p_count<1)//no bounding-box for person
    {
        printf("NO BOX\n");
        return 0;
    }

    int *labels= new int[MAX_PROBE_IMAGES*MAX_HEIGHT*MAX_NODES];
    svm_data.clear();
    vector<im_feature> probe_im;
    extract_im_feature(probe_im,
                       probe_features,
                       model_name, labels, opt, p_count, MAX_PROBE_IMAGES);

    norm_im_f(probe_im, opt);

#pragma omp parallel for
    for (int i=0;i<p_count;i++)
    {
        test_it(in_data.im_boxes,
                probe_im,
                probe_features,
                i,
                opt);
    }

    delete []labels;
    printf("OK!\n");
    return 0;
}


//train models for linearSVM and RankSVM
void person_recognize::train_models(const option &opt)
{
    //change size, realloc memory
    svm_data.resize(MAX_TRAIN_ITEMS);

    model_name.clear();
    rank_model_name="";

    memset(last_node, -1, sizeof(int)*MAX_TRAIN_IMAGES);//serve for adjacency table
    int count=0;
    char temp_str[MAX_FILENAME];
    FILE * file;
    vector<feature> stripe[10];
    vector<im_feature> im_features;

    sprintf(temp_str, "%s/train", opt.current_path.c_str());
    vector<vector<feature> > features=Read_Extract_Train(temp_str, opt, count);//import data form files and extract features, vector: coef 1--which image 2--which patch in each image
    sprintf(temp_str, "%s/%s/Rank_train", opt.current_path.c_str(), CACHE_DIR);
    file=fopen(temp_str, "w");
    fclose(file);
    if (TRAIN_FLAG==1)//flag==1 means need train linearsvm
    {
        vector<vector<vector<feature> > > nodes;//coef 1--which height 2--which small set (node) 3--each patch in one set
        vector<vector<vector<sortAssistant> > >neighbor;//coef 1--which height 2--which patch 3--which neighbor

        //trainning

        for (int i=0;i<opt.nstripe;i++)
        {
            vector<vector<sortAssistant> >temp1;
            vector<vector<feature> > temp2;
            stripe[i]=select_all(features, i, opt); //select all patch in given height into a package
            clustering(stripe[i], temp1, temp2, opt);//cluster these patches with hierarchical structure
            nodes.push_back(temp2);
            neighbor.push_back(temp1);
            printf("height %d has done cnt=%d \n", i, temp2.size());
        }
        int *lsvm_count= new int[opt.nstripe];//record numbers of linear svm  for each height
        memset(lsvm_count, 0, sizeof(int)*opt.nstripe);
        for (int j=0;j<opt.nstripe;j++)
        {
            vector<string> temp_name;
            for (uint i=0;i<nodes[j].size();i++)
            {
                temp_name.push_back(train_lsvm(collect_aux(nodes[j], i, neighbor[j], features, opt), opt, lsvm_count[j], j));//select aux_+ and aux_- samples and use it for trainning linear SVM
                sprintf(temp_str, "%s/%s/%s.model", opt.current_path.c_str(), CACHE_DIR, temp_name[i].c_str());
                models[num_models++]=svm_load_model(temp_str);
                lsvm_count[j]++;
                printf("height=%d cnt%d=%d \n", j, i, nodes[j][i].size());
            }
            model_name.push_back(temp_name);
        }
        printf("done 1 \n");
        delete []lsvm_count;
    }
    if (TRAIN_FLAG==2)//just need to train Rank SVM
    {
        read_models(opt);
    }
    int *labels=new int[MAX_TRAIN_IMAGES*MAX_HEIGHT*MAX_NODES];
    extract_im_feature(im_features, features, model_name, labels, opt, count, MAX_TRAIN_IMAGES);
    delete [] labels;
    norm_im_f(im_features, opt);//normalization
    for (int i=0;i<count;i++)
    {
        prepare_rank(im_features, features ,i, opt);//prepare data for trainning RankSVM
    }

    rank_model_name=train_ranksvm(opt);//train ranksvm

    //save models
    sprintf(temp_str, "%s/model_list", opt.current_path.c_str());
    FILE *train_model=fopen(temp_str, "w");
    fprintf(train_model, "%s\n", rank_model_name.c_str());
    fprintf(train_model, "%d\n", model_name.size());
    for (int i=0;i<model_name.size();i++)
    {
        fprintf(train_model, "%d\n", model_name[i].size());
        for (int j=0;j<model_name[i].size();j++)
            fprintf(train_model, "%s\n", model_name[i][j].c_str());
    }
    fclose(train_model);
    svm_data.resize(MAX_ITEMS);
}

//read models for linear SVM and Rank SVM
void person_recognize::read_models(const option & opt)
{
    char temp[MAX_FILENAME];
    sprintf(temp, "%s/model_list", opt.current_path.c_str());
    FILE *train_model=fopen(temp, "r");

    if (train_model!=NULL)
    {
        //read models from files
        char temp_str[MAX_FILENAME];
        int num1, num2, count=0;
        fscanf(train_model, "%s", temp_str);
        rank_model_name=string(temp_str);
        sprintf(temp, "%s/%s/%s", opt.current_path.c_str(), CACHE_DIR, temp_str);
        ranksvm_model=svm_load_model(temp);
        fscanf(train_model, "%d", &num1);
        for (int i=0;i<num1;i++)
        {
            vector<string> temp_name;
            fscanf(train_model, "%d", &num2);
            for (int j=0;j<num2;j++)
            {
                fscanf(train_model, "%s",temp_str);
                sprintf(temp, "%s/%s/%s.model", opt.current_path.c_str(), CACHE_DIR, temp_str);
                models[count++]=svm_load_model(temp);//read model by libsvm API
                temp_name.push_back(string(temp_str));
            }
            model_name.push_back(temp_name);
        }
        num_models=count;
        fclose(train_model);
    }
    else
    {
        printf("No models!\n");
    }
}

//recognize n-th region
void person_recognize::test_it(vector<bbox> & result_box, //o
                               const vector<im_feature > & features, //i:mid-level features
                               const vector<vector<feature> > & patch_feats, //i:low-level features
                               int n, //i:n-th regions
                               const option & opt)
{
    int Par_num=n%PAR_SIZE;

    FILE * debugf=fopen("debug", "a+");
    fprintf(debugf, "%d\n", n);
    sortsupport *dist=new sortsupport[gallery_im.size()];
    for (int i=0;i<gallery_im.size();i++)
    {
        dist[i].filter_dist=gauss_filter_dist(gallery_im[i],features[n]);
        dist[i].patch_dist=gauss_patch_dist(gallery_features[i], patch_feats[n]);
        fprintf(debugf, "%d %5.7lf  %5.7lf\n", i, dist[i].filter_dist, dist[i].patch_dist);
    }
    fclose(debugf);

    ranksvm_data[Par_num].resize(gallery_im.size()*gallery_im.size());
    ranksvm_data[Par_num].clear();
    for (int i=0;i<gallery_im.size();i++)
    {
        for (int j=0;j<gallery_im.size();j++)
        {
            ranksvm_data[Par_num].index[2*ranksvm_data[Par_num].L_items]=1;
            ranksvm_data[Par_num].data[2*ranksvm_data[Par_num].L_items]=dist[i].filter_dist-dist[j].filter_dist;
            ranksvm_data[Par_num].index[2*ranksvm_data[Par_num].L_items+1]=2;
            ranksvm_data[Par_num].data[2*ranksvm_data[Par_num].L_items+1]=dist[i].patch_dist-dist[j].patch_dist;
            ranksvm_data[Par_num].L_dim[ranksvm_data[Par_num].L_items++]=2;
        }
    }

    int items=0;
    svm_test(ranksvm_model, ranksvm_data[Par_num].L_items, ranksvm_data[Par_num].L_dim, ranksvm_data[Par_num].index, ranksvm_data[Par_num].data, ranksvm_data[Par_num].output);//call libSVM
    //calc partial ordering
    //dist just for sort
    for (int i=0;i<gallery_im.size();i++)
    {
        int count=0;
        for (int j=0;j<gallery_im.size();j++)
        {
            if (ranksvm_data[Par_num].output[items++]>0)
            {
                count++;
            }
        }
        dist[i].index=gallery_im[i].im_index;
        dist[i].filter_dist=count;
        dist[i].patch_dist=0;
    }

    sort(dist, dist+gallery_im.size());

    result_box[box_index[patch_feats[n][0].im_index]].type_label=TYPE_TRACK;
    for (int i=0;i<opt.ntop;i++)
    {
        if (i==0&&dist[i].filter_dist/gallery_im.size()<opt.tol_confidence)
        {
            result_box[box_index[patch_feats[n][0].im_index]].result_label.push_back(-1);
            break;
        }
        result_box[box_index[patch_feats[n][0].im_index]].result_label.push_back(dist[i].index);
        result_box[box_index[patch_feats[n][0].im_index]].prob.push_back(dist[i].filter_dist/gallery_im.size());
    }
    delete[] dist;
}

//extract mid-level features
void person_recognize::extract_im_feature(vector<im_feature> & im, //o:mid-level features
                                          const vector<vector<feature> > & features, //i:low-level features
                                          const vector<vector<string> > & model_name, //i:linear SVM
                                          int * & labels, //temp array, avoid to new big array many times
                                          const option & opt,
                                          int count,
                                          int max_images)
{

    for (int i=0;i<count;i++)
        im.push_back(im_feature());

    int i=0;

    for (i=0;i<count;i++)
    {
            pre_mid_filter(svm_data, features[i], opt);//prepare data, convert data to svm-style
    }

    exec_mid_filter(im, model_name, opt, 0, count, labels, max_images);

    //printf("im_feature done\n");
}

//add new node in adjacency talbe
void person_recognize::add_node(int label,
                                int index)
{
    num_samples++;
    samples[num_samples].next=last_node[label];
    samples[num_samples].im_index=index;
    last_node[label]=num_samples;
}

//read and extrat trainning data from file
vector<vector<feature> > person_recognize::Read_Extract_Train(const char * filename,
                                                              const option & opt,
                                                              int &count)
{
    FILE * file=fopen(filename, "r");
    vector<vector<feature> > result;
    char str1[MAX_FILENAME], str2[MAX_FILENAME];
    Mat Image;
    count=0;
    int temp=0;
    while (fscanf(file, "%d%s", &temp, str2)!=EOF) //have labels
    {
        add_node(temp, count);
        train_label[count]=temp;
        //sprintf(str1, "%s/%s" ,opt.current_path.c_str(), str2);
        Mat temp;
        temp=imread(str1, -1);
        Image.create(120, 40, CV_64FC3);
        resize(temp, Image, Image.size());
        Image=norm_image(Image, opt);
        result.push_back(get_features(Image, opt, count));
        count++;
        cout << str2 << endl;
    }
    fclose(file);
    return result;
}

//select all patches at given height
vector<feature> person_recognize::select_all(vector<vector<feature> > &features,
                                             int high,
                                             const option & opt)
{
    vector<feature> result;
    result.reserve(opt.Nx/opt.nstripe*opt.h*opt.Ny);
    for (int i=0;i<features.size();i++)
    {
        for (int height=(high-opt.h)*opt.Nx/opt.nstripe;height<(high+opt.h+1)*opt.Nx/opt.nstripe;height++)
        {
            if (height<0||height>=opt.Nx)
                continue;
            for (int col=0;col<opt.Ny;col++)
                result.push_back(features[i][height*opt.Ny+col]);
        }
    }
    return result;
}



//collect pos samples, neg samples, aux_pos, aux_neg samples for train linear svm
vector<vector<feature> > person_recognize::collect_aux(const vector<vector<feature> >& nodes,//i:clustering result
                                                       int n,//i:n-th cluster
                                                       const vector<vector<sortAssistant> >& neighbor,//i:distance table
                                                       const vector<vector<feature> > & features,//i:low-level features
                                                       const option &opt)
{
    vector<feature> pos=nodes[n]; //+
    vector<feature> neg;
    vector<vector<feature> > result;

    int n_index=0, tt=0, im_index=0, pos_index=0;
    for (int i=0;i<nodes[n].size();i++)
    {
        n_index=nodes[n][i].im_index;
        //aux+
        tt=last_node[train_label[n_index]];
        while (tt!=-1)
        {
            im_index=samples[tt].im_index;
            pos_index=nodes[n][i].pos_index;
            for (int j=pos_index-opt.h*opt.Ny*opt.Nx/opt.nstripe;j<pos_index+opt.h*opt.Ny*opt.Nx/opt.nstripe;j++)
            {
                if (j<0||j>=opt.Ny*opt.Nx)
                    continue;
                if ((features[im_index][j]-nodes[n][i]<opt.tol*(nodes[n][i]-0))&&((nodes[n][i]-0)>opt.Epsi))
                    pos.push_back(features[im_index][j]);
            }
            tt=samples[tt].next;
        }
        //aux-
        pos_index=b_search(neighbor, nodes[n][i]);
        for (int j=0, k=0;k<opt.naux_neg&&j<opt.dist_threshold_KNN_for_affinity;j++) //wait to modify
        {
            if (neighbor[pos_index][j].dist<opt.Epsi)
                break;
            if (train_label[neighbor[pos_index][j].im_index]!=train_label[n_index])
            {
                neg.push_back(features[neighbor[pos_index][j].im_index][neighbor[pos_index][j].pos_index]);
                k++;
            }
        }
    }
    //-
    int t=0;
    for (int j=0;j<opt.rand_neg;j++)
    {
        t=rand()%nodes.size();
        tt=rand()%nodes[t].size();
        if (train_label[nodes[t][tt].im_index]==train_label[n_index])
            continue;
        neg.push_back(nodes[t][tt]);
    }

    result.push_back(pos);
    result.push_back(neg);
    return result;
}

//train linear svm for each node
string person_recognize::train_lsvm(const vector<vector<feature> >& samples, //i:pos and neg samples for trainning linear SVM
                                    const option & opt,
                                    int num,//i
                                    int h)//i
{
    char model_name[MAX_FILENAME];
    char filename[MAX_FILENAME];
    sprintf(model_name, LSVM_MODEL_PATTERN, h, num);
    sprintf(filename, "%s/%s/%s", opt.current_path.c_str(), CACHE_DIR, model_name);
    FILE *file=fopen(filename, "w");

    for (int i=0;i<samples[0].size();i++)//pos
    {
        fprintf(file, "+1");
        const double *dptr=samples[0][i].feat.ptr<double>(0);
        for (int j=0;j<samples[0][i].feat.cols;j++)
            if (dptr[j]!=0)
            fprintf(file, " %d:%5.7lf", j, dptr[j]);
        fprintf(file,"\n");
    }

    for (int i=0;i<samples[1].size();i++)//neg
    {
        fprintf(file, "-1");
        const double *dptr=samples[1][i].feat.ptr<double>(0);
        for (int j=0;j<samples[1][i].feat.cols;j++)
            if (dptr[j]!=0)
            fprintf(file, " %d:%5.7lf", j, dptr[j]);
        fprintf(file,"\n");
    }

    fclose(file);
    char cmd[MAX_FILENAME];
    sprintf(cmd, "./%s/svm_train -s 0 -t 0 -c 5 %s/%s/%s %s/%s/%s.model", opt.current_path.c_str(), opt.current_path.c_str(), CACHE_DIR ,model_name, opt.current_path.c_str(), CACHE_DIR ,model_name);
    system(cmd);
    //command line
    return string(model_name);
}

//prepare data for train RankSVM
void person_recognize::prepare_rank(const vector<im_feature> & features,//i:mid-level features
                                    const vector<vector<feature> > & patch_feats, //i:low-level features
                                    int n,
                                    const option & opt)
{
    sortsupport *dist=new sortsupport[features.size()];
    for (int i=0;i<features.size();i++)
    {
            dist[i].filter_dist=gauss_filter_dist(features[n],features[i]);
            dist[i].patch_dist=gauss_patch_dist(patch_feats[n], patch_feats[i]);
            dist[i].index=i;
    }

    char filename[MAX_FILENAME];
    sprintf(filename, "%s/%s/Rank_train", opt.current_path.c_str(), CACHE_DIR);
    FILE *file;
    file=fopen(filename, "a+");
    sort(dist, dist+features.size());
    double temp1, temp2;
    int tt;
    for (int i=0;i<opt.nRank_neg;i++)
    {
        if (train_label[dist[i].index]==train_label[n])
            continue;
        tt=last_node[n];
        while (tt!=-1)
        {
            temp1=dist[samples[tt].im_index].filter_dist, temp2=dist[samples[tt].im_index].patch_dist;
            if (temp1>dist[i].filter_dist&&temp2>dist[i].patch_dist)
            {
                fprintf(file, "+1 1:%5.7lf 2:%5.7lf\n", temp1-dist[i].filter_dist, temp2-dist[i].patch_dist);
                fprintf(file, "-1 1:%5.7lf 2:%5.7lf\n", dist[i].filter_dist-temp1, dist[i].patch_dist-temp2);
            }
            tt=samples[tt].next;
        }
        fprintf(file, "+1 1:%5.7lf 2:%5.7lf\n", 1-dist[i].filter_dist, 1-dist[i].patch_dist);
        fprintf(file, "-1 1:%5.7lf 2:%5.7lf\n", dist[i].filter_dist-1, dist[i].patch_dist-1);
    }

    delete[] dist;
    fclose(file);
}

string person_recognize::train_ranksvm(const option &opt)
{
    char cmd[MAX_FILENAME];
    sprintf(cmd, "./%s/ranksvm_train -s 0 -t 0 %s/%s/Rank_train %s/%s/Rank_train.model", opt.current_path.c_str(), opt.current_path.c_str(), CACHE_DIR, opt.current_path.c_str(), CACHE_DIR);
    system(cmd);
    return string("Rank_train.model");
}

void person_recognize::pre_mid_filter(SVM_data & data,//o:data in svm-style
                                      const vector<feature> & f_image,//i:low-level features
                                      const option & opt)
{

    for (int i=0;i<f_image.size();i++)
    {
        const double *dptr=f_image[i].feat.ptr<double>(0);
        for (int j=0;j<f_image[i].feat.cols;j++)
            if (dptr[j]!=0||j==0)
            {
                data.L_dim[data.L_items]++;
                data.data[data.count]= dptr[j];
                data.index[data.count]=j+1;
                data.count++;
            }
        data.L_items++;
    }
}

//extract mid-level features
void person_recognize::exec_mid_filter(vector<im_feature> &im_f,//o
                                       const vector<vector<string> >& model_name,//i:linear SVMs
                                       const option & opt,
                                       int start,//i:start image, start and end are prepare for parallel(not appear in this version)
                                       int end,
                                       int * & labels,//temp array
                                       int max_images)
{
    int sum=0;

    int sizes[3]={max_images, MAX_HEIGHT, MAX_NODES};
    memset(labels, 0, sizeof(int)*max_images*MAX_HEIGHT*MAX_NODES);
    int  count=0, im_count=start, lsum=0;
    double label=0;
    for (int i=0;i<opt.nstripe;i++)
    {
        for (int j=0;j<model_name[i].size();j++)
        {
            count=0, im_count=0, lsum=0;
            svm_test(models[sum+j], svm_data.L_items, svm_data.L_dim, svm_data.index, svm_data.data, svm_data.output);//libSVM
            for (int k=0;k<svm_data.L_items;k++)
            {
                label=svm_data.output[k];
                if (label>0&&count>=(i-opt.h)*opt.Nx/opt.nstripe&&count<(i+opt.h+1)*opt.Nx/opt.nstripe)
                    *DIM3(labels, sizes, im_count,i,j)=*DIM3(labels, sizes, im_count,i,j)+1;
                lsum+=(label>0);
                count++;
                if (count==opt.Nx*opt.Ny)
                {
                    count=0;
                    im_count++;
                }
            }
            if (1.0*lsum/((end-start)*opt.Nx*opt.Ny)>0.5)
            {
                for (int t=start;t<end;t++)
                    *DIM3(labels, sizes, t, i, j)=0;
            }
        }
        sum+=model_name[i].size();
    }
    int temp=0;
    for (int j=start;j<end;j++)
    {
            Mat result(1, sum, CV_64FC1);
            double *dptr=result.ptr<double>(0);
            //int temp=0;
            temp=0;
            for (int t1=0;t1<opt.nstripe;t1++)
                for (int t2=0;t2<model_name[t1].size();t2++)
                {
                    dptr[temp++]=*DIM3(labels, sizes, j, t1, t2);
                }
            im_f[j].feat=result;
            im_f[j].im_index=j;
    }
}

//Read data from file and extract features
vector<vector<feature> > person_recognize::Read_Extract(frame & data , //i:origin image
                                                        vector<bbox> boxes, //i:regions
                                                        const option & opt,
                                                        int &count)
{
    vector<vector<feature> > result;
    vector<bbox> my_boxes;
    count=0;
    for (int i=0;i<boxes.size();i++)
        if (boxes[i].type_label==TYPE_PERSON) //type_label=2 means it's a bounding-box for person
        {
            result.push_back(vector<feature>());
            my_boxes.push_back(boxes[i]);
            box_index[count]=i;
            count++;
        }
#pragma omp parallel for
   for (int i=0;i<my_boxes.size();i++)
    {
       Mat Image;
       Mat im_data(data.image(my_boxes[i]));
       Mat box_data(120,40, CV_64FC3);//not appear in next version, the trainning dataset are 128x48 pixels
       resize(im_data, box_data, box_data.size(),0, 0, CV_INTER_NN);
       Image=norm_image(box_data, opt);
       result[i]=get_features(Image, opt, i);
    }
    return result;
}

//read and extract testing data(gallery) from file
vector<vector<feature> > person_recognize::Read_Extract(const char * filename,
                                                        const option & opt,
                                                        int &count)
{
    FILE * file=fopen(filename, "r");
    vector<vector<feature> > result;
    char str1[MAX_FILENAME], str2[MAX_FILENAME];
    Mat Image;
    count=0;
    while (fscanf(file, "%s", str2)!=EOF)
    {
        sprintf(str1, "%s", str2);
        Mat temp;
        temp=imread(str1, -1);
        Image.create(120, 40, CV_64FC3);
        resize(temp, Image, Image.size());
        Image=norm_image(Image, opt);
        result.push_back(get_features(Image, opt, count));
        count++;
        cout << str1 << endl;
    }
    fclose(file);
    return result;
}

//binary_search
int person_recognize::b_search(const vector<vector<sortAssistant> > &neighbor, //i:distance table
                               const feature & x)
{
    int l=0, r=neighbor.size(), mid=0;
    while (l<r)
    {
        mid=(l+r)>>1;
        if (neighbor[mid][0].im_index==x.im_index&&neighbor[mid][0].pos_index==x.pos_index)
            return mid;
        if (neighbor[mid][0].im_index<x.im_index||(neighbor[mid][0].im_index==x.im_index&&neighbor[mid][0].pos_index<x.pos_index))
            l=mid+1;
        else
            r=mid-1;
    }
    if (l<neighbor.size()&&neighbor[l][0].im_index==x.im_index&&neighbor[l][0].pos_index==x.pos_index)
        return l;
    else
        return r;
}

//calc square euclidean distance between tow patches
double person_recognize::patch_distance(double *p1,
                                        double *p2,
                                        int dim)
{
    double sum=0;
    for (int i=0;i<dim;i++)
    {
        sum=sum+(p1[i]-p2[i])*(p1[i]-p2[i]);
    }
    return sum;
}

//calc gauss distance for mid-level features
double person_recognize::gauss_filter_dist(const im_feature &a,
                                           const im_feature &b)
{
    double sum=0;
    const double *ptr1=a.feat.ptr<double>(0), *ptr2=b.feat.ptr<double>(0);
    double temp=0;
    for (int i=0;i<a.feat.cols;i++)
    {
        temp=ptr1[i]-ptr2[i];
        sum+=temp*temp;
    }
    temp=GAUSSF_COEF*exp(-(sum/(SIGMAF*SIGMAF)));
    return temp;
}

//calc gauss distance for patch features
double person_recognize::gauss_patch_dist(const vector<feature> &a,
                                          const vector<feature> &b)
{
    double sum=0, ans=0;
    const double *ptr1, *ptr2;
    double temp=0;
    for (int i=0;i<a.size();i++)
    {
        ptr1=a[i].feat.ptr<double>(0);
        ptr2=b[i].feat.ptr<double>(0);
        for (int j=0;j<a[i].feat.cols;j++)
        {
            if (a[i].sAUC>0&&b[i].sAUC>0)
                temp=a[i].sAUC*ptr1[j]-b[i].sAUC*ptr2[j];
            else
                temp=ptr1[j]-ptr2[j];
            sum+=temp*temp;
        }
        ans+=exp(-(sum/(SIGMAP*SIGMAP)));
    }
    if (ans!=a.size())
    {
        ans=ans*GAUSSP_COEF/a.size();
    }
    else
        ans=1;

    return ans;
}

bool person_recognize::cmp_sortsupport(sortsupport a,
                                       sortsupport b)
{
    return a.filter_dist+a.patch_dist>b.filter_dist+b.patch_dist;
}


//normalize mid-level features
void person_recognize::norm_im_f(vector<im_feature> & im_feat,
                                 const option & opt)
{
    double sum[MAX_GALLERY_IMAGES], sum_all=0;
    double *dptr;
    for (int i=0;i<im_feat.size();i++)
    {
        sum[i]=0;
        dptr=im_feat[i].feat.ptr<double>(0);
        for (int j=0;j<im_feat[i].feat.cols;j++)
            sum[i]+=*(dptr++);
        sum_all+=sum[i];
    }
    double temp=0;
    for (int i=0;i<im_feat.size();i++)
    {
        if (sum[i]!=0)
            temp=1.0/sum[i];
        else
            temp=0;
        dptr=im_feat[i].feat.ptr<double>(0);
        for (int j=0;j<im_feat[i].feat.cols;j++)
        {
            dptr[j]*=temp;
        }
    }
}


//normalize one image by hist equlalization
Mat person_recognize::norm_image(Mat &image,
                                 const option & opt)
{
    Mat temp_image(image.size(), CV_8UC3);
    vector<Mat> hsv_vec;
    //split origin image into three channels
    image.convertTo(temp_image, CV_8UC3, 1.0, 0);
    cvtColor(temp_image, temp_image, COLOR_BGR2HSV);
    split(temp_image, hsv_vec);

    //Histogram Equalize
    equalizeHist(hsv_vec[2], hsv_vec[2]);
    cv::merge(hsv_vec, temp_image);
    cvtColor(temp_image, temp_image, COLOR_HSV2RGB);
    return temp_image;
}


//get sift from one Image
Mat person_recognize::get_sift(Mat &Image,
                               const option & opt)
{
    int patch_size = opt.Patch_Size;
    Mat sigma_edge = opt.Sigma_edge;
    int Nx=opt.Nx;
    int Ny=opt.Ny;
    double stepx=(0.0+Image.size[0]-patch_size)/Nx;
    double stepy=(0.0+Image.size[1]-patch_size)/Ny;

    int num_angles = opt.num_angles;
    int num_bins = opt.sift_nBins;
    int alpha = opt.alpha;

    double angle_step = 2.0*M_PI/num_angles;
    double temp_angle=0;
    double angles[360];
    int count =0;
    while (temp_angle<=2.0*M_PI)
    {
        angles[count++]=temp_angle;
        temp_angle+=angle_step;
    }

    Mat G_X, G_Y;
    generate_dgauss(G_X, G_Y, sigma_edge); //wait modify
    Mat I_mag, I_theta, sinI, cosI, I_orientation, weight_x, single_arr, I_X, I_Y, sift_arr;
    I_mag.create(Image.rows, Image.cols, CV_64FC1);
    I_theta.create(Image.rows, Image.cols, CV_64FC1);
    sinI.create(Image.rows, Image.cols, CV_64FC1);
    cosI.create(Image.rows, Image.cols, CV_64FC1);
    weight_x.create(1, patch_size+1, CV_64FC1);
    int size1[3]={Image.rows,Image.cols, num_angles};
    int size2[3]={Nx*Ny, opt.color, num_angles*Image.rows*Image.cols};
    int size3[3]={Nx, Ny, num_angles*num_bins*num_bins};
    I_orientation.create(3,  size1, CV_64FC1);
    I_X.create(Image.rows, Image.cols, CV_64FC1);
    I_Y.create(Image.rows, Image.cols, CV_64FC1);
    sift_arr.create(3, size2, CV_64FC1);
    single_arr.create(3, size3, CV_64FC1);
    vector<Mat> lab;


    double temp=0, temp1=0, temp2=0, temp3=0;

    split(Image, lab);
    for (int ch=0;ch<opt.color;ch++)// channels
    {
        Image=lab[ch];
        filter2D(Image, I_X, I_X.depth(), G_X);
        filter2D(Image, I_Y, I_Y.depth(), G_Y);
        for (int count_row=0;count_row<Image.rows;count_row++)
        {
            for (int count_col=0;count_col<Image.cols;count_col++)
            {
                temp1=I_X.at<double>(count_row, count_col), temp2=I_Y.at<double>(count_row, count_col);
                I_mag.at<double>(count_row, count_col) = sqrt(temp1*temp1+temp2*temp2);
                temp3 = atan2(temp2, temp1);
                I_theta.at<double>(count_row, count_col)=temp3;
                cosI.at<double>(count_row, count_col)=cos(temp3);
                sinI.at<double>(count_row, count_col)=sin(temp3);
            }
        }

        for (int count_row=0;count_row<Image.rows;count_row++)
        {
            for (int count_col=0;count_col<Image.cols;count_col++)
            {
                for (int angle_count=0;angle_count<num_angles;angle_count++)
                {
                    temp1=cosI.at<double>(count_row, count_col), temp2=sinI.at<double>(count_row, count_col);
                    temp=temp1*cos(angles[angle_count])+temp2*pow(sin(angles[angle_count]), alpha);
                    temp*=temp>0;
                    I_orientation.at<double>(count_row,count_col,angle_count)=temp*I_mag.at<double>(count_row, count_col);
                }
            }
        }

        double r=patch_size/2.0;
        double cx=r-0.5;
        double sample_res=1.0*patch_size/num_bins;
        for (int i=1;i<=patch_size;i++)
        {
            temp=fabs(i-cx)/sample_res;
            temp=(1.0-temp)*(temp<=1);
            weight_x.at<double>(0, i)=temp;
        }
        Mat weight_xx=weight_x.t()*weight_x;
        Mat T_I;//wait debug
        T_I.create(size1[0], size1[1], CV_64FC1);

        for (int angle_count=0;angle_count<num_angles;angle_count++)
        {
            double *srcptr=I_orientation.ptr<double>(0)+angle_count;
            double *desptr=T_I.ptr<double>(0);
            for (int i=0;i<size1[0];i++)
                for (int j=0;j<size1[1];j++)
                    if (i!=size1[0]-1&&j!=size1[1]-1)   *(desptr++)=*(srcptr),srcptr+=num_angles;
            filter2D(T_I, T_I, T_I.depth(), weight_xx);
            srcptr=I_orientation.ptr<double>(0)+angle_count;
            desptr=T_I.ptr<double>(0);
            for (int i=0;i<size1[0];i++)
                for (int j=0;j<size1[1];j++)
                    if (i!=size1[0]-1&&j!=size1[1]-1)   *(srcptr)=*(desptr++),srcptr+=num_angles;
        }

        double sample_step=(patch_size-1.0)/(num_bins+1);
        double *sptr=single_arr.ptr<double>(0);
        for (int x=0;x<Nx;x++)
            for (int y=0;y<Ny;y++)
                for (int i=0;i<num_bins;i++)
                    for (int j=0;j<num_bins;j++)
                        for (int k=0;k<num_angles;k++)
                        {
                            int t1=cvCeil(patch_size/2.0+x*stepx+cvCeil(1+i*sample_step-patch_size/2.0)), t2=cvCeil(patch_size/2.0+y*stepy+cvCeil(1+j*sample_step-patch_size/2.0));
                            double aaa=I_orientation.at<double>(t2, t1, k);
                            //(*sptr)=I_orientation.at<double>(t2, t1, k);
                            (*sptr)=aaa;
                            sptr++;
                        }

        double ct = 0.000001;
        single_arr+=ct;
        double *sift_ptr=sift_arr.ptr<double>(0);
        for (int i=0;i<Nx;i++)
        {
            for (int j=0;j<Ny;j++)
            {
                double sum=0;
                for (int k=0;k<num_angles*num_bins*num_bins;k++)
                {
                    temp=single_arr.at<double>(i, j, k);
                    sum+=temp*temp;
                }
                sum=sqrt(sum);
                for (int k=0;k<num_angles*num_bins*num_bins;k++)
                {
                    temp=single_arr.at<double>(i, j, k);
                    if (sum!=0)
                        temp/=sum;
                    *(DIM3(sift_ptr,size2,i*Ny+j, ch, k))=temp;
                }
            }
        }
    }

    return sift_arr;
}

Mat person_recognize::generate_gauss(const Mat &sigma)
{
    int f_wid_x, f_wid_y;
    double s1, s2;
    if (sigma.cols>1)
    {
        s1=sigma.at<double>(0,0);
        s2=sigma.at<double>(0,1);
    }
    else
    {
        s1=sigma.at<double>(0,0);
        int f_wid=4*cvCeil(s1)+1;
        Mat G;
        G.create(1, f_wid, CV_64FC1);
        int t=0.0-f_wid/2;
        for (int i=0;i<f_wid;i++)
        {
            G.at<double>(0, i)=exp(-0.5*pow((t/s1),2))/(sqrt(2.0*M_PI)*s1);
            t++;
        }
        return G.t()*G;
    }
    f_wid_x=2*cvCeil(s1)+1;
    f_wid_y=2*cvCeil(s2)+1;
    Mat G_x, G_y;
    G_x.create(1, 2*f_wid_x, CV_64FC1);
    G_y.create(1, 2*f_wid_y, CV_64FC1);
    for (int i=-f_wid_x;i<=f_wid_x;i++)
    {
        G_x.at<double>(0, i+f_wid_x)=exp(-0.5*pow((i/s1),2))/(sqrt(2.0*M_PI)*s1);
    }
    for (int i=-f_wid_y;i<=f_wid_y;i++)
    {
        G_y.at<double>(0, i+f_wid_y)=exp(-0.5*pow((i/s2),2))/(sqrt(2.0*M_PI)*s2);
    }
    return G_y.t()*G_x;
}


Mat person_recognize::generate_gauss(double sigma)
{
    double s1;
    s1=sigma;
    int f_wid=4*cvCeil(s1)+1;
    Mat G;
    G.create(1, f_wid, CV_64FC1);
    int t=0.0-f_wid/2;
    for (int i=0;i<f_wid;i++)
    {
        G.at<double>(0, i)=exp(-0.5*pow((t/s1),2))/(sqrt(2.0*M_PI)*s1);
        t++;
    }
    Mat temp=G.t()*G;
    return G.t()*G;
}

void person_recognize::generate_dgauss(Mat &G_x,
                                       Mat &G_y,
                                       const Mat &sigma)
{
    Mat G=generate_gauss(sigma).clone();
    G_x.create(G.rows,G.cols, CV_64FC1);
    G_y.create(G.rows,G.cols, CV_64FC1);
    double sumx=0, sumy=0;
    for (int i=0;i<G.rows;i++)
    {
        for (int j=0;j<G.cols;j++)
        {
            int count=0;
            double t1=0;
            double t2=0;
            if (i-1>=0)
                t1=G.at<double>(i-1,j), count++;
            if (i+1<G.rows)
                t2=G.at<double>(i+1,j), count++;
            G_y.at<double>(i,j)=((t2-t1)/count);
            sumy+=fabs(G_y.at<double>(i,j));
            t1=0;
            t2=0;
            count=0;
            if (j-1>=0)
                t1=G.at<double>(i,j-1), count++;
            if (j+1<G.cols)
                t2=G.at<double>(i,j+1), count++;
            G_x.at<double>(i,j)=((t2-t1)/count);
            sumx+=fabs(G_x.at<double>(i,j));
        }
    }
    G_x=G_x*2.0/sumx;
    G_y=G_y*2.0/sumy;
}

void person_recognize::generate_dgauss(Mat &G_x,
                                       Mat &G_y,
                                       double sigma)
{
    Mat G=generate_gauss(sigma).clone();
    G_x.create(G.rows,G.cols, CV_64FC1);
    G_y.create(G.rows,G.cols, CV_64FC1);
    double sumx=0, sumy=0;
    for (int i=0;i<G.rows;i++)
    {
        for (int j=0;j<G.cols;j++)
        {
            int count=0;
            double t1=0;
            double t2=0;
            if (i-1>=0)
                t1=G.at<double>(i-1,j), count++;
            if (i+1<G.rows)
                t2=G.at<double>(i+1,j), count++;
            G_y.at<double>(i,j)=((t2-t1)/count);
            sumy+=fabs(G_y.at<double>(i,j));
            t1=0;
            t2=0;
            count=0;
            if (j-1>=0)
                t1=G.at<double>(i,j-1), count++;
            if (j+1<G.cols)
                t2=G.at<double>(i,j+1), count++;
            G_x.at<double>(i,j)=((t2-t1)/count);
            sumy+=fabs(G_y.at<double>(i,j));
        }
    }
    G_x=G_x*2.0/sumx;
    G_y=G_y*2.0/sumy;
}

//get_features for each Image
vector<feature> person_recognize::get_features(Mat &I,
                                               const option & opt,
                                               int im_index)
{
    // color features


    Mat I_1=rgb2lab_1(I);
    Mat result_color =get_colorhist(I_1, opt);

    // sift features


    Mat I_2=rgb2lab_2(I);
    Mat result_sift = get_sift(I_2, opt);

    //merge two features
    int size[3]={opt.Nx*opt.Ny, opt.color_nBins*I.channels()*opt.nScale,opt.sift_nBins*opt.sift_nBins*opt.num_angles*opt.color};
    double *cptr=result_color.ptr<double>(0), *sptr=result_sift.ptr<double>(0);
    vector<feature> result;
    result.reserve(size[0]);
    for (int i=0;i<size[0];i++)
    {
        Mat temp;
        temp.create(1,size[1]+size[2], CV_64FC1);
        double *tt=temp.ptr<double>(0);
        for (int j=0;j<size[1];j++)
            *(tt++)=*(cptr++);
        for (int j=0;j<size[2];j++)
            *(tt++)=*(sptr++);
        feature ftemp(temp, im_index, i, i/opt.Ny);
        result.push_back(ftemp);
    }

    return result;
}

Mat person_recognize::rgb2lab_1(const Mat & I)
{
    vector<Mat> rgb, lab;
    Mat I_F;
    I.convertTo(I_F, CV_64FC1);
    split(I_F, rgb);
    Mat ans_lab;
    lab.push_back(Mat());
    lab.push_back(Mat());
    lab.push_back(Mat());
    lab[0].create(rgb[0].rows, rgb[0].cols, CV_64FC1);
    lab[1].create(rgb[1].rows, rgb[1].cols, CV_64FC1);
    lab[2].create(rgb[2].rows, rgb[2].cols, CV_64FC1);
    lab[0]=(rgb[0]-rgb[1])/sqrt(2);
    lab[1]=(rgb[0]+rgb[1]-2*rgb[2])/sqrt(6);
    lab[2]=(rgb[0]+rgb[1]+rgb[2])/sqrt(3);
    cv::merge(lab, ans_lab);
    return ans_lab;
}

void person_recognize::norm(Mat & I)
{
    double min=255, max=0;
    for (int i=0;i<I.size[0];i++)
        for (int j=0;j<I.size[1];j++)
        {
            double temp=I.at<double>(i,j);
            if (min>temp)
            {
                min=temp;
            }
            if (max<temp)
            {
                max=temp;
            }
        }
    I=(I-min)/(max-min);
}

Mat person_recognize::rgb2lab_2(const Mat & I)
{
    Mat I_lab, I_flab;
    I_lab.create(I.rows, I.cols, I.type());
    vector<Mat> lab;
    cvtColor(I, I_lab, CV_RGB2Lab);
    I_lab.convertTo(I_flab, CV_64FC1);
    split(I_flab, lab);
    Mat ans_lab;
    ans_lab.create(I_flab.rows, I_flab.cols, I_flab.type());
    norm(lab[0]);
    norm(lab[1]);
    norm(lab[2]);
    cv::merge(lab, ans_lab);
    return ans_lab;
}

//get color histogram features for each Image
Mat person_recognize::get_colorhist(Mat &Image,
                                    const option & opt)
{
    int patch_size = opt.Patch_Size;
    int Nx=opt.Nx, Ny=opt.Ny;
    const double *scale = opt.Scale;
    int nScale = opt.nScale;
    double clamp = opt.clamp;
    Mat sigma = opt.Sigma;
    int num_bins =opt.color_nBins;
    double epsi = opt.Epsi;
    double hist[NUM_BINS]; //Warning!
    //double *hist=new double[num_bins];
    Mat color_features;
    vector<Mat> lab;
    split(Image, lab);
    int nch=Image.channels();
    int size1[4]={Nx*Ny, nScale, nch, num_bins};
    color_features.create(4, size1, CV_64FC1);


    Mat G, Image_s, ROI;
    double temp=0;
    int t1=0, t2=0;
    for (int i=0;i<nScale;i++)//scale is base on the origin image ,not the last one
    {
        for (int j=0;j<nch;j++)//n channels
        {

            Image=lab[j];
            if (scale[i]!=1)
            {
                G=generate_gauss(sigma.at<double>(0,0)/scale[i]); //wait modify
                filter2D(Image, Image, Image.depth(), G);
            }

            Image_s.create(Image.rows*scale[i], Image.cols*scale[i], CV_64FC1);


            resize(Image, Image_s, Image_s.size(), 0, 0, CV_INTER_NN);

            int patch_size_s=patch_size*scale[i];
            double half_patch=(patch_size_s-1.0)/2;

            double stepx=((0.0+Image_s.size[0]-patch_size_s)/Nx);
            double stepy=((0.0+Image_s.size[1]-patch_size_s)/Ny);

            for (int row=0;row<Nx;row++)
            {
                for (int col=0;col<Ny;col++)
                {
                    t1=cvCeil(cvCeil(patch_size_s/2.0+stepx*row)-half_patch+1);
                    t2=cvCeil(cvCeil(patch_size_s/2.0+stepy*col)-half_patch+1);
                    //vps warning!
                    ROI=Image_s(Rect(t2-1, t1-1, patch_size_s-2, patch_size_s-2));//select a small region

                    memset(hist, 0, sizeof(double)*num_bins);
                    colorHist(hist, ROI, num_bins, j+1);

                    //            L2 Normalization
                    //            hist = hist./scale_s;
                    //            norm_tmp = hist/sqrt(sum(hist.^2)+epsi^2);
                    double sum=0;
                    for (int count=0;count<num_bins;count++)
                    {
                        hist[count]/=scale[i];
                        sum+=hist[count]*hist[count];
                    }
                    sum+=epsi*epsi;

                    //            norm_tmp(norm_tmp >= clamp) = clamp;
                    //            norm_tmp = norm_tmp/sqrt(sum(norm_tmp.^2)+epsi^2);
                    double nsum=0;
                    for (int count=0;count<num_bins;count++)
                    {
                        temp=hist[count]/sqrt(sum);
                        if (temp>clamp)
                            temp=clamp;
                        nsum+=temp*temp;
                        hist[count]=temp;
                    }
                    nsum+=epsi*epsi;

                    double *pos=color_features.ptr<double>();
                    for (int count=0;count<num_bins;count++)
                    {
                        temp=hist[count]/sqrt(nsum);
                        hist[count]=temp;
                        *(DIM4(pos, size1, row*Ny+col, i, j, count))=temp;
                    }
                }
            }
        }
    }
    //delete []hist;
    return color_features;
}

//calc colorHist
void person_recognize::colorHist(double * hist,//i
                                 Mat &cvt,//o
                                 double K,//i
                                 int dim)//i
{
    double mini, maxi;
    if (dim==1)
    {
        mini=-255/sqrt(2);
        maxi=255/sqrt(2);
    }
    else
    {
        if (dim==2)
        {
            mini=-510/sqrt(6);
            maxi=510/sqrt(6);
        }
        else
        {
            mini=0;
            maxi=765;
        }
    }
    for (int i=0;i<cvt.rows;i++)
    {
        for (int j=0;j<cvt.cols;j++)
        {
            double temp=cvt.at<double>(i,j);
            int temp1=cvFloor((K+1e-8)*(temp-mini)/(maxi-mini))+1;
            if (temp1>K)
                temp1=K;
            hist[temp1-1]++;
        }
    }

}


//-------------------------------------------------//


// 'feature' is a class, contain image info. and image features


//-------------------------------------------------//

void person_recognize::clustering( vector<feature>& feats, // i: all features distracted from patches
                                   vector< vector<sortAssistant> >& KNN_array, // o: KNN of features respect, for further use
                                   vector< vector<feature> >& cluster_result, // o: clustered features,
                                   const option& opt )
{
    // Zhang W, Wang X, Zhao D, et al.
    // Graph degree linkage: Agglomerative clustering on a directed graph
    // implemented by: Yanye.Li from Fudan University, yyli12@fudan.edu.cn

    int numOfFeats;

    Mat allWeight;
    list<cluster> result;
    vector<feature> mid_level_feats;

    midLevelSelect( feats, mid_level_feats, opt );
    /* select those feats which are mid-level
     * mid-level: the sum of KNN-dist local in the 5 level ( out of 10 )
     *
     * do cluster base on the mid-level features (patches)
     */

    numOfFeats = mid_level_feats.size();
    allWeight.create( numOfFeats, numOfFeats, CV_64FC1 );

    for( int i = 0; i < numOfFeats; i++ )
        mid_level_feats[i].index = i;

    double* weight;
    double sigmaSquare = 0.0;
    weight = (double*)allWeight.data;

    calcAllPairDist( weight, mid_level_feats, &sigmaSquare, opt.dist_threshold_KNN_for_affinity, KNN_array );
    /* ( not the final weights )
     * w_ij = dist(i,j)^2, if j in KNN of i;
     *      = 0, otherwise.
     */

    /* debug code
     * ofstream KNN_file("KNN.txt");
    for( int i = 0; i < numOfFeats; i++ )
    {
        for( int j = 0; j < KNN_array[i].size() ; j++ )
            KNN_file << KNN_array[i][j].index << " " << KNN_array[i][j].im_index << " " << KNN_array[i][j].pos_index << " " << KNN_array[i][j].dist << " ";
        KNN_file << endl;
    } */

    calcWeight( weight, numOfFeats, opt.dist_threshold_KNN_for_affinity, &sigmaSquare, opt.scaling_a );
    /* w_ij is final weight of edge(i,j)
     * w_ij = exp( - dist(i,j)^2 / sigma^2 ), if j in KNN of i;
     *      = 0, otherwise.
     */


    hierachy( mid_level_feats, allWeight, result, opt );
    for( auto i : result )
    {
        vector<feature> temp;
        for( auto j : i.feats )
            temp.push_back( j );

        cluster_result.push_back( temp );
    }

    // do hierachy-cluster, rough -> fine


}

void person_recognize::hierachy( vector<feature>& allFeats, // i: all features at 0-level
                                 Mat& allWeight, // i: precalculated weights b/w features
                                 list<cluster>& result, // o: clustered features
                                 const option& opt )
{
    list<cluster> buffer[2];
    list<cluster> allCluster;
    int a, b;
    cluster clst;
    initialCluster( allFeats, allCluster, opt );
    // aggregate similar features as initial cluster

    doCluster( allCluster, opt.split_into, allWeight );
    for( auto &i : allCluster )
        buffer[0].push_back( i );

    for( int i = 1; i < opt.hierachy_level; i++ )
    {
        // 10-level of cluster
        a = i % 2;
        b = a == 0 ? 1 : 0;

        while( !buffer[a].empty() )
        {
            clst = buffer[a].back();
            buffer[a].pop_back();
            if( clst.feats.size() < opt.lower_bound )
                continue; // abandon cluster with few features
            /*else if( clst.feats.size() < 30 )
                result.push_back( clst ); // do not split cluster with 15~29 features
            */
            else
            {
                // split the cluster in to 4 smaller cluster
                list<cluster> split;

                for( auto i : clst.feats )
                {
                    cluster newC;
                    newC.feats.push_back( i );
                    split.push_back( newC );
                }

                doCluster( split, opt.split_into, allWeight );
                for( auto i : split )
                    buffer[b].push_back( i );
            }
        }
    }
    for( auto i : buffer[0] )
        if( i.feats.size() >= opt.lower_bound && i.feats.size() <= opt.upper_bound )
            result.push_back( i );
    for( auto i : buffer[1] )
        if( i.feats.size() >= opt.lower_bound && i.feats.size() <= opt.upper_bound )
            result.push_back( i );

}





void person_recognize::doCluster( list<cluster>& allCluster, // i: clusters wait to be aggregated
                                  int k, // o: target No. of final clusters
                                  Mat& w )
{
    // combine all cluster until ( No. of cluster == k )
    priority_queue<affinityNode> Q;
    double affi;

    // valid bit of cluster, 1: in Q, 0: not in Q.
    vector<bool> valid;

    // save the addr of specified cluster in list<cluster>
    vector<cluster*> ref;

    /* initial priority-queue Q with all possible pairs of all clusters
     * key of Q is affinity b/w two cluster
     */
    for( list<cluster>::iterator i = allCluster.begin(); i != allCluster.end(); i++ )
    {
        int i_id = valid.size();
        int j_id = i_id;
        // initiate the assistant vector 'valid' and 'ref'
        valid.push_back( true );
        ref.push_back( &(*i) );

        (*i).index = i_id;

        for( auto j = i; j != allCluster.end(); j++, j_id++ )
        {
            // calculate all pairs
            if( j == i ) continue;
            (*j).index = j_id;
            double affi = affinity( &(*i), &(*j), w );
            affinityNode n( i_id, j_id, affi );
            Q.push( n );
        }

    }


    while( allCluster.size() > k )
    {
        /* distract pair of cluster with highest affinity and merge them
         * until number of clusters is fewer than k
         */
        affinityNode n = Q.top();
        Q.pop();

        // cluster #n.a and #n.b are both valid
        bool found = valid[n.a] && valid[n.b];
        bool notTooLarge = true; // reserve flag to control the size of cluster

        if( found && notTooLarge )
        {
            valid[n.a] = valid[n.b] = false;
            // pop from Queue, make them invalid
            merge( ref[n.a], ref[n.b] );
            ref[n.a]->index = valid.size();
            ref.push_back( ref[n.a] );
            n.a = ref[n.a]->index;
            // push newly-merged cluster into Q, and create valid bit and ref record.
            valid.push_back( true );

            // delete cluster #n.b, because merged to cluster #n.a
            allCluster.remove( *ref[n.b] );
            for( auto i : allCluster )
            {
                if( n.a == i.index ) continue;
                double affi = affinity( ref[n.a], ref[i.index], w );
                affinityNode nn( n.a, i.index, affi );
                Q.push( nn );
            }
        }
    }

}


void person_recognize::initialCluster( vector<feature>& feats, // i: all input features
                                       list<cluster>& allCluster, // o: initialized clusters
                                       const option& opt )
{
    /* initial cluster with feature-graph
     * use BFS to find weak connected component in the graph,
     * each WCC is a cluster
     */
    int numOfFeats = feats.size();
    Mat allWeight;
    allWeight.create( numOfFeats, numOfFeats, CV_64FC1 );


    double* weight = (double*)allWeight.data;
    double s; // useless here

    calcAllPairDist( weight, feats, &s, opt.dist_threshold_KNN_for_init_cluster );

    Mat trans = allWeight.t();

    Mat m = trans | allWeight;

    bool* visited = new bool[ numOfFeats ];
    memset( visited, 0, numOfFeats*sizeof(bool) );

    for( int i = 0; i < numOfFeats; i++ )
    {
        if( !visited[i] )
        {
            cluster newCluster;
            BFS( i, newCluster, feats, visited, (double*)m.data );
            allCluster.push_back( newCluster );
        }
    }

    delete[] visited;

}

void person_recognize::BFS( int start, // i: start od BFS
                            cluster& c, // i/o: newly-generated cluster during the process of BFS
                            vector<feature>& feats, // i: all features
                            bool* visited, // i/o: visited flags
                            double* w ) // i: edge info.
{
    // use BFS to find weakly connected component (WCC)
    int numOfFeats = feats.size();
    queue<int> Q;
    int f;
    Q.push( start );
    while( !Q.empty() )
    {
        f = Q.front();
        Q.pop();
        if( !visited[f] )
        {
            c.feats.push_back( feats[f] );
            visited[f] = true;
            for( int i = 0; i < numOfFeats; i++ )
            {
                if( !visited[i] && ( w[f*numOfFeats+i] != 0) )
                    Q.push(i);
            }
        }
    }
}



int* person_recognize::getCluster( cluster& c ) // i: cluster input
// o: a int array of features' index
{
    int* f = new int[c.feats.size()];
    int i = 0;
    for( auto & a : c.feats )
        f[i++] = a.index;
    return f;
}

void person_recognize::merge( cluster* a, cluster* b ) // i: two clusters need to be merged
{
    a->feats.splice( a->feats.end(), b->feats );
}

double person_recognize::affinity( cluster* a, cluster* b, // i: two clusters
                                   Mat& w ) // i: weights b/w features
{
    // matrix-operation to get affinity b/w two clusters

    Mat Wab;
    Mat Wba;
    int* lista = getCluster( *a );
    int* listb = getCluster( *b );
    int Na = a->feats.size();
    int Nb = b->feats.size();
    Wab.create( Na, Nb, CV_64FC1 );
    Wba.create( Nb, Na, CV_64FC1 );

    // get weight b/w specified nodes, and store them in Mat Wab and Wba
    for( int i = 0; i < Na; i++ )
    {
        int row = lista[i];
        for( int j = 0; j < Nb; j++ )
        {
            Wab.at<double>(i,j) = w.at<double>(row,listb[j]);
        }
    }

    for( int i = 0; i < Nb; i++ )
    {
        int row = listb[i];
        for( int j = 0; j < Na; j++ )
        {
            Wba.at<double>(i,j) = w.at<double>(row,lista[j]);
        }
    }

    // vector ones with length Na and Nb
    Mat ones_a( Na, 1, CV_64FC1, Scalar(1.0) );
    Mat ones_b( Nb, 1, CV_64FC1, Scalar(1.0) );

    // calculate affinity, results m1 and m2 are both 1x1 matrix
    Mat m1 = ones_a.t() * Wab * Wba * ones_a;
    Mat m2 = ones_b.t() * Wba * Wab * ones_b;

    delete []lista;
    delete []listb;
    return  m1.at<double>(0,0) / (Na*Na) + m2.at<double>(0,0) / (Nb*Nb);

}

void person_recognize::calcAllPairDist( double* w, // o: weights b/w features ( Euclidean dist, not final result )
                                        vector<feature>& f, // i:all features
                                        double* s, // o: squared sigma
                                        int k_KNN ) // i: k for KNN
{
    /* ( not the final weights )
     * w_ij = dist(i,j)^2, if j in KNN of i;
     *      = 0, otherwise.
     */
    int n = f.size();
    sortAssistant* neighbor = new sortAssistant[n];
    for( int i = 0; i < n; i++ )
    {
#pragma omp parallel for
        for( int j = 0; j < n; j++ )
        {
            neighbor[j].index = j;
            neighbor[j].dist = getEuclidDistSquare( f[i], f[j] );
        }

        std::sort( neighbor, neighbor+n);

        for( int j = 0; j < n; j++ )
        {
            if( j < k_KNN + 1 )
            {
                *s += neighbor[j].dist;
            }
            else
                neighbor[j].dist = 0.0;
            w[n*i+neighbor[j].index] = neighbor[j].dist;

        }

    }
    delete [] neighbor;
}

void person_recognize::calcAllPairDist( double* w, // o: weights b/w features ( Euclidean dist, not final result )
                                        vector<feature>& f, // i:all features
                                        double* s, // o: squared sigma
                                        int k_KNN, // i: k for KNN
                                        vector< vector<sortAssistant> >& KNN_array ) // o: KNN of features respect
{
    /* ( not the final weights )
     * w_ij = dist(i,j)^2, if j in KNN of i;
     *      = 0, otherwise.
     */
    int n = f.size();
    sortAssistant* neighbor = new sortAssistant[n];
    sortAssistant first;
    for( int i = 0; i < n; i++ )
    {
#pragma omp parallel for
        for( int j = 0; j < n; j++ )
        {
            neighbor[j].im_index = f[j].im_index;
            neighbor[j].pos_index = f[j].pos_index;
            neighbor[j].index = j;
            neighbor[j].dist = getEuclidDistSquare( f[i], f[j] );
        }

        std::sort( neighbor, neighbor+n);

        vector<sortAssistant> vec;
        KNN_array.push_back( vec );
        first.dist=0;
        first.im_index=f[i].im_index;
        first.index=i;
        first.pos_index=f[i].pos_index;
        KNN_array[i].push_back(first);
        for( int j = 0; j < n; j++ )
        {
            if (neighbor[j].im_index==f[i].im_index&&neighbor[j].pos_index==f[i].pos_index)
                continue;
            if( j < k_KNN + 1 )
            {
                *s += neighbor[j].dist;
                KNN_array[i].push_back( neighbor[j] );
            }
            else
                neighbor[j].dist = 0.0;

            w[n*i+neighbor[j].index] = neighbor[j].dist;

        }

    }
    delete [] neighbor;
}


void person_recognize::calcWeight( double* w, // i/o: precalculated Euclidean dist, further calculation here
                                   int n, // i: No. of features
                                   int KNN, // i: k for KNN
                                   double* s, // i: squared sigma
                                   double scaling_a ) // i: scaling parameter a
{
    /* w_ij is final weight of edge(i,j)
     * w_ij = exp( - dist(i,j)^2 / sigma^2 ), if j in KNN of i;
     *      = 0, otherwise.
     */
    *s = scaling_a * (*s) / n / KNN;
    for( int i = 0; i < n; i++ )
    {
        for( int j = 0; j < n; j++ )
        {
            double* weigh = w + n*i+j;
            if( *weigh == 0.0 )
                continue;
            else
                *weigh = exp( 0 - *weigh / *s );

        }
    }

}

double person_recognize::getEuclidDistSquare( feature& a, feature& b ) // i: two features
{
    // calculate squared euclidean-distance b/w feat.a and feat.b
    double *ptr1, *ptr2, sum=0;
    ptr1=a.feat.ptr<double>(0);
    ptr2=b.feat.ptr<double>(0);

    int l=a.feat.cols;
    if (l==1)
        l=a.feat.rows;
    // row vector or column vector

    for (int i=0;i<l;i++)
    {
        sum=sum+(ptr1[i]-ptr2[i])*(ptr1[i]-ptr2[i]);
    }
    return sum;
}

void person_recognize::midLevelSelect( vector<feature>& all_feats, // i: all features
                                       vector<feature>& mid_level_feats, // i: selected mid-level features
                                       const option& opt )
{
    /* mid-level: the sum of KNN-dist local in the 5 level ( out of 10 )
     */

    int numOfFeats = all_feats.size();
    double* sum = new double[ numOfFeats ];
    double* dist = new double[ numOfFeats ];

    for( int i = 0; i < numOfFeats; i++ )
    {
        double tsum=0.0;
#pragma omp parallel for
        for( int j = 0; j < numOfFeats; j++ )
            dist[j] = getEuclidDistSquare( all_feats[i], all_feats[j] );
        sort( dist, dist + numOfFeats );
#pragma omp parallel for reduction(+:tsum)
        for( int j = 0; j < (int)(opt.dist_threshold_KNN_rate_for_midselect*numOfFeats); j++ )
            tsum += dist[j];
        sum[i]=tsum;
    }

    double max = -1.0;
    double min = 99999.9;
    for( int i = 0; i < numOfFeats; i++ )
    {
        if( sum[i] > max )
            max = sum[i];
        else if( sum[i] < min )
            min = sum[i];
    }
    for (int i=0;i<numOfFeats;i++)
    {
        all_feats[i].sAUC=(sum[i]-min)*10.0/(max-min);
    }

    // selecet mid-level features
    double step = ( max - min ) / 10;
    double lower = min + opt.seleted_level*step, upper = lower + step;//+step*2
    int cnt = 0;
    for( int j = 0; j < numOfFeats; j++ )
        if( sum[j] >= lower && sum[j] <= upper )
        {
            mid_level_feats.push_back( all_feats[j] );
            cnt++;
        }
    delete []sum;
    delete []dist;

}
