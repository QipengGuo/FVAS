#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "svm.h"


void exit_input_error(int line_num)
{
    fprintf(stderr,"Wrong input format at line %d\n", line_num);
    exit(1);
}

void predict(int length_item, int *length_dim, int * in_index, double * in_data, double *out_data, struct svm_node* & x, int & max_nr_attr, struct svm_model* & model)
{
    int correct = 0;
    int total = 0;
    double error = 0;
    double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

    int temp_i, temp_j;
    long in_count=0;
    for ( temp_i=0;temp_i<length_item;temp_i++)
    {
        double target_label, predict_label;
        int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

        target_label = -1;

        for(temp_j=0;temp_j<length_dim[temp_i];temp_j++)
        {
            if(temp_j>=max_nr_attr-1)	// need one more for index = -1
            {
                max_nr_attr *= 2;
                x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
            }
            errno = 0;
            x[temp_j].index = in_index[in_count];
            if (inst_max_index<x[temp_j].index)
                inst_max_index = x[temp_j].index;

            errno = 0;
            x[temp_j].value = in_data[in_count];
            in_count++;
        }
        x[temp_j].index = -1;
        predict_label = svm_predict(model,x);

        out_data[total]=predict_label;

        if(predict_label == target_label)
            ++correct;
        error += (predict_label-target_label)*(predict_label-target_label);
        sump += predict_label;
        sumt += target_label;
        sumpp += predict_label*predict_label;
        sumtt += target_label*target_label;
        sumpt += predict_label*target_label;
        ++total;
    }
}

void exit_with_help()
{
    printf(
    "Usage: svm-predict [options] test_file model_file output_file\n"
    "options:\n"
    "-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
    "-q : quiet mode (no outputs)\n"
    );
    exit(1);
}

int svm_test(struct svm_model * in_model, int length_item, int *length_dim, int * in_index, double * in_data, double *out_data)
{
    struct svm_node *x;
    int max_nr_attr = 64;

    struct svm_model* model;

    model=in_model;
    x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
    predict(length_item, length_dim, in_index, in_data, out_data, x, max_nr_attr, model);
    free(x);
    return 0;
}
