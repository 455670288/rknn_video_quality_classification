#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include "resnet.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "resize_function.h"

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

typedef struct {
    float value;
    int index;
} element_t;

void swap(element_t* a, element_t* b) {
    element_t temp = *a;
    *a = *b;
    *b = temp;
}

int partition(element_t arr[], int low, int high) {
    float pivot = arr[high].value;
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j].value >= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quick_sort(element_t arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}


void sigmoid(float* array, int size, resnet_result* result_cls) {
    for (int i = 0; i < size; ++i) {
        array[i] = 1.0f / (1.0f + std::exp(-array[i]));
        // std::cout << i << ": " << array[i] << std::endl;
        result_cls[i].cls = i;
        result_cls[i].score = array[i]; 
        // std::cout << result_cls[i].cls << ": " << result_cls[i].score << std::endl;
    }
}


void softmax(float* array, int size) {
    // Find the maximum value in the array
    float max_val = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
    }

    // Subtract the maximum value from each element to avoid overflow
    for (int i = 0; i < size; i++) {
        array[i] -= max_val;
    }

    // Compute the exponentials and sum
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        array[i] = expf(array[i]);
        sum += array[i];
    }

    // Normalize the array by dividing each element by the sum
    for (int i = 0; i < size; i++) {
        array[i] /= sum;
    }
}

void get_topk_with_indices(float arr[], int size, int k, resnet_result* result) {

    // 创建元素数组，保存值和索引号
    element_t* elements = (element_t*)malloc(size * sizeof(element_t));
    for (int i = 0; i < size; i++) {
        elements[i].value = arr[i];
        elements[i].index = i;
    }

    // 对元素数组进行快速排序
    quick_sort(elements, 0, size - 1);

    // 获取前K个最大值和它们的索引号
    for (int i = 0; i < k; i++) {
        result[i].score = elements[i].value;
        result[i].cls = elements[i].index;
    }

    free(elements);
}

Abnormal::AbnormalAnalysis::Impl::Impl(){
    app_ctx = (rknn_app_context_t*)calloc(1, sizeof(rknn_app_context_t));
}

Abnormal::AbnormalAnalysis::Impl::~Impl(){
    release_resnet_model();
    free(app_ctx);
}


int Abnormal::AbnormalAnalysis::Impl::init_resnet_model(const char* model_path,const int &NPUcore)
{
    int ret;
    int model_len = 0;
    char* model;
    rknn_context ctx = 0;


    //读取类别列表
    // lines = read_lines_from_file(classes_file_path, &line_count);
    // if (lines == NULL) {
    //     printf("read classes label file fail! path=%s\n", classes_file_path);
    //     return -1;
    // }   


    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // 设置模型绑定的核心/Set the core of the model that needs to be bound
    rknn_core_mask core_mask;
    switch (NPUcore)
    {
    case -1:
        core_mask = RKNN_NPU_CORE_AUTO;
        break;
    case 0:
        core_mask = RKNN_NPU_CORE_0;
        break;
    case 1:
        core_mask = RKNN_NPU_CORE_1;
        break;
    case 2:
        core_mask = RKNN_NPU_CORE_2;
        break;
    case 3:
        core_mask = RKNN_NPU_CORE_0_1_2;
        break;
    }
    ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);



    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height  = input_attrs[0].dims[2];
        app_ctx->model_width   = input_attrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height  = input_attrs[0].dims[1];
        app_ctx->model_width   = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
        app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int Abnormal::AbnormalAnalysis::Impl::release_resnet_model()
{
    if (app_ctx->rknn_ctx != 0) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    return 0;
}

int Abnormal::AbnormalAnalysis::Impl::inference_resnet_model(const cv::Mat &img, resnet_result* result_cls, int topk)
{
    int ret;
    rknn_input inputs[1];
    rknn_output outputs[1];

    //defualt initialized
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));
    

    //letter_box裁剪resize
    LETTER_BOX letter_box;
    letter_box.in_height = img.rows;
    letter_box.in_width = img.cols;
    letter_box.channel = img.channels();
    letter_box.target_width = app_ctx->model_width;
    letter_box.target_height = app_ctx->model_height;

    cv::Mat resized_img;
    if(img.cols != app_ctx->model_width || img.rows != app_ctx->model_height){
        compute_letter_box(&letter_box);
        letter_box.reverse_available = true;
        if (img.rows != letter_box.resize_height || img.cols != letter_box.resize_width)
        {
            cv::resize(img, resized_img, cv::Size(letter_box.resize_width, letter_box.resize_height));
            cv::copyMakeBorder(resized_img, resized_img, letter_box.h_pad_top, letter_box.h_pad_bottom, letter_box.w_pad_left,
                               letter_box.w_pad_right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        }
        else{
            cv::copyMakeBorder(img, resized_img, letter_box.h_pad_top,       
                               letter_box.h_pad_bottom, letter_box.w_pad_left,
                               letter_box.w_pad_right, cv::BORDER_CONSTANT,
                               cv::Scalar(114, 114, 114));           
        }
        inputs[0].buf = resized_img.data;        
    }
    else{
        inputs[0].buf = img.data;
    }
        
    
    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    // inputs[0].buf   = img.data;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    // printf("rknn_run\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }
    
    // Get Output
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
    }

    // Post Process
    // softmax((float*)outputs[0].buf, app_ctx->output_attrs[0].n_elems);
    // get_topk_with_indices((float*)outputs[0].buf, app_ctx->output_attrs[0].n_elems, topk, result_cls);
    
    //多标签分类后处理
    sigmoid((float*)outputs[0].buf, app_ctx->output_attrs[0].n_elems, result_cls);


    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);


    // for (int i = 0; i < topk; i++) {
    //     printf("[%d] score=%.6f class=%s\n", result_cls[i].cls, result_cls[i].score, classes_labels[result_cls[i].cls].c_str());
    // }


    return ret;
}