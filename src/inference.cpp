#include "inference.h"
#include <regex>

#define benchmark


YOLOV8_INFERENCE_CORE::YOLOV8_INFERENCE_CORE() {

}

YOLOV8_INFERENCE_CORE::~YOLOV8_INFERENCE_CORE() {
    delete session;
}

#ifdef USE_CUDA
namespace Ort
{
    template<>
    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}
#endif

template<typename T>
T *ImageToBlob(cv::Mat iImg, T &iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWight = iImg.cols;

    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < imgHeight; h++) {
            for (int w = 0; w < imgWight; w++) {
                iBlob[c * imgHeight * imgWight + h * imgWight + w] = typename std::remove_pointer<T>::type(
                        iImg.at<cv::Vec3b>(h, w)[c] / 255.0f);
            }
        }
    }
    return RET_OK;
}

Mat PostProcess(Mat &iImg, vector<int> imgSize) {
    cv::Mat oImage;
    Mat img = iImg.clone();
    cv::resize(img, oImage, cv::Size(imgSize.at(0), imgSize.at(1)));
    if (img.channels() == 1) {
        cv::cvtColor(oImage, oImage, COLOR_GRAY2BGR);
    }
    cvtColor(oImage, oImage, COLOR_BGR2RGB);
    return oImage;
}


char *YOLOV8_INFERENCE_CORE::CreatSession(YOLOV8_INIT_PARAM &iparams) {
    char *Ret = RET_OK;
    std::regex pattern ("[\u4e00-\u9fa5]");
    bool result = std::regex_search(iparams.ModelPath, pattern);
    if (result) {
        Ret = "[YOLOV8_INFERENCE_CORE]:Model path error.Change your model path without Chinese characters.";
        cerr << Ret << endl;
        return Ret;
    }

    try {
        rectConfidenceThreshold = iparams.RectConfidenceThreshold;
        iouThreshold = iparams.iouThreshold;
        modelType = iparams.ModelType;
        imgSize = iparams.ImgSize;
        env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Yolov8_inference");

        if (iparams.CUDAEnable) {
            cudaEnable = iparams.CUDAEnable;
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
        } else {cudaEnable = iparams.CUDAEnable;}

        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel ::ORT_ENABLE_ALL);
        sessionOptions.SetIntraOpNumThreads(iparams.IntraOpNumThreds);
        sessionOptions.SetLogSeverityLevel(iparams.LogSeverityLevel);

#ifdef _WIN32
        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.ModelPath.c_str(), static_cast<int>(iParams.ModelPath.length()), nullptr, 0);
        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
        MultiByteToWideChar(CP_UTF8, 0, iParams.ModelPath.c_str(), static_cast<int>(iParams.ModelPath.length()), wide_cstr, ModelPathSize);
        wide_cstr[ModelPathSize] = L'\0';
        const wchar_t* modelPath = wide_cstr;
#else
        const char *modelPath = iparams.ModelPath.c_str();
#endif

        session = new Ort::Session(env, modelPath, sessionOptions);
        Ort::AllocatorWithDefaultOptions allocator;

        size_t inputNodeCount = session->GetInputCount();
        for (size_t i = 0; i < inputNodeCount; i++)
        {
            Ort::AllocatedStringPtr  input_node_name = session->GetInputNameAllocated(i, allocator);
            char *temp_buf = new char[50];
            strcpy(temp_buf, input_node_name.get());
            inputNodeNames.emplace_back(temp_buf);
        }

        size_t outputNodeCount = session->GetOutputCount();
        for (size_t i = 0; i < outputNodeCount; i++) {
            Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
            char *temp_buf = new char[50];
            strcpy(temp_buf, output_node_name.get());
            outputNodeNames.emplace_back(temp_buf);
        }

        options = Ort::RunOptions{nullptr};
        //WarmUpSession();
        return RET_OK;
    }
    catch (const std::exception &e) {
        const char *str1 = "[YOLOV8_ORT_INFERENCE]:";
        const char *str2 = e.what();
        string result = string(str1) + string(str2);
        char *merged = new char[result.length() + 1];
        strcpy(merged, result.c_str());
        cerr << merged <<endl;
        delete [] merged;
        return "[YOLOV8_ORT_INFERENCE]:Create session failed.";
    }
}

char *YOLOV8_INFERENCE_CORE::RunSession(cv::Mat iImg, vector<YOLOV8_RESULT> &oResult) {
#ifdef benchmark
    clock_t starttime1 = clock();
#endif

    char *Ret = RET_OK;
    cv::Mat processImg = PostProcess(iImg, imgSize);

    if (modelType < 2) {
        float *blob  = new float [processImg.total() * 3];
        ImageToBlob(processImg, blob);
        vector<int64_t> inputNodeDims = {1, 3, imgSize.at(0), imgSize.at(1)};
        oResult = TensorProcess(starttime1, iImg, blob, inputNodeDims);
    }

}

template<typename N>
std::vector<YOLOV8_RESULT> YOLOV8_INFERENCE_CORE::TensorProcess(clock_t &starttime1, cv::Mat &iImg, N &blob, vector<int64_t> &inputNodeDims) {
//    1 创建 inputTensor
//    2 Run得到 outputTensor
//    3 处理 得到 信息

    vector<YOLOV8_RESULT> outResult;
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename remove_pointer<N>::type>(
                Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
                inputNodeDims.data(), inputNodeDims.size());
#ifdef benchmark
    clock_t starttime2 = clock();
#endif
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), outputNodeNames.size());

#ifdef benchmark
    clock_t starttime3 = clock();
#endif
    vector<int64_t> outputNodeDims = outputTensor.front().GetTensorTypeAndShapeInfo().GetShape();
    auto output = outputTensor.front().GetTensorMutableData<typename remove_pointer<N>::type>();

    delete blob;
    switch (modelType) {
        case YOLO_ORIGIN_V8:
        {
            int resultNum = outputNodeDims[2];
            int classNum = outputNodeDims[1];

            vector<int> class_ids;
            vector<float> confidences;
            vector<cv::Rect> boxes;

            cv::Mat rowData(classNum, resultNum, CV_32F, output);
            rowData = rowData.t();
            float *data = (float *)rowData.data;

            float x_factor = iImg.cols / 640.;
            float y_factor = iImg.rows / 640.;

            std::cout << iImg.cols << ' ' << x_factor << std::endl;

            for (int i = 0; i < resultNum; i++)
            {
                float *classSocre = data + 4;
                cv::Mat socre(1, this->classes.size(), CV_32F, classSocre);
                double maxConfidence;
                cv::Point class_id;
                cv::minMaxLoc(socre, 0, &maxConfidence, 0, &class_id);
                if (maxConfidence > rectConfidenceThreshold)
                {
                    confidences.push_back(maxConfidence);
                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    float left = ((x - 0.5 * w) * x_factor);

                    float top = ((y - 0.5 * h) * y_factor);
                    float width = w * x_factor;
                    float height = h * y_factor;

                    boxes.emplace_back(left, top, width, height);
                }
                data += classNum;
            }

            vector<int> nmsResult;
            cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);

            for (int i = 0; i < nmsResult.size(); i++){
                int idx = nmsResult[i];
                YOLOV8_RESULT res;
                res.classId = class_ids[idx];
                res.confidece = confidences[idx];
                res.box = boxes[idx];
                outResult.push_back(res);

            }
#ifdef benchmark
            clock_t starttime4 = clock();
            double pre_process_time = (double)(starttime2 - starttime1) / CLOCKS_PER_SEC * 1000;
            double process_time = (double)(starttime3 - starttime2) / CLOCKS_PER_SEC * 1000;
            double post_process_time = (double)(starttime4 - starttime3) / CLOCKS_PER_SEC * 1000;

            if (cudaEnable) {
                cout << "[YOLOV8_ORT(CUDA)]: " << pre_process_time << "ms pre_process_time, " << process_time <<
                process_time << "ms process_time, " << post_process_time << "ms post_process_time."<< endl;
            }
            else {
                cout << "[YOLOV8_ORT(CPU)]: " << pre_process_time << "ms pre_process_time, " << process_time <<
                     process_time << "ms process_time, " << post_process_time << "ms post_process_time."<< endl;
            }
#endif
            break;
        }

    }

    return outResult;
}

//  1、前处理  2、推理
char *YOLOV8_INFERENCE_CORE::WarmUpSession() {

    clock_t starttime1 = clock();

    Mat img(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    Mat postImg = PostProcess(img, imgSize);

    if (modelType < 2) {
        float *blob = new float [img.total() * 3];
        ImageToBlob(postImg, blob);
        vector<int64_t> warm_tenor_node = {1, 3, imgSize.at(0), imgSize.at(1)};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                blob, 3 * imgSize.at(0) * imgSize.at(1), warm_tenor_node.data(), warm_tenor_node.size());
        auto out_tensor = session->Run(Ort::RunOptions(nullptr), inputNodeNames.data(),
                                       &input_tensor, 1, outputNodeNames.data(),outputNodeNames.size());
        delete [] blob;
        clock_t starttime4 = clock();
        double warm_time = (starttime4 - starttime1) / CLOCKS_PER_SEC * 1000;
        cout << starttime4 <<endl;
        cout << starttime1 <<endl;
        if(cudaEnable) {
            cout << "[YOLOV8_ORT_INFERENCE+(CUDA)]: Warm-up cost " << warm_time << " ms. " <<endl;
        }
        else{
            cout << "[YOLOV8_ORT_INFERENCE+(CPU)]: Warm-up cost " << warm_time << " ms. " <<endl;
        }
    } else{
#ifdef USE_CUDA
        half *blob = new half [imgSize.at(0) * imgSize.at(1) * 3];
        ImageToBlob(img, blob);
        vector<int64_t> warm_node = {1, 3, imgSize.at(0), imgSize.at(1)};
        Ort::Value intpue_tensor = Ort::Value::CreateTensor(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
                                                            blob, imgSize.at(0) * imgSize.at(1) * 3, warm_node.data(), warm_node.size());
        auto output_tensor = session->Run(options, inputNodeNames.data(),
                                          &intpue_tensor, inputNodeNames.size(), outputNodeNames.data(), outputNodeNames.size());

        delete [] blob;
        clock_t starttime4 = clock();
        double warm_time = (starttime4 - starttime1) / CLOCKS_PER_SEC * 1000;
        if(cudaEnable) {
            cout << "[YOLOV8_ORT_INFERENCE(CUDA)]: Warm-up cost " << warm_time << " ms. " <<endl;
        }
        else{
            cout << "[YOLOV8_ORT_INFERENCE(CPU)]: Warm-up cost " << warm_time << " ms. " <<endl;
        }
#endif
    }
    return RET_OK;
}

