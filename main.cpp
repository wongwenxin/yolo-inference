#include "inference.h"
#include "tools.h"



int main() {


    string modelpath = "/home/wwx/CLionProjects/yolo-ort-inference/models/yolov8n.onnx";
    string yamlpath = "/home/wwx/CLionProjects/yolo-ort-inference/yaml/coco.yaml";


    YOLOV8_INFERENCE_CORE *yolov8 = new YOLOV8_INFERENCE_CORE;
    YOLOV8_INIT_PARAM params{modelpath};
    tools::read_yaml(yamlpath, yolov8);

    yolov8->CreatSession(params);
    tools::ResultVisualization(yolov8);

    return 0;
}
