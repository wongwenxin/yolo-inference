#pragma once
#include "opencv2/opencv.hpp"
#include <iostream>
#include <filesystem>
#include <vector>
#include <fstream>
#include "inference.h"
using namespace cv;
using namespace std;

namespace tools {

   int read_yaml(std::string yamlPath, YOLOV8_INFERENCE_CORE *&p);
   void ResultVisualization(YOLOV8_INFERENCE_CORE *&p);
}
