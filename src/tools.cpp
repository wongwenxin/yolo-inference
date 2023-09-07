#include "tools.h"




int tools::read_yaml(std::string yamlPath, YOLOV8_INFERENCE_CORE *&p) {

    std::ifstream file(yamlPath);
    if(!file.is_open()) {
        cout << "Yaml file is empty, please check path. " << endl;
        return 0;
    }

    string line;
    vector<string> lines;
    while(getline(file, line)) {
        lines.push_back(line);
    }

    size_t start = 0;
    size_t end = 0;

    for (size_t i = 0; i < lines.size(); i++)
    {
        if(lines[i].find("names:") != string::npos) {
            start = i + 1;
        } else if(start > 0 && lines[i].find(":") == string::npos) {
            end = i;
            break;
        }
    }

    vector<string> names;
    for (size_t i = start; i < end; i++) {
        std::stringstream ss(lines[i]);
        string name;
        getline(ss, name, ':');
        getline(ss, name);
        names.push_back(name);
    }

    p->classes = names;

    return 1;
}

void tools::ResultVisualization(YOLOV8_INFERENCE_CORE *&p) {

//    std::filesystem::path newWorkingDir = "/home/wwx/CLionProjects/yolo-ort-inference";
//    filesystem::current_path(newWorkingDir);
//    std::filesystem::path current_path = filesystem::current_path();
    filesystem::path imgs_dir = "/home/wwx/CLionProjects/yolo-ort-inference/images";

    for (auto &i : filesystem::directory_iterator(imgs_dir)) {
        if(i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg") {
            string img_path = i.path().string();
            Mat img = imread(img_path);

            vector<YOLOV8_RESULT> result;
            p->RunSession(img, result);


            for (auto &re : result) {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
                cv::rectangle(img, re.box, color, 2);

                auto conf = floor(100 * re.confidece) / 100;
                std::cout << std::fixed << std::setprecision(2);
                string label = p->classes[re.classId] + " " + to_string(conf).substr(0, to_string(conf).size() - 4);

                cout << label <<endl;

                cv::rectangle(
                        img,
                        cv::Point(re.box.x, re.box.y - 25),
                        cv::Point(re.box.x + label.length() * 15, re.box.y),
                        color,
                        cv::FILLED
                        );

                cv::putText(
                        img,
                        label,
                        cv::Point(re.box.x, re.box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.75,
                        cv::Scalar(0, 0, 0),
                        2
                        );
            }

            cv::imshow("result", img);
            cv::waitKey(0);
            cv::destroyAllWindows();

        }

    }

}
