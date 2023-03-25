#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include "../include/face_recognition.hpp"

using namespace dlib;
namespace fs = std::filesystem;

using encoding = matrix<double, 0L, 1L>;
using name_encoding_map = std::map<std::string, std::vector<encoding>>;

std::vector<fs::path> get_image_paths(fs::path& root_dir, std::vector<std::string>& class_names)
{
    static std::vector<std::string> available_ext = {"png", "jpg", "jpeg"};
    std::vector<fs::path> result;

    for (const std::string& class_name : class_names)
    {
        fs::path folder = root_dir / class_name;
        for (const auto& entry : fs::directory_iterator(folder))
        {
            fs::path path = entry.path();
            std::string ext = path.extension();

            for (auto& ext : available_ext)
                if (path.string().find(ext) != std::string::npos)
                {
                    result.emplace_back(path);
                    break;
                }
        }
    }

    return result;
}

std::vector<rectangle> face_rects(cv::Mat& img, frontal_face_detector& predictor)
{
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    std::vector<rectangle> rects = predictor(cv_image<unsigned char>(gray), 1);
    
    return rects;
}

std::vector<full_object_detection> face_landmarks(cv::Mat& img, std::vector<rectangle>& rects, shape_predictor& sp)
{
    std::vector<full_object_detection> result;

    for (auto& rect : rects)
        result.emplace_back(sp(cv_image<bgr_pixel>(img), rect));

    return result;
}

std::vector<encoding> face_encodings(cv::Mat& img, std::vector<full_object_detection>& face_landmarks, face_recognition_model_v1& face_encoder)
{
    std::vector<encoding> result;

    for (auto& landmark : face_landmarks)
        result.emplace_back(face_encoder.compute_face_descriptor(cv_image<rgb_pixel>(img), landmark, 1));     

    return result;
}

float nb_of_matches(std::vector<encoding>& known_encodings, encoding& unknow_encoding)
{
    float result = 0.f;

    for (auto& known_encoding : known_encodings)
    {
        float distance = length(known_encoding - unknow_encoding);
        if (distance < 0.6f)
            result += distance;
    }

    return result;
}

int main()
{
    frontal_face_detector ffd = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("../resources/models/shape_predictor_68_face_landmarks.dat") >> sp;
    face_recognition_model_v1 frm = face_recognition_model_v1("../resources/models/dlib_face_recognition_resnet_model_v1.dat");

    name_encoding_map name_encodings_dict;

    //Read dataset
    std::ifstream ifs("../resources/name_encodings");
    if (ifs.is_open())
    {
        deserialize(name_encodings_dict, ifs);
    }
    else
    {
        fs::path root("../resources/dataset/");
        std::vector<std::string> class_names
        {
            "Arnold_Schwarzenegger",
            "Bill_Clinton",
            "Bill_Gates",
            "Nancy_Pelosi"
        };

        std::vector<fs::path> img_paths = get_image_paths(root, class_names);
        for (auto& path : img_paths)
        {
            std::cout << path << std::endl;
            cv::Mat img = cv::imread(path);

            auto rects = face_rects(img, ffd);
            auto landmarks = face_landmarks(img, rects, sp);
            auto encodings = face_encodings(img, landmarks, frm);

            std::cout << "rects: " << rects.size() << std::endl;
            std::cout << "landmarks: " << landmarks.size() << std::endl;
            std::cout << "encoding: " << encodings.size() << std::endl;

            std::string name = path.parent_path().filename();

            if (name_encodings_dict.contains(name))
                name_encodings_dict[name].insert(name_encodings_dict[name].end(), encodings.begin(), encodings.end());
            else
                name_encodings_dict.insert({ name, { encodings } });
        }

        //Save dict
        std::ofstream ofs("../resources/name_encodings");
        serialize(name_encodings_dict, ofs);
    }

    for (auto& pair : name_encodings_dict)
        std::cout << pair.first << " " << pair.second.size() << std::endl;


    //Try sample
    fs::path root("../resources/");
    std::vector<std::string> subfolders_names{"examples"};
    std::vector<fs::path> sample_paths = get_image_paths(root, subfolders_names);

    for (auto& path : sample_paths)
    {
        cv::Mat sample = cv::imread(path);
        auto rects = face_rects(sample, ffd);
        auto landmarks = face_landmarks(sample, rects, sp);
        auto encodings = face_encodings(sample, landmarks, frm);

        std::vector<std::string> names;

        for (auto& encoding : encodings)
        {
            std::map<std::string, float> counts;

            for (auto& pair : name_encodings_dict)
                counts[pair.first] = nb_of_matches(pair.second, encoding);

            float max_val = 0;
            std::string name = "Unknow";
            
            for (auto& count : counts)
            {
                float val = count.second;
                if (val != 0 && val > max_val)
                {
                    max_val = val;
                    name = count.first;
                    
                    std::cout << name << ": " << max_val << std::endl;
                }
            }

            names.emplace_back(name);
        }

        for (size_t i = 0; i < std::min(rects.size(), names.size()); i++)
        {
            auto& rect = rects[i];
            auto& name = names[i];

            cv::rectangle(sample, cv::Rect(cv::Point2i(rect.left(), rect.top()),
                cv::Point2i(rect.right() + 1, rect.bottom() + 1)), cv::Scalar(255, 0, 0), 2);
            cv::putText(sample, name, cv::Point(rect.left(), rect.top() - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Detection", sample);
        cv::waitKey();
    }

    return 0;
}