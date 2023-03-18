#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

static std::string haar_cascades_path = "../../opencv/data/haarcascades/";

int main()
{
    //1 Photo face + smile + eye detection 
    cv::Mat img_base = cv::imread("../resources/photo1.jpg");
    cv::Mat img_resized;
    float scaling_factor = 0.5f;

    cv::resize(img_base, img_resized, cv::Size(0, 0), scaling_factor, scaling_factor, cv::INTER_AREA);

    cv::CascadeClassifier face_cascade(haar_cascades_path + "haarcascade_frontalface_default.xml");
    cv::CascadeClassifier smile_cascade(haar_cascades_path + "haarcascade_smile.xml");
    cv::CascadeClassifier eye_cascade(haar_cascades_path + "haarcascade_eye.xml");

    cv::Mat img_gs;
    cv::cvtColor(img_resized, img_gs, cv::COLOR_RGB2GRAY);

    cv::Mat img_final;
    img_resized.copyTo(img_final);

    std::vector<cv::Rect> face_rects;
    face_cascade.detectMultiScale(img_gs, face_rects);

    for (auto const& rect : face_rects) {
        cv::rectangle(img_final, rect, cv::Scalar(0, 255, 0), 2);

        cv::Mat face_roi = img_resized(
            cv::Range(rect.y, rect.y + rect.height),
            cv::Range(rect.x, rect.x + rect.width)
        );

        std::vector<cv::Rect> smile_rects;
        smile_cascade.detectMultiScale(face_roi, smile_rects);
        std::vector<cv::Rect> eye_rects;
        eye_cascade.detectMultiScale(face_roi, eye_rects);

        for (auto const& rect_s : smile_rects) {
            cv::Rect full_rect = cv::Rect(rect.x + rect_s.x, rect.y + rect_s.y,
                rect_s.width, rect_s.height);
            cv::rectangle(img_final, full_rect, cv::Scalar(255, 255, 0), 2);
        }
        for (auto const& rect_e : eye_rects) {
            cv::Rect full_rect = cv::Rect(rect.x + rect_e.x, rect.y + rect_e.y,
                rect_e.width, rect_e.height);
            cv::rectangle(img_final, full_rect, cv::Scalar(0, 255, 255), 2);
        }
    }

    cv::imshow("Final", img_final);

    auto hog = cv::HOGDescriptor();
    auto detector = cv::HOGDescriptor::getDefaultPeopleDetector();
    hog.setSVMDetector(detector);

    img_base = cv::imread("../resources/photo3.jpg");
    scaling_factor = 0.2f;

    cv::resize(img_base, img_resized, cv::Size(0, 0), scaling_factor, scaling_factor, cv::INTER_AREA);
    cv::cvtColor(img_resized, img_gs, cv::COLOR_RGB2GRAY);

    img_resized.copyTo(img_final);

    std::vector<cv::Rect> people_rects;
    hog.detectMultiScale(img_gs, people_rects, 0, cv::Size(8, 8), cv::Size(30, 30));

    for (auto const& rect : people_rects) {
        cv::rectangle(img_final, rect, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("PeopleDetection", img_final);

    //2 Video face detection
    cv::Mat frame;
    cv::Mat frame_scaled;
    cv::Mat frame_gs;
    
    cv::VideoCapture webcam_cap(0);
    while(true)
    {
        webcam_cap.read(frame);

        std::cout << frame.size() << std::endl;
        if (frame.empty())
            break;
        
        cv::resize(frame, frame_scaled, cv::Size(640, 360));
        cv::cvtColor(frame_scaled, frame_gs, cv::COLOR_BGR2GRAY);

        hog.detectMultiScale(frame_gs, people_rects, 0, cv::Size(8, 8));
        for (auto const& rect : people_rects)
            cv::rectangle(frame_scaled, rect, cv::Scalar(0, 255, 0), 2);

        face_cascade.detectMultiScale(frame_gs, face_rects);
        for (auto const& rect : face_rects)
            cv::rectangle(frame_scaled, rect, cv::Scalar(0, 0, 255), 2);

        cv::imshow("VideoFaceDetection", frame_scaled);

        char c = (char)cv::waitKey(25);
        if (c == 27) //Esc
            break;
    }

    webcam_cap.release();
    //3 walking people + faces video detection
    cv::VideoCapture cap("../resources/video1.mp4");

    while(true)
    {
        cap.read(frame);

        std::cout << frame.size() << std::endl;
        if (frame.empty())
            break;
        
        cv::resize(frame, frame_scaled, cv::Size(640, 360));
        cv::cvtColor(frame_scaled, frame_gs, cv::COLOR_BGR2GRAY);

        hog.detectMultiScale(frame_gs, people_rects, 0, cv::Size(8, 8));
        for (auto const& rect : people_rects)
            cv::rectangle(frame_scaled, rect, cv::Scalar(0, 255, 0), 2);

        face_cascade.detectMultiScale(frame_gs, face_rects);
        for (auto const& rect : face_rects)
            cv::rectangle(frame_scaled, rect, cv::Scalar(0, 0, 255), 2);

        cv::imshow("VideoPeopleDetection", frame_scaled);

        char c = (char)cv::waitKey(25);
        if (c == 27) //Esc
            break;
    }

    cap.release();

    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}