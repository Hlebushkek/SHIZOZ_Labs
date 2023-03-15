#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

int main()
{
    //Read photo in base mode
    cv::Mat img_base = cv::imread("../images/DSC_1993.jpeg");
    cv::imshow("Show Base Image", img_base);

    //Read photo in gray-scale mode
    cv::Mat img_gs = cv::imread("../images/DSC_1993.jpeg", 0);
    cv::imshow("Show GrayScale Image", img_gs);
    
    //Save photo in gray-scale mode
    cv::imwrite("../images/DSC_1993_gs.jpeg", img_gs);

    //Read image pixels
#ifdef ITERATE_PIXELS
    int rows = img_gs.rows;
    int cols = img_gs.cols;
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
            std::cout << (float)img_gs.at<cv::Vec3b>(i,j)[0] << " "
                      << (float)img_gs.at<cv::Vec3b>(i,j)[1] << " "
                      << (float)img_gs.at<cv::Vec3b>(i,j)[2] << std::endl;
#endif

    //ROI
    cv::Mat cropped_image = img_base(cv::Range(80,800), cv::Range(80,800));
    cv::imshow("ROI", cropped_image);

    //Resize
    cv::Mat img_resized;

    cv::Size img_size = cv::Size(img_base.cols, img_base.rows);
    int new_width = 200;
    float k = (float)img_size.height / (float)img_size.width; // height / width
    int new_height = new_width * k;

    cv::resize(img_base, img_resized, cv::Size(new_width, new_height));
    cv::imshow("Resized", img_resized);

    //Rotation
    cv::Mat img_rotated;

    // cv::Mat rotation = cv::getRotationMatrix2D(cv::Point2f(img_size.width / 2, img_size.height / 2), -45, 1.0);
    // cv::warpAffine(img_base, img_rotated, rotation, img_size);
    cv::rotate(img_base, img_rotated, cv::ROTATE_90_CLOCKWISE);
    cv::imshow("Rotated", img_rotated);

    //Blur
    cv::Mat img_blurred, img_stacked;
    // cv::GaussianBlur(img_base, img_blurred, cv::Size(121, 121), 0);
    // cv::vconcat(img_base, img_blurred, img_stacked);
    // cv::imshow("Blurred", img_stacked);

    //Rectangle
    cv::Mat img_rect;
    img_base.copyTo(img_rect);
    cv::rectangle(img_rect, cv::Rect(120, 120, 360, 360), cv::Scalar(0, 0, 255), 8);
    cv::imshow("Rect", img_rect);

    //Line
    cv::Mat img_line(400, 400, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::line(img_line, cv::Point(0, 0), cv::Point(400, 400), cv::Scalar(255, 0, 0), 10);
    cv::imshow("Line", img_line);

    //Polylines
    cv::Mat img_polylines(400, 400, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<cv::Point> pts;
    pts.push_back(cv::Point(50, 50));
    pts.push_back(cv::Point(300, 50));
    pts.push_back(cv::Point(350, 200));
    pts.push_back(cv::Point(300, 150));
    pts.push_back(cv::Point(150, 350));
    pts.push_back(cv::Point(100, 100));

    cv::polylines(img_polylines, pts, true, cv::Scalar(0, 255, 0), 8);
    cv::imshow("Polylines", img_polylines);

    //Circle
    cv::Mat img_circle(400, 400, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(img_circle, cv::Point(200, 200), 50, cv::Scalar(0, 0, 255), 8);
    cv::imshow("Circle", img_circle);

    //Text
    cv::Mat img_text;
    img_base.copyTo(img_text);
    int font0 = cv::FONT_HERSHEY_SIMPLEX;
    int font1 = cv::FONT_HERSHEY_COMPLEX;
    int font2 = cv::FONT_HERSHEY_SCRIPT_COMPLEX;
    cv::putText(img_text, "Somewhere in Kyiv", cv::Point(10,300),
        font0, 10, cv::Scalar(0), 8, cv::LINE_4);
        cv::putText(img_text, "Somewhere in Kyiv", cv::Point(10,600),
        font1, 10, cv::Scalar(0), 8, cv::LINE_4);
    cv::putText(img_text, "Somewhere in Kyiv", cv::Point(10,900),
        font2, 10, cv::Scalar(0), 8, cv::LINE_4);
    
    cv::imshow("Text", img_text);

    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}