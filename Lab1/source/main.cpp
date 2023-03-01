#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    //Read photo in base mode
    cv::Mat img_base = cv::imread("../images/DSC_1993.jpeg");
    cv::imshow("Show Base Image", img_base);

    //Read photo in gray-scale mode
    cv::Mat img_gs = cv::imread("../images/DSC_1993.jpeg", 0);
    cv::imshow("Show GrayScale Image", img_gs);
    cv::waitKey();
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

    cv::destroyAllWindows();

    return 0;
}