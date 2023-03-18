#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

void draw_lines(cv::Mat& img, std::vector<cv::Vec4f>& lines)
{
    float y_bottom = 540;
    float y_upper = 315;

    std::vector<float> x_bottom_pos;
    std::vector<float> x_upper_pos;
    std::vector<float> x_bottom_neg;
    std::vector<float> x_upper_neg;

    for (cv::Vec4f& line : lines)
    {
        float x1 = line[0]; float y1 = line[1];
        float x2 = line[2]; float y2 = line[3];

        float slope = (y2 - y1) / (x2 - x1);
        float b = y1 - slope * x1;

        float x_bottom = (y_bottom - b) / slope;
        float x_upper =  (y_upper - b) / slope;

        if (slope > 0.5 && slope < 0.8)
        {
            x_bottom_pos.emplace_back(x_bottom);
            x_upper_pos.emplace_back(x_upper);
        }
        else if (slope < -0.5 && slope > -0.8)
        {
            x_bottom_neg.emplace_back(x_bottom);
            x_upper_neg.emplace_back(x_upper);
        }
    }

    if (x_bottom_pos.size() == 0 || x_bottom_neg.size() == 0)
        return;

    std::vector<std::vector<int>> lines_mean = {
        {
            (int)cv::mean(x_bottom_pos)[0], (int)cv::mean(y_bottom)[0],
            (int)cv::mean(x_upper_pos)[0], (int)cv::mean(y_upper)[0]
        },
        {
            (int)cv::mean(x_bottom_neg)[0], (int)cv::mean(y_bottom)[0],
            (int)cv::mean(x_upper_neg)[0], (int)cv::mean(y_upper)[0]
        }
    };

    for (int i = 0; i < lines_mean.size(); i++)
        cv::line(
            img,
            cv::Size(lines_mean[i][0], lines_mean[i][1]),
            cv::Size(lines_mean[i][2], lines_mean[i][3]),
            cv::Scalar(255, 0, 0), 7
        );
}

void process_image(cv::Mat& img)
{
    cv::Mat img_gs;
    cv::cvtColor(img, img_gs, cv::COLOR_BGR2GRAY);

    cv::Mat img_blurred;
    cv::GaussianBlur(img_gs, img_blurred, cv::Size(5, 5), 0, 0);

    cv::Mat edges;
    int low_t = 10, high_t = 200;
    cv::Canny(img_blurred, edges, low_t, high_t);

    cv::imshow("Edges", edges);

    std::vector vertices = {
        std::vector {
            cv::Point(0, img_gs.rows), cv::Point(450, 310),
            cv::Point(490, 310), cv::Point(img_gs.cols, img_gs.rows)
        }
    };

    cv::Mat mask = cv::Mat::zeros(edges.size(), 0);
    int ignore_mask_color = 255;
    cv::fillPoly(mask, vertices, ignore_mask_color);

    cv::Mat masked_edges = cv::Mat::zeros(edges.size(), CV_8U);
    cv::bitwise_and(edges, mask, masked_edges);
    cv::imshow("Mask", mask);
    cv::imshow("Masked edges", masked_edges);

    cv::Mat img_masked = cv::Mat::zeros(edges.size(), 0);
    cv::bitwise_and(img_gs, mask, img_masked);
    cv::imshow("Masked image", img_masked);

    int rho = 3;
    float theta = M_PI / 180;
    int threshold = 15;
    int min_line_length = 150;
    int max_line_gap = 60;

    std::vector<cv::Vec4f> lines;
    cv::HoughLinesP(masked_edges, lines, rho, theta, threshold, min_line_length, max_line_gap);
    draw_lines(img, lines);
}

int main()
{
    cv::Mat img_base = cv::imread("../resources/road2.jpg");
    process_image(img_base);
    cv::imshow("Lines", img_base);

    //Video
    cv::VideoCapture cap = cv::VideoCapture("../resources/road.mp4");
    cv::Mat frame;
    while(cap.isOpened())
    {
        cap.read(frame);

        // if (!frame.empty())
        // {
            process_image(frame);
            cv::imshow("Video", frame);

            char c = (char)cv::waitKey(250);
            if (c == 27)
                break;
        // }
    }

    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}