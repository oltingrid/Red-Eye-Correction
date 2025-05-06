#ifndef PROIECT_RED_EYE_H
#define PROIECT_RED_EYE_H
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;

image_channels_hsv bgr_2_hsv(image_channels_bgr bgr_channels);
Mat detect_red_pixels_custom(Mat eye_roi);
Mat correct_red_eye(Mat image, const vector<Rect>& eye_regions);

#endif //PROIECT_RED_EYE_H
