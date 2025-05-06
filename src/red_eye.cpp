#include <iostream>
#include <opencv2/opencv.hpp>
#include "pre_processing.h"
#include "red_eye.h"
using namespace std;
using namespace cv;

image_channels_hsv bgr_2_hsv(image_channels_bgr bgr_channels){

    int rows=bgr_channels.B.rows, cols=bgr_channels.B.cols;
    Mat H, S, V;
    image_channels_hsv hsv_channels;

    // folosim float pt HSV channels si dupa facem conversie
    hsv_channels.H = Mat(rows, cols, CV_32F);
    hsv_channels.S = Mat(rows, cols, CV_32F);
    hsv_channels.V = Mat(rows, cols, CV_32F);


    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float r = (float)bgr_channels.R.at<uchar>(i, j) / 255.0f;
            float g = (float)bgr_channels.G.at<uchar>(i, j) / 255.0f;
            float b = (float)bgr_channels.B.at<uchar>(i, j) / 255.0f;

            float M = max(max(r, g), b);
            float m = min(min(r, g), b);
            float c = M - m;
            float v = M;
            float s;
            if(v!=0)
                s=c/v;
            else s=0;

            float h=0;
            if (c != 0) {
                if (M == r) h = 60 * (g - b) / c;
                else if (M == g) h = 120 + 60 * (b - r) / c;
                else if (M == b) h = 240 + 60 * (r - g) / c;
            }
            else h=0;
            if (h < 0) h += 360;

            hsv_channels.H.at<float>(i, j) = h;
            hsv_channels.S.at<float>(i, j) = s;
            hsv_channels.V.at<float>(i, j) = v;
        }
    }

    return hsv_channels;
}

Mat detect_red_pixels_custom(Mat eye_roi) {

    image_channels_bgr bgr;
    bgr.B = Mat(eye_roi.rows, eye_roi.cols, CV_8UC1);
    bgr.G = Mat(eye_roi.rows, eye_roi.cols, CV_8UC1);
    bgr.R = Mat(eye_roi.rows, eye_roi.cols, CV_8UC1);

    for (int i = 0; i < eye_roi.rows; i++) {
        for (int j = 0; j < eye_roi.cols; j++) {
            Vec3b pixel = eye_roi.at<Vec3b>(i, j);
            bgr.B.at<uchar>(i, j) = pixel[0];
            bgr.G.at<uchar>(i, j) = pixel[1];
            bgr.R.at<uchar>(i, j) = pixel[2];
        }
    }

    image_channels_hsv hsv = bgr_2_hsv(bgr);

    Mat red_mask = Mat::zeros(eye_roi.size(), CV_8UC1);
    for (int i = 0; i < eye_roi.rows; i++) {
        for (int j = 0; j < eye_roi.cols; j++) {
            float H = hsv.H.at<float>(i, j);
            float S = hsv.S.at<float>(i, j);
            float V = hsv.V.at<float>(i, j);

            if (
                    ((H >= 0 && H <= 10) || (H >= 340 && H <= 360)) &&
                    S > 0.4 && V > 0.4
                    ) {
                red_mask.at<uchar>(i, j) = 255;
            }
        }
    }

    return red_mask;
}


Mat correct_red_eye(Mat image, const vector<Rect>& eye_regions) {
    Mat result = image.clone();

    for (const Rect& region : eye_regions) {
        Mat eye_roi = result(region);
        Mat red_mask = detect_red_pixels_custom(eye_roi);

        // Correction step: reduce red channel
        for (int i = 0; i < eye_roi.rows; i++) {
            for (int j = 0; j < eye_roi.cols; j++) {
                if (red_mask.at<uchar>(i, j) == 255) {
                    Vec3b& pix = eye_roi.at<Vec3b>(i, j);
                    int avg = (pix[0] + pix[1]) / 2;
                    pix[2] = saturate_cast<uchar>(avg);                    // R devine medie
                    pix[1] = saturate_cast<uchar>(pix[1] + 0.2 * (pix[2])); // G putin mai mare
                    pix[0] = saturate_cast<uchar>(pix[0] + 0.2 * (pix[2])); //B mai mare
                }
            }
        }
    }

    return result;
}

