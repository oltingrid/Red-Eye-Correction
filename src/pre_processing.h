#ifndef PROIECT_PRE_PROCESSING_H
#define PROIECT_PRE_PROCESSING_H
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

const int n8_di[8] = {0,-1,-1, -1, 0, 1, 1, 1};
const int n8_dj[8] = {1, 1, 0, -1, -1,-1, 0, 1};

typedef struct {
    Mat B;
    Mat G;
    Mat R;
} image_channels_bgr;

typedef struct {
    Mat H;
    Mat S;
    Mat V;
} image_channels_hsv;

typedef struct {
    Mat labels;
    int no_labels;
}Labels;

typedef struct{
    int c_min;
    int c_max;
    int r_min;
    int r_max;
} circumscribed_rectangle_coord;

typedef struct{
    int size;
    int* di;
    int* dj;
} neighborhood_structure;

typedef struct{
    Mat contour;
    int length;
} perimeter;

Mat bgr_2_grayscale(Mat source);
Mat grayscale_2_binary(Mat source, int threshold);
Mat dilation(Mat source, neighborhood_structure neighborhood, int no_iter);
Mat erosion(Mat source, neighborhood_structure neighborhood, int no_iter);
Mat opening(Mat source, neighborhood_structure neighborhood, int no_iter);
Labels Two_pass_labeling(Mat source);
perimeter naive_perimeter(Mat binary_object);
int compute_area(Mat binary_object);
Point compute_center_of_mass(Mat binary_object);
circumscribed_rectangle_coord compute_circumscribed_rectangle_coord(Mat binary_object);
float compute_aspect_ratio(circumscribed_rectangle_coord coord);
float compute_thinness_ratio(int area, int perimeter);
Mat closing(Mat source, neighborhood_structure neighborhood, int no_iter);
vector<Rect> detect_eye_candidates(Mat labels, int num_labels);

#endif //PROIECT_PRE_PROCESSING_H
