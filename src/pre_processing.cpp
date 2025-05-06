#include <iostream>
#include <opencv2/opencv.hpp>
#include "pre_processing.h"
using namespace std;
using namespace cv;

#define PI 3.14

Mat bgr_2_grayscale(Mat source){

    int rows=source.rows, cols=source.cols;
    Mat grayscale_image(rows, cols, CV_8UC1);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Vec3b pixel = source.at<Vec3b>(i, j);
            int gri = (pixel[2] + pixel[1] + pixel[0]) / 3;
            grayscale_image.at<uchar>(i, j) = (uchar)gri;
        }
    }

    return grayscale_image;

}

Mat grayscale_2_binary(Mat source, int threshold){

    int rows=source.rows, cols=source.cols;
    Mat binary(rows, cols, CV_8UC1);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (source.at<uchar>(i, j) >= threshold) {
                binary.at<uchar>(i, j) = 255;
            } else {
                binary.at<uchar>(i, j) = 0;
            }
        }
    }

    return binary;
}

Mat dilation(Mat source, neighborhood_structure neighborhood, int no_iter){

    Mat dst, aux;
    int rows = source.rows, cols = source.cols;
    aux = source.clone();
    for (int iter = 0; iter < no_iter; iter++) {
        dst = Mat::ones(rows, cols, CV_8UC1) * 255;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (aux.at<uchar>(i, j) == 0) {
                    for (int k = 0; k < neighborhood.size; k++) {
                        int ni = i + neighborhood.di[k];
                        int nj = j + neighborhood.dj[k];
                        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                            dst.at<uchar>(ni, nj) = 0;
                        }
                    }
                }
            }
        }

        aux = dst.clone();
    }
    return dst;

}

Mat erosion(Mat source, neighborhood_structure neighborhood, int no_iter){

    Mat dst, aux;
    int rows = source.rows, cols = source.cols;
    aux = source.clone();
    for (int iter = 0; iter < no_iter; iter++) {
        dst = Mat::zeros(rows, cols, CV_8UC1);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                bool ok = true;
                for (int k = 0; k < neighborhood.size; k++) {
                    int ni = i + neighborhood.di[k];
                    int nj = j + neighborhood.dj[k];
                    if (ni < 0 || ni >= rows || nj < 0 || nj >= cols || aux.at<uchar>(ni, nj) != 0) {
                        ok = false;
                        break;
                    }
                }

                if (ok) {
                    dst.at<uchar>(i, j) = 0;
                } else {
                    dst.at<uchar>(i, j) = 255;
                }
            }
        }

        aux = dst.clone();
    }

    return dst;
}

Mat opening(Mat source, neighborhood_structure neighborhood, int no_iter) {

    Mat dst, aux;
    aux = erosion(source, neighborhood, no_iter);
    dst = dilation(aux, neighborhood, no_iter);
    return dst;

}


Labels Two_pass_labeling(Mat source){
    Mat labels;
    int no_newlabels=0;

    int height = source.rows;
    int  width = source.cols;
    int label = 0;
    labels = Mat::zeros(height, width, CV_32SC1);

    vector<set<int>> edges;
    edges.resize(height * width + 1);
    for(int i=0; i<height-1; i++)
        for(int j=0; j<width-1; j++)
            if(source.at<uchar>(i, j)==0 && labels.at<int>(i, j)==0)
            {
                vector<int> L;
                for (int k = 0; k < 7; k++) {
                    int ni = i + n8_di[k];
                    int nj = j + n8_dj[k];
                    if (ni >= 0 && nj >= 0 && ni < height && nj < width) {
                        if (labels.at<int>(ni, nj) > 0) {
                            L.push_back(labels.at<int>(ni, nj));
                        }
                    }
                }
                if(L.empty())
                {
                    label++;
                    labels.at<int>(i, j)=label;
                }
                else
                {
                    int min=INT_MAX;
                    for(int k=0; k<L.size(); k++)
                        if(L[k]<min) min=L[k];
                    labels.at<int>(i, j) = min;
                    for (int z = 0; z < L.size(); z++) {
                        if (L[z] != min) {
                            edges[min].insert(L[z]);
                            edges[L[z]].insert(min);
                        }
                    }
                }


            }

    vector<int> newlabels(label + 1, 0);

    for (int i = 1; i <= label; i++) {
        if (newlabels[i] == 0) {
            no_newlabels++;
            queue<int> Q;
            newlabels[i] = no_newlabels;
            Q.push(i);

            while (!Q.empty()) {
                int x = Q.front();
                Q.pop();
                for (int y : edges[x]) {
                    if (newlabels[y] == 0) {
                        newlabels[y] = no_newlabels;
                        Q.push(y);
                    }
                }
            }
        }
    }

    for (int i = 0; i < height-1; i++) {
        for (int j = 0; j < width-1; j++) {
            if (labels.at<int>(i, j) > 0) {
                labels.at<int>(i, j) = newlabels[labels.at<int>(i, j)];
            }
        }
    }

    return {labels, no_newlabels};
}

perimeter naive_perimeter(Mat binary_object){

    perimeter object_perimeter;
    object_perimeter.length = 0;
    object_perimeter.contour = Mat::zeros(binary_object.size(), CV_8UC1);
    int rows=binary_object.rows;
    int cols=binary_object.cols;
    for(int i=1; i<rows-1; i++)
        for(int j=1; j<cols-1; j++)
            if(binary_object.at<uchar>(i, j)==255){
                int contur=0;
                for(int x=i-1; x<=i+1; x++)
                    for(int y=j-1; y<=j+1; y++)
                        if(binary_object.at<uchar>(x, y)==0)
                        {
                            contur=1;
                            break;
                        }
                if(contur==1)
                {
                    object_perimeter.length++;
                    object_perimeter.contour.at<uchar>(i, j)=255;
                }
            }

    return object_perimeter;

}

int compute_area(Mat binary_object){

    int area=0;
    int rows = binary_object.rows;
    int cols = binary_object.cols;
    for(int i=0; i<rows; i++)
        for(int j=0; j<cols; j++)
            if(binary_object.at<uchar>(i, j)==255)
                area++;

    return area;
}

Point compute_center_of_mass(Mat binary_object){

    int rows = binary_object.rows;
    int cols = binary_object.cols;
    Point center_mass;
    int aria=0;
    int sum_rows=0;
    int sum_cols=0;
    for(int i=0; i<rows; i++)
        for(int j=0; j<cols; j++)
            if(binary_object.at<uchar>(i, j)==255)
            {
                sum_rows+=i;
                sum_cols+=j;
                aria++;
            }
    center_mass.x=sum_cols/aria;
    center_mass.y=sum_rows/aria;

    return center_mass;

}

circumscribed_rectangle_coord compute_circumscribed_rectangle_coord(Mat binary_object){

    int rows = binary_object.rows;
    int cols = binary_object.cols;
    circumscribed_rectangle_coord coords;
    coords.r_min = rows;
    coords.r_max = 0;
    coords.c_min = cols;
    coords.c_max = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (binary_object.at<uchar>(i, j) == 255) {
                if(i<coords.r_min)
                    coords.r_min=i;
                if(i>coords.r_max)
                    coords.r_max=i;
                if(j<coords.c_min)
                    coords.c_min=j;
                if(j>coords.c_max)
                    coords.c_max=j;
            }
        }
    }

    return coords;
}

float compute_aspect_ratio(circumscribed_rectangle_coord coord){

    float ratio;
    ratio = (float)(coord.c_max - coord.c_min + 1)/(float)(coord.r_max - coord.r_min + 1);

    return ratio;
}

float compute_thinness_ratio(int area, int perimeter){

    float thinness_ratio;
    thinness_ratio = 4*PI*((float)area/(float)(perimeter*perimeter));
    return thinness_ratio;
}


Mat closing(Mat source, neighborhood_structure neighborhood, int no_iter) {

    Mat dst, aux;
    aux = dilation(source, neighborhood, no_iter);
    dst = erosion(aux, neighborhood, no_iter);
    return dst;
}


//vector<Rect> detect_eye_candidates(Mat labels, int num_labels) {
//    vector<Rect> eye_regions;
//
//    for (int lbl = 1; lbl <= num_labels; lbl++) {
//        Mat component_mask = (labels == lbl);
//        int area = compute_area(component_mask);
//        if (area < 50 || area > 5000) continue;
//
//        circumscribed_rectangle_coord rect = compute_circumscribed_rectangle_coord(component_mask);
//        float aspect = compute_aspect_ratio(rect);
//        if (aspect < 1.0 || aspect > 6.0) continue;
//
//        Point center = compute_center_of_mass(component_mask);
//        if (center.y > labels.rows * 0.7 || center.y < labels.rows * 0.2) continue;
//
//        perimeter perim = naive_perimeter(component_mask);
//        float thinness = compute_thinness_ratio(area, perim.length);
//        if (thinness < 0.3 || thinness > 0.8) continue;
//
//        if ((rect.c_max - rect.c_min) < 10 || (rect.r_max - rect.r_min) < 10) continue;
//
//        eye_regions.push_back(Rect(rect.c_min, rect.r_min,
//                                   rect.c_max - rect.c_min + 1,
//                                   rect.r_max - rect.r_min + 1));
//    }
//
//    // Sort the regions by area (largest first)
//    sort(eye_regions.begin(), eye_regions.end(), [](const Rect& a, const Rect& b) {
//        return a.area() > b.area();
//    });
//
//    // Keep only the top 2 candidates
//    if (eye_regions.size() > 2)
//        eye_regions.resize(2);
//
//    return eye_regions;
//}

vector<Rect> detect_eye_candidates(Mat labels, int num_labels) {
    vector<Rect> eye_regions;

    for (int lbl = 1; lbl <= num_labels; lbl++) {
        Mat component_mask = (labels == lbl);
        int area = compute_area(component_mask);
        if (area < 50 || area > 5000) continue;

        circumscribed_rectangle_coord rect = compute_circumscribed_rectangle_coord(component_mask);
        float aspect = compute_aspect_ratio(rect);
        if (aspect < 1.0 || aspect > 6.0) continue;

        Point center = compute_center_of_mass(component_mask);
        if (center.y > labels.rows * 0.7 || center.y < labels.rows * 0.2) continue;

        perimeter perim = naive_perimeter(component_mask);
        float thinness = compute_thinness_ratio(area, perim.length);
        if (thinness < 0.2 || thinness > 0.8) continue;

        if ((rect.c_max - rect.c_min) < 10 || (rect.r_max - rect.r_min) < 10) continue;

        eye_regions.push_back(Rect(rect.c_min, rect.r_min, rect.c_max - rect.c_min + 1, rect.r_max - rect.r_min + 1));
    }

    return eye_regions;
}

