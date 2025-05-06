#include <iostream>
#include <opencv2/opencv.hpp>
#include "pre_processing.h"
#include "red_eye.h"

using namespace std;
using namespace cv;

int main() {

    Mat image = imread(R"(D:\Ingrid\AA_AN_3\PI\Proiect\test.jpg)",  IMREAD_COLOR); // Put your actual image path here
    if (image.empty()) {
        cerr << "Failed to load image!" << endl;
        return -1;
    }

    // === pas 1: Grayscale and Binarizare===
    Mat gray = bgr_2_grayscale(image);
    //
    //equalizeHist(gray, gray);

    //
    //GaussianBlur(gray, gray, Size(3,3), 0);
    Mat binary= grayscale_2_binary(gray, 80);
    //threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    // === pas 2: Morphological Opening ===
    int n_di[8] = {0, -1, -1, -1, 0, 1, 1, 1};
    int n_dj[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    neighborhood_structure neighborhood = {8, n_di, n_dj};

    Mat cleaned = opening(binary, neighborhood, 1);
    //Mat cleaned = closing(opened, neighborhood, 1);

    // === pas 3: Labeling ===
    Labels labeled = Two_pass_labeling(cleaned);
    //
//    Mat label_display;
//    labeled.labels.convertTo(label_display, CV_8U, 255.0 / labeled.no_labels);  // normalize for visualization
//    imshow("Labels (debug view)", label_display);
    //

    // === pas 4: Detectia Ochilor ===
    vector<Rect> eye_regions = detect_eye_candidates(labeled.labels, labeled.no_labels);

    // === pas 5: Croectare ochi rosii ===
    Mat corrected = correct_red_eye(image, eye_regions);

    // === Desenez dreptunghiuri pe zona corectata ===
//    for (const Rect& region : eye_regions) {
//        rectangle(corrected, region, Scalar(0, 255, 0), 2);
//    }

    // === Show results ===
    imshow("Original", image);
    imshow("Grayscale", gray);
    imshow("Equalized", gray);
    imshow("Binary", binary);
    //imshow("After Opening", cleaned);
    imshow("Red-Eye Corrected", corrected);

    Mat label_vis;
    normalize(labeled.labels, label_vis, 0, 255, NORM_MINMAX, CV_8UC1);
    applyColorMap(label_vis, label_vis, COLORMAP_JET);  // Optional: adds colors
    imshow("Labels Colored", label_vis);

    for (const Rect& region : eye_regions) {
        rectangle(corrected, region, Scalar(0, 255, 0), 2);  // Green box
    }
    imshow("Detected Eye Regions", corrected);

    waitKey(0);
    return 0;
}
