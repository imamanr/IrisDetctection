  //
//  main.cpp
//  IrisDetctection
//
//  Created by Imama Noor on 4/1/16.
//  Copyright © 2016 Imama. All rights reserved.
//

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <math.h>



/** Constants **/
using namespace cv;
using namespace std;
/** Function Headers */
void detectAndDisplayEye( cv::Mat & frame );
void detectPupil(cv::Mat coloredEye, Rect eyeArea, RotatedRect & eclipseFound);

/** Global variables */
//-- Note, either copy this file from haarscascades_eye.html to your current folder, or change these locations
cv::String eye_cascade_name = "/Users/imamanoor/Documents/workspace_Xcode/IrisDetctection/IrisDetctection/haarcascade_eye.xml";
cv::CascadeClassifier eye_cascade;
cv::RNG rng(12345);
cv::Mat debugImage;

/**
 * @function main
 */
int main( int argc, const char** argv ) {
    cv::Mat frame;
    bool image = 0; //set 0 for video and 1 for image
    String image_path = "/Users/imamanoor/Documents/jobApplications/medicaBio/810nm dark.png";
    String video_path = "/Users/imamanoor/Documents/jobApplications/medicaBio/S2L_Dark.avi";
    String video_path_out = "/Users/imamanoor/Documents/jobApplications/medicaBio/S2L_Dark_Iris.avi";
    // Load the cascades
    if( !eye_cascade.load( eye_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };

    

    if(image) {
        frame = cv::imread(image_path, CV_LOAD_IMAGE_UNCHANGED | CV_LOAD_IMAGE_ANYDEPTH);
        if( !frame.empty() ) {
            detectAndDisplayEye( frame );
        }
        else {
            printf(" --(!) No captured frame -- Break!");
            return 0;
        }
        
        namedWindow("Display frame", WINDOW_NORMAL);
        cv::resizeWindow("Display frame", 480, 640);
        imshow("Display frame", frame);
        imwrite("OutputImage.jpg", frame);
        waitKey(0);
        return 0;
    }
    else {
        VideoCapture capture(video_path);
        int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form
        Size S = Size((int) capture.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                      (int) capture.get(CV_CAP_PROP_FRAME_HEIGHT));

        VideoWriter cap_out(video_path_out,capture.get(CV_CAP_PROP_FOURCC),
                            capture.get(CV_CAP_PROP_FPS),
                            cv::Size(capture.get(CV_CAP_PROP_FRAME_WIDTH),
                                     capture.get(CV_CAP_PROP_FRAME_HEIGHT)));
        cap_out.open(video_path_out, ex=-1, capture.get(CV_CAP_PROP_FPS), S, true);
        if( !capture.isOpened() )
            throw "Error when reading steam_avi";
    
        for( ; ; )
        {
            capture >> frame;
            if(frame.empty())
            break;
    
    
            // mirror it
            cv::flip(frame, frame, 1);
            frame.copyTo(debugImage);
                    
            // Apply the classifier to the frame
            if( !frame.empty() ) {
                detectAndDisplayEye( frame );
            }
            else {
                printf(" --(!) No captured frame -- Break!");
                return 0;
            }
        
            namedWindow("Display frame", WINDOW_NORMAL);
            cv::resizeWindow("Display frame", 480, 640);
            imshow("Display frame", frame);
            cap_out << frame;
            //waitKey(0);
        
        }
        cap_out.release();
     }
    
    return 0;

    }


    /**
     * @function detectAndDisplay
     */
    void detectAndDisplayEye( cv::Mat & frame ) {
        std::vector<cv::Rect> eyes;
        RotatedRect eclipseFound;
        vector<RotatedRect> pupilsDetect(2);
        std::vector<cv::Mat> rgbChannels(3);
        cv::split(frame, rgbChannels);
        cv::Mat frame_gray = rgbChannels[2];
        
        //-- Detect eyes
        eye_cascade.detectMultiScale( frame_gray, eyes, 1.1, 2, 0,cv::Size(5, 5) );
        
        for( int i = 0; i < eyes.size(); i++ )
        {
            rectangle(debugImage, eyes[i], 1234);
        }

        long detEyes = eyes.size();
        while(detEyes > 0) {
            Rect iterat;
            iterat = eyes[detEyes-1];
            cv::Mat frameCrop = frame(iterat);
            detectPupil(frameCrop, eyes[detEyes],eclipseFound);
            eclipseFound.center.x = eclipseFound.center.x + eyes[detEyes-1].x;
            eclipseFound.center.y = eclipseFound.center.y + eyes[detEyes-1].y;
            eclipseFound.size.width = eclipseFound.size.width;
            eclipseFound.size.height = eclipseFound.size.height;
            pupilsDetect.push_back(eclipseFound);
            detEyes--;
        }
        
        for (int j =1;j<eyes.size();j++){
            ellipse(frame, pupilsDetect[j], CV_RGB(255,0,0), 2, 8 );

        }

        
    }

void detectPupil(cv::Mat coloredEye, Rect eyeArea, RotatedRect & eclipseFound) {
    
    double largest = 0; // Largest ellipse area
    int indx = 0; //Index of largest ellipse area
    
    // Load image
    if (coloredEye.empty())
        return;
    
    Mat srccopy = coloredEye.clone();
    // Invert the source image and convert to grayscale
    Mat gray;
    cvtColor(~coloredEye, gray, CV_BGR2GRAY);
    Mat graycopy = gray.clone();
    
    // Convert to binary image by thresholding it
    threshold(gray, gray, 190, 250,  THRESH_BINARY);
    
    // Find all contours
    vector<vector<Point> > contours;
    findContours(gray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
    // Fill holes in each contour
    drawContours(gray, contours, -1, CV_RGB(255,255,255), -1);
    
    // Modification to look for ellipses
    vector<RotatedRect> minRect( contours.size() );
    vector<RotatedRect> minEllipse( contours.size() );
    
    for( int i = 0; i < contours.size(); i++ )
    { minRect[i] = minAreaRect( Mat(contours[i]) );
        if( contours[i].size() > 5 && contours[i].size() > 150 )
         minEllipse[i] = fitEllipse( Mat(contours[i]) );
        
    }
    
    
    // Back to Normal
    for (int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area > largest){
            largest = area;
            indx = i;
        }
        Rect rect = boundingRect(contours[i]);
        int radius = rect.width/2;
        
        // If contour is big enough and has round shape
        // Then it is the pupil
        if (area >= 30 &&
            abs(1 - ((double)rect.width / (double)rect.height)) <= 0.2 &&
            abs(1 - (area / (CV_PI * pow((double)radius, (double)2)))) <= 0.2)
        {
            circle(coloredEye, Point(rect.x + radius, rect.y + radius), radius, CV_RGB(255,0,0), 2);
        }
    }
    
    if (indx == 0)
        return;
    eclipseFound = minEllipse[indx];
}
    


