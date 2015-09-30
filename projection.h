//this file is newly edited by Yuhang He on Mar. 16, 2015
//and this file is created for depth map, height above the ground
//and surface normal estimation;

#ifndef _PROJECTION_H_
#define _PROJECTION_H_
#include <iostream>
#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <cmath>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <armadillo>
#include <glog/logging.h>
struct boundingBox
{
    float left;
    float top;
    float right;
    float bottom;
    boundingBox()
    {
        left = 0.;
        top = 0.;
        right = 0.;
        bottom = 0.;
    }
};

struct dimension3D
{
    float height;
    float width;
    float length;
    dimension3D()
    {
        height = 0.;
        width = 0.;
        length = 0.;
    }
};

struct location3D
{
    float x;
    float y;
    float z;
    location3D()
    {
        x = 0.;
        y = 0.;
        z = 0.;
    }
};

struct objectLabel
{
    std::string objectType;
    float truncation;
    float occlusion;
    float alphaAngle;
    struct boundingBox bbox;
    struct dimension3D dimension;
    struct location3D location;
    float ry; //ry is rotation_y;
    Eigen::MatrixXf boundingBox3D;
    Eigen::Vector3d boundingBoxColor;

    objectLabel()
    {
        objectType = "No Type";
        truncation = -1;
        occlusion = -1;
        alphaAngle = -2*CV_PI;
        ry = -1.;
        boundingBox3D = Eigen::MatrixXf::Zero(3,8);
        boundingBoxColor(0) = 0.;
        boundingBoxColor(1) = 0.;
        boundingBoxColor(2) = 0.;
    }

};
//this function is implemented to project 3D points onto 2D image plane;
bool projection2D23D( arma::Cube<float>& matrix2D, const arma::Mat<float>& projectionMatrix, const std::string& veloPointFile,
                      const arma::Mat<float>& cameraMatrix_arma, const arma::Mat<float>& R0_rect_arma, const arma::Mat<float>& velo2camera_arma);
//this function is created for projecting 2D image point to 3D world;
bool reprojection(const Eigen::Vector2f& location, float depVal, const cv::Mat& projectMatrix, pcl::PointXYZ& point3D );

//this function is created for reading labelFile for each image;
bool readObjectLabel( std::vector<struct objectLabel>& objectLabels, const std::string& labelName, const Eigen::MatrixXf& R0_rect, const Eigen::MatrixXf& velo2camera );

//this function is created for reading projectionMatrix for each image;
//bool readProjectionMatrix( cv::Mat& projectionMatrix, const std::string& cameraFile);
bool readProjectionMatrix( cv::Mat& projectionMatrix, cv::Mat& cameraMatrix, cv::Mat& R0_rect, cv::Mat& velo2camera, const std::string& cameraFile);

//compute 3D BoundingBox of each label for each image;
bool compute3DBoundingBox( std::vector<struct objectLabel>& objectLabels, const Eigen::MatrixXf& R0_rect, const Eigen::MatrixXf& velo2camera );

//this function is implemented for reading pointXYZ from .bin velodyne files;
bool readPointXYZ(const std::string veloDataDir, pcl::PointCloud<pcl::PointXYZ>::Ptr veloCloudPtr, int imgRows, int imgCols, const arma::Mat<float>& projectionMatrix);

//this function is implemented for generating pointXYZRGB from RGB image and denseDepImg;
bool generatePointXYZRGB(const cv::Mat& rgbImg, const cv::Mat& denseDepImg, const cv::Mat& projectionMatrix, pcl::PointCloud<pcl::PointXYZRGB>::Ptr veloCloudPtr);
#endif
