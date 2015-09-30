/*Author: Yuhang He
 * Email: yuhanghe@whu.edu.cn
 * Date: March, 2015
 *
 */
#ifndef _UPSAMPLE_H_
#define _UPSAMPLE_H_

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include"projection.h"

//this function is implemented for depth estimation by using geodesic distance;
bool geoDepEstimation(const cv::Mat& rgbImg, arma::Cube<float>& matrix2D, arma::Mat<float>& geoImg);

cv::Mat imgPadding(cv::Mat& labImg, int paddingRows, int paddingCols);

arma::Mat<float> matPadding(arma::Mat<float>& unPadMat, int paddingRows = 0, int paddingCols = 0);

arma::Mat<float> computeMeanColor( cv::Mat& imgPatch );

arma::Mat<float> depUpsampling( arma::Cube<float>& matrix2D,  arma::Mat<float>& geoEstImg, const cv::Mat& rgbImg );

arma::Mat<float> depUpsamplingDynamic( arma::Cube<float>& matrix2DNoPadding, arma::Mat<float>& geoEstImgNoPadding, const cv::Mat& rgbImgNoPadding );

arma::Mat<float> depUpsamplingRandom( arma::Cube<float>& matrix2DNoPadding, arma::Mat<float>& geoEstImgNoPadding, const cv::Mat& rgbImgNoPadding );

void inhomogeneityClean( arma::Mat<float>& sparseDepImg, int paddingSize );

arma::Cube<float> imgNormalize( cv::Mat& labImg );

arma::Cube<float> cubeAdjust( arma::Cube<float>& matPad );

arma::Mat<float> matNormalize( arma::Mat<float>& normImg );

arma::Mat<float> computeBilateralFilter( arma::Cube<float>& imgPatch, float sigma );
#endif
