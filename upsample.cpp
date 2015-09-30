#include"upsample.h"

cv::Mat imgPadding(cv::Mat& labImg, int paddingRows, int paddingCols)
{
    /*
    for( int i = 0; i < labImg.rows; i++ )
      {
        for( int j = 0; j < labImg.cols; j++ )
        {
          if( labImg.at<cv::Vec3b>(i,j)[0] > 0 ||
              labImg.at<cv::Vec3b>(i,j)[1] > 0 ||
              labImg.at<cv::Vec3b>(i,j)[2] >0 )
          {
            std::cout << "finished rgbTest in the imgPadding, it's OK!\n";
            char ch3;
            std::cin.get(ch3);
          }
        }
      }
      */

    //std::cout << "begin to pad rgbImg!\n";
    //std::cout << "before rgb image padding, the size is: " <<  labImg.rows  << "x" << labImg.cols << std::endl;
    cv::Mat padImg(cv::Size((labImg.cols+2*paddingCols),(labImg.rows+2*paddingRows)), CV_8UC3, cv::Scalar::all(0));

    //padImg(cv::Range(paddingRows,padImg.rows-paddingRows), cv::Range(paddingCols, padImg.cols-paddingCols)) = labImg.clone();
    labImg.copyTo(padImg(cv::Range(paddingRows,padImg.rows-paddingRows), cv::Range(paddingCols, padImg.cols-paddingCols)));

    // padImg(cv::Range(0,paddingRows), cv::Range::all()) = labImg(cv::Range(0,paddingRows), cv::Range::all()).clone();
    labImg(cv::Range(0,paddingRows), cv::Range::all()).copyTo(padImg(cv::Range(0,paddingRows), cv::Range(paddingCols, padImg.cols - paddingCols)));
    // padImg(cv::Range(padImg.rows - paddingRows, padImg.rows), cv::Range::all()) = labImg(cv::Range(labImg.rows - paddingRows, labImg.rows), cv::Range::all()).clone();
    labImg(cv::Range(labImg.rows - paddingRows, labImg.rows), cv::Range::all()).copyTo(padImg(cv::Range(padImg.rows - paddingRows, padImg.rows),
            cv::Range(paddingCols, padImg.cols - paddingCols)));

    //padImg(cv::Range::all(), cv::Range(0,paddingCols)) = labImg(cv::Range::all(), cv::Range(0,paddingCols)).clone();
    labImg(cv::Range::all(), cv::Range(0,paddingCols)).copyTo(padImg(cv::Range(paddingRows, padImg.rows - paddingRows), cv::Range(0,paddingCols)));
    labImg(cv::Range::all(), cv::Range(labImg.cols-paddingCols, labImg.cols)).copyTo(padImg(cv::Range(paddingRows, padImg.rows - paddingRows),
            cv::Range(padImg.cols-paddingCols, padImg.cols)) );
    /*
    for( int i = 0; i < padImg.rows; i++ )
    {
      for( int j = 0; j < padImg.cols; j++ )
      {
        if( padImg.at<cv::Vec3b>(i,j)[0] > 0 ||
            padImg.at<cv::Vec3b>(i,j)[1] > 0 ||
            padImg.at<cv::Vec3b>(i,j)[2] >0 )
        {
          std::cout << "finished padImg test in the imgPadding, it's OK!\n";
          char ch3;
          std::cin.get(ch3);
        }
      }
    }
    */
    return padImg;
}

arma::Mat<float> matPadding( arma::Mat<float>& unPadMat, int paddingRows, int paddingCols)
{
    CHECK( arma::accu(unPadMat) > 0 ) << "the input unPadMat is not OK!\n";
    arma::Mat<float> matPad((unPadMat.n_rows+2*paddingRows),(unPadMat.n_cols + 2*paddingCols));
    matPad(arma::span(paddingRows,matPad.n_rows-paddingRows-1), arma::span(paddingCols,matPad.n_cols-paddingCols-1)) = unPadMat;

    matPad.rows(0,paddingRows-1) = matPad.rows(paddingRows,2*paddingRows-1);
    matPad.rows(matPad.n_rows-paddingRows,matPad.n_rows-1) = matPad.rows(matPad.n_rows-2*paddingRows, matPad.n_rows-paddingRows-1);

    matPad.cols(0,paddingCols-1) = matPad.cols(paddingCols,2*paddingCols-1);
    matPad.cols(matPad.n_cols-paddingCols,matPad.n_cols-1) = matPad.cols(matPad.n_cols-2*paddingCols, matPad.n_cols - paddingCols -1);

    CHECK( arma::accu(matPad) > 0 ) << "failed to pad the mat!\n";


    /*
    for( int i = 0; i < matPad.n_rows; i++ )
    {
      for( int j = 0; j < matPad.n_cols; j++ )
      {
        float rowValTmp = matPad(i,j);
        int rowTmp = int( rowValTmp + 0.5 );
        float rowTag = rowValTmp - (float)rowTmp;
        matPad(i,j) = float(i) + rowTag;

        float colValTmp = matPad.slice(1)(i,j);
        int colTmp = int( colValTmp + 0.5 );
        float colTag = colValTmp - (float)colTmp;
        matPad.slice(1)(i,j) = float(j) + colTag;
      }
    }
    */
    //std::cout << "padding result: " << matPad.n_rows << "x" << matPad.n_cols << std::endl;
    return matPad;
}

arma::Cube<float> cubeAdjust( arma::Cube<float>& matPad )
{
    for( int i = 0; i < matPad.slice(0).n_rows; i++ )
    {
        for( int j = 0; j < matPad.slice(0).n_cols; j++ )
        {
            float rowValTmp = matPad.slice(0)(i,j);
            int rowTmp = int( rowValTmp + 0.5 );
            float rowTag = rowValTmp - (float)rowTmp;
            matPad.slice(0)(i,j) = float(i) + rowTag;

            float colValTmp = matPad.slice(1)(i,j);
            int colTmp = int( colValTmp + 0.5 );
            float colTag = colValTmp - (float)colTmp;
            matPad.slice(1)(i,j) = float(j) + colTag;
        }
    }

    return matPad;
}
bool geoDepEstimation(const cv::Mat& rgbImgNoPadding, arma::Cube<float>& matrix2DNoPadding, arma::Mat<float>& geoImgNoPadding)
{
    // std::cout << "begin to estimate the geoDepImg!\n";
    if( (rgbImgNoPadding.rows != matrix2DNoPadding.slice(0).n_rows) || (rgbImgNoPadding.cols != geoImgNoPadding.n_cols) )
    {
        fprintf(stderr, "the input rgbImg, matrix2D and geoImg do not share the same size!\n");
        return false;
    }
    //define the distance weight for spatial distance and color distance, respectively;
    const float thetaSpatial = 0.5;
    const float thetaColor = 15.;
    //cv::imshow("inputRGB", rgbImgNoPadding);
    //cv::waitKey(0);
    cv::Mat labImg;
    rgbImgNoPadding.copyTo(labImg);
    //cv::imshow("labImg", labImg);
    //cv::waitKey(0);
    //cv::cvtColor(rgbImgNoPadding, labImg, CV_BGR2Lab);
    //padding the RGB image and sparseDepImg;
    int paddingSize = 11;
    labImg = imgPadding(labImg, paddingSize, paddingSize);
    //cv::imshow("labImgAfterPadding", labImg);
    //cv::waitKey(0);

    //labImg = imgPadding(rgbImgNoPadding, paddingSize, paddingSize);
    //std::cout << "After image padding, the image size is: " << labImg.rows << "x" << labImg.cols << std::endl;
    arma::Cube<float> matrix2D( labImg.rows, labImg.cols, 3 );
    //std::cout << "begin padding matrix2D!\n";
    matrix2D.slice(0) = matPadding(matrix2DNoPadding.slice(0), paddingSize, paddingSize);
    matrix2D.slice(1) = matPadding(matrix2DNoPadding.slice(1), paddingSize, paddingSize);
    matrix2D.slice(2) = matPadding(matrix2DNoPadding.slice(2), paddingSize, paddingSize);
    matrix2D = cubeAdjust(matrix2D);
    cv::Mat testImg( cv::Size( matrix2D.slice(2).n_cols, matrix2D.slice(2).n_rows ), CV_8UC1, cv::Scalar::all(0) );
    int maxValueImg = 0;
    int minValueImg = 255;
    for( int i = 0; i < testImg.rows; i++ )
    {
        for( int j = 0; j < testImg.cols; j++ )
        {
            testImg.at<uchar>(i,j) = int(matrix2D.slice(2)(i,j) + 0.5);
            if( int(matrix2D.slice(2)(i,j) + 0.5) > 0 && int( matrix2D.slice(2)(i,j) + 0.5 ) < minValueImg )
                minValueImg = int( matrix2D.slice(2)(i,j) + 0.5 );
            if( int(matrix2D.slice(2)(i,j) + 0.5) > 0 && int( matrix2D.slice(2)(i,j) + 0.5 ) > maxValueImg )
                maxValueImg = int( matrix2D.slice(2)(i,j) + 0.5 );
            //std::cout << "the depth value is " << int(geoImgNoPadding(i,j) + 0.5) << std::endl;
        }
    }
    testImg = testImg*(255.0/maxValueImg);
    //cv::imshow("sparseWindowAfter Padding", testImg);
    //cv::waitKey(0);

    arma::Mat<float> labelImg( matrix2D.slice(2).n_rows, matrix2D.slice(2).n_cols );
    labelImg = matrix2D.slice(2);
    CHECK( arma::accu( matrix2D.slice(2)) > 0. ) << " in the geoDepEstimation, the matrix2D.slice(2) is failed!\n";
    //std::cout << "end padding matrix2D!\n";
    //std::cout << "the matrix2D size is: " << matrix2D.slice(2).n_rows << "x" << matrix2D.slice(2).n_cols << std::endl;
    //normalise the labImg
    // cv::normalize(labImg, labImg);
    arma::Cube<float> cubeImg = imgNormalize(labImg);
    /*
    for(int i = 0; i < labImg.rows; i++)
    {
      for(int j = 0; j < labImg.cols; j++)
      {
        std::cout << "R= " << cubeImg.slice(0)(i,j) << " G=  "
                  << cubeImg.slice(1)(i,j)  << " B= " <<  cubeImg.slice(2)(i,j) << std::endl;

      }
    }
    */
    //the first channel stores the geoDistance, while the second channel stores the relevant depth value;
    arma::Cube<float> geoDisImg(labImg.rows, labImg.cols, 2);
    geoDisImg.slice(0).fill(-1.0);
    //geoDisImg.slice(1).fill(-1);
    geoDisImg.slice(1) = matrix2D.slice(2);
    for(int i = 0; i < geoDisImg.slice(1).n_rows; i++)
    {
        for(int j = 0; j < geoDisImg.slice(1).n_cols; j++)
        {
            if( geoDisImg.slice(1)(i,j) > 0. )
                geoDisImg.slice(0)(i,j) = 0.;
        }
    }
    CHECK( arma::accu( geoDisImg.slice(1) ) > 0 ) << "the input geoEstImg is failed!\n";
    //arma::Mat<float> labelDepImg = matrix2D.slice(2);
    int iterNum = 0;
    std::cout << "begin to geoEstimation!!!\n";
    while( iterNum < 4 )
    {
        iterNum++;
        //forwarsPassing
        for(int i = 2 + 0; i < geoDisImg.slice(1).n_rows-1; i++)
        {
            for(int j = 2; j < geoDisImg.slice(1).n_cols-1; j++)
            {
                if( labelImg(i,j) > 0. )
                    continue;
                float geoDisTmp[4];
                float depValTmp[4];
                for(int k = 0; k < 4; k++)
                {
                    geoDisTmp[k] = -1.0;
                    depValTmp[k] = -1.0;
                }
                arma::Mat<float> colorVal(3,1);
                colorVal << cubeImg.slice(2)(i,j) << arma::endr << cubeImg.slice(1)(i,j) << arma::endr << cubeImg.slice(0)(i,j);

                if( geoDisImg.slice(0)(i-1,j-1) != -1.0 )
                {
                    arma::Mat<float> colorValTmp(3,1);
                    colorValTmp << cubeImg.slice(2)(i-1,j-1) << arma::endr << cubeImg.slice(1)(i-1,j-1) << arma::endr << cubeImg.slice(0)(i-1,j-1);
                    float spatialDis = sqrt(2);
                    float colorDis = sqrt( arma::accu((colorVal - colorValTmp)%(colorVal - colorValTmp)) );
                    geoDisTmp[0] = thetaSpatial*spatialDis + thetaColor*colorDis + geoDisImg.slice(0)(i-1,j-1);
                    depValTmp[0] = geoDisImg.slice(1)(i-1,j-1);
                    //std::cout << "the spatialDis = " << spatialDis << " colorDis = " << colorDis << std::endl;
                }

                if( geoDisImg.slice(0)(i-1,j) != -1.0 )
                {
                    arma::Mat<float> colorValTmp(3,1);
                    colorValTmp << cubeImg.slice(2)(i-1,j) << arma::endr << cubeImg.slice(1)(i-1,j) << arma::endr << cubeImg.slice(0)(i-1,j);
                    float spatialDis = 1.0;
                    float colorDis = sqrt( arma::accu((colorVal - colorValTmp)%(colorVal - colorValTmp)) );
                    geoDisTmp[1] = thetaSpatial*spatialDis + thetaColor*colorDis + geoDisImg.slice(0)(i-1,j);
                    depValTmp[1] = geoDisImg.slice(1)(i-1,j);
                }

                if( geoDisImg.slice(0)(i-1,j+1) != -1.0 )
                {
                    arma::Mat<float> colorValTmp(3,1);
                    colorValTmp << cubeImg.slice(2)(i-1,j+1) << arma::endr << cubeImg.slice(1)(i-1,j+1) << arma::endr << cubeImg.slice(0)(i-1,j+1);
                    float spatialDis = sqrt(2);
                    float colorDis = sqrt( arma::accu((colorVal - colorValTmp)%(colorVal - colorValTmp)) );
                    geoDisTmp[2] = thetaSpatial*spatialDis + thetaColor*colorDis + geoDisImg.slice(0)(i-1,j+1);
                    depValTmp[2] = geoDisImg.slice(1)(i-1,j+1);
                }

                if( geoDisImg.slice(0)(i,j-1) != -1.0 )
                {
                    arma::Mat<float> colorValTmp(3,1);
                    colorValTmp << cubeImg.slice(2)(i,j-1) << arma::endr << cubeImg.slice(1)(i,j-1) << arma::endr << cubeImg.slice(0)(i,j-1);
                    float spatialDis = 1.0;
                    float colorDis = sqrt( arma::accu((colorVal - colorValTmp)%(colorVal - colorValTmp)) );
                    geoDisTmp[3] = thetaSpatial*spatialDis + thetaColor*colorDis + geoDisImg.slice(0)(i,j-1);
                    depValTmp[3] = geoDisImg.slice(1)(i,j-1);
                }

                float maxGeoDis = 0.;
                float minGeoDis = 1000000.0;
                int minLocation = 0;
                for(int k = 0; k < 4; k++)
                {
                    if( geoDisTmp[k] > maxGeoDis )
                        maxGeoDis = geoDisTmp[k];
                    if( geoDisTmp[k] < minGeoDis && geoDisTmp[k] > 0. )
                    {
                        minGeoDis = geoDisTmp[k];
                        minLocation = k;
                    }

                }
                if( maxGeoDis == 0.0 )
                    continue;
                geoDisImg.slice(1)(i,j) = depValTmp[minLocation];
                geoDisImg.slice(0)(i,j) = geoDisTmp[minLocation];
            }
        }


        std::cout << "finished forwardpassing!\n";
        //backward passing
        for(int i = geoDisImg.slice(1).n_rows - 2; i >= 2 + 0; i--)
        {
            for(int j = geoDisImg.slice(1).n_cols - 2; j >= 2; j--)
            {
                if( labelImg(i,j) > 0. )
                    continue;
                float geoDisTmp[4];
                float depValTmp[4];
                for(int k = 0; k < 4; k++)
                {
                    geoDisTmp[k] = -1.0;
                    depValTmp[k] = -1.0;
                }
                arma::Mat<float> colorVal(3,1);
                colorVal << cubeImg.slice(2)(i,j) << arma::endr << cubeImg.slice(1)(i,j) << arma::endr << cubeImg.slice(0)(i,j);

                if( geoDisImg.slice(0)(i,j+1) != -1.0 )
                {
                    arma::Mat<float> colorValTmp(3,1);
                    colorValTmp << cubeImg.slice(2)(i,j+1) << arma::endr << cubeImg.slice(1)(i,j+1) << arma::endr << cubeImg.slice(0)(i,j+1);
                    float spatialDis = 1.0;
                    float colorDis = sqrt( arma::accu((colorVal - colorValTmp)%(colorVal - colorValTmp)) );
                    geoDisTmp[0] = thetaSpatial*spatialDis + thetaColor*colorDis + geoDisImg.slice(0)(i,j+1);
                    depValTmp[0] = geoDisImg.slice(1)(i,j+1);
                }

                if( geoDisImg.slice(0)(i+1,j-1) != -1.0 )
                {
                    arma::Mat<float> colorValTmp(3,1);
                    colorValTmp << cubeImg.slice(2)(i+1,j-1) << arma::endr << cubeImg.slice(1)(i+1,j-1) << arma::endr << cubeImg.slice(0)(i+1,j-1);
                    float spatialDis = sqrt(2);
                    float colorDis = sqrt( arma::accu((colorVal - colorValTmp)%(colorVal - colorValTmp)) );
                    geoDisTmp[1] = thetaSpatial*spatialDis + thetaColor*colorDis + geoDisImg.slice(0)(i+1,j-1);
                    depValTmp[1] = geoDisImg.slice(1)(i+1,j-1);
                }

                if( geoDisImg.slice(0)(i+1,j) != -1.0 )
                {
                    arma::Mat<float> colorValTmp(3,1);
                    colorValTmp << cubeImg.slice(2)(i+1,j) << arma::endr << cubeImg.slice(1)(i+1,j) << arma::endr << cubeImg.slice(0)(i+1,j);
                    float spatialDis = 1.0;
                    float colorDis = sqrt( arma::accu((colorVal - colorValTmp)%(colorVal - colorValTmp)) );
                    geoDisTmp[2] = thetaSpatial*spatialDis + thetaColor*colorDis + geoDisImg.slice(0)(i+1,j);
                    depValTmp[2] = geoDisImg.slice(1)(i+1,j);
                }

                if( geoDisImg.slice(0)(i+1,j+1) != -1.0 )
                {
                    arma::Mat<float> colorValTmp(3,1);
                    colorValTmp << cubeImg.slice(2)(i+1,j+1) << arma::endr << cubeImg.slice(1)(i+1,j+1) << arma::endr << cubeImg.slice(0)(i+1,j+1);
                    float spatialDis = sqrt(2);
                    float colorDis = sqrt( arma::accu((colorVal - colorValTmp)%(colorVal - colorValTmp)) );
                    geoDisTmp[3] = thetaSpatial*spatialDis + thetaColor*colorDis + geoDisImg.slice(0)(i+1,j+1);
                    depValTmp[3] = geoDisImg.slice(1)(i+1,j+1);
                }

                float maxGeoDis = 0.;
                float minGeoDis = 1000000.0;
                int minLocation = 0;
                for(int k = 0; k < 4; k++)
                {
                    if( geoDisTmp[k] > maxGeoDis )
                        maxGeoDis = geoDisTmp[k];
                    if( geoDisTmp[k] < minGeoDis && geoDisTmp[k] > 0. )
                    {
                        minGeoDis = geoDisTmp[k];
                        minLocation = k;
                    }

                }
                if( maxGeoDis == 0.0 )
                    continue;
                geoDisImg.slice(1)(i,j) = depValTmp[minLocation];
                geoDisImg.slice(0)(i,j) = geoDisTmp[minLocation];
            }
        }
        std::cout << "finished backward passing...\n";
    }
    //std::cout << "finished backward passing!\n";
    geoImgNoPadding = geoDisImg.slice(1)(arma::span(paddingSize, geoDisImg.slice(0).n_rows - paddingSize -1),
                                         arma::span(arma::span(paddingSize, geoDisImg.slice(0).n_cols - paddingSize -1)));
    //check the geodesic estimation result!

    CHECK( arma::accu( geoImgNoPadding) > 0. ) << "the geoEstImg estimation is failed!\n";
    return true;

}

arma::Mat<float> computeBilateralFilter( arma::Cube<float>& imgPatch, float sigma )
{
    int patchRows = imgPatch.slice(0).n_rows;
    int patchCols = imgPatch.slice(1).n_cols;

    arma::Mat<float> bilateralFilter( patchRows, patchCols );
    bilateralFilter.fill( 0.0 );

    arma::Mat<float> dR( patchRows, patchCols );
    dR.fill( 0.0 );

    arma::Mat<float> dG( patchRows, patchCols );
    dG.fill( 0.0 );

    arma::Mat<float> dB( patchRows, patchCols );
    dB.fill( 0.0 );

    dR = imgPatch.slice(0) - imgPatch.slice(0)( (patchRows - 1)/2, (patchCols - 1)/2 );
    dG = imgPatch.slice(1) - imgPatch.slice(1)( (patchRows - 1)/2, (patchCols - 1)/2 );
    dB = imgPatch.slice(2) - imgPatch.slice(2)( (patchRows - 1)/2, (patchCols - 1)/2 );

// std::cout << "dR = \n " << dR << std::endl;
// std::cout << "dG = \n " << dG << std::endl;
// std::cout << "dB = \n " << dB << std::endl;
    arma::Mat<float> finalPatch( patchRows, patchCols );
    finalPatch.fill( 0.0 );

    finalPatch = dR%dR + dG%dG + dB%dB;
    //std::cout << "before normalize, the finalPatch = \n" << finalPatch << std::endl;
    finalPatch = matNormalize( finalPatch );
    //std::cout << "After normalize, the finalPatch = \n" << finalPatch << std::endl;
    //char ch; std::cin.get(ch); std::cin.get(ch);
    float finalMaxVal = 0.0;
    float finalMinVal = 10000.0;
    for( int i = 0; i < finalPatch.n_rows; i++ )
    {
        for( int j = 0; j < finalPatch.n_cols; j++ )
        {
            if( finalMaxVal < finalPatch(i,j) )
                finalMaxVal = finalPatch(i,j);
            if( finalMinVal > finalPatch(i,j) && finalPatch(i,j) > 0. )
                finalMinVal = finalPatch(i,j);
        }
    }
    //std::cout << "finalMaxVal = " << finalMaxVal << " finalMinVal = " << finalMinVal << std::endl;
    float finalSigma = 0.;
    if( finalMaxVal == finalMinVal )
        finalSigma = 1.0;
    else
        //finalSigma = -log( finalMaxVal - finalMinVal );
        //finalSigma = (finalMaxVal - finalMinVal);
        finalSigma = (finalMaxVal + finalMinVal)/2;

    //std::cout << " finalSigma = " << finalSigma << std::endl;
    bilateralFilter = exp( -(finalPatch)/(2*finalSigma*finalSigma) );
    //bilateralFilter = exp( -( dR%dR + dG%dG + dB%dB )/(2*sigma*sigma) );

    //std::cout << "after normalize, bilateralFilter = \n" << bilateralFilter << std::endl;
    //char ch; std::cin.get(ch); std::cin.get(ch);
    return bilateralFilter;
}

arma::Mat<float> computeMeanColor( arma::Cube<float>& imgPatch )
{
    float R = 0.;
    float G = 0.;
    float B = 0.;
    for( int i = 0; i < imgPatch.n_rows; i++ )
    {
        for( int j = 0; j < imgPatch.n_cols; j++ )
        {
            R += imgPatch.slice(0)(i,j);
            G += imgPatch.slice(1)(i,j);
            B += imgPatch.slice(2)(i,j);
        }
    }
    R /= (imgPatch.n_rows*imgPatch.n_cols);
    G /= (imgPatch.n_rows*imgPatch.n_cols);
    B /= (imgPatch.n_rows*imgPatch.n_cols);

    // std::cout << "R = " << R << " G = " << G << " B = " << B << std::endl;
    arma::Mat<float> meanColor(3,1);
    meanColor << R << arma::endr << G << arma::endr << B;

    return meanColor;
}

arma::Mat<float> depUpsampling( arma::Cube<float>& matrix2DNoPadding, arma::Mat<float>& geoEstImgNoPadding, const cv::Mat& rgbImgNoPadding )
{
    int paddingSize = 15;
    int colorPatchSize = 7;
    const float sigmaSpatial = 2.0;
    /*
    arma::Mat<float> spatialMatRow(paddingSize, paddingSize);
    arma::Mat<float> spatialMatCol(paddingSize, paddingSize);
    for(int i = 0; i < paddingSize; i++)
    {
     spatialMatRow.col(i) = arma::linspace<arma::Mat<float> >(1,paddingSize, paddingSize);
     spatialMatCol.row(i) = arma::linspace<arma::Mat<float> >(1,paddingSize, paddingSize).t();
    }
    arma::Mat<float> spatialWeight = exp(-((spatialMatRow - 5)%(spatialMatRow - 5) + (spatialMatCol - 5)%(spatialMatCol -5))/(2*sigmaSpatial*sigmaSpatial));
    */
    //padding relevant matrixs;
    cv::Mat labImg;
    rgbImgNoPadding.copyTo(labImg);
    //cv::cvtColor(labImg, labImg, CV_RGB2Lab);
    labImg = imgPadding(labImg, paddingSize, paddingSize);
    arma::Cube<float> imgCube( labImg.rows, labImg.cols, 3 );
    imgCube = imgNormalize( labImg );
    arma::Cube<float> matrix2D(labImg.rows, labImg.cols, 3);
    matrix2D.slice(0) = matPadding(matrix2DNoPadding.slice(0), paddingSize, paddingSize);
    matrix2D.slice(1) = matPadding(matrix2DNoPadding.slice(1), paddingSize, paddingSize);
    matrix2D.slice(2) = matPadding(matrix2DNoPadding.slice(2), paddingSize, paddingSize);

    //std::cout << "begin to adjust ... " << std::endl;
    matrix2D = cubeAdjust( matrix2D );
    //test matPadding result;
    /*
    for( int i = 100; i < 110; i++ )
    {
      for( int j = 100; j < 110; j++ )
      {
        std::cout << "(i,j) = (" << i << " , " << j << " )" << std::endl;
        std::cout << "NoPadding(i,j) = (" << matrix2DNoPadding.slice(0)(i,j) << " , " << matrix2DNoPadding.slice(1)(i,j) << " )" << std::endl;
        //std::cout << "(i,j) = (" << i << " , " << j << " )" << std::endl;
      }
    }
    */
    arma::Mat<float> geoEstImg;
    geoEstImg = matPadding(geoEstImgNoPadding, paddingSize, paddingSize);
    //geoEstImg = matNormalise(geoEstImg);
    arma::Mat<float> denseDepImg = matrix2D.slice(2);
    for(int i = paddingSize + 1 + 0; i < denseDepImg.n_rows - paddingSize; i++)
    {
        for(int j = paddingSize + 1; j < denseDepImg.n_cols - paddingSize; j++)
        {
            if( denseDepImg(i,j) > 0 ) // here the denseDepImg serves as a labelImg;
                continue;
            //std::cout << "estimate point: " << i << "x" << j << std::endl;
            arma::Mat<float> depPatch = matrix2D.slice(2)(arma::span(i-(paddingSize-1)/2,i+(paddingSize-1)/2),
                                        arma::span(j-(paddingSize-1)/2,j+(paddingSize-1)/2));
            // std::cout << "depPatch size is: " << depPatch.n_rows << "x" << depPatch.n_cols << std::endl;
            arma::Mat<float> depPatchOri = depPatch;
            // std::cout << "the input depPatch =\n " << depPatchOri << std::endl;
            bool centerExist = false;
            if(depPatch((paddingSize-1)/2,(paddingSize-1)/2) > 0)
                centerExist = true;
            // depPatch = matNormalise(depPatch);
            // geoEstImg = arma::normalise(geoEstImg);
            if( arma::accu( depPatchOri ) == 0 )
                continue;
            arma::uvec seedMat = arma::find(depPatchOri > 0);
            if( seedMat.n_rows == 1 )
            {
                denseDepImg(i,j) = depPatchOri( seedMat(0,0) );
                continue;
            }
            //calculate the spatial distance weight;
            arma::Cube<float> cubePatch( paddingSize, paddingSize, 3 );
            cubePatch.fill(0.0);
            cubePatch.slice(0) = matrix2D.slice(0)(arma::span((i-(paddingSize-1)/2),(i+(paddingSize-1)/2)),
                                                   arma::span((j-(paddingSize-1)/2), (j+(paddingSize-1)/2)));
            cubePatch.slice(1) = matrix2D.slice(1)(arma::span((i-(paddingSize-1)/2),(i+(paddingSize-1)/2)),
                                                   arma::span((j-(paddingSize-1)/2), (j+(paddingSize-1)/2)));
            cubePatch.slice(2) = matrix2D.slice(2)(arma::span((i-(paddingSize-1)/2),(i+(paddingSize-1)/2)),
                                                   arma::span((j-(paddingSize-1)/2), (j+(paddingSize-1)/2)));
            //std::cout << "cubePatch size is: " << cubePatch.slice(2).n_rows << "x" << cubePatch.slice(2).n_cols << std::endl;
            arma::Mat<float> spaWeiPatch(paddingSize, paddingSize);
            spaWeiPatch.fill(0.0);
            arma::Mat<float> depWeiPatch(paddingSize, paddingSize);
            depWeiPatch.fill(0.0);
            arma::Mat<float> colorWeiPatch(paddingSize, paddingSize);
            colorWeiPatch.fill(0.0);
            //compute DepWeiPatch;
            arma::Mat<float> depPatchTmp = depPatch;
            depPatchTmp((paddingSize-1)/2, (paddingSize-1)/2) = geoEstImg(i,j);
            depPatch = matNormalize( depPatch );
            depPatchTmp = matNormalize( depPatchTmp );
            //std::cout << "after normalization, the depPatch is: \n" << depPatch << std::endl;
            // char chDep; std::cin.get(chDep); std::cin.get(chDep);
            float iniDepVal = depPatchTmp((paddingSize-1)/2, (paddingSize-1)/2);
            float maxDepVal = 0.;
            maxDepVal = arma::max(arma::max( depPatchTmp ));
            float minDepVal = 0.;
            if( maxDepVal > 0. )
                minDepVal = arma::min( arma::min( depPatchTmp( arma::find( depPatchTmp > 0 ) ) ) );
            float sigmaDep = fabs(maxDepVal - minDepVal) < 0.0001?1:(-log(maxDepVal - minDepVal));
            depWeiPatch = exp( -((depPatch - iniDepVal)%(depPatch - iniDepVal))/(2*sigmaDep*sigmaDep));
            for( int i1 = 0; i1 < depWeiPatch.n_rows; i1++ )
            {
                for( int j1 = 0; j1 < depWeiPatch.n_cols; j1++ )
                {
                    if( depPatchOri(i1,j1) == 0. )
                        depWeiPatch(i1,j1) = 0.;
                }
            }
            //depWeiPatch( arma::find( depPatch == 0 ) )  = 0.;

            //compute spaWeiPatch and colorWerPatch;
            arma::Mat<float> colorPatch(3,1);
            colorPatch.fill(0.0);

            cv::Mat labImgTmp; // = labImg(cv::Range(i-(colorPatchSize-1)/2, i + (colorPatchSize+1)/2), cv::Range(j-(colorPatchSize-1)/2, j + (colorPatchSize+1)/2)).clone();
            labImg(cv::Range(i-(colorPatchSize-1)/2, i + (colorPatchSize+1)/2), cv::Range(j-(colorPatchSize-1)/2, j + (colorPatchSize+1)/2)).copyTo( labImgTmp );
            arma::Cube<float> imgCubeTmp( colorPatchSize, colorPatchSize, 3 );
            imgCubeTmp.fill( 0.0 );
            imgCubeTmp.slice(0)= imgCube.slice(0)( arma::span( i - (colorPatchSize-1)/2, i + (colorPatchSize-1)/2),
                                                   arma::span( j - (colorPatchSize-1)/2, j + (colorPatchSize-1)/2) );
            imgCubeTmp.slice(1)= imgCube.slice(1)( arma::span( i - (colorPatchSize-1)/2, i + (colorPatchSize-1)/2),
                                                   arma::span( j - (colorPatchSize-1)/2, j + (colorPatchSize-1)/2) );
            imgCubeTmp.slice(2)= imgCube.slice(2)( arma::span( i - (colorPatchSize-1)/2, i + (colorPatchSize-1)/2),
                                                   arma::span( j - (colorPatchSize-1)/2, j + (colorPatchSize-1)/2) );
            //std::cout << "labImgTmp size is: " << labImgTmp.rows << "x" << labImgTmp.cols << std::endl;
            //std::cout << "the labImgTmp is:\n" << labImgTmp << std::endl;
            colorPatch = computeMeanColor( imgCubeTmp );
            //std::cout << "the colorPatch is " << colorPatch << std::endl;
            for(int i1 = (i - (paddingSize-1)/2); i1 <= (i + (paddingSize-1)/2); i1++)
            {
                for(int j1 = (j - (paddingSize-1)/2); j1 <= (j + (paddingSize-1)/2); j1++)
                {
                    if( matrix2D.slice(2)(i1, j1) == 0.0 )
                        continue;

                    float locRow = matrix2D.slice(0)(i1,j1);
                    float locCol = matrix2D.slice(1)(i1,j1);
                    //  std::cout << "(i1, j1) = (" << i1 << " , " << j1 << ")" << std::endl;
                    //  std::cout << "(i, j) = (" << i << " , " << j << ")" << std::endl;
                    //  std::cout << "(locRow, lowCol) = (" << locRow << " , " << locCol << ")" << std::endl;
                    //char chLoc; std::cin.get(chLoc); std::cin.get(chLoc);
                    int rowTmp = i1 - i + (paddingSize-1)/2;
                    int colTmp = j1 - j + (paddingSize-1)/2;
                    spaWeiPatch(rowTmp, colTmp) = exp(-((locRow - i)*(locRow -i) + (locCol - j)*(locCol - j))/(2*sigmaSpatial*sigmaSpatial));
                    //depWeiPatch(rowTmp, colTmp) = exp(-(iniDepVal - matrix2D.slice(2)(i1,j1))*(iniDepVal - matrix2D.slice(2)(i1,j1))/(2*sigmaDep*sigmaDep));

                    arma::Mat<float> colorPatchTmp(3,1);
                    arma::Cube<float> imgCubeTmp1( colorPatchSize, colorPatchSize, 3 );
                    imgCubeTmp1.slice(0) = imgCube.slice(0)( arma::span( i1 - (colorPatchSize-1)/2, i1 + (colorPatchSize-1)/2),
                                           arma::span( j1 - (colorPatchSize-1)/2, j1 + (colorPatchSize-1)/2) );
                    imgCubeTmp1.slice(1) = imgCube.slice(1)( arma::span( i1 - (colorPatchSize-1)/2, i1 + (colorPatchSize-1)/2),
                                           arma::span( j1 - (colorPatchSize-1)/2, j1 + (colorPatchSize-1)/2) );
                    imgCubeTmp1.slice(2) = imgCube.slice(2)( arma::span( i1 - (colorPatchSize-1)/2, i1 + (colorPatchSize-1)/2),
                                           arma::span( j1 - (colorPatchSize-1)/2, j1 + (colorPatchSize-1)/2) );

                    //std::cout << "labImgTmp size is: " << labImgTmp.rows << "x" << labImgTmp.cols << std::endl;
                    //std::cout << "the labImgTmp is:\n" << labImgTmp << std::endl;
                    colorPatchTmp = computeMeanColor( imgCubeTmp1 );

                    //cv::Mat labImgTmp1;
                    //labImg(cv::Range(i1-(colorPatchSize-1)/2, i1 + (colorPatchSize+1)/2), cv::Range(j1-(colorPatchSize-1)/2, j1 + (colorPatchSize+1)/2)).copyTo(labImgTmp1);
                    //colorPatchTmp = computeMeanColor( labImgTmp1 );
                    colorWeiPatch( rowTmp, colTmp ) = sqrt(arma::accu((colorPatch - colorPatchTmp)%(colorPatch - colorPatchTmp)));
                }
            }
            if( arma::max(arma::max(colorWeiPatch)) == 0. )
            {
                colorWeiPatch.fill(1.0);
            }
            colorWeiPatch = matNormalize(colorWeiPatch);
            //std::cout << "finished weight computing!\n";
            float maxColorVal = 0.;
            if( arma::accu(colorWeiPatch) > 0 )
                maxColorVal = arma::max( arma::max( colorWeiPatch ) );
            //std::cout << "the maxColorVal = " << maxColorVal << std::endl;
            float minColorVal = 0.;
            if( arma::accu(colorWeiPatch) > 0 )
                minColorVal = arma::min( colorWeiPatch( arma::find(colorWeiPatch > 0) ) );
            float sigmaColor = 0.0;
            if( maxColorVal == minColorVal )
                sigmaColor = 1.0;
            else
            {
                sigmaColor = -log( maxColorVal - minColorVal );
                if( sigmaColor == 0. )
                    sigmaColor = 0.0001;
            }
            colorWeiPatch = exp(-(colorWeiPatch%colorWeiPatch)/(2*sigmaColor*sigmaColor));
            for( int i1 = 0; i1 < colorWeiPatch.n_rows; i1++ )
            {
                for( int j1 = 0; j1 < colorWeiPatch.n_cols; j1++ )
                {
                    if( depPatchOri(i1,j1) == 0. )
                        colorWeiPatch(i1,j1) = 0.;
                }
            }


            arma::Mat<float> weightPatch = spaWeiPatch%depWeiPatch%colorWeiPatch;
            //std::cout << "spaWeiPatch = \n" << spaWeiPatch << std::endl;
            //std::cout << "depWeiPatch = \n" << depWeiPatch << std::endl;
            //std::cout << "colorWeiPatch = \n" << colorWeiPatch << std::endl;
            //char chTmp; std::cin.get(chTmp); std::cin.get(chTmp);
            //arma::Mat<float> weightPatch = spaWeiPatch*depWeiPatch;
            if( arma::accu(weightPatch) == 0 )
            {
                fprintf(stderr, "cannot calculate depth value!\n");
                continue;
            }
            //normalize weightPatch;
            weightPatch = weightPatch/(arma::accu(weightPatch));
            float finalDepVal = arma::accu(depPatchOri%weightPatch);
            // std::cout << "finalDepVal = " << finalDepVal << std::endl;
            denseDepImg(i,j) = finalDepVal;
        }
    }
    arma::Mat<float> denseDepImgFinal(rgbImgNoPadding.rows, rgbImgNoPadding.cols);
    denseDepImgFinal.fill(0);
    denseDepImgFinal = denseDepImg( arma::span(paddingSize, denseDepImg.n_rows-paddingSize-1),
                                    arma::span(paddingSize, denseDepImg.n_cols-paddingSize-1));
    return denseDepImgFinal;
}


void inhomogeneityClean( arma::Mat<float>& sparseDepImgNoPadding, int paddingSize )
{

    //the depThd is very important!!!!
    float depThd = 1.0;
    CHECK( arma::accu( sparseDepImgNoPadding ) > 0 ) << "in the homogeneity, the sparseDepImgNoPadding is not OK!\n";
    arma::Mat<float> sparseDepImg = matPadding(sparseDepImgNoPadding, paddingSize, paddingSize);
    //CHECK( arma::accu( sparseDepImg ) > 0 ) << "in the homogeneity, the sparseDepImg is not OK! before clean\n";
    CHECK( arma::accu(sparseDepImg( arma::span(paddingSize, sparseDepImg.n_rows - paddingSize -1),
                                    arma::span(paddingSize, sparseDepImg.n_cols - paddingSize -1))) > 0 ) << " Before Clean, the submaxtrix of sparseDepImg is not OK!\n";

    //std::cout << "After padding, the matrix size is: " << sparseDepImg.n_rows << "x" << sparseDepImg.n_cols << std::endl;

    for(int i = paddingSize + 0; i < sparseDepImg.n_rows - paddingSize; i++)
    {
        for(int j = paddingSize; j < sparseDepImg.n_cols - paddingSize; j++)
        {
            arma::Mat<float> depPatch = sparseDepImg( arma::span(i - (paddingSize-1)/2, i + (paddingSize-1)/2),
                                        arma::span(j - (paddingSize-1)/2, j + (paddingSize-1)/2) );
            //std::cout << "the depPatch size is: " << depPatch.n_rows << "x" << depPatch.n_cols << std::endl;
            float maxDepVal = 0.;
            maxDepVal = arma::max( arma::max(depPatch) );
            float minDepVal = 0.;
            if( maxDepVal > 0. )
                minDepVal = arma::min( depPatch( arma::find(depPatch > 0) ) );

            if( maxDepVal - minDepVal < depThd )
                continue;
            float meanDepVal = (maxDepVal + minDepVal)/2;
            for( int i1 = 0; i1 < depPatch.n_rows; i1++ )
            {
                for( int j1 = 0; j1 < depPatch.n_cols; j1++ )
                {
                    if( depPatch(i1,j1) <= meanDepVal )
                        continue;
                    bool left = false;
                    bool right = false;
                    bool top = false;
                    bool bottom = false;

                    int rowTmp = i + i1 - (paddingSize-1)/2;
                    int colTmp = j + j1 - (paddingSize-1)/2;
                    //check left
                    for( int i2 = 0; i2 < depPatch.n_rows; i2++ )
                    {
                        for( int j2 = 0; j2 < j1; j2++ )
                        {
                            if( depPatch(i2,j2) < meanDepVal && depPatch(i2,j2) > 0. )
                            {
                                left = true;
                                break;
                            }
                        }
                    }

                    //check right
                    for( int i2 = 0; i2 < depPatch.n_rows; i2++ )
                    {
                        for( int j2 = j1+1; j2 < depPatch.n_cols; j2++ )
                        {
                            if( depPatch(i2,j2) < meanDepVal && depPatch(i2,j2) > 0. )
                            {
                                right = true;
                                break;
                            }
                        }
                    }

                    //check top
                    for( int i2 = 0; i2 < i1; i2++ )
                    {
                        for( int j2 = 0; j2 < depPatch.n_cols; j2++ )
                        {
                            if( depPatch(i2,j2) < meanDepVal && depPatch(i2,j2) > 0. )
                            {
                                top = true;
                                break;
                            }
                        }
                    }

                    //check bottom
                    for( int i2 = i1+1; i2 < depPatch.n_rows; i2++ )
                    {
                        for( int j2 = 0; j2 < depPatch.n_cols; j2++ )
                        {
                            if( depPatch(i2,j2) < meanDepVal && depPatch(i2,j2) > 0. )
                            {
                                bottom = true;
                                break;
                            }
                        }
                    }

                    //sparseDepImg(rowTmp, colTmp) = 1.;
                    if( left && right && top && bottom )
                    {
                        sparseDepImg(rowTmp, colTmp) = 0.;
                    }
                }
            }
        }
    }

    //std::cout << "After clean, the sparseDepImg size is: " << sparseDepImg.n_rows << "x" << sparseDepImg.n_cols << std::endl;
    //std::cout << "the paddingSize is: " << paddingSize << std::endl;
    //std::cout << "finished inhomogeneityClean and begin to assign value!\n";
    sparseDepImgNoPadding = sparseDepImg( arma::span(paddingSize, sparseDepImg.n_rows - paddingSize -1),
                                          arma::span(paddingSize, sparseDepImg.n_cols - paddingSize -1));
    CHECK( arma::accu( sparseDepImg ) > 0 ) << "in the homogeneity, the sparseDepImg is not OK! after clean\n";
    CHECK( arma::accu(sparseDepImg( arma::span(paddingSize, sparseDepImg.n_rows - paddingSize -1),
                                    arma::span(paddingSize, sparseDepImg.n_cols - paddingSize -1))) > 0 ) << "the submaxtrix of sparseDepImg is not OK!\n";
    CHECK( arma::accu(sparseDepImgNoPadding) > 0 ) << "the sparseDepImgNoPadding after Clean is not OK! it's size is: "
            << sparseDepImgNoPadding.n_rows << "x" << sparseDepImgNoPadding.n_cols;
    //std::cout << "After clean, the sparseDepImg size is: " << sparseDepImg.n_rows << "x" << sparseDepImg.n_cols << std::endl;

    //sparseDepImgNoPadding = imgTmp;

    //std::cout << "!!!After inhomogeneityClean the size is " << imgTmp.n_rows << "x" << imgTmp.n_cols << std::endl;
}

arma::Mat<float> matNormalize( arma::Mat<float>& normImg )
{
    float maxVal = -10000.;
    float minVal = 10000.;

    for( int i = 0; i < normImg.n_rows; i++ )
    {
        for( int j = 0; j < normImg.n_cols; j++ )
        {
            if( normImg(i,j) > maxVal )
                maxVal = normImg(i,j);
            if( normImg(i,j) < minVal )
                minVal = normImg(i,j);
        }
    }

    if( minVal > 0. )
        minVal = 0.0;
    if( fabs( maxVal - minVal ) < 1.0 )
    {
        normImg.fill(1.0);
        return normImg;
    }
    else if( maxVal > minVal )
        return (normImg-minVal)/(maxVal - minVal);
    else
        return normImg;
}

arma::Cube<float> imgNormalize( cv::Mat& labImg )
{
    arma::Cube<float> normImg( labImg.rows, labImg.cols, 3 );
    normImg.fill(0.);
    float maxChannel1 = -255.;
    float maxChannel2 = -255.;
    float maxChannel3 = -255.;

    float minChannel1 = 255.;
    float minChannel2 = 255.;
    float minChannel3 = 255.;

    for(int i = 0; i < labImg.rows; i++)
    {
        for(int j = 0; j < labImg.cols; j++)
        {
            normImg.slice(0)(i,j) = (float)labImg.at<cv::Vec3b>(i,j)[0];
            normImg.slice(1)(i,j) = (float)labImg.at<cv::Vec3b>(i,j)[1];
            normImg.slice(2)(i,j) = (float)labImg.at<cv::Vec3b>(i,j)[2];
            if( normImg.slice(0)(i,j) > maxChannel1 )
                maxChannel1 = normImg.slice(0)(i,j);
            if( normImg.slice(0)(i,j) < minChannel1 )
                minChannel1 = normImg.slice(0)(i,j);

            if( normImg.slice(1)(i,j) > maxChannel2 )
                maxChannel2 = normImg.slice(1)(i,j);
            if( normImg.slice(1)(i,j) < minChannel2 )
                minChannel2 = normImg.slice(1)(i,j);

            if( normImg.slice(2)(i,j) > maxChannel3 )
                maxChannel3 = normImg.slice(2)(i,j);
            if( normImg.slice(2)(i,j) < minChannel3 )
                minChannel3 = normImg.slice(2)(i,j);
        }
    }

    /*
    if( maxChannel1 - minChannel1 != 0. )
      normImg.slice(0) = (normImg.slice(0) - minChannel1)/(maxChannel1 - minChannel1);

    if( maxChannel2 - minChannel2 != 0. )
      normImg.slice(1) = (normImg.slice(1) - minChannel2)/(maxChannel2 - minChannel2);

    if( maxChannel3 - minChannel3 != 0. )
      normImg.slice(2) = (normImg.slice(2) - minChannel3)/(maxChannel3 - minChannel3);
    */
    return normImg;
}

arma::Mat<float> depUpsamplingDynamic( arma::Cube<float>& matrix2DNoPadding, arma::Mat<float>& geoEstImgNoPadding, const cv::Mat& rgbImgNoPadding )
{
    int paddingSize = 15;
    int colorPatchSize = 7;
    const float sigmaSpatial = 2.0;
    /*
    arma::Mat<float> spatialMatRow(paddingSize, paddingSize);
    arma::Mat<float> spatialMatCol(paddingSize, paddingSize);
    for(int i = 0; i < paddingSize; i++)
    {
     spatialMatRow.col(i) = arma::linspace<arma::Mat<float> >(1,paddingSize, paddingSize);
     spatialMatCol.row(i) = arma::linspace<arma::Mat<float> >(1,paddingSize, paddingSize).t();
    }
    arma::Mat<float> spatialWeight = exp(-((spatialMatRow - 5)%(spatialMatRow - 5) + (spatialMatCol - 5)%(spatialMatCol -5))/(2*sigmaSpatial*sigmaSpatial));
    */
    //padding relevant matrixs;
    cv::Mat labImg;
    rgbImgNoPadding.copyTo(labImg);
    //cv::cvtColor(labImg, labImg, CV_RGB2Lab);
    labImg = imgPadding(labImg, paddingSize, paddingSize);
    arma::Cube<float> imgCube( labImg.rows, labImg.cols, 3 );
    imgCube = imgNormalize( labImg );
    arma::Cube<float> matrix2D(labImg.rows, labImg.cols, 3);
    matrix2D.slice(0) = matPadding(matrix2DNoPadding.slice(0), paddingSize, paddingSize);
    matrix2D.slice(1) = matPadding(matrix2DNoPadding.slice(1), paddingSize, paddingSize);
    matrix2D.slice(2) = matPadding(matrix2DNoPadding.slice(2), paddingSize, paddingSize);

    //std::cout << "begin to adjust ... " << std::endl;
    matrix2D = cubeAdjust( matrix2D );
    //test matPadding result;
    /*
    for( int i = 100; i < 110; i++ )
    {
      for( int j = 100; j < 110; j++ )
      {
        std::cout << "(i,j) = (" << i << " , " << j << " )" << std::endl;
        std::cout << "NoPadding(i,j) = (" << matrix2DNoPadding.slice(0)(i,j) << " , " << matrix2DNoPadding.slice(1)(i,j) << " )" << std::endl;
        //std::cout << "(i,j) = (" << i << " , " << j << " )" << std::endl;
      }
    }
    */
    arma::Mat<float> geoEstImg;
    geoEstImg = matPadding(geoEstImgNoPadding, paddingSize, paddingSize);
    //geoEstImg = matNormalise(geoEstImg);
    arma::Mat<float> denseDepImg = matrix2D.slice(2);
    for(int i = paddingSize + 1 ; i < denseDepImg.n_rows - paddingSize; i++)
    {
        for(int j = paddingSize + 1; j < denseDepImg.n_cols - paddingSize; j++)
        {
            if( denseDepImg(i,j) > 0 ) // here the denseDepImg serves as a labelImg;
                continue;
            //std::cout << "estimate point: " << i << "x" << j << std::endl;
            arma::Mat<float> depPatch = matrix2D.slice(2)(arma::span(i-(paddingSize-1)/2,i+(paddingSize-1)/2),
                                        arma::span(j-(paddingSize-1)/2,j+(paddingSize-1)/2));
            // std::cout << "depPatch = \n " << depPatch << std::endl;
            // std::cout << "depPatch size is: " << depPatch.n_rows << "x" << depPatch.n_cols << std::endl;
            arma::Mat<float> depPatchOri = depPatch;
            // std::cout << "the input depPatch =\n " << depPatchOri << std::endl;
            bool centerExist = false;
            if(depPatch((paddingSize-1)/2,(paddingSize-1)/2) > 0)
                centerExist = true;
            // depPatch = matNormalise(depPatch);
            // geoEstImg = arma::normalise(geoEstImg);
            if( arma::accu( depPatchOri ) == 0 )
                continue;
            arma::uvec seedMat = arma::find(depPatchOri > 0);
            if( seedMat.n_rows == 1 )
            {
                denseDepImg(i,j) = depPatchOri( seedMat(0,0) );
                continue;
            }
            //calculate the spatial distance weight;
            arma::Cube<float> cubePatch( paddingSize, paddingSize, 3 );
            cubePatch.fill(0.0);
            cubePatch.slice(0) = matrix2D.slice(0)(arma::span((i-(paddingSize-1)/2),(i+(paddingSize-1)/2)),
                                                   arma::span((j-(paddingSize-1)/2), (j+(paddingSize-1)/2)));
            cubePatch.slice(1) = matrix2D.slice(1)(arma::span((i-(paddingSize-1)/2),(i+(paddingSize-1)/2)),
                                                   arma::span((j-(paddingSize-1)/2), (j+(paddingSize-1)/2)));
            cubePatch.slice(2) = matrix2D.slice(2)(arma::span((i-(paddingSize-1)/2),(i+(paddingSize-1)/2)),
                                                   arma::span((j-(paddingSize-1)/2), (j+(paddingSize-1)/2)));
            //std::cout << "cubePatch size is: " << cubePatch.slice(2).n_rows << "x" << cubePatch.slice(2).n_cols << std::endl;
            arma::Mat<float> spaWeiPatch(paddingSize, paddingSize);
            spaWeiPatch.fill(0.0);
            arma::Mat<float> depWeiPatch(paddingSize, paddingSize);
            depWeiPatch.fill(0.0);
            arma::Mat<float> colorWeiPatch(paddingSize, paddingSize);
            colorWeiPatch.fill(0.0);
            //compute DepWeiPatch;
            arma::Mat<float> depPatchTmp = depPatch;
            depPatchTmp((paddingSize-1)/2, (paddingSize-1)/2) = geoEstImg(i,j);
            depPatch = matNormalize( depPatch );
            depPatchTmp = matNormalize( depPatchTmp );
            //std::cout << "after normalization, the depPatch is: \n" << depPatch << std::endl;
            // char chDep; std::cin.get(chDep); std::cin.get(chDep);
            float iniDepVal = depPatchTmp((paddingSize-1)/2, (paddingSize-1)/2);
            float maxDepVal = 0.;
            maxDepVal = arma::max(arma::max( depPatchTmp ));
            float minDepVal = 0.;
            if( maxDepVal > 0. )
                minDepVal = arma::min( arma::min( depPatchTmp( arma::find( depPatchTmp > 0 ) ) ) );

            arma::Mat<float> sigmaDepPatch( paddingSize, paddingSize );
            sigmaDepPatch.fill( 0.0 );
            for( int i1 = 0; i1 < paddingSize; i1++ )
            {
                for( int j1 = 0; j1 < paddingSize; j1++ )
                {
                    if( depPatchOri(i1,j1) == 0. )
                    {
                        sigmaDepPatch( i1, j1 ) = 1.0;
                        continue;
                    }
                    if( iniDepVal == depPatch(i1,j1) )
                    {
                        sigmaDepPatch( i1, j1 ) = 1.0;
                        continue;
                    }
                    //std::cout << " -log10(0.2) = " << -log10(0.2) << std::endl;
                    //std::cout << " fabs(iniDepVal - depPatch(i1,j1)) = " << fabs( iniDepVal - depPatch(i1,j1)) << std::endl;
                    float sigmaTmp = -log2( fabs(iniDepVal - depPatch(i1,j1)) );
                    // std::cout << " iniDepVal = " << iniDepVal << " depPatch(i1,j1) = " << depPatch(i1,j1) << std::endl;
                    // std::cout << "sigmaTmp = " << sigmaTmp << std::endl;
                    if( sigmaTmp == 0.0 )
                        sigmaTmp = 0.00001;
                    sigmaDepPatch( i1, j1 ) = sigmaTmp;
                    // char chPatch; std::cin.get(chPatch); std::cin.get(chPatch);
                }
            }
            float sigmaDep = fabs(maxDepVal - minDepVal) < 0.0001?1:(-log(maxDepVal - minDepVal));
            //depWeiPatch = exp( -((depPatch - iniDepVal)%(depPatch - iniDepVal))/(2*sigmaDep*sigmaDep));
            depWeiPatch = exp( -((depPatch - iniDepVal)%(depPatch - iniDepVal))/(2*sigmaDepPatch%sigmaDepPatch));
            for( int i1 = 0; i1 < depWeiPatch.n_rows; i1++ )
            {
                for( int j1 = 0; j1 < depWeiPatch.n_cols; j1++ )
                {
                    if( depPatchOri(i1,j1) == 0. )
                        depWeiPatch(i1,j1) = 0.;
                }
            }

            //std::cout << " - - - - - - - - - - - - -- - - -- - - - - - - \n";
            //std::cout << "sigmaDepPatch = \n " << sigmaDepPatch << std::endl;
            //std::cout << "depWeiPatch = \n " << depWeiPatch << std::endl;
            //std::cout << "depPatchOri = \n " << depPatchOri << std::endl;
            //char chPatch; std::cin.get(chPatch); std::cin.get(chPatch);
            //std::cout << "* * * * * * * * *  * * * * * * * * * * * * * * \n";
            // char chPatch; std::cin.get(chPatch); std::cin.get(chPatch);
            //depWeiPatch( arma::find( depPatch == 0 ) )  = 0.;

            //compute spaWeiPatch and colorWerPatch;
            //arma::Mat<float> colorPatch(3,1);
            //colorPatch.fill(0.0);

            cv::Mat labImgTmp; // = labImg(cv::Range(i-(colorPatchSize-1)/2, i + (colorPatchSize+1)/2), cv::Range(j-(colorPatchSize-1)/2, j + (colorPatchSize+1)/2)).clone();
            labImg(cv::Range(i-(colorPatchSize-1)/2, i + (colorPatchSize+1)/2), cv::Range(j-(colorPatchSize-1)/2, j + (colorPatchSize+1)/2)).copyTo( labImgTmp );
            arma::Cube<float> imgCubeTmp( colorPatchSize, colorPatchSize, 3 );
            imgCubeTmp.fill( 0.0 );
            imgCubeTmp.slice(0)= imgCube.slice(0)( arma::span( i - (colorPatchSize-1)/2, i + (colorPatchSize-1)/2),
                                                   arma::span( j - (colorPatchSize-1)/2, j + (colorPatchSize-1)/2) );
            imgCubeTmp.slice(1)= imgCube.slice(1)( arma::span( i - (colorPatchSize-1)/2, i + (colorPatchSize-1)/2),
                                                   arma::span( j - (colorPatchSize-1)/2, j + (colorPatchSize-1)/2) );
            imgCubeTmp.slice(2)= imgCube.slice(2)( arma::span( i - (colorPatchSize-1)/2, i + (colorPatchSize-1)/2),
                                                   arma::span( j - (colorPatchSize-1)/2, j + (colorPatchSize-1)/2) );
            //std::cout << "labImgTmp size is: " << labImgTmp.rows << "x" << labImgTmp.cols << std::endl;
            //std::cout << "the labImgTmp is:\n" << labImgTmp << std::endl;
            //colorPatch = computeMeanColor( imgCubeTmp );
            arma::Mat<float> colorPatch( imgCubeTmp.slice(0).n_rows, imgCubeTmp.slice(0).n_cols );
            colorPatch = computeBilateralFilter( imgCubeTmp, 0.1 );
            //std::cout << "the colorPatch is:\n " << colorPatch << std::endl;
            //char ch; std::cin.get(ch); std::cin.get(ch);
            for(int i1 = (i - (paddingSize-1)/2); i1 <= (i + (paddingSize-1)/2); i1++)
            {
                for(int j1 = (j - (paddingSize-1)/2); j1 <= (j + (paddingSize-1)/2); j1++)
                {
                    if( matrix2D.slice(2)(i1, j1) == 0.0 )
                        continue;

                    float locRow = matrix2D.slice(0)(i1,j1);
                    float locCol = matrix2D.slice(1)(i1,j1);
                    //  std::cout << "(i1, j1) = (" << i1 << " , " << j1 << ")" << std::endl;
                    //  std::cout << "(i, j) = (" << i << " , " << j << ")" << std::endl;
                    //  std::cout << "(locRow, lowCol) = (" << locRow << " , " << locCol << ")" << std::endl;
                    //char chLoc; std::cin.get(chLoc); std::cin.get(chLoc);
                    int rowTmp = i1 - i + (paddingSize-1)/2;
                    int colTmp = j1 - j + (paddingSize-1)/2;
                    spaWeiPatch(rowTmp, colTmp) = exp(-((locRow - i)*(locRow -i) + (locCol - j)*(locCol - j))/(2*sigmaSpatial*sigmaSpatial));
                    //depWeiPatch(rowTmp, colTmp) = exp(-(iniDepVal - matrix2D.slice(2)(i1,j1))*(iniDepVal - matrix2D.slice(2)(i1,j1))/(2*sigmaDep*sigmaDep));

                    arma::Mat<float> colorPatchTmp(3,1);
                    arma::Cube<float> imgCubeTmp1( colorPatchSize, colorPatchSize, 3 );
                    imgCubeTmp1.slice(0) = imgCube.slice(0)( arma::span( i1 - (colorPatchSize-1)/2, i1 + (colorPatchSize-1)/2),
                                           arma::span( j1 - (colorPatchSize-1)/2, j1 + (colorPatchSize-1)/2) );
                    imgCubeTmp1.slice(1) = imgCube.slice(1)( arma::span( i1 - (colorPatchSize-1)/2, i1 + (colorPatchSize-1)/2),
                                           arma::span( j1 - (colorPatchSize-1)/2, j1 + (colorPatchSize-1)/2) );
                    imgCubeTmp1.slice(2) = imgCube.slice(2)( arma::span( i1 - (colorPatchSize-1)/2, i1 + (colorPatchSize-1)/2),
                                           arma::span( j1 - (colorPatchSize-1)/2, j1 + (colorPatchSize-1)/2) );

                    //std::cout << "labImgTmp size is: " << labImgTmp.rows << "x" << labImgTmp.cols << std::endl;
                    //std::cout << "the labImgTmp is:\n" << labImgTmp << std::endl;

                    imgCubeTmp1.slice(0) = imgCubeTmp1.slice(0) - imgCubeTmp.slice(0);
                    imgCubeTmp1.slice(1) = imgCubeTmp1.slice(1) - imgCubeTmp.slice(1);
                    imgCubeTmp1.slice(2) = imgCubeTmp1.slice(2) - imgCubeTmp.slice(2);

                    arma::Mat<float> imgCubeTmpFinal = imgCubeTmp1.slice(0)%imgCubeTmp1.slice(0) +
                                                       imgCubeTmp1.slice(1)%imgCubeTmp1.slice(1) +
                                                       imgCubeTmp1.slice(2)%imgCubeTmp1.slice(2);
                    //imgCubeTmpFinal = matNormalize( imgCubeTmpFinal );
                    //colorPatchTmp = computeMeanColor( imgCubeTmp1 );

                    //cv::Mat labImgTmp1;
                    //labImg(cv::Range(i1-(colorPatchSize-1)/2, i1 + (colorPatchSize+1)/2), cv::Range(j1-(colorPatchSize-1)/2, j1 + (colorPatchSize+1)/2)).copyTo(labImgTmp1);
                    //colorPatchTmp = computeMeanColor( labImgTmp1 );
                    //colorWeiPatch( rowTmp, colTmp ) = sqrt(arma::accu((colorPatch - colorPatchTmp)%(colorPatch - colorPatchTmp)));
                    //colorWeiPatch( rowTmp, colTmp ) = exp( -( arma::accu( colorPatch%( imgCubeTmpFinal%imgCubeTmpFinal )) )/(2*3*6.66*6.66));
                    colorWeiPatch( rowTmp, colTmp ) = arma::accu( colorPatch%imgCubeTmpFinal );
                }
            }
            if( arma::max(arma::max(colorWeiPatch)) == 0. )
            {
                std::cout << "the colorPatch is zeros!\n";
                colorWeiPatch.fill(1.0);
            }

            colorWeiPatch = matNormalize(colorWeiPatch);
            //std::cout << "finished weight computing!\n";
            float maxColorVal = 0.;
            if( arma::accu(colorWeiPatch) > 0 )
                maxColorVal = arma::max( arma::max( colorWeiPatch ) );
            //std::cout << "the maxColorVal = " << maxColorVal << std::endl;
            float minColorVal = 0.;
            if( arma::accu(colorWeiPatch) > 0 )
                minColorVal = arma::min( colorWeiPatch( arma::find(colorWeiPatch > 0) ) );
            float sigmaColor = 0.0;
            if( maxColorVal == minColorVal )
                sigmaColor = 1.0;
            else
            {
                sigmaColor = -log2( maxColorVal - minColorVal );
                if( sigmaColor == 0. )
                    sigmaColor = 0.0001;
            }
            colorWeiPatch = exp(-(colorWeiPatch%colorWeiPatch)/(2*sigmaColor*sigmaColor));

            //std::cout << "the colorWeiPatch = \n " << colorWeiPatch << std::endl;
            for( int i1 = 0; i1 < colorWeiPatch.n_rows; i1++ )
            {
                for( int j1 = 0; j1 < colorWeiPatch.n_cols; j1++ )
                {
                    if( depPatchOri(i1,j1) == 0. )
                        colorWeiPatch(i1,j1) = 0.;
                }
            }


            arma::Mat<float> weightPatch = spaWeiPatch%depWeiPatch%colorWeiPatch;
            //std::cout << "spaWeiPatch = \n" << spaWeiPatch << std::endl;
            //std::cout << "depWeiPatch = \n" << depWeiPatch << std::endl;
            //std::cout << "colorWeiPatch = \n" << colorWeiPatch << std::endl;
            //char chTmp; std::cin.get(chTmp); std::cin.get(chTmp);
            //arma::Mat<float> weightPatch = spaWeiPatch*depWeiPatch;
            if( arma::accu(weightPatch) == 0 )
            {
                fprintf(stderr, "cannot calculate depth value!\n");
                continue;
            }
            //normalize weightPatch;
            weightPatch = weightPatch/(arma::accu(weightPatch));
            float finalDepVal = arma::accu(depPatchOri%weightPatch);
            // std::cout << "finalDepVal = " << finalDepVal << std::endl;
            denseDepImg(i,j) = finalDepVal;
        }
    }
    arma::Mat<float> denseDepImgFinal(rgbImgNoPadding.rows, rgbImgNoPadding.cols);
    denseDepImgFinal.fill(0);
    denseDepImgFinal = denseDepImg( arma::span(paddingSize, denseDepImg.n_rows-paddingSize-1),
                                    arma::span(paddingSize, denseDepImg.n_cols-paddingSize-1));
    return denseDepImgFinal;
}

arma::Mat<float> depUpsamplingRandom( arma::Cube<float>& matrix2DNoPadding, arma::Mat<float>& geoEstImgNoPadding, const cv::Mat& rgbImgNoPadding )
{
    int paddingSize = 15;
    int colorPatchSize = 7;
    const float sigmaSpatial = 2.0;
    //padding relevant matrixs;
    cv::Mat labImg;
    rgbImgNoPadding.copyTo(labImg);
    //cv::cvtColor(labImg, labImg, CV_RGB2Lab);
    labImg = imgPadding(labImg, paddingSize, paddingSize);
    arma::Cube<float> imgCube( labImg.rows, labImg.cols, 3 );
    imgCube = imgNormalize( labImg );
    arma::Cube<float> matrix2D(labImg.rows, labImg.cols, 3);
    matrix2D.slice(0) = matPadding(matrix2DNoPadding.slice(0), paddingSize, paddingSize);
    matrix2D.slice(1) = matPadding(matrix2DNoPadding.slice(1), paddingSize, paddingSize);
    matrix2D.slice(2) = matPadding(matrix2DNoPadding.slice(2), paddingSize, paddingSize);

    //std::cout << "begin to adjust ... " << std::endl;
    matrix2D = cubeAdjust( matrix2D );
    //test matPadding result;
    /*
    for( int i = 100; i < 110; i++ )
    {
      for( int j = 100; j < 110; j++ )
      {
        std::cout << "(i,j) = (" << i << " , " << j << " )" << std::endl;
        std::cout << "NoPadding(i,j) = (" << matrix2DNoPadding.slice(0)(i,j) << " , " << matrix2DNoPadding.slice(1)(i,j) << " )" << std::endl;
        //std::cout << "(i,j) = (" << i << " , " << j << " )" << std::endl;
      }
    }
    */
    arma::Mat<float> geoEstImg;
    geoEstImg = matPadding(geoEstImgNoPadding, paddingSize, paddingSize);
    arma::Mat<int> orderImg( geoEstImg.n_rows, geoEstImg.n_cols );
    orderImg.fill(0); //orderImg is used to store the upsampling order;
    int beginRow = paddingSize + 1 + 0;
    int endRow = geoEstImg.n_rows - paddingSize - 1;
    int beginCol = paddingSize + 1;
    int endCol = geoEstImg.n_cols - paddingSize - 1;

    //std::cout << "begin to assign the value -1 !\n";
    //orderImg( arma::span( beginRow, endRow ), arma::span(beginCol, endCol) ) = -1;
    for( int i1 = beginRow; i1 <= endRow; i1++ )
    {
        for( int j1 = beginCol; j1 <= endCol; j1++ )
        {
            orderImg(i1,j1) = -1;
        }
    }
    //std::cout << "finished to assign the value -1 !\n";
    int order = 1;
    std::cout << "begin orderImg assign !\n";
    srand( int(time(0)) );
    while(1)
    {
        // srand( int(time(0)) );
        int rowTmp =  int( (double(rand())/RAND_MAX)*(endRow - beginRow) + beginRow + 0.5 );
        //srand( int(time(0)) );
        int colTmp = int( (double(rand())/RAND_MAX)*(endCol - beginCol) + beginCol + 0.5 );

        //std::cout << "rowTmp = " << rowTmp << " colTmp= " << colTmp << std::endl;
        if( orderImg(rowTmp, colTmp) > 0 )
            continue;
        //std::cout << "skipped continue!\n";
        orderImg( rowTmp, colTmp ) = order;
        order++;

        if( !arma::any( arma::vectorise(orderImg) == -1 ) )
            break;
        //arma::uvec labelVec = arma::find( orderImg == -1 );
        //std::cout << "the -1 number is: " << labelVec.n_rows << std::endl;
    }
    //geoEstImg = matNormalise(geoEstImg);
    std::cout << "finished orderImg assigning !\n";
    //std::cout << "orderImg(150,527) = " << orderImg(150,527) << std::endl;
    arma::Mat<float> denseDepImg = matrix2D.slice(2);

    //the labelImg is used to store the calculated depth value;
    arma::Mat<float> labelImg( denseDepImg.n_rows, denseDepImg.n_cols );
    labelImg.fill(0);
    order = 1;
    int orderMax = arma::max( arma::max(orderImg) );
    std::cout << "orderMax = " << orderMax << std::endl;
    while(1)
    {
        arma::uvec location = arma::find( orderImg == order );
        //std::cout << "the current order = " << order << std::endl;
        order++;

        if( order == orderMax )
            break;
        if( location.n_rows > 1 )
        {
            fprintf(stderr, "strange! find more than one value equals to %d !\n", order );
            exit( 0 );
        }
        if( location.n_rows == 0 )
        {
            fprintf( stderr, "strange! cannot find the relevant value equals to %d !\n", order );
            exit( 0 );
        }
        int i = (location(0,0)+1)%(orderImg.n_rows) - 1;
        int j = (location(0,0)+1)/(orderImg.n_rows);
        /*
        if( i == 150 && j == 527 )
        {
          std::cout << "i = " << i << " ,j = " << j << std::endl;
          std::cout << "the geoEstimated value is: " << geoEstImg(i,j) << std::endl;
          std::cout << "the matrix2D.slice(2)(i,j) =  " << matrix2D.slice(2)(i,j) << std::endl;
          char ch; std::cin.get(ch);
        }
        */
        //std::cout << " i = " << i << " j = " << j << std::endl;
        // if( denseDepImg(i,j) > 0 ) // here the denseDepImg serves as a labelImg;
        //     continue;
        if( matrix2D.slice(2)(i,j) > 0. )
            continue;
        //std::cout << "estimate point: " << i << "x" << j << std::endl;
        arma::Mat<float> depPatch = matrix2D.slice(2)(arma::span(i-(paddingSize-1)/2,i+(paddingSize-1)/2),
                                    arma::span(j-(paddingSize-1)/2,j+(paddingSize-1)/2)) +
                                    labelImg(arma::span(i-(paddingSize-1)/2,i+(paddingSize-1)/2),
                                             arma::span(j-(paddingSize-1)/2,j+(paddingSize-1)/2));
        //if( i == 150 && j == 527 )
        //    std::cout << "the depPatch = \n " << depPatch << std::endl;
        //std::cout << "depPatch = \n" << depPatch << std::endl;
        // std::cout << "depPatch size is: " << depPatch.n_rows << "x" << depPatch.n_cols << std::endl;
        arma::Mat<float> depPatchOri = depPatch;

        arma::Mat<float> labelPatch = labelImg(arma::span(i-(paddingSize-1)/2,i+(paddingSize-1)/2),
                                               arma::span(j-(paddingSize-1)/2,j+(paddingSize-1)/2));
        // std::cout << "the input depPatch =\n " << depPatchOri << std::endl;
        bool centerExist = false;
        if(depPatch((paddingSize-1)/2,(paddingSize-1)/2) > 0.)
            centerExist = true;
        // depPatch = matNormalise(depPatch);
        // geoEstImg = arma::normalise(geoEstImg);
        if( arma::accu( depPatchOri ) == 0. )
            continue;
        arma::uvec seedMat = arma::find(depPatchOri > 0.);
        if( seedMat.n_rows == 1 )
        {
            denseDepImg(i,j) = depPatchOri( seedMat(0,0) );
            continue;
        }
        //calculate the spatial distance weight;
        arma::Cube<float> cubePatch( paddingSize, paddingSize, 3 );
        cubePatch.fill(0.0);
        cubePatch.slice(0) = matrix2D.slice(0)(arma::span((i-(paddingSize-1)/2),(i+(paddingSize-1)/2)),
                                               arma::span((j-(paddingSize-1)/2), (j+(paddingSize-1)/2)));
        cubePatch.slice(1) = matrix2D.slice(1)(arma::span((i-(paddingSize-1)/2),(i+(paddingSize-1)/2)),
                                               arma::span((j-(paddingSize-1)/2), (j+(paddingSize-1)/2)));
        cubePatch.slice(2) = matrix2D.slice(2)(arma::span((i-(paddingSize-1)/2),(i+(paddingSize-1)/2)),
                                               arma::span((j-(paddingSize-1)/2), (j+(paddingSize-1)/2)));
        //std::cout << "cubePatch size is: " << cubePatch.slice(2).n_rows << "x" << cubePatch.slice(2).n_cols << std::endl;
        arma::Mat<float> spaWeiPatch(paddingSize, paddingSize);
        spaWeiPatch.fill(0.0);
        arma::Mat<float> depWeiPatch(paddingSize, paddingSize);
        depWeiPatch.fill(0.0);
        arma::Mat<float> colorWeiPatch(paddingSize, paddingSize);
        colorWeiPatch.fill(0.0);
        //compute DepWeiPatch;
        arma::Mat<float> depPatchTmp = depPatch;
        depPatchTmp((paddingSize-1)/2, (paddingSize-1)/2) = geoEstImg(i,j);
        arma::Mat<float> depPatchTmpBefore = depPatchTmp;
        depPatch = matNormalize( depPatch );
        depPatchTmp = matNormalize( depPatchTmp );
        //std::cout << "after normalization, the depPatch is: \n" << depPatch << std::endl;
        // char chDep; std::cin.get(chDep); std::cin.get(chDep);
        float iniDepVal = depPatchTmp((paddingSize-1)/2, (paddingSize-1)/2);
        if( iniDepVal < 1e-3 )
        {
            float minDis = 1000.;
            int minRow = 0;
            int minCol = 0;
            for(int i1 = 0; i1 < paddingSize; i1++ )
            {
                for(int j1 = 0; j1 < paddingSize; j1++ )
                {
                    if( depPatch(i1,j1) > 0. && sqrt( (i1 - (paddingSize-1)/2)*(i1 - (paddingSize-1)/2) + ( j1 - (paddingSize-1)/2)*(j1 - (paddingSize-1)/2)) < minDis )
                    {
                        minRow = i1;
                        minCol = j1;
                    }
                }
            }
            if( minRow != 0 && minCol != 0 )
                iniDepVal = depPatch(minRow, minCol);
        }
        float maxDepVal = 0.;
        maxDepVal = arma::max(arma::max( depPatchTmp ));
        float minDepVal = 0.;
        if( maxDepVal > 0. )
            minDepVal = arma::min( arma::min( depPatchTmp( arma::find( depPatchTmp > 0 ) ) ) );

        arma::Mat<float> sigmaDepPatch( paddingSize, paddingSize );
        sigmaDepPatch.fill( 0.0 );
        for( int i1 = 0; i1 < paddingSize; i1++ )
        {
            for( int j1 = 0; j1 < paddingSize; j1++ )
            {
                if( depPatchOri(i1,j1) == 0. )
                {
                    sigmaDepPatch( i1, j1 ) = 1.0;
                    continue;
                }
                if( iniDepVal == depPatch(i1,j1) )
                {
                    sigmaDepPatch( i1, j1 ) = 1.0;
                    continue;
                }
                //std::cout << " -log10(0.2) = " << -log10(0.2) << std::endl;
                //std::cout << " fabs(iniDepVal - depPatch(i1,j1)) = " << fabs( iniDepVal - depPatch(i1,j1)) << std::endl;
                float sigmaTmp = -log2( fabs(iniDepVal - depPatch(i1,j1)) );
                // std::cout << " iniDepVal = " << iniDepVal << " depPatch(i1,j1) = " << depPatch(i1,j1) << std::endl;
                // std::cout << "sigmaTmp = " << sigmaTmp << std::endl;
                if( sigmaTmp == 0.0 )
                    sigmaTmp = 0.00001;
                sigmaDepPatch( i1, j1 ) = sigmaTmp;
                // char chPatch; std::cin.get(chPatch); std::cin.get(chPatch);
            }
        }
        float sigmaDep = fabs(maxDepVal - minDepVal) < 0.0001?1:(-log(maxDepVal - minDepVal));
        //depWeiPatch = exp( -((depPatch - iniDepVal)%(depPatch - iniDepVal))/(2*sigmaDep*sigmaDep));
        depWeiPatch = exp( -((depPatch - iniDepVal)%(depPatch - iniDepVal))/(2*sigmaDepPatch%sigmaDepPatch));
        for( int i1 = 0; i1 < depWeiPatch.n_rows; i1++ )
        {
            for( int j1 = 0; j1 < depWeiPatch.n_cols; j1++ )
            {
                if( depPatchOri(i1,j1) == 0. )
                    depWeiPatch(i1,j1) = 0.;
            }
        }

        //std::cout << " - - - - - - - - - - - - -- - - -- - - - - - - \n";
        //std::cout << "sigmaDepPatch = \n " << sigmaDepPatch << std::endl;
        //std::cout << "depWeiPatch = \n " << depWeiPatch << std::endl;
        //std::cout << "depPatchOri = \n " << depPatchOri << std::endl;
        //char chPatch; std::cin.get(chPatch); std::cin.get(chPatch);
        //std::cout << "* * * * * * * * *  * * * * * * * * * * * * * * \n";
        // char chPatch; std::cin.get(chPatch); std::cin.get(chPatch);
        //depWeiPatch( arma::find( depPatch == 0 ) )  = 0.;

        //compute spaWeiPatch and colorWerPatch;
        arma::Mat<float> colorPatch(3,1);
        colorPatch.fill(0.0);

        cv::Mat labImgTmp; // = labImg(cv::Range(i-(colorPatchSize-1)/2, i + (colorPatchSize+1)/2), cv::Range(j-(colorPatchSize-1)/2, j + (colorPatchSize+1)/2)).clone();
        labImg(cv::Range(i-(colorPatchSize-1)/2, i + (colorPatchSize+1)/2), cv::Range(j-(colorPatchSize-1)/2, j + (colorPatchSize+1)/2)).copyTo( labImgTmp );
        arma::Cube<float> imgCubeTmp( colorPatchSize, colorPatchSize, 3 );
        imgCubeTmp.fill( 0.0 );
        imgCubeTmp.slice(0)= imgCube.slice(0)( arma::span( i - (colorPatchSize-1)/2, i + (colorPatchSize-1)/2),
                                               arma::span( j - (colorPatchSize-1)/2, j + (colorPatchSize-1)/2) );
        imgCubeTmp.slice(1)= imgCube.slice(1)( arma::span( i - (colorPatchSize-1)/2, i + (colorPatchSize-1)/2),
                                               arma::span( j - (colorPatchSize-1)/2, j + (colorPatchSize-1)/2) );
        imgCubeTmp.slice(2)= imgCube.slice(2)( arma::span( i - (colorPatchSize-1)/2, i + (colorPatchSize-1)/2),
                                               arma::span( j - (colorPatchSize-1)/2, j + (colorPatchSize-1)/2) );
        //std::cout << "labImgTmp size is: " << labImgTmp.rows << "x" << labImgTmp.cols << std::endl;
        //std::cout << "the labImgTmp is:\n" << labImgTmp << std::endl;
        colorPatch = computeMeanColor( imgCubeTmp );
        //std::cout << "the colorPatch is " << colorPatch << std::endl;
        for(int i1 = (i - (paddingSize-1)/2); i1 <= (i + (paddingSize-1)/2); i1++)
        {
            for(int j1 = (j - (paddingSize-1)/2); j1 <= (j + (paddingSize-1)/2); j1++)
            {
                float locRow = matrix2D.slice(0)(i1,j1);
                float locCol = matrix2D.slice(1)(i1,j1);
                //  std::cout << "(i1, j1) = (" << i1 << " , " << j1 << ")" << std::endl;
                //  std::cout << "(i, j) = (" << i << " , " << j << ")" << std::endl;
                //  std::cout << "(locRow, lowCol) = (" << locRow << " , " << locCol << ")" << std::endl;
                //char chLoc; std::cin.get(chLoc); std::cin.get(chLoc);
                int rowTmp = i1 - i + (paddingSize-1)/2;
                int colTmp = j1 - j + (paddingSize-1)/2;
                if( matrix2D.slice(2)(i1, j1) == 0.0 && depPatch(rowTmp, colTmp) == 0.0 )
                    continue;

                if( locRow == 0.0 && locCol == 0.0 )
                    spaWeiPatch(rowTmp, colTmp) = exp(-((i1 - i)*(i1 -i) + (j1 - j)*(j1 - j))/(2*sigmaSpatial*sigmaSpatial));
                else
                    spaWeiPatch(rowTmp, colTmp) = exp(-((locRow - i)*(locRow -i) + (locCol - j)*(locCol - j))/(2*sigmaSpatial*sigmaSpatial));
                //depWeiPatch(rowTmp, colTmp) = exp(-(iniDepVal - matrix2D.slice(2)(i1,j1))*(iniDepVal - matrix2D.slice(2)(i1,j1))/(2*sigmaDep*sigmaDep));

                arma::Mat<float> colorPatchTmp(3,1);
                arma::Cube<float> imgCubeTmp1( colorPatchSize, colorPatchSize, 3 );
                imgCubeTmp1.slice(0) = imgCube.slice(0)( arma::span( i1 - (colorPatchSize-1)/2, i1 + (colorPatchSize-1)/2),
                                       arma::span( j1 - (colorPatchSize-1)/2, j1 + (colorPatchSize-1)/2) );
                imgCubeTmp1.slice(1) = imgCube.slice(1)( arma::span( i1 - (colorPatchSize-1)/2, i1 + (colorPatchSize-1)/2),
                                       arma::span( j1 - (colorPatchSize-1)/2, j1 + (colorPatchSize-1)/2) );
                imgCubeTmp1.slice(2) = imgCube.slice(2)( arma::span( i1 - (colorPatchSize-1)/2, i1 + (colorPatchSize-1)/2),
                                       arma::span( j1 - (colorPatchSize-1)/2, j1 + (colorPatchSize-1)/2) );

                //std::cout << "labImgTmp size is: " << labImgTmp.rows << "x" << labImgTmp.cols << std::endl;
                //std::cout << "the labImgTmp is:\n" << labImgTmp << std::endl;
                colorPatchTmp = computeMeanColor( imgCubeTmp1 );
                //std::cout << "the colorPatchTmp is " << colorPatchTmp << std::endl;
                //cv::Mat labImgTmp1;
                //labImg(cv::Range(i1-(colorPatchSize-1)/2, i1 + (colorPatchSize+1)/2), cv::Range(j1-(colorPatchSize-1)/2, j1 + (colorPatchSize+1)/2)).copyTo(labImgTmp1);
                //colorPatchTmp = computeMeanColor( labImgTmp1 );
                colorWeiPatch( rowTmp, colTmp ) = sqrt(arma::accu((colorPatch - colorPatchTmp)%(colorPatch - colorPatchTmp)));
            }
        }
        if( arma::max(arma::max(colorWeiPatch)) == 0. )
        {
            colorWeiPatch.fill(1.0);
        }
        //arma::Mat<float> colorOri = colorWeiPatch;
        colorWeiPatch = matNormalize(colorWeiPatch);
        colorWeiPatch( (paddingSize - 1)/2, (paddingSize - 1)/2 ) = 1.0;
        arma::Mat<float> colorOri = colorWeiPatch;
        //std::cout << "finished color  weight computing!\n";
        float maxColorVal = 0.;
        if( arma::accu(colorWeiPatch) > 0 )
            maxColorVal = arma::max( arma::max( colorWeiPatch ) );
        //std::cout << "the maxColorVal = " << maxColorVal << std::endl;
        float minColorVal = 0.;
        if( arma::accu(colorWeiPatch) > 0 )
            minColorVal = arma::min( colorWeiPatch( arma::find(colorWeiPatch > 0) ) );
        float sigmaColor = 0.0;
        if( maxColorVal == minColorVal )
            sigmaColor = 1.0;
        else
        {
            sigmaColor = -log2( maxColorVal - minColorVal );
            if( sigmaColor == 0. )
                sigmaColor = 0.0001;
        }
        colorWeiPatch = exp(-(colorWeiPatch%colorWeiPatch)/(2*sigmaColor*sigmaColor));
        for( int i1 = 0; i1 < colorWeiPatch.n_rows; i1++ )
        {
            for( int j1 = 0; j1 < colorWeiPatch.n_cols; j1++ )
            {
                if( depPatchOri(i1,j1) == 0. )
                    colorWeiPatch(i1,j1) = 0.;
            }
        }
        arma::Mat<float> weightPatch = spaWeiPatch%depWeiPatch%colorWeiPatch;
        //std::cout << "weightPatch = \n" << weightPatch << std::endl;
        //reassign the weight for newly calculated depth value;
        for( int i1 = 0; i1 < weightPatch.n_rows; i1++ )
        {
            for( int j1 = 0; j1 < weightPatch.n_cols; j1++ )
            {
                if( labelPatch(i1,j1) > 0 )
                    weightPatch(i1,j1) *= 0.1;
            }
        }
        //std::cout << "spaWeiPatch = \n" << spaWeiPatch << std::endl;
        //std::cout << "depWeiPatch = \n" << depWeiPatch << std::endl;
        //std::cout << "colorWeiPatch = \n" << colorWeiPatch << std::endl;
        //char chTmp; std::cin.get(chTmp); std::cin.get(chTmp);
        //arma::Mat<float> weightPatch = spaWeiPatch*depWeiPatch;
        if( arma::accu(weightPatch) == 0 )
        {
            fprintf(stderr, "cannot calculate depth value!\n");
            std::cout << " i = " << i << " , j = " << j << std::endl;
            std::cout << " geoEstImg = " << geoEstImg(i,j) << std::endl;
            std::cout << "depPatchOri = \n " << depPatchOri << std::endl;
            std::cout << "depPatch = \n " << depPatch << std::endl;
            std::cout << "depPatchTmpBefore = \n " << depPatchTmpBefore << std::endl;
            std::cout << "depPatchTmp = \n " << depPatchTmp << std::endl;
            std::cout << "iniDepVal = \n " << iniDepVal << std::endl;
            std::cout << "sigmaDepPatch = \n " << sigmaDepPatch << std::endl;
            //std::cout << "labelPatch = \n " << labelPatch << std::endl;
            //std::cout << "weightPatch = \n " << weightPatch << std::endl;
            //std::cout << "spaWeiPatch = \n " << spaWeiPatch << std::endl;
            std::cout << "depWeiPatch = \n " << depWeiPatch << std::endl;
            //std::cout << "colorWeiPatch = \n " << colorWeiPatch << std::endl;
            //std::cout << "colorOri = \n " << colorOri << std::endl;
            char ch;
            std::cin.get(ch);
            std::cin.get(ch);

            continue;
        }
        //normalize weightPatch;
        weightPatch = weightPatch/(arma::accu(weightPatch));
        float finalDepVal = arma::accu(depPatchOri%weightPatch);
        if( !arma::is_finite(finalDepVal) )
        {
            fprintf(stderr, "encountered NaN!\n");
            std::cout << " i = " << i << " , j = " << j << std::endl;
            //std::cout << "depPatchOri = \n " << depPatchOri << std::endl;
            //std::cout << "labelPatch = \n " << labelPatch << std::endl;
            //std::cout << "weightPatch = \n " << weightPatch << std::endl;
            std::cout << "spaWeiPatch = \n " << spaWeiPatch << std::endl;
            std::cout << "depWeiPatch = \n " << depWeiPatch << std::endl;
            std::cout << "colorWeiPatch = \n " << colorWeiPatch << std::endl;
            std::cout << "colorOri = \n " << colorOri << std::endl;
            //char ch; std::cin.get(ch); std::cin.get(ch);
        }
        //std::cout << "finalDepVal = " << finalDepVal << std::endl;
        //char ch; std::cin.get(ch); std::cin.get(ch);

        if( finalDepVal == 0.0 )
        {
            std::cout << " i = " << i << " , j = " << j << std::endl;
            std::cout << " geoEstImg = " << geoEstImg(i,j) << std::endl;
            std::cout << "depPatchOri = \n " << depPatchOri << std::endl;
            std::cout << "depPatch = \n " << depPatch << std::endl;
            std::cout << "depPatchTmpBefore = \n " << depPatchTmpBefore << std::endl;
            std::cout << "depPatchTmp = \n " << depPatchTmp << std::endl;
            std::cout << "iniDepVal = \n " << iniDepVal << std::endl;
            std::cout << "sigmaDepPatch = \n " << sigmaDepPatch << std::endl;
            //std::cout << "labelPatch = \n " << labelPatch << std::endl;
            //std::cout << "weightPatch = \n " << weightPatch << std::endl;
            //std::cout << "spaWeiPatch = \n " << spaWeiPatch << std::endl;
            std::cout << "depWeiPatch = \n " << depWeiPatch << std::endl;
            //std::cout << "colorWeiPatch = \n " << colorWeiPatch << std::endl;
            //std::cout << "colorOri = \n " << colorOri << std::endl;
            char ch;
            std::cin.get(ch);
            std::cin.get(ch);

        }
        labelImg(i,j) = finalDepVal;
        denseDepImg(i,j) = finalDepVal;
    }

    //final check;
    for( int i = beginRow; i <= endRow; i++ )
    {
        for( int j = beginCol; j <= endCol; j++ )
        {
            if( denseDepImg(i,j) == 0.0 )
            {
                if( matrix2D.slice(2)(i,j) > 0. )
                    continue;
                //std::cout << "estimate point: " << i << "x" << j << std::endl;
                arma::Mat<float> depPatch = matrix2D.slice(2)(arma::span(i-(paddingSize-1)/2,i+(paddingSize-1)/2),
                                            arma::span(j-(paddingSize-1)/2,j+(paddingSize-1)/2)) +
                                            labelImg(arma::span(i-(paddingSize-1)/2,i+(paddingSize-1)/2),
                                                     arma::span(j-(paddingSize-1)/2,j+(paddingSize-1)/2));
                // if( i == 150 && j == 527 )
                //     std::cout << "the depPatch = \n " << depPatch << std::endl;
                //std::cout << "depPatch = \n" << depPatch << std::endl;
                // std::cout << "depPatch size is: " << depPatch.n_rows << "x" << depPatch.n_cols << std::endl;
                arma::Mat<float> depPatchOri = depPatch;

                arma::Mat<float> labelPatch = labelImg(arma::span(i-(paddingSize-1)/2,i+(paddingSize-1)/2),
                                                       arma::span(j-(paddingSize-1)/2,j+(paddingSize-1)/2));
                // std::cout << "the input depPatch =\n " << depPatchOri << std::endl;
                bool centerExist = false;
                if(depPatch((paddingSize-1)/2,(paddingSize-1)/2) > 0.)
                    centerExist = true;
                // depPatch = matNormalise(depPatch);
                // geoEstImg = arma::normalise(geoEstImg);
                if( arma::accu( depPatchOri ) == 0. )
                    continue;
                arma::uvec seedMat = arma::find(depPatchOri > 0.);
                if( seedMat.n_rows == 1 )
                {
                    denseDepImg(i,j) = depPatchOri( seedMat(0,0) );
                    continue;
                }
                //calculate the spatial distance weight;
                arma::Cube<float> cubePatch( paddingSize, paddingSize, 3 );
                cubePatch.fill(0.0);
                cubePatch.slice(0) = matrix2D.slice(0)(arma::span((i-(paddingSize-1)/2),(i+(paddingSize-1)/2)),
                                                       arma::span((j-(paddingSize-1)/2), (j+(paddingSize-1)/2)));
                cubePatch.slice(1) = matrix2D.slice(1)(arma::span((i-(paddingSize-1)/2),(i+(paddingSize-1)/2)),
                                                       arma::span((j-(paddingSize-1)/2), (j+(paddingSize-1)/2)));
                cubePatch.slice(2) = matrix2D.slice(2)(arma::span((i-(paddingSize-1)/2),(i+(paddingSize-1)/2)),
                                                       arma::span((j-(paddingSize-1)/2), (j+(paddingSize-1)/2)));
                //std::cout << "cubePatch size is: " << cubePatch.slice(2).n_rows << "x" << cubePatch.slice(2).n_cols << std::endl;
                arma::Mat<float> spaWeiPatch(paddingSize, paddingSize);
                spaWeiPatch.fill(0.0);
                arma::Mat<float> depWeiPatch(paddingSize, paddingSize);
                depWeiPatch.fill(0.0);
                arma::Mat<float> colorWeiPatch(paddingSize, paddingSize);
                colorWeiPatch.fill(0.0);
                //compute DepWeiPatch;
                arma::Mat<float> depPatchTmp = depPatch;
                depPatchTmp((paddingSize-1)/2, (paddingSize-1)/2) = geoEstImg(i,j);
                arma::Mat<float> depPatchTmpBefore = depPatchTmp;
                depPatch = matNormalize( depPatch );
                depPatchTmp = matNormalize( depPatchTmp );
                //std::cout << "after normalization, the depPatch is: \n" << depPatch << std::endl;
                // char chDep; std::cin.get(chDep); std::cin.get(chDep);
                float iniDepVal = depPatchTmp((paddingSize-1)/2, (paddingSize-1)/2);
                if( iniDepVal < 1e-3 )
                {
                    float minDis = 1000.;
                    int minRow = 0;
                    int minCol = 0;
                    for(int i1 = 0; i1 < paddingSize; i1++ )
                    {
                        for(int j1 = 0; j1 < paddingSize; j1++ )
                        {
                            if( depPatch(i1,j1) > 0. && sqrt( (i1 - (paddingSize-1)/2)*(i1 - (paddingSize-1)/2) + ( j1 - (paddingSize-1)/2)*(j1 - (paddingSize-1)/2)) < minDis )
                            {
                                minRow = i1;
                                minCol = j1;
                            }
                        }
                    }
                    if( minRow != 0 && minCol != 0 )
                        iniDepVal = depPatch(minRow, minCol);
                }
                float maxDepVal = 0.;
                maxDepVal = arma::max(arma::max( depPatchTmp ));
                float minDepVal = 0.;
                if( maxDepVal > 0. )
                    minDepVal = arma::min( arma::min( depPatchTmp( arma::find( depPatchTmp > 0 ) ) ) );

                arma::Mat<float> sigmaDepPatch( paddingSize, paddingSize );
                sigmaDepPatch.fill( 0.0 );
                for( int i1 = 0; i1 < paddingSize; i1++ )
                {
                    for( int j1 = 0; j1 < paddingSize; j1++ )
                    {
                        if( depPatchOri(i1,j1) == 0. )
                        {
                            sigmaDepPatch( i1, j1 ) = 1.0;
                            continue;
                        }
                        if( iniDepVal == depPatch(i1,j1) )
                        {
                            sigmaDepPatch( i1, j1 ) = 1.0;
                            continue;
                        }
                        //std::cout << " -log10(0.2) = " << -log10(0.2) << std::endl;
                        //std::cout << " fabs(iniDepVal - depPatch(i1,j1)) = " << fabs( iniDepVal - depPatch(i1,j1)) << std::endl;
                        float sigmaTmp = -log2( fabs(iniDepVal - depPatch(i1,j1)) );
                        // std::cout << " iniDepVal = " << iniDepVal << " depPatch(i1,j1) = " << depPatch(i1,j1) << std::endl;
                        // std::cout << "sigmaTmp = " << sigmaTmp << std::endl;
                        if( sigmaTmp == 0.0 )
                            sigmaTmp = 0.00001;
                        sigmaDepPatch( i1, j1 ) = sigmaTmp;
                        // char chPatch; std::cin.get(chPatch); std::cin.get(chPatch);
                    }
                }
                float sigmaDep = fabs(maxDepVal - minDepVal) < 0.0001?1:(-log(maxDepVal - minDepVal));
                //depWeiPatch = exp( -((depPatch - iniDepVal)%(depPatch - iniDepVal))/(2*sigmaDep*sigmaDep));
                depWeiPatch = exp( -((depPatch - iniDepVal)%(depPatch - iniDepVal))/(2*sigmaDepPatch%sigmaDepPatch));
                for( int i1 = 0; i1 < depWeiPatch.n_rows; i1++ )
                {
                    for( int j1 = 0; j1 < depWeiPatch.n_cols; j1++ )
                    {
                        if( depPatchOri(i1,j1) == 0. )
                            depWeiPatch(i1,j1) = 0.;
                    }
                }

                //std::cout << " - - - - - - - - - - - - -- - - -- - - - - - - \n";
                //std::cout << "sigmaDepPatch = \n " << sigmaDepPatch << std::endl;
                //std::cout << "depWeiPatch = \n " << depWeiPatch << std::endl;
                //std::cout << "depPatchOri = \n " << depPatchOri << std::endl;
                //char chPatch; std::cin.get(chPatch); std::cin.get(chPatch);
                //std::cout << "* * * * * * * * *  * * * * * * * * * * * * * * \n";
                // char chPatch; std::cin.get(chPatch); std::cin.get(chPatch);
                //depWeiPatch( arma::find( depPatch == 0 ) )  = 0.;

                //compute spaWeiPatch and colorWerPatch;
                arma::Mat<float> colorPatch(3,1);
                colorPatch.fill(0.0);

                cv::Mat labImgTmp; // = labImg(cv::Range(i-(colorPatchSize-1)/2, i + (colorPatchSize+1)/2), cv::Range(j-(colorPatchSize-1)/2, j + (colorPatchSize+1)/2)).clone();
                labImg(cv::Range(i-(colorPatchSize-1)/2, i + (colorPatchSize+1)/2), cv::Range(j-(colorPatchSize-1)/2, j + (colorPatchSize+1)/2)).copyTo( labImgTmp );
                arma::Cube<float> imgCubeTmp( colorPatchSize, colorPatchSize, 3 );
                imgCubeTmp.fill( 0.0 );
                imgCubeTmp.slice(0)= imgCube.slice(0)( arma::span( i - (colorPatchSize-1)/2, i + (colorPatchSize-1)/2),
                                                       arma::span( j - (colorPatchSize-1)/2, j + (colorPatchSize-1)/2) );
                imgCubeTmp.slice(1)= imgCube.slice(1)( arma::span( i - (colorPatchSize-1)/2, i + (colorPatchSize-1)/2),
                                                       arma::span( j - (colorPatchSize-1)/2, j + (colorPatchSize-1)/2) );
                imgCubeTmp.slice(2)= imgCube.slice(2)( arma::span( i - (colorPatchSize-1)/2, i + (colorPatchSize-1)/2),
                                                       arma::span( j - (colorPatchSize-1)/2, j + (colorPatchSize-1)/2) );
                //std::cout << "labImgTmp size is: " << labImgTmp.rows << "x" << labImgTmp.cols << std::endl;
                //std::cout << "the labImgTmp is:\n" << labImgTmp << std::endl;
                colorPatch = computeMeanColor( imgCubeTmp );
                //std::cout << "the colorPatch is " << colorPatch << std::endl;
                for(int i1 = (i - (paddingSize-1)/2); i1 <= (i + (paddingSize-1)/2); i1++)
                {
                    for(int j1 = (j - (paddingSize-1)/2); j1 <= (j + (paddingSize-1)/2); j1++)
                    {

                        float locRow = matrix2D.slice(0)(i1,j1);
                        float locCol = matrix2D.slice(1)(i1,j1);
                        //  std::cout << "(i1, j1) = (" << i1 << " , " << j1 << ")" << std::endl;
                        //  std::cout << "(i, j) = (" << i << " , " << j << ")" << std::endl;
                        //  std::cout << "(locRow, lowCol) = (" << locRow << " , " << locCol << ")" << std::endl;
                        //char chLoc; std::cin.get(chLoc); std::cin.get(chLoc);
                        int rowTmp = i1 - i + (paddingSize-1)/2;
                        int colTmp = j1 - j + (paddingSize-1)/2;
                        if( matrix2D.slice(2)(i1, j1) == 0.0 && depPatch(rowTmp, colTmp) == 0.0 )
                            continue;

                        if( locRow == 0.0 && locCol == 0.0 )
                            spaWeiPatch(rowTmp, colTmp) = exp(-((i1 - i)*(i1 -i) + (j1 - j)*(j1 - j))/(2*sigmaSpatial*sigmaSpatial));
                        else
                            spaWeiPatch(rowTmp, colTmp) = exp(-((locRow - i)*(locRow -i) + (locCol - j)*(locCol - j))/(2*sigmaSpatial*sigmaSpatial));
                        //depWeiPatch(rowTmp, colTmp) = exp(-(iniDepVal - matrix2D.slice(2)(i1,j1))*(iniDepVal - matrix2D.slice(2)(i1,j1))/(2*sigmaDep*sigmaDep));

                        arma::Mat<float> colorPatchTmp(3,1);
                        arma::Cube<float> imgCubeTmp1( colorPatchSize, colorPatchSize, 3 );
                        imgCubeTmp1.slice(0) = imgCube.slice(0)( arma::span( i1 - (colorPatchSize-1)/2, i1 + (colorPatchSize-1)/2),
                                               arma::span( j1 - (colorPatchSize-1)/2, j1 + (colorPatchSize-1)/2) );
                        imgCubeTmp1.slice(1) = imgCube.slice(1)( arma::span( i1 - (colorPatchSize-1)/2, i1 + (colorPatchSize-1)/2),
                                               arma::span( j1 - (colorPatchSize-1)/2, j1 + (colorPatchSize-1)/2) );
                        imgCubeTmp1.slice(2) = imgCube.slice(2)( arma::span( i1 - (colorPatchSize-1)/2, i1 + (colorPatchSize-1)/2),
                                               arma::span( j1 - (colorPatchSize-1)/2, j1 + (colorPatchSize-1)/2) );

                        //std::cout << "labImgTmp size is: " << labImgTmp.rows << "x" << labImgTmp.cols << std::endl;
                        //std::cout << "the labImgTmp is:\n" << labImgTmp << std::endl;
                        colorPatchTmp = computeMeanColor( imgCubeTmp1 );
                        //std::cout << "the colorPatchTmp is " << colorPatchTmp << std::endl;
                        //cv::Mat labImgTmp1;
                        //labImg(cv::Range(i1-(colorPatchSize-1)/2, i1 + (colorPatchSize+1)/2), cv::Range(j1-(colorPatchSize-1)/2, j1 + (colorPatchSize+1)/2)).copyTo(labImgTmp1);
                        //colorPatchTmp = computeMeanColor( labImgTmp1 );
                        colorWeiPatch( rowTmp, colTmp ) = sqrt(arma::accu((colorPatch - colorPatchTmp)%(colorPatch - colorPatchTmp)));
                    }
                }
                if( arma::max(arma::max(colorWeiPatch)) == 0. )
                {
                    colorWeiPatch.fill(1.0);
                }
                //arma::Mat<float> colorOri = colorWeiPatch;
                colorWeiPatch = matNormalize(colorWeiPatch);
                colorWeiPatch( (paddingSize - 1)/2, (paddingSize - 1)/2 ) = 1.0;
                arma::Mat<float> colorOri = colorWeiPatch;
                //std::cout << "finished color  weight computing!\n";
                float maxColorVal = 0.;
                if( arma::accu(colorWeiPatch) > 0 )
                    maxColorVal = arma::max( arma::max( colorWeiPatch ) );
                //std::cout << "the maxColorVal = " << maxColorVal << std::endl;
                float minColorVal = 0.;
                if( arma::accu(colorWeiPatch) > 0 )
                    minColorVal = arma::min( colorWeiPatch( arma::find(colorWeiPatch > 0) ) );
                float sigmaColor = 0.0;
                if( maxColorVal == minColorVal )
                    sigmaColor = 1.0;
                else
                {
                    sigmaColor = -log2( maxColorVal - minColorVal );
                    if( sigmaColor == 0. )
                        sigmaColor = 0.0001;
                }
                colorWeiPatch = exp(-(colorWeiPatch%colorWeiPatch)/(2*sigmaColor*sigmaColor));
                for( int i1 = 0; i1 < colorWeiPatch.n_rows; i1++ )
                {
                    for( int j1 = 0; j1 < colorWeiPatch.n_cols; j1++ )
                    {
                        if( depPatchOri(i1,j1) == 0. )
                            colorWeiPatch(i1,j1) = 0.;
                    }
                }
                arma::Mat<float> weightPatch = spaWeiPatch%depWeiPatch%colorWeiPatch;
                //std::cout << "weightPatch = \n" << weightPatch << std::endl;
                //reassign the weight for newly calculated depth value;
                for( int i1 = 0; i1 < weightPatch.n_rows; i1++ )
                {
                    for( int j1 = 0; j1 < weightPatch.n_cols; j1++ )
                    {
                        if( labelPatch(i1,j1) > 0 )
                            weightPatch(i1,j1) *= 0.1;
                    }
                }
                //std::cout << "spaWeiPatch = \n" << spaWeiPatch << std::endl;
                //std::cout << "depWeiPatch = \n" << depWeiPatch << std::endl;
                //std::cout << "colorWeiPatch = \n" << colorWeiPatch << std::endl;
                //char chTmp; std::cin.get(chTmp); std::cin.get(chTmp);
                //arma::Mat<float> weightPatch = spaWeiPatch*depWeiPatch;
                if( arma::accu(weightPatch) == 0 )
                {
                    fprintf(stderr, "cannot calculate depth value!\n");
                    std::cout << " i = " << i << " , j = " << j << std::endl;
                    std::cout << " geoEstImg = " << geoEstImg(i,j) << std::endl;
                    std::cout << "depPatchOri = \n " << depPatchOri << std::endl;
                    std::cout << "depPatch = \n " << depPatch << std::endl;
                    std::cout << "depPatchTmpBefore = \n " << depPatchTmpBefore << std::endl;
                    std::cout << "depPatchTmp = \n " << depPatchTmp << std::endl;
                    std::cout << "iniDepVal = \n " << iniDepVal << std::endl;
                    std::cout << "sigmaDepPatch = \n " << sigmaDepPatch << std::endl;
                    //std::cout << "labelPatch = \n " << labelPatch << std::endl;
                    //std::cout << "weightPatch = \n " << weightPatch << std::endl;
                    //std::cout << "spaWeiPatch = \n " << spaWeiPatch << std::endl;
                    std::cout << "depWeiPatch = \n " << depWeiPatch << std::endl;
                    //std::cout << "colorWeiPatch = \n " << colorWeiPatch << std::endl;
                    //std::cout << "colorOri = \n " << colorOri << std::endl;
                    char ch;
                    std::cin.get(ch);
                    std::cin.get(ch);

                    continue;
                }
                //normalize weightPatch;
                weightPatch = weightPatch/(arma::accu(weightPatch));
                float finalDepVal = arma::accu(depPatchOri%weightPatch);
                if( !arma::is_finite(finalDepVal) )
                {
                    fprintf(stderr, "encountered NaN!\n");
                    std::cout << " i = " << i << " , j = " << j << std::endl;
                    //std::cout << "depPatchOri = \n " << depPatchOri << std::endl;
                    //std::cout << "labelPatch = \n " << labelPatch << std::endl;
                    //std::cout << "weightPatch = \n " << weightPatch << std::endl;
                    std::cout << "spaWeiPatch = \n " << spaWeiPatch << std::endl;
                    std::cout << "depWeiPatch = \n " << depWeiPatch << std::endl;
                    std::cout << "colorWeiPatch = \n " << colorWeiPatch << std::endl;
                    std::cout << "colorOri = \n " << colorOri << std::endl;
                    //char ch; std::cin.get(ch); std::cin.get(ch);
                }
                //std::cout << "finalDepVal = " << finalDepVal << std::endl;
                //char ch; std::cin.get(ch); std::cin.get(ch);

                if( finalDepVal == 0.0 )
                {
                    std::cout << " i = " << i << " , j = " << j << std::endl;
                    std::cout << " geoEstImg = " << geoEstImg(i,j) << std::endl;
                    std::cout << "depPatchOri = \n " << depPatchOri << std::endl;
                    std::cout << "depPatch = \n " << depPatch << std::endl;
                    std::cout << "depPatchTmpBefore = \n " << depPatchTmpBefore << std::endl;
                    std::cout << "depPatchTmp = \n " << depPatchTmp << std::endl;
                    std::cout << "iniDepVal = \n " << iniDepVal << std::endl;
                    std::cout << "sigmaDepPatch = \n " << sigmaDepPatch << std::endl;
                    //std::cout << "labelPatch = \n " << labelPatch << std::endl;
                    //std::cout << "weightPatch = \n " << weightPatch << std::endl;
                    //std::cout << "spaWeiPatch = \n " << spaWeiPatch << std::endl;
                    std::cout << "depWeiPatch = \n " << depWeiPatch << std::endl;
                    //std::cout << "colorWeiPatch = \n " << colorWeiPatch << std::endl;
                    //std::cout << "colorOri = \n " << colorOri << std::endl;
                    char ch;
                    std::cin.get(ch);
                    std::cin.get(ch);

                }
                labelImg(i,j) = finalDepVal;
                denseDepImg(i,j) = finalDepVal;

            }
        }
    }
    arma::Mat<float> denseDepImgFinal(rgbImgNoPadding.rows, rgbImgNoPadding.cols);
    denseDepImgFinal.fill(0);
    denseDepImgFinal = denseDepImg( arma::span(paddingSize, denseDepImg.n_rows-paddingSize-1),
                                    arma::span(paddingSize, denseDepImg.n_cols-paddingSize-1));
    return denseDepImgFinal;
}

