#include"projection.h"


bool reprojection(const Eigen::Vector2f& location, float depVal, const cv::Mat& projectMatrix, pcl::PointXYZ& point3D )
{
// std::cout << "entered the reprojection function!\n" << std::endl;
    int j = location(0);//j is the col number;
    int i = location(1);//i is the row number;

    if( j < 0 || i < 0 || depVal < 0 )
    {
        return false;
    }

    float a1 = projectMatrix.at<float>(0,0);
    float a2 = projectMatrix.at<float>(1,0);
    float a3 = projectMatrix.at<float>(2,0);
    float b1 = projectMatrix.at<float>(0,1);
    float b2 = projectMatrix.at<float>(1,1);
    float b3 = projectMatrix.at<float>(2,1);
    float c1 = projectMatrix.at<float>(0,2);
    float c2 = projectMatrix.at<float>(1,2);
    float c3 = projectMatrix.at<float>(2,2);
    float d1 = projectMatrix.at<float>(0,3);
    float d2 = projectMatrix.at<float>(1,3);
    float d3 = projectMatrix.at<float>(2,3);

    //we construct two equations of Y and Z, they are:
    //A1*Y + B1*Z = C1;
    //A2*Y + B2*Z = C2;
    float A1 = b1-b3*j;
    float B1 = c1-c3*j;
    float C1 = (a3*j - a1)*depVal + d3*j - d1;

    float A2 = b2 - b3*i;
    float B2 = c2 - c3*i;
    float C2 = (a3*i - a2)*depVal + d3*i - d2;

    float Z = 0.0;
    float Y = 0.0;
    float X = depVal;

    if( (B1*A2 - B2*A1) != 0.0 )
        Z = (C1*A2 - C2*A1)/(B1*A2 - B2*A1);
    else
        Z = 1.0;

    if( A1 != 0.0 )
        Y = (C1 - B1*Z)/A1;
    else
        Y = 1.0;

    if( (B1*A2 - B2*A1 == 0.0) && A1 == 0.0  )
        return false;
    else
    {
        point3D.x = X;
        point3D.y = Y;
        point3D.z = Z;

        return true;
    }
}


bool readObjectLabel( std::vector<struct objectLabel>& objectLabels, const std::string& labelName, const Eigen::MatrixXf& R0_rect, const Eigen::MatrixXf& velo2camera)
{
    std::ifstream labelFile;
    labelFile.open(labelName.c_str(), std::ios_base::in);
    if( !labelFile.is_open() )
    {
        fprintf(stderr, "failed to open the label file: %s!\n", labelName.c_str());
        return false;
    }

    while( !labelFile.eof() )
    {
        std::string line_tmp;
        std::getline(labelFile, line_tmp);
        if( line_tmp.size() < 3 )
            continue;
        std::istringstream inputString(line_tmp);
        struct objectLabel objectLabel_tmp;
        inputString >> objectLabel_tmp.objectType >> objectLabel_tmp.truncation >> objectLabel_tmp.occlusion >> objectLabel_tmp.alphaAngle
                    >> objectLabel_tmp.bbox.left >> objectLabel_tmp.bbox.top >> objectLabel_tmp.bbox.right >> objectLabel_tmp.bbox.bottom
                    >> objectLabel_tmp.dimension.height >> objectLabel_tmp.dimension.width >> objectLabel_tmp.dimension.length
                    >> objectLabel_tmp.location.x >> objectLabel_tmp.location.y >> objectLabel_tmp.location.z
                    >> objectLabel_tmp.ry;

        /*
        std::cout << "successfully load one label:\n " << objectLabel_tmp.objectType << " " << objectLabel_tmp.truncation << " "
                  << objectLabel_tmp.occlusion << " " << objectLabel_tmp.alphaAngle << " " << objectLabel_tmp.bbox.left << " "
                  << objectLabel_tmp.bbox.top << " " << objectLabel_tmp.bbox.right << " " <<  objectLabel_tmp.bbox.bottom << " "
                  << objectLabel_tmp.dimension.height << " " << objectLabel_tmp.dimension.width << " " << objectLabel_tmp.dimension.length << " "
                  << objectLabel_tmp.location.x << " " << objectLabel_tmp.location.y << " " << objectLabel_tmp.location.z << " "
                  << objectLabel_tmp.ry << std::endl;
                  */
        //define the color for each object
        if( objectLabel_tmp.objectType == "Pedestrian" )
        {
            objectLabel_tmp.boundingBoxColor(0) = 255.;
            objectLabel_tmp.boundingBoxColor(1) = 0.;
            objectLabel_tmp.boundingBoxColor(2) = 0.;
        }
        else if( objectLabel_tmp.objectType == "Car" )
        {
            objectLabel_tmp.boundingBoxColor(0) = 0.;
            objectLabel_tmp.boundingBoxColor(1) = 255.;
            objectLabel_tmp.boundingBoxColor(2) = 0.;
        }
        else if( objectLabel_tmp.objectType == "Cyclist" )
        {
            objectLabel_tmp.boundingBoxColor(0) = 0.;
            objectLabel_tmp.boundingBoxColor(1) = 0.;
            objectLabel_tmp.boundingBoxColor(2) = 255.;
        }
        else if( objectLabel_tmp.objectType == "Tram" )
        {
            objectLabel_tmp.boundingBoxColor(0) = 125.;
            objectLabel_tmp.boundingBoxColor(1) = 255.;
            objectLabel_tmp.boundingBoxColor(2) = 0.;
        }

        objectLabels.push_back(objectLabel_tmp);
    }
    // std::cout << "finished reading labels!\n";
    labelFile.close();

    // std::cout << "the labelFile is closed!\n";
    if( compute3DBoundingBox(objectLabels, R0_rect, velo2camera) )
        return true;
    else
        return false;
}


bool readProjectionMatrix( cv::Mat& projectionMatrix, cv::Mat& cameraMatrix, cv::Mat& R0_rect, cv::Mat& velo2camera, const std::string& cameraFile)
{
  /*
    cv::Mat cameraMatrix(3,4,CV_32F,cv::Scalar::all(0));
    cv::Mat R0_rect(4,4,CV_32F,cv::Scalar::all(0));
    R0_rect.at<float>(3,3) = 1.0;
    cv::Mat velo2camera(4,4,CV_32F,cv::Scalar::all(0));
    velo2camera.at<float>(3,3) = 1.0;
    */
    std::ifstream inputFile;
    inputFile.open(cameraFile.c_str(),std::ios_base::in);
    if( !inputFile.is_open() )
    {
        fprintf(stderr, "cannot open the calibration file: %s!\n", cameraFile.c_str());
        return false;
    }

    while( !inputFile.eof() )
    {
        std::string line_tmp;
        std::getline(inputFile,line_tmp);
        std::istringstream inputString(line_tmp);
        std::string tag;
        inputString>>tag;
        if ( tag == "P2:")
        {
            inputString>> cameraMatrix.at<float>(0,0) >> cameraMatrix.at<float>(0,1) >> cameraMatrix.at<float>(0,2) >> cameraMatrix.at<float>(0,3)
                       >> cameraMatrix.at<float>(1,0) >> cameraMatrix.at<float>(1,1) >> cameraMatrix.at<float>(1,2) >> cameraMatrix.at<float>(1,3)
                       >> cameraMatrix.at<float>(2,0) >> cameraMatrix.at<float>(2,1) >> cameraMatrix.at<float>(2,2) >> cameraMatrix.at<float>(2,3);
        }

        if( tag == "R0_rect:" )
        {
            inputString>> R0_rect.at<float>(0,0) >> R0_rect.at<float>(0,1) >> R0_rect.at<float>(0,2)
                       >> R0_rect.at<float>(1,0) >> R0_rect.at<float>(1,1) >> R0_rect.at<float>(1,2)
                       >> R0_rect.at<float>(2,0) >> R0_rect.at<float>(2,1) >> R0_rect.at<float>(2,2);
        }

        if( tag == "Tr_velo_to_cam:")
        {
            inputString >> velo2camera.at<float>(0,0) >> velo2camera.at<float>(0,1) >> velo2camera.at<float>(0,2) >> velo2camera.at<float>(0,3)
                        >> velo2camera.at<float>(1,0) >> velo2camera.at<float>(1,1) >> velo2camera.at<float>(1,2) >> velo2camera.at<float>(1,3)
                        >> velo2camera.at<float>(2,0) >> velo2camera.at<float>(2,1) >> velo2camera.at<float>(2,2) >> velo2camera.at<float>(2,3);
        }
    }
    inputFile.close();

    projectionMatrix = cameraMatrix*R0_rect*velo2camera;

    //std::cout << "in the reading, the projectionMatrix = \n " << projectionMatrix << std::endl;
    return true;
}


bool compute3DBoundingBox( std::vector<struct objectLabel>& objectLabels, const Eigen::MatrixXf& R0_rect, const  Eigen::MatrixXf& velo2camera )
{
    if( objectLabels.size() < 1 )
        return false;
    for( int i = 0; i < objectLabels.size(); i++ )
    {
        Eigen::MatrixXf R(3,3);
        R(0,0) = cos(objectLabels[i].ry);
        R(0,1) = 0.;
        R(0,2) = sin(objectLabels[i].ry);
        R(1,0) = 0;
        R(1,1) = 1.;
        R(1,2) = 0;
        R(2,0) = -sin(objectLabels[i].ry);
        R(2,1) = 0.;
        R(2,2) = cos(objectLabels[i].ry);

        float height = objectLabels[i].dimension.height;
        float width = objectLabels[i].dimension.width;
        float length = objectLabels[i].dimension.length;

        Eigen::MatrixXf matrixTmp(3,8);
        matrixTmp.row(0) << length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2;
        matrixTmp.row(1) << 0, 0, 0, 0, -height, -height, -height,  -height;
        matrixTmp.row(2) << width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2;

        objectLabels[i].boundingBox3D = R*matrixTmp;
        objectLabels[i].boundingBox3D.row(0) = objectLabels[i].boundingBox3D.row(0) + objectLabels[i].location.x*Eigen::MatrixXf::Ones(1,8);
        objectLabels[i].boundingBox3D.row(1) = objectLabels[i].boundingBox3D.row(1) + objectLabels[i].location.y*Eigen::MatrixXf::Ones(1,8);
        objectLabels[i].boundingBox3D.row(2) = objectLabels[i].boundingBox3D.row(2) + objectLabels[i].location.z*Eigen::MatrixXf::Ones(1,8);

        Eigen::Matrix4f trsMatrix = (R0_rect*velo2camera).inverse();
        Eigen::MatrixXf camera3Dpoints(4,8);
        camera3Dpoints.row(0) = objectLabels[i].boundingBox3D.row(0);
        camera3Dpoints.row(1) = objectLabels[i].boundingBox3D.row(1);
        camera3Dpoints.row(2) = objectLabels[i].boundingBox3D.row(2);
        camera3Dpoints.row(3) = Eigen::MatrixXf::Ones(1,8);

        Eigen::MatrixXf velo3Dpoints(4,8);
        velo3Dpoints = trsMatrix*camera3Dpoints;
        for(int i = 0; i < 8; i++)
        {
            velo3Dpoints(0,i) /= velo3Dpoints(3,i);
            velo3Dpoints(1,i) /= velo3Dpoints(3,i);
            velo3Dpoints(2,i) /= velo3Dpoints(3,i);
        }

        objectLabels[i].boundingBox3D.row(0) = velo3Dpoints.row(0);
        objectLabels[i].boundingBox3D.row(1) = velo3Dpoints.row(1);
        objectLabels[i].boundingBox3D.row(2) = velo3Dpoints.row(2);
    }

    return true;
}


bool readPointXYZ(const std::string veloDataDir, pcl::PointCloud<pcl::PointXYZ>::Ptr veloCloudPtr, int imgRows, int imgCols, const arma::Mat<float>& projectionMatrix)
{
    FILE* veloData;
    veloData = fopen(veloDataDir.c_str(),"rb");

    if( veloData == NULL )
    {
        fprintf(stderr,"falied to load the velodyne file %s!\n", veloDataDir.c_str());
        return false;
    }
    int32_t veloDataNum = 1000000;
    float* data = (float*)malloc(veloDataNum*sizeof(float));
    float* px = data + 0;
    float* py = data + 1;
    float* pz = data + 2;
    float* pr = data + 3;

    veloDataNum = fread(data,sizeof(float),veloDataNum,veloData)/4;

    //pcl::PointCloud<pcl::PointXYZ>::Ptr veloCloudPtr (new pcl::PointCloud<pcl::PointXYZ>);
    for(int32_t i = 0; i < veloDataNum; i++)
    {
        pcl::PointXYZ pointTmp;
        pointTmp.x = *px;
        pointTmp.y = *py;
        pointTmp.z = *pz;
        
        arma::Mat<float> veloPoint3D(4,1);
        arma::Mat<float> imgPoint2D(3,1);
        veloPoint3D.fill(1.0);

        veloPoint3D(0,0) = *px;
        veloPoint3D(1,0) = *py;
        veloPoint3D(2,0) = *pz;
       // std::cout << "the depth value is: " << *px << std::endl;
        if( *px < 0 )
        {
            px += 4;
            py += 4;
            pz += 4;
            pr += 4;
            continue;
        }
       // std::cout << "the depth value is: " << *px << std::endl;
        imgPoint2D = projectionMatrix*veloPoint3D;
        //imgPoint2D = cameraMatrix_arma*R0_rect_arma*velo2camera_arma*veloPoint3D;
        if( imgPoint2D(2,0) == 0 )
        {
            fprintf(stderr,"the calculated 2D image points are wrong!\n");
            exit(0);
        }

        imgPoint2D(0,0) /= imgPoint2D(2,0);
        imgPoint2D(1,0) /= imgPoint2D(2,0);

        int colTmp = int(imgPoint2D(0,0)+0.5);

        int rowTmp = int(imgPoint2D(1,0)+0.5);

        if( rowTmp < 0 || rowTmp >= imgRows || colTmp < 0 || colTmp >= imgCols )
        {
            px += 4;
            py += 4;
            pz += 4;
            pr += 4;
            continue;
        }

        veloCloudPtr->points.push_back(pointTmp);


        px += 4;
        py += 4;
        pz += 4;
        pr += 4;
    }
    free(data);
    fclose(veloData);
    veloCloudPtr->width = (int) veloCloudPtr->points.size();
    veloCloudPtr->height = 1;

    return true;
}


bool generatePointXYZRGB(const cv::Mat& rgbImg, const cv::Mat& denseDepImg, const cv::Mat& projectionMatrix, pcl::PointCloud<pcl::PointXYZRGB>::Ptr veloCloudPtr)
{
    if( !rgbImg.data || !denseDepImg.data )
    {
        fprintf(stderr, "the rgbImg or the denseDepImg is incorrectly loaded!\n");
        return false;
    }
    for(int32_t i = 100; i < denseDepImg.rows; i++)
    {
        for(int32_t j = 0; j < denseDepImg.cols; j++)
        {
            if( denseDepImg.at<uchar>(i,j) == 0)
                continue;
             Eigen::Vector2f location( j, i );
            // location(0,0) = j;
            // location(1,0) = i;
            //std::cout << location << std::endl;
            float depVal = float(denseDepImg.at<uchar>(i,j));
            pcl::PointXYZ pointTmp;
            if( reprojection( location, depVal, projectionMatrix, pointTmp ) )
            {
                //veloCloudPtr->points.push_back(pointTmp);
                pcl::PointXYZRGB pointTmp_rgb;
                pointTmp_rgb.x = pointTmp.x;
                pointTmp_rgb.y = pointTmp.y;
                pointTmp_rgb.z = pointTmp.z;
                uint8_t R = rgbImg.at<cv::Vec3b>(i,j)[2];
                uint8_t G = rgbImg.at<cv::Vec3b>(i,j)[1];
                uint8_t B = rgbImg.at<cv::Vec3b>(i,j)[0];
                uint32_t rgbTmp = ((uint32_t)R << 16 | (uint32_t)G << 8 | (uint32_t)B);
                pointTmp_rgb.rgb = *reinterpret_cast<float*>(&rgbTmp);

                veloCloudPtr->points.push_back(pointTmp_rgb);
            }
        }
    }

    veloCloudPtr->width = (int) veloCloudPtr->points.size ();
    veloCloudPtr->height = 1;
    veloCloudPtr->width = (int) veloCloudPtr->points.size ();
    veloCloudPtr->height = 1;

    return true;
}

//matrix2D is a 3-channel matrix, which the first channel store the row-value, second channel store the col-value and
//the third channel store the depth value;
bool projection2D23D( arma::Cube<float>& matrix2D, const arma::Mat<float>& projectionMatrix, const std::string& veloPointFile,
                      const arma::Mat<float>& cameraMatrix_arma, const arma::Mat<float>& R0_rect_arma, const arma::Mat<float>& velo2camera_arma)
{
    FILE* stream;
    stream = fopen(veloPointFile.c_str(),"rb");
    if( stream == NULL )
    {
        fprintf(stderr, "failed to open veloDyne file: %s !\n", veloPointFile.c_str());
        return false;
    }
    int32_t num = 1000000;

    float* data = (float*)malloc(num*sizeof(float));
    float* px = data+0;
    float* py = data+1;
    float* pz = data+2;
    float* pr = data+3;

    num = fread(data,sizeof(float),num,stream)/4;

    int minDepth = 10000;
    int maxDepth = 0;
    int minIntensity = 10000;
    int maxIntensity = 0;

    int imgRows = matrix2D.slice(2).n_rows;
    int imgCols = matrix2D.slice(2).n_cols;

   // std::cout << "rows x cols : " << imgRows << "x" << imgCols << std::endl;
    cv::Mat testImg(cv::Size(imgCols, imgRows), CV_8UC1, cv::Scalar::all(0));

   // std::cout << "the point num is: " << num << std::endl;
    for(int32_t i=0; i<num; i++)
    {
        arma::Mat<float> veloPoint3D(4,1);
        arma::Mat<float> imgPoint2D(3,1);
        veloPoint3D.fill(1.0);

        veloPoint3D(0,0) = *px;
        veloPoint3D(1,0) = *py;
        veloPoint3D(2,0) = *pz;
       // std::cout << "the depth value is: " << *px << std::endl;
        if( *px < 0 )
        {
            px += 4;
            py += 4;
            pz += 4;
            pr += 4;
            continue;
        }
       // std::cout << "the depth value is: " << *px << std::endl;
        imgPoint2D = projectionMatrix*veloPoint3D;
        //imgPoint2D = cameraMatrix_arma*R0_rect_arma*velo2camera_arma*veloPoint3D;
        if( imgPoint2D(2,0) == 0 )
        {
            fprintf(stderr,"the calculated 2D image points are wrong!\n");
            exit(0);
        }

        imgPoint2D(0,0) /= imgPoint2D(2,0);
        imgPoint2D(1,0) /= imgPoint2D(2,0);

        int colTmp = int(imgPoint2D(0,0)+0.5);
        float colValTmp = imgPoint2D(0,0);

        int rowTmp = int(imgPoint2D(1,0)+0.5);
        float rowValTmp = imgPoint2D(1,0);

        //std::cout << "the rowTmp = " << rowTmp << " colTmp = " << colTmp << std::endl;
        //std::cout << "the rowValTmp = " << rowValTmp << " colValTmp = " << colValTmp << std::endl;
        //char ch; std::cin.get(ch); std::cin.get(ch);
        //if the image point exceeds the image plane;
        if( rowTmp < 0 || rowTmp >= imgRows || colTmp < 0 || colTmp >= imgCols )
        {
            px += 4;
            py += 4;
            pz += 4;
            pr += 4;
            continue;
        }

       // std::cout << "the rowTmp = " << rowTmp << " colTmp = " << colTmp << std::endl;
       // std::cout << "the rowValTmp = " << rowValTmp << " colValTmp = " << colValTmp << std::endl;
       // char ch; std::cin.get(ch); std::cin.get(ch);
        float  depth = fabs(*px);

       // std::cout << "assign values!\n";
        /*
        int rowColTmp = rowTmp;
        rowTmp = colTmp;
        colTmp = rowColTmp;
        */
        //std::cout << "the depth value is: " << depth << std::endl;
        if( (matrix2D.slice(2)(rowTmp,colTmp) == 0. ) || (matrix2D.slice(2)(rowTmp,colTmp) > depth) )
        //if( testImg.at<uchar>(rowTmp,colTmp) == 0 )
        {
       // std::cout << "the rowTmp = " << rowTmp << " colTmp = " << colTmp << std::endl;
         // std::cout << "the depth value is: " << depth << std::endl;
            //testImg.at<uchar>(rowTmp, colTmp) = depth;
            matrix2D.slice(2)(rowTmp,colTmp) = depth;
            matrix2D.slice(1)(rowTmp,colTmp) = colValTmp;
            matrix2D.slice(0)(rowTmp,colTmp) = rowValTmp;
        }
        px+=4;
        py+=4;
        pz+=4;
        pr+=4;
    }
    free(data);
    fclose(stream);

    float maxVal = 0.;
    for( int i = 0; i < testImg.rows; i++ )
    {
      for( int j = 0; j < testImg.cols; j++ )
      {
        if( testImg.at<uchar>(i,j) > maxVal )
          maxVal = testImg.at<uchar>(i,j);
      }
    }
    std::cout << "the maxVal = " << maxVal << std::endl;
    for( int i = 0; i < testImg.rows; i++ )
    {
      for( int j = 0; j < testImg.cols; j++ )
      {
        testImg.at<uchar>(i,j) = testImg.at<uchar>(i,j)*(255.0/maxVal);
      }
    }

    //cv::imshow("proRst", testImg);
    //cv::waitKey(0);
    CHECK( arma::accu( matrix2D.slice(2) ) > 0 ) << "failed to create matrix2D!!!\n";
    return true;
}
