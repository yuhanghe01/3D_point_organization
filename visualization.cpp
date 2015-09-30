/* @author: Yuhang He 
 * @Email: yuhanghe@whu.edu.cn
 *
 */

#include<strstream>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>
#include "projection.h"
#include "upsample.h"

struct saveFormat
{
    int i;
    int j;
    float depth;
    float height;
};

// --------------
// -----Help-----
// --------------
void
printUsage (const char* progName)
{
    std::cout << "\n\nUsage: "<<progName<<" [options]\n\n"
              << "Options:\n"
              << "-------------------------------------------\n"
              << "-h           this help\n"
              << "-s           Simple visualisation example\n"
              << "-r           RGB colour visualisation example\n"
              << "-c           Custom colour visualisation example\n"
              << "-n           Normals visualisation example\n"
              << "-a           Shapes visualisation example\n"
              << "-v           Viewports example\n"
              << "-i           Interaction Customization example\n"
              << "\n\n";
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, std::vector<struct objectLabel> objectLabels,
        const  Eigen::Matrix3f& intrinsicMat,const  Eigen::Matrix4f& extrinsicMat )
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters();
    viewer->setCameraParameters( intrinsicMat, extrinsicMat );
    //viewer->initCameraParameters();

    //std::cout << "the objects have been read are: " << objectLabels.size() << std::endl;
    for(int j = 0; j < objectLabels.size(); j++)
    {
        if( objectLabels[j].objectType == "DontCare" )
            continue;
        pcl::PointXYZRGB bboxPoint[8];
        for(int i = 0; i < 8; i++)
        {
            bboxPoint[i].x = objectLabels[j].boundingBox3D(0,i);
            bboxPoint[i].y = objectLabels[j].boundingBox3D(1,i);
            bboxPoint[i].z = objectLabels[j].boundingBox3D(2,i);

            //std::cout << "------ the points are: ------\n";
            //std::cout << "x= " << bboxPoint[i].x << " y= " << bboxPoint[i].y << " z= " << bboxPoint[i].z << std::endl;
            uint8_t R = 255;
            uint8_t G = 255;
            uint8_t B = 0;
            uint32_t rgbTmp = ((uint32_t)R << 16 | (uint32_t)G << 8 | (uint32_t)B);
            bboxPoint[i].rgb = *reinterpret_cast<float*>(&rgbTmp);
        }
        std::string lineName[12];
        for( int i = 0; i < 12; i++ )
        {
            int digitTmp = j*100 + i;
            std::strstream ss;
            std::string s;
            ss << digitTmp;
            ss >> s;
            lineName[i] = "line" + s;
        }
        double r = objectLabels[j].boundingBoxColor(0);
        double g = objectLabels[j].boundingBoxColor(1);
        double b = objectLabels[j].boundingBoxColor(2);
        //std::cout << "r= " << r << " g= " << g << " b= " << b << std::endl;
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[0],bboxPoint[1], r, g, b, lineName[0].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[1],bboxPoint[5], r, g, b, lineName[1].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[5],bboxPoint[4], r, g, b, lineName[2].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[4],bboxPoint[0], r, g, b, lineName[3].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[5],bboxPoint[6], r, g, b, lineName[4].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[6],bboxPoint[2], r, g, b, lineName[5].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[2],bboxPoint[1], r, g, b, lineName[6].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[4],bboxPoint[7], r, g, b, lineName[7].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[7],bboxPoint[3], r, g, b, lineName[8].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[3],bboxPoint[0], r, g, b, lineName[9].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[6],bboxPoint[7], r, g, b, lineName[10].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[3],bboxPoint[2], r, g, b, lineName[11].c_str());
    }
    return (viewer);
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,  std::vector<struct objectLabel> objectLabels)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    for(int j = 0; j < objectLabels.size(); j++)
    {
        if( objectLabels[j].objectType == "DontCare" )
            continue;
        pcl::PointXYZRGB bboxPoint[8];
        for(int i = 0; i < 8; i++)
        {
            bboxPoint[i].x = objectLabels[j].boundingBox3D(0,i);
            bboxPoint[i].y = objectLabels[j].boundingBox3D(1,i);
            bboxPoint[i].z = objectLabels[j].boundingBox3D(2,i);

            //std::cout << "------ the points are: ------\n";
            //std::cout << "x= " << bboxPoint[i].x << " y= " << bboxPoint[i].y << " z= " << bboxPoint[i].z << std::endl;
            uint8_t R = 255;
            uint8_t G = 255;
            uint8_t B = 0;
            uint32_t rgbTmp = ((uint32_t)R << 16 | (uint32_t)G << 8 | (uint32_t)B);
            bboxPoint[i].rgb = *reinterpret_cast<float*>(&rgbTmp);
        }
        std::string lineName[12];
        for( int i = 0; i < 12; i++ )
        {
            int digitTmp = j*100 + i;
            std::strstream ss;
            std::string s;
            ss << digitTmp;
            ss >> s;
            lineName[i] = "line" + s;
        }
        double r = objectLabels[j].boundingBoxColor(0);
        double g = objectLabels[j].boundingBoxColor(1);
        double b = objectLabels[j].boundingBoxColor(2);
        //std::cout << "r= " << r << " g= " << g << " b= " << b << std::endl;
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[0],bboxPoint[1], r, g, b, lineName[0].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[1],bboxPoint[5], r, g, b, lineName[1].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[5],bboxPoint[4], r, g, b, lineName[2].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[4],bboxPoint[0], r, g, b, lineName[3].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[5],bboxPoint[6], r, g, b, lineName[4].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[6],bboxPoint[2], r, g, b, lineName[5].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[2],bboxPoint[1], r, g, b, lineName[6].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[4],bboxPoint[7], r, g, b, lineName[7].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[7],bboxPoint[3], r, g, b, lineName[8].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[3],bboxPoint[0], r, g, b, lineName[9].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[6],bboxPoint[7], r, g, b, lineName[10].c_str());
        viewer->addLine<pcl::PointXYZRGB> (bboxPoint[3],bboxPoint[2], r, g, b, lineName[11].c_str());
    }


    return (viewer);
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> customColourVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> normalsVis (
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
    // --------------------------------------------------------
    // -----Open 3D viewer and add point cloud and normals-----
    // --------------------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals, 1, 0.2, "normals");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> shapesVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

    //------------------------------------
    //-----Add shapes at cloud points-----
    //------------------------------------
    viewer->addLine<pcl::PointXYZRGB> (cloud->points[0],
                                       cloud->points[cloud->size() - 1], "line");
    viewer->addSphere (cloud->points[0], 0.2, 0.5, 0.5, 0.0, "sphere");

    //---------------------------------------
    //-----Add shapes at other locations-----
    //---------------------------------------
    pcl::ModelCoefficients coeffs;
    coeffs.values.push_back (0.0);
    coeffs.values.push_back (0.0);
    coeffs.values.push_back (1.0);
    coeffs.values.push_back (0.0);
    viewer->addPlane (coeffs, "plane");
    coeffs.values.clear ();
    coeffs.values.push_back (0.3);
    coeffs.values.push_back (0.3);
    coeffs.values.push_back (0.0);
    coeffs.values.push_back (0.0);
    coeffs.values.push_back (1.0);
    coeffs.values.push_back (0.0);
    coeffs.values.push_back (5.0);
    viewer->addCone (coeffs, "cone");

    return (viewer);
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> viewportsVis (
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals1, pcl::PointCloud<pcl::Normal>::ConstPtr normals2)
{
    // --------------------------------------------------------
    // -----Open 3D viewer and add point cloud and normals-----
    // --------------------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->initCameraParameters ();

    int v1(0);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->setBackgroundColor (0, 0, 0, v1);
    viewer->addText("Radius: 0.01", 10, 10, "v1 text", v1);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud1", v1);

    int v2(0);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer->setBackgroundColor (0.3, 0.3, 0.3, v2);
    viewer->addText("Radius: 0.1", 10, 10, "v2 text", v2);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, single_color, "sample cloud2", v2);

    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud1");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud2");
    viewer->addCoordinateSystem (1.0);

    viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals1, 10, 0.05, "normals1", v1);
    viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals2, 10, 0.05, "normals2", v2);

    return (viewer);
}


unsigned int text_id = 0;
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
    if (event.getKeySym () == "r" && event.keyDown ())
    {
        std::cout << "r was pressed => removing all text" << std::endl;

        char str[512];
        for (unsigned int i = 0; i < text_id; ++i)
        {
            sprintf (str, "text#%03d", i);
            viewer->removeShape (str);
        }
        text_id = 0;
    }
}

void mouseEventOccurred (const pcl::visualization::MouseEvent &event,
                         void* viewer_void)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
    if (event.getButton () == pcl::visualization::MouseEvent::LeftButton &&
            event.getType () == pcl::visualization::MouseEvent::MouseButtonRelease)
    {
        std::cout << "Left mouse button released at position (" << event.getX () << ", " << event.getY () << ")" << std::endl;

        char str[512];
        sprintf (str, "text#%03d", text_id ++);
        viewer->addText ("clicked here", event.getX (), event.getY (), str);
    }
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> interactionCustomizationVis ()
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (1.0);

    viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
    viewer->registerMouseCallback (mouseEventOccurred, (void*)&viewer);

    return (viewer);
}


// --------------
// -----Main-----
// --------------
// this file is newly edited for KITTI point cloud visulization;
int main (int argc, char** argv)
{
  //std::string veloDir("0000001_rst.bin");

    if( argc != 6 )
    {
        fprintf(stderr, "Usage: ./visulizationTest veloPointDir rgbImgDir calibFileDir labelFileDir saveFileDir !\n");
        exit(0);
    }

    std::string veloPointDir = argv[1];
    std::string rgbImgDir = argv[2];
    std::string calibFileDir = argv[3];
    std::string labelFileDir = argv[4];
    std::string saveFileDir = argv[5];

    veloPointDir.append("/");
    rgbImgDir.append("/");
    calibFileDir.append("/");
    labelFileDir.append("/");
    saveFileDir.append("/");

    struct stat fileStat;
    if( !(stat(veloPointDir.c_str(),&fileStat) == 0) || !S_ISDIR(fileStat.st_mode) )
    {
        fprintf(stderr,"the veloPointDir file does not exist: %s\n",veloPointDir.c_str());
        exit(0);
    }

    if( !(stat(rgbImgDir.c_str(),&fileStat) == 0) || !S_ISDIR(fileStat.st_mode) )
    {
        fprintf(stderr,"the rgbImgDir file does not exist: %s\n",rgbImgDir.c_str());
        exit(0);
    }

    if( !(stat(calibFileDir.c_str(),&fileStat) == 0) || !S_ISDIR(fileStat.st_mode) )
    {
        fprintf(stderr,"the calibFileDir file does not exist: %s\n",calibFileDir.c_str());
        exit(0);
    }

    if( !(stat(labelFileDir.c_str(),&fileStat) == 0) || !S_ISDIR(fileStat.st_mode) )
    {
        fprintf(stderr,"the labelFileDir file does not exist: %s\n",labelFileDir.c_str());
        exit(0);
    }
    if( !(stat(saveFileDir.c_str(),&fileStat) == 0) || !S_ISDIR(fileStat.st_mode) )
    {
        fprintf(stderr,"the saveFileDir file does not exist: %s\n",saveFileDir.c_str());
        exit(0);
    }

    struct dirent* ptrDir;
    DIR* dir = NULL;
    if(NULL == (dir = opendir(veloPointDir.c_str())))
    {
        fprintf(stderr,"failed to open veloPointDir:  %s\n",veloPointDir.c_str());
        exit(1);
    }

    if(NULL == (dir = opendir(rgbImgDir.c_str())))
    {
        fprintf(stderr,"failed to open rgbImgDir:  %s\n",rgbImgDir.c_str());
        exit(0);
    }

    if( NULL == (dir = opendir(calibFileDir.c_str())) )
    {
        fprintf(stderr,"failed to open the calibFileDir:  %s\n", calibFileDir.c_str());
        exit(1);
    }

    if( NULL == (dir = opendir(labelFileDir.c_str())) )
    {
        fprintf(stderr,"failed to open the labelFileDir: %s\n", labelFileDir.c_str());
        exit(1);
    }

    if( NULL == (dir = opendir(saveFileDir.c_str())) )
    {
        fprintf(stderr,"failed to open the saveFileDir: %s\n", saveFileDir.c_str());
        exit(1);
    }

    google::InitGoogleLogging((const char*) argv[0]);
    // --------------------------------------
    // -----Parse Command Line Arguments-----
    // --------------------------------------
    /*
    if (pcl::console::find_argument (argc, argv, "-h") >= 0)
    {
        printUsage (argv[0]);
        return 0;
    }
    */
    bool simple(false), rgb(false), custom_c(false), normals(true),
         shapes(false), viewports(false), interaction_customization(false);
    /*
    if (pcl::console::find_argument (argc, argv, "-s") >= 0)
    {
        simple = true;
        std::cout << "Simple visualisation example\n";
    }
    else if (pcl::console::find_argument (argc, argv, "-c") >= 0)
    {
        custom_c = true;
        std::cout << "Custom colour visualisation example\n";
    }
    else if (pcl::console::find_argument (argc, argv, "-r") >= 0)
    {
        rgb = true;
        std::cout << "RGB colour visualisation example\n";
    }
    else if (pcl::console::find_argument (argc, argv, "-n") >= 0)
    {
        normals = true;
        std::cout << "Normals visualisation example\n";
    }
    else if (pcl::console::find_argument (argc, argv, "-a") >= 0)
    {
        shapes = true;
        std::cout << "Shapes visualisation example\n";
    }
    else if (pcl::console::find_argument (argc, argv, "-v") >= 0)
    {
        viewports = true;
        std::cout << "Viewports example\n";
    }
    else if (pcl::console::find_argument (argc, argv, "-i") >= 0)
    {
        interaction_customization = true;
        std::cout << "Interaction Customization example\n";
    }
    else
    {
        printUsage (argv[0]);
        return 0;
    }
    */

    dir = opendir( veloPointDir.c_str() );
    int tag = 0;
    while( (ptrDir = readdir(dir)) != NULL )
    {
      if( tag%10 == 0 )
      {
        tag++;
        continue;
      }

      tag++;
        std::string tName = ptrDir->d_name;
        std::cout << "processing the veloFile: " << tName << std::endl;

        if( tName == ".." || tName == "." )
            continue;

        size_t s = tName.rfind(".");
        if( s == std::string::npos )
            continue;
        /*
        if( !(strcmp(tName.c_str(),".bin") == 0) )
        {
            fprintf(stderr, "Failed to load the corresponding .bin velodyne file!\n");
            exit(0);
        }
        */

        // read the camera calibration file;
        tName = tName.substr(0,s);
        //std::string calibFile = calibFileDir + tName + ".txt";
        std::string calibFile = "0000.txt";
        cv::Mat projectMatrix;
        cv::Mat cameraMatrix(3,4,CV_32F,cv::Scalar::all(0));
        cv::Mat R0_rect(4,4,CV_32F,cv::Scalar::all(0));
        R0_rect.at<float>(3,3) = 1.0;
        cv::Mat velo2camera(4,4,CV_32F,cv::Scalar::all(0));
        velo2camera.at<float>(3,3) = 1.0;

        if( !readProjectionMatrix( projectMatrix, cameraMatrix, R0_rect, velo2camera, calibFile) )
        {
            fprintf(stderr, "failed to read the calibration file: %s !\n", calibFile.c_str());
            exit(0);
        }

        Eigen::MatrixXf R0_rect_eigen(R0_rect.rows, R0_rect.cols);
        Eigen::MatrixXf velo2camera_eigen(velo2camera.rows, velo2camera.cols);
        arma::Mat<float> projectionMatrix_arma( projectMatrix.rows, projectMatrix.cols );
        Eigen::MatrixXf projectionMatrix_eigen(3,4);
        arma::Mat<float> cameraMatrix_arma(3,4);
        cameraMatrix_arma.fill(0);
        arma::Mat<float> R0_rect_arma(4,4);
        R0_rect_arma.fill(0);
        arma::Mat<float> velo2camera_arma(4,4);
        velo2camera_arma.fill(0);
        for(int i = 0; i < 4; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                R0_rect_arma(i,j) = R0_rect.at<float>(i,j);
                velo2camera_arma(i,j) = R0_rect.at<float>(i,j);

                R0_rect_eigen(i,j) = R0_rect.at<float>(i,j);
                velo2camera_eigen(i,j) = velo2camera.at<float>(i,j);
                // extrinsicMat(i,j) = velo2camera.at<float>(i,j);
                if( i < 3 )
                {
                    projectionMatrix_arma(i,j) = projectMatrix.at<float>(i,j);
                    projectionMatrix_eigen(i,j) = projectMatrix.at<float>(i,j);
                    cameraMatrix_arma(i,j) = cameraMatrix.at<float>(i,j);
                }
            }
        }


        //std::cout << "projectMatrix = \n" << projectMatrix << std::endl; 
        // read the RGB image;
        std::string rgbImg_dir = rgbImgDir + tName + ".png";
        cv::Mat rgbImg = cv::imread(rgbImg_dir, CV_LOAD_IMAGE_COLOR);
        if( !rgbImg.data )
        {
          fprintf( stderr, "falied th load the RGB image: %s \n", rgbImg_dir.c_str() );
          exit(0);
        }
        //read the velodyne binary file;
        std::string veloDataDir = veloPointDir + ptrDir -> d_name;
        pcl::PointCloud<pcl::PointXYZ>::Ptr veloCloudPtr_bin (new pcl::PointCloud<pcl::PointXYZ>);
        if( !readPointXYZ(veloDataDir, veloCloudPtr_bin, rgbImg.rows, rgbImg.cols, projectionMatrix_arma) )
        {
            fprintf(stderr,"failed to load the velodyne .bin file: %s !\n", veloDataDir.c_str());
            return 0;
        }

        //project the 3D veloDyne data into 2D image plane;
        arma::Cube<float> matrix2D( rgbImg.rows, rgbImg.cols, 3 );
        matrix2D.slice(0).fill(0.0);
        matrix2D.slice(1).fill(0.0);
        matrix2D.slice(2).fill(0.0);
        if( !projection2D23D( matrix2D, projectionMatrix_arma, veloDataDir, cameraMatrix_arma, R0_rect_arma, velo2camera_arma ) )
        {
            fprintf(stderr, "failed to project 3D point cloud into 2D image plane!\n");
            exit(1);
        }
        int paddingSize = 11;

        inhomogeneityClean( matrix2D.slice(2), paddingSize );

        
        cv::Mat sparseDepImg( cv::Size(matrix2D.slice(2).n_cols, matrix2D.slice(2).n_rows), CV_8UC1, cv::Scalar::all(0));
        float maxSparseDepVal = 0.;
        for(int i1 = 0; i1 < matrix2D.slice(2).n_rows; i1++)
        {
          for(int j1 = 0; j1 < matrix2D.slice(2).n_cols; j1++)
          {
            sparseDepImg.at<uchar>(i1,j1) = uchar( matrix2D.slice(2)(i1,j1) + 0.5 );
            if( sparseDepImg.at<uchar>(i1,j1) > maxSparseDepVal )
              maxSparseDepVal = sparseDepImg.at<uchar>(i1,j1);
          }
        }
        sparseDepImg = sparseDepImg*(255.0/maxSparseDepVal);

        //cv::imshow("sparseDepImg", sparseDepImg);
        //cv::waitKey(0);
        //std::string saveSparseImgName = "SparseRst/" + tName + ".png";
        //cv::imwrite( saveSparseImgName.c_str(), sparseDepImg );
        // */
        //arma::Cube<float> sparseMat = matrix2D;
        //std::cout << "finished inhomogeneityClean!!!!\n";

        arma::Mat<float> geoEstImg( rgbImg.rows, rgbImg.cols);
        geoEstImg.fill(0.0);
        if( !geoDepEstimation(rgbImg, matrix2D, geoEstImg) )
        {
            fprintf(stderr,"failed to geodesic-estimate depth map!\n");
            exit(0);
        }


        
        cv::Mat testImg( cv::Size( matrix2D.slice(2).n_cols, matrix2D.slice(2).n_rows ), CV_8UC1, cv::Scalar::all(0) );
        int maxValueImg = 0;
        int minValueImg = 255;
        for( int i = 0; i < testImg.rows; i++ )
        {
            for( int j = 0; j < testImg.cols; j++ )
            {
                testImg.at<uchar>(i,j) = int(geoEstImg(i,j)+0.5);
                if( int(geoEstImg(i,j) + 0.5) > 0 && int( geoEstImg(i,j) + 0.5 ) > maxValueImg )
                    maxValueImg = int( geoEstImg(i,j) + 0.5 );
                //std::cout << "the depth value is " << int(geoImgNoPadding(i,j) + 0.5) << std::endl;
            }
        }
        testImg = testImg*(255.0/maxValueImg);
       // std::string saveGeoImgName = "geoRst/" + tName + ".png";
       // cv::imwrite( saveGeoImgName.c_str(), testImg );
        //cv::imshow("geoEstImg", testImg);
        //cv::waitKey(0);
        //  */
        arma::Mat<float> denseDepImg(rgbImg.rows, rgbImg.cols);
        //arma::Mat<float> denseDepImg1;
        denseDepImg.fill(0.0);
        //denseDepImg1 = depUpsampling( matrix2D, geoEstImg, rgbImg);
        //inhomogeneityClean( matrix2D.slice(2), paddingSize );
        //std::cout << "begin to check rgbImg:\n";
        
        denseDepImg = depUpsamplingRandom( matrix2D, geoEstImg, rgbImg );

        std::cout << "finished depUpsamplingDynamic!!!\n";
        arma::Mat<float> denseDepImgRandom( rgbImg.rows, rgbImg.cols );
        denseDepImgRandom.fill( 0.0 );

        //denseDepImgRandom = depUpsamplingRandom( matrix2D, geoEstImg, rgbImg );


        std::cout << "finished depUpsamplingRandom!\n";
        arma::Mat<float> heightImgDynamic( rgbImg.rows, rgbImg.cols );
        heightImgDynamic.fill( 0.0 );
       // arma::Mat<float> heightImgRandom( rgbImg.rows, rgbImg.cols );
       //  heightImgRandom.fill( 0.0 );

        for(int i1 = 0; i1 < rgbImg.rows; i1++ )
        {
          for(int j1 = 0; j1 < rgbImg.cols; j1++ )
          {
            if( denseDepImg(i1,j1) > 0. )
            {
              pcl::PointXYZ point3DTmp;
              Eigen::Vector2f locationTmp;
              locationTmp(0) = j1;
              locationTmp(1) = i1;
              float depValTmp = denseDepImg(i1,j1);

              reprojection( locationTmp, depValTmp, projectMatrix, point3DTmp );

              heightImgDynamic(i1,j1) = point3DTmp.z;
            }
            /*
            if( denseDepImgRandom(i1,j1) > 0. )
            {
              pcl::PointXYZ point3DTmp;
              Eigen::Vector2f locationTmp;
              locationTmp(0) = j1;
              locationTmp(1) = i1;
              float depValTmp = denseDepImgRandom(i1,j1);

              reprojection( locationTmp, depValTmp, projectMatrix, point3DTmp );

              heightImgRandom(i1,j1) = point3DTmp.z;
            }
            */

          }
        }

        cv::Mat denseDepImgSave( cv::Size( rgbImg.cols, rgbImg.rows), CV_8UC1, cv::Scalar::all(0) );
        cv::Mat denseHeightImgSave( cv::Size( rgbImg.cols, rgbImg.rows), CV_8UC1, cv::Scalar::all(0) );
        //cv::Mat denseDepImgSaveRandom( cv::Size( rgbImg.cols, rgbImg.rows), CV_8UC1, cv::Scalar::all(0) );
        //cv::Mat denseHeightImgSaveRandom( cv::Size( rgbImg.cols, rgbImg.rows), CV_8UC1, cv::Scalar::all(0) );

        float maxDynamic = -255.0;
        float minDynamic = 255.0;
        float maxDynamicHeight = -255.0;
        float minDynamicHeight = 255.0;

        float maxRandom = -255.0;
        float minRandom = 255.0;
        float maxRandomHeight = -255.0;
        float minRandomHeight = 255.0;

        for(int i1 = 0; i1 < rgbImg.rows; i1++ )
        {
          for(int j1 = 0; j1 < rgbImg.cols; j1++ )
          {
            denseDepImgSave.at<uchar>(i1,j1) = uchar( denseDepImg(i1,j1) + 0.5 );
            //denseDepImgSaveRandom.at<uchar>(i1,j1) = uchar( denseDepImgRandom(i1,j1) + 0.5 );
            if( denseDepImg(i1,j1) > maxDynamic )
              maxDynamic = denseDepImg(i1,j1);
            if( denseDepImg(i1,j1) < minDynamic )
              minDynamic = denseDepImg(i1,j1);

            if( heightImgDynamic(i1,j1) > maxDynamicHeight )
              maxDynamicHeight = heightImgDynamic(i1,j1);
            if( heightImgDynamic(i1,j1) < minDynamicHeight )
              minDynamicHeight = heightImgDynamic(i1,j1);

           // if( denseDepImgRandom(i1,j1) > maxRandom )
           //   maxRandom = denseDepImgRandom(i1,j1);
           // if( denseDepImgRandom(i1,j1) < minRandom )
           //   minRandom = denseDepImgRandom(i1,j1);

           // if( heightImgRandom(i1,j1) > maxRandomHeight )
           //   maxRandomHeight = heightImgRandom(i1,j1);
           // if( heightImgRandom(i1,j1) < minRandomHeight )
           //   minRandomHeight = heightImgRandom(i1,j1);
          }
        }

        denseDepImgSave = denseDepImgSave*(255.0/maxDynamic);
        //denseDepImgSaveRandom = denseDepImgSaveRandom*(255.0/maxRandom);
        
        float kDynamic = 255.0/(maxDynamicHeight - minDynamicHeight );
        float bDynamic = -kDynamic*minDynamicHeight;

       // float kRandom = 255.0/(maxRandomHeight - minRandomHeight);
       // float bRandom = -kRandom*minRandomHeight;

        for(int i1 = 0; i1 < rgbImg.rows; i1++ )
        {
          for(int j1 = 0; j1 < rgbImg.cols; j1++ )
          {
            denseHeightImgSave.at<uchar>(i1,j1) = uchar( kDynamic*heightImgDynamic(i1,j1) + bDynamic);
           // denseHeightImgSaveRandom.at<uchar>(i1,j1) = uchar( kRandom*heightImgRandom(i1,j1) + bRandom);
          }
        }

        std::string saveDyDep = "DynamicDep02/" + tName + ".png";
        std::string saveDyHei = "DynamicHei02/" + tName + ".png";
        std::string saveRaDep = "RandomDep01/" + tName + ".png";
        std::string saveRaHei = "RandomHei01/" + tName + ".png";
        cv::imwrite( saveDyDep.c_str(), denseDepImgSave );
        cv::imwrite( saveDyHei.c_str(), denseHeightImgSave );
        //cv::imwrite( saveRaDep.c_str(), denseDepImgSaveRandom );
        //cv::imwrite( saveRaHei.c_str(), denseHeightImgSaveRandom );

        std::string saveName = "dynamicBin02/" + tName + ".bin";
        std::ofstream outputFile( saveName.c_str(),std::ios::binary);
        if( !outputFile )
        {
            fprintf( stderr, "failed to open the outputFile!\n");
            exit(0);
        }

        for(int i = 0; i < rgbImg.rows; i++)
        {
            for(int j = 0; j < rgbImg.cols; j++)
            {
                struct saveFormat saveFormatTmp = { i,j, denseDepImg(i,j), heightImgDynamic(i,j)};
                outputFile.write((char*) &saveFormatTmp, sizeof(saveFormatTmp) );
            }
        }
        outputFile.close();

        /*
        std::string saveNameRandom = "randomBin/" + tName + ".bin";
        std::ofstream outputFile1( saveNameRandom.c_str(),std::ios::binary);
        if( !outputFile1 )
        {
            fprintf( stderr, "failed to open the outputFile!\n");
            exit(0);
        }

        for(int i = 0; i < rgbImg.rows; i++)
        {
            for(int j = 0; j < rgbImg.cols; j++)
            {
                struct saveFormat saveFormatTmp = { i,j, denseDepImgRandom(i,j), heightImgRandom(i,j)};
                outputFile1.write((char*) &saveFormatTmp, sizeof(saveFormatTmp) );
            }
        }
        outputFile1.close();
          */

        /*
        cv::Mat testImg1( cv::Size( denseDepImg.n_cols, denseDepImg.n_rows ), CV_8UC1, cv::Scalar::all(0) );
        int maxValueImg1 = 0;
        int minValueImg1 = 255;
        for( int i = 0; i < testImg1.rows; i++ )
        {
            for( int j = 0; j < testImg1.cols; j++ )
            {
                testImg1.at<uchar>(i,j) = int(denseDepImg(i,j)+0.5);
                if( int(denseDepImg(i,j) + 0.5) > 0 && int( denseDepImg(i,j) + 0.5 ) > maxValueImg1 )
                    maxValueImg1 = int( denseDepImg(i,j) + 0.5 );
                //std::cout << "the depth value is " << int(geoImgNoPadding(i,j) + 0.5) << std::endl;
            }
        }
        testImg1 = testImg1*(255.0/maxValueImg1);
        std::string saveFileName = saveFileDir + tName + ".png";
        cv::imwrite( saveFileName.c_str(), testImg1 );
        //cv::namedWindow("upsampleRst", cv::WINDOW_AUTOSIZE);
        //cv::imshow("upSampleRst", testImg1);
        //cv::waitKey(0);
        //cv::destroyWindow("upsampleRst");
         */
        //read the objectLabel file;
        /*
        std::string labelFile = labelFileDir + tName + ".txt";
        std::vector<struct objectLabel> objectLabels;
        if( !readObjectLabel( objectLabels, labelFile, R0_rect_eigen, velo2camera_eigen ) )
        {
            fprintf(stderr, "failed to load the labelFile: %s !\n", labelFile.c_str());
            return 0;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr veloCloudPtr_denseRGB(new pcl::PointCloud<pcl::PointXYZRGB>);

        for(int32_t i = 0; i < matrix2D.slice(2).n_rows; i++)
        {
            for(int32_t j = 0; j < matrix2D.slice(2).n_cols; j++)
            {
                if( denseDepImg(i,j) == 0. )
                    continue;
                //if( sparseMat.slice(2)(i,j) == 0. )
                //  continue;
                Eigen::Vector2f location( matrix2D.slice(1)(i,j), matrix2D.slice(0)(i,j) );
                float depVal = denseDepImg(i,j);
                // float depVal = sparseMat.slice(2)(i,j);
                pcl::PointXYZ pointTmp;
                if( reprojection(location, depVal, projectMatrix, pointTmp) )
                {
                    //veloCloudPtr1->points.push_back(pointTmp);
                    pcl::PointXYZRGB pointTmp_rgb;
                    pointTmp_rgb.x = pointTmp.x;
                    pointTmp_rgb.y = pointTmp.y;
                    pointTmp_rgb.z = pointTmp.z;
                    uint8_t R = rgbImg.at<cv::Vec3b>(i,j)[2];
                    uint8_t G = rgbImg.at<cv::Vec3b>(i,j)[1];
                    uint8_t B = rgbImg.at<cv::Vec3b>(i,j)[0];
                    uint32_t rgbTmp = ((uint32_t)R << 16 | (uint32_t)G << 8 | (uint32_t)B);
                    pointTmp_rgb.rgb = *reinterpret_cast<float*>(&rgbTmp);

                    veloCloudPtr_denseRGB->points.push_back(pointTmp_rgb);
                }
            }
        }
        veloCloudPtr_denseRGB->width = (int) veloCloudPtr_denseRGB->points.size ();
        veloCloudPtr_denseRGB->height = 1;
        veloCloudPtr_denseRGB->width = (int) veloCloudPtr_denseRGB->points.size ();
        veloCloudPtr_denseRGB->height = 1;

        // ----------------------------------------------------------------
        // -----Calculate surface normals with a search radius of 0.05-----
        // ----------------------------------------------------------------
        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
        ne.setInputCloud (veloCloudPtr_denseRGB);
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
        ne.setSearchMethod (tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        ne.setRadiusSearch(0.3);
        ne.compute(*cloud_normals);
        */

        /*
        for(int32_t l = 0; l < 1; l++)
        {
            float left = objectLabels[l].bbox.left;
            float right = objectLabels[l].bbox.right;
            float top = objectLabels[l].bbox.top;
            float bottom = objectLabels[l].bbox.bottom;
            for(int32_t i = int(top+0.5); i < int(bottom+0.5); i++)
            {
                for(int32_t j = int(left + 0.5); j < int(right + 0.5); j++)
                {
                    if( denseDepImg1(i,j) == 0. )
                        continue;
                    //if( sparseMat.slice(2)(i,j) == 0. )
                    //  continue;
                    Eigen::Vector2f location( matrix2D.slice(1)(i,j), matrix2D.slice(0)(i,j) );
                    float depVal = denseDepImg1(i,j);
                    // float depVal = sparseMat.slice(2)(i,j);
                    pcl::PointXYZ pointTmp;
                    if( reprojection(location, depVal, projectMatrix, pointTmp) )
                    {
                        veloCloudPtr1->points.push_back(pointTmp);
                        pcl::PointXYZRGB pointTmp_rgb;
                        pointTmp_rgb.x = pointTmp.x;
                        pointTmp_rgb.y = pointTmp.y;
                        pointTmp_rgb.z = pointTmp.z;
                        uint8_t R = rgbImg.at<cv::Vec3b>(i,j)[2];
                        uint8_t G = rgbImg.at<cv::Vec3b>(i,j)[1];
                        uint8_t B = rgbImg.at<cv::Vec3b>(i,j)[0];
                        uint32_t rgbTmp = ((uint32_t)R << 16 | (uint32_t)G << 8 | (uint32_t)B);
                        pointTmp_rgb.rgb = *reinterpret_cast<float*>(&rgbTmp);

                        veloCloudPtr_denseRGB->points.push_back(pointTmp_rgb);
                    }
                }
            }
        }
        */

        /*
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
        viewer = normalsVis( veloCloudPtr_denseRGB, cloud_normals );
        //--------------------
        // -----Main loop-----
        //--------------------
        while (!viewer->wasStopped ())
        {
            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }
        */

    }
    // ------------------------------------
    // -----read the camera matrix---------
    // ------------------------------------


    /*
    std::string calibFile = "000074.txt";

    cv::Mat projectMatrix;
    cv::Mat cameraMatrix(3,4,CV_32F,cv::Scalar::all(0));
    cv::Mat R0_rect(4,4,CV_32F,cv::Scalar::all(0));
    R0_rect.at<float>(3,3) = 1.0;
    cv::Mat velo2camera(4,4,CV_32F,cv::Scalar::all(0));
    velo2camera.at<float>(3,3) = 1.0;

    if( !readProjectionMatrix( projectMatrix, cameraMatrix, R0_rect, velo2camera, calibFile) )
    {
        fprintf(stderr, "failed to read the calibration file: %s !\n", calibFile.c_str());
        exit(0);
    }

    Eigen::MatrixXf R0_rect_eigen(R0_rect.rows, R0_rect.cols);
    Eigen::MatrixXf velo2camera_eigen(velo2camera.rows, velo2camera.cols);
    arma::Mat<float> projectionMatrix_arma( projectMatrix.rows, projectMatrix.cols );
    Eigen::MatrixXf projectionMatrix_eigen(3,4);
    arma::Mat<float> cameraMatrix_arma(3,4);
    cameraMatrix_arma.fill(0);
    arma::Mat<float> R0_rect_arma(4,4);
    R0_rect_arma.fill(0);
    arma::Mat<float> velo2camera_arma(4,4);
    velo2camera_arma.fill(0);

    Eigen::Matrix3f intrinsicMat;
    Eigen::Matrix3f intrinsicMat1;
    Eigen::Matrix4f extrinsicMat;

    //cv::Mat matTmp = cameraMatrix*R0_rect;
    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            R0_rect_arma(i,j) = R0_rect.at<float>(i,j);
            velo2camera_arma(i,j) = R0_rect.at<float>(i,j);

            R0_rect_eigen(i,j) = R0_rect.at<float>(i,j);
            velo2camera_eigen(i,j) = velo2camera.at<float>(i,j);
            extrinsicMat(i,j) = velo2camera.at<float>(i,j);
            if( i < 3 )
            {
                projectionMatrix_arma(i,j) = projectMatrix.at<float>(i,j);
                projectionMatrix_eigen(i,j) = projectMatrix.at<float>(i,j);
                cameraMatrix_arma(i,j) = cameraMatrix.at<float>(i,j);
            }
            if( i < 3 && j < 3 )
            {
                intrinsicMat(i,j) = cameraMatrix.at<float>(i,j);
            }
        }
    }

    if( !calculateKRT(projectionMatrix_eigen, intrinsicMat1, extrinsicMat) )
    {
        fprintf(stderr, "failed to calculate the intrinsic/extrinsic Matrix!\n");
        exit(0);
    }


    std::cout << "the intrinsic = \n " << intrinsicMat1 << std::endl;
    intrinsicMat = intrinsicMat1;
    cv::Mat RT = R0_rect*velo2camera;
    cv::Mat R = RT( cv::Range(0,3), cv::Range(0,3) );
    R = R.t();
    R = -R;
    //R = R.t();
    //R = R.inv();
    //R.copyTo( extrinsic.at<float>(cv::Range(0,3),cv::Range(0,3)));
    cv::Mat T = RT( cv::Range(0,3), cv::Range(3,4) );
    //T.copyTo( extrinsic.at<float>(cv::Range(0,3), cv::Range(3,4)) );

    for(int i = 0; i < 3; i++ )
    {
        for(int j = 0; j < 3; j++ )
        {
            extrinsicMat(i,j) = R.at<float>(i,j);
        }
    }
    extrinsicMat(0,3) = T.at<float>(0,0);
    extrinsicMat(1,3) = T.at<float>(1,0);
    extrinsicMat(2,3) = T.at<float>(2,0);

    std::cout << " after transpose, the intrinsic = \n " << intrinsicMat << std::endl;
    std::cout << "the extrinsic = \n " << extrinsicMat << std::endl;
    //std::cout << "cameraMatrix = \n " << cameraMatrix << std::endl;
    //std::cout << "R0_rect*velo2camera = \n " << R0_rect*velo2camera << std::endl;
    //std::cout << "velo2camera = \n" << velo2camera << std::endl;
    //std::cout << "the projectionMatrix read is:\n" << projectMatrix << std::endl;
    //std::cout << "the projectionMatrix transfered is:\n" << projectionMatrix_arma << std::endl;
    //char chTmp; std::cin.get(chTmp);

    cv::Mat denseDepImg = cv::imread("000074tgvl.png",CV_LOAD_IMAGE_GRAYSCALE);
    if( !denseDepImg.data )
    {
        fprintf(stderr,"failed to load the denseDepImg!\n");
        exit(1);
    }
    //cv::imshow("denseDep",denseDepImg);
    //cv::waitKey(0);

    pcl::PointCloud<pcl::PointXYZ>::Ptr veloCloudPtr1(new pcl::PointCloud<pcl::PointXYZ>);
    cv::Mat rgbImg = cv::imread("000074RGB.png", CV_LOAD_IMAGE_COLOR);
    std::cout << "the image size is: " << rgbImg.rows << "x" << rgbImg.cols << std::endl;
    if( !rgbImg.data )
    {
        fprintf(stderr,"failed to load the RGB image!\n");
        exit(0);
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr veloCloudPtr_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    for(int32_t i = 100; i < denseDepImg.rows; i++)
    {
        for(int32_t j = 0; j < denseDepImg.cols; j++)
        {
            if( denseDepImg.at<uchar>(i,j) == 0)
                continue;
            Eigen::Vector2f location(j,i);
            float depVal = denseDepImg.at<uchar>(i,j);
            pcl::PointXYZ pointTmp;
            if( reprojection(location, depVal, projectMatrix, pointTmp) )
            {
                veloCloudPtr1->points.push_back(pointTmp);
                pcl::PointXYZRGB pointTmp_rgb;
                pointTmp_rgb.x = pointTmp.x;
                pointTmp_rgb.y = pointTmp.y;
                pointTmp_rgb.z = pointTmp.z;
                uint8_t R = rgbImg.at<cv::Vec3b>(i,j)[2];
                uint8_t G = rgbImg.at<cv::Vec3b>(i,j)[1];
                uint8_t B = rgbImg.at<cv::Vec3b>(i,j)[0];
                uint32_t rgbTmp = ((uint32_t)R << 16 | (uint32_t)G << 8 | (uint32_t)B);
                pointTmp_rgb.rgb = *reinterpret_cast<float*>(&rgbTmp);

                veloCloudPtr_rgb->points.push_back(pointTmp_rgb);
            }
        }
    }

    veloCloudPtr1->width = (int) veloCloudPtr1->points.size ();
    veloCloudPtr1->height = 1;
    veloCloudPtr1->width = (int) veloCloudPtr1->points.size ();
    veloCloudPtr1->height = 1;

    // ------------------------------------
    // -----Create example point cloud-----
    // ------------------------------------
    FILE* veloData;
    std::string veloDataDir = "000074.bin";
    pcl::PointCloud<pcl::PointXYZ>::Ptr veloCloudPtr_bin (new pcl::PointCloud<pcl::PointXYZ>);
    if( !readPointXYZ(veloDataDir, veloCloudPtr_bin, 370, 1224, projectionMatrix_arma) )
    {
        fprintf(stderr,"failed to load the velodyne .bin file: %s !\n", veloDataDir.c_str());
        return 0;
    }

    //project the 3D veloDyne data into 2D image plane;
    arma::Cube<float> matrix2D( rgbImg.rows, rgbImg.cols, 3 );
    matrix2D.slice(0).fill(0);
    matrix2D.slice(1).fill(0);
    matrix2D.slice(2).fill(0);
    if( !projection2D23D( matrix2D, projectionMatrix_arma, veloDataDir, cameraMatrix_arma, R0_rect_arma, velo2camera_arma ) )
    {
        fprintf(stderr, "failed to project 3D point cloud into 2D image plane!\n");
        exit(1);
    }
    int paddingSize = 11;

    //std::cout << "begin enter inhomogeneityClean! the matrix2D.slice(2) is " << matrix2D.slice(2).n_rows << "x" << matrix2D.slice(2).n_cols << std::endl;
    inhomogeneityClean( matrix2D.slice(2), paddingSize );

    arma::Cube<float> sparseMat = matrix2D;
    //std::cout << "finished inhomogeneityClean!!!!\n";

    arma::Mat<float> geoEstImg( rgbImg.rows, rgbImg.cols);
    geoEstImg.fill(0.0);
    if( !geoDepEstimation(rgbImg, matrix2D, geoEstImg) )
    {
        fprintf(stderr,"failed to geodesic-estimate depth map!\n");
        exit(0);
    }


    cv::Mat testImg( cv::Size( matrix2D.slice(2).n_cols, matrix2D.slice(2).n_rows ), CV_8UC1, cv::Scalar::all(0) );
    int maxValueImg = 0;
    int minValueImg = 255;
    for( int i = 0; i < testImg.rows; i++ )
    {
        for( int j = 0; j < testImg.cols; j++ )
        {
            testImg.at<uchar>(i,j) = int(geoEstImg(i,j)+0.5);
            if( int(geoEstImg(i,j) + 0.5) > 0 && int( geoEstImg(i,j) + 0.5 ) > maxValueImg )
                maxValueImg = int( geoEstImg(i,j) + 0.5 );
            //std::cout << "the depth value is " << int(geoImgNoPadding(i,j) + 0.5) << std::endl;
        }
    }
    testImg = testImg*(255.0/maxValueImg);
    //cv::imshow("geoEstImg", testImg);
    //cv::waitKey(0);

    arma::Mat<float> denseDepImg1(rgbImg.rows, rgbImg.cols);
    //arma::Mat<float> denseDepImg1;
    denseDepImg1.fill(0.0);
    //denseDepImg1 = depUpsampling( matrix2D, geoEstImg, rgbImg);
    //inhomogeneityClean( matrix2D.slice(2), paddingSize );
    //std::cout << "begin to check rgbImg:\n";
    denseDepImg1 = depUpsamplingRandom( matrix2D, geoEstImg, rgbImg );
    cv::Mat testImg1( cv::Size( denseDepImg1.n_cols, denseDepImg1.n_rows ), CV_8UC1, cv::Scalar::all(0) );
    int maxValueImg1 = 0;
    int minValueImg1 = 255;
    for( int i = 0; i < testImg.rows; i++ )
    {
        for( int j = 0; j < testImg.cols; j++ )
        {
            testImg1.at<uchar>(i,j) = int(denseDepImg1(i,j)+0.5);
            if( int(denseDepImg1(i,j) + 0.5) > 0 && int( denseDepImg1(i,j) + 0.5 ) > maxValueImg1 )
                maxValueImg1 = int( denseDepImg1(i,j) + 0.5 );
            //std::cout << "the depth value is " << int(geoImgNoPadding(i,j) + 0.5) << std::endl;
        }
    }
    testImg1 = testImg1*(255.0/maxValueImg1);
    cv::imshow("upSampleRst", testImg1);
    cv::waitKey(0);

    std::string labelFile = "000074label.txt";
    std::vector<struct objectLabel> objectLabels;
    if( !readObjectLabel( objectLabels, labelFile, R0_rect_eigen, velo2camera_eigen ) )
    {
        fprintf(stderr, "failed to load the labelFile: %s !\n", labelFile.c_str());
        return 0;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr veloCloudPtr_denseRGB(new pcl::PointCloud<pcl::PointXYZRGB>);

    for(int32_t l = 0; l < 1; l++)
    {
        float left = objectLabels[l].bbox.left;
        float right = objectLabels[l].bbox.right;
        float top = objectLabels[l].bbox.top;
        float bottom = objectLabels[l].bbox.bottom;
        for(int32_t i = int(top+0.5); i < int(bottom+0.5); i++)
        {
            for(int32_t j = int(left + 0.5); j < int(right + 0.5); j++)
            {
                if( denseDepImg1(i,j) == 0. )
                    continue;
                //if( sparseMat.slice(2)(i,j) == 0. )
                //  continue;
                Eigen::Vector2f location( matrix2D.slice(1)(i,j), matrix2D.slice(0)(i,j) );
                float depVal = denseDepImg1(i,j);
                // float depVal = sparseMat.slice(2)(i,j);
                pcl::PointXYZ pointTmp;
                if( reprojection(location, depVal, projectMatrix, pointTmp) )
                {
                    veloCloudPtr1->points.push_back(pointTmp);
                    pcl::PointXYZRGB pointTmp_rgb;
                    pointTmp_rgb.x = pointTmp.x;
                    pointTmp_rgb.y = pointTmp.y;
                    pointTmp_rgb.z = pointTmp.z;
                    uint8_t R = rgbImg.at<cv::Vec3b>(i,j)[2];
                    uint8_t G = rgbImg.at<cv::Vec3b>(i,j)[1];
                    uint8_t B = rgbImg.at<cv::Vec3b>(i,j)[0];
                    uint32_t rgbTmp = ((uint32_t)R << 16 | (uint32_t)G << 8 | (uint32_t)B);
                    pointTmp_rgb.rgb = *reinterpret_cast<float*>(&rgbTmp);

                    veloCloudPtr_denseRGB->points.push_back(pointTmp_rgb);
                }
            }
        }
    }


    //std::cout << "The Boundingbox is:\n";
    //std::cout << objectLabels[0].boundingBox3D << std::endl;

    //char ch1; std::cin.get(ch1);
    pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    std::cout << "Genarating example point clouds.\n";
    // We're going to make an ellipse extruded along the z-axis. The colour for
    // the XYZRGB cloud will gradually go from red to green to blue.
    uint8_t r(255), g(15), b(15);
    for (float z(-1.0); z <= 1.0; z += 0.05)
    {
        for (float angle(0.0); angle <= 360.0; angle += 5.0)
        {
            pcl::PointXYZ basic_point;
            basic_point.x = 0.5 * cosf (pcl::deg2rad(angle));
            basic_point.y = sinf (pcl::deg2rad(angle));
            basic_point.z = z;
            basic_cloud_ptr->points.push_back(basic_point);

            pcl::PointXYZRGB point;
            point.x = basic_point.x;
            point.y = basic_point.y;
            point.z = basic_point.z;
            uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                            static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
            point.rgb = *reinterpret_cast<float*>(&rgb);
            point_cloud_ptr->points.push_back (point);
        }
        if (z < 0.0)
        {
            r -= 12;
            g += 12;
        }
        else
        {
            g -= 12;
            b += 12;
        }
    }
    basic_cloud_ptr->width = (int) basic_cloud_ptr->points.size ();
    basic_cloud_ptr->height = 1;
    point_cloud_ptr->width = (int) point_cloud_ptr->points.size ();
    point_cloud_ptr->height = 1;

    // ----------------------------------------------------------------
    // -----Calculate surface normals with a search radius of 0.05-----
    // ----------------------------------------------------------------
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setInputCloud (veloCloudPtr_denseRGB);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
    ne.setSearchMethod (tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1 (new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch (1.0);
    //ne.compute (*cloud_normals1);

    // ---------------------------------------------------------------
    // -----Calculate surface normals with a search radius of 0.1-----
    // ---------------------------------------------------------------
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch (0.1);
    ne.compute (*cloud_normals2);
    

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    if (simple)
    {
        viewer = simpleVis( veloCloudPtr_bin, objectLabels, intrinsicMat, extrinsicMat );
    }
    else if (rgb)
    {
        viewer = rgbVis(veloCloudPtr_denseRGB, objectLabels);
    }
    else if (custom_c)
    {
        viewer = customColourVis(basic_cloud_ptr);
    }
    else if (normals)
    {
        viewer = normalsVis( veloCloudPtr_denseRGB, cloud_normals1 );
    }
    else if (shapes)
    {
        viewer = shapesVis(point_cloud_ptr);
    }
    else if (viewports)
    {
        viewer = viewportsVis(point_cloud_ptr, cloud_normals1, cloud_normals1);
    }
    else if (interaction_customization)
    {
        viewer = interactionCustomizationVis();
    }

    //--------------------
    // -----Main loop-----
    //--------------------
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    */
      
}
