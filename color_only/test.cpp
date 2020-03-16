#include "./include/color_only/color_only.h"
typedef pcl::PointXYZRGB PointT;
std::vector<float> PreprocessModel(const pcl::PolygonMesh::Ptr &mesh_in) {
  pcl::PointCloud<PointT>::Ptr cloud_in (new
                                         pcl::PointCloud<PointT>);

  pcl::fromPCLPointCloud2(mesh_in->cloud, *cloud_in);
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud_in, centroid);
  double x_translation = centroid[0];
  double y_translation = centroid[1];

  PointT min_pt, max_pt;
  pcl::getMinMax3D(*cloud_in, min_pt, max_pt);
  double z_translation = min_pt.z;
  std::cout << "Preprocessing Model, z : " << z_translation << std::endl;

  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  Eigen::Vector3f translation;
  translation << -x_translation, -y_translation, -z_translation;
  transform.translation() = translation;
  std::cout << "Preprocess done" << std::endl;
  std::cout << "Preprocessing transform : " << transform.matrix() << std::endl;
  std::vector<float> result;
  result.push_back(-x_translation);
  result.push_back(-y_translation);
  result.push_back(-z_translation);
  return result;
}





color_only::~color_only()
{

}
color_only::color_only(){
    sub = n.subscribe("color_only_image", 1, &color_only::imageCallback,this);
    cuda_renderer::Model model(prefix+"textured.ply");
    pcl::PolygonMesh mesh;
    pcl::io::loadPolygonFilePLY (prefix+"textured.ply", mesh);
    pcl::PolygonMesh::Ptr mesh_in(new pcl::PolygonMesh(mesh));
    mode_trans = PreprocessModel(mesh_in);
    std::cout<<mode_trans[0]<<","<<mode_trans[1]<<","<<mode_trans[2];
    models.push_back(model);
    // model = aaa;
    width = 960;
    height = 540;
    float kCameraFX=768.1605834960938;
    float kCameraFY=768.1605834960938;
    float kCameraCX=480;
    float kCameraCY=270;
    cam_intrinsic=(cv::Mat_<float>(3,3) << kCameraFX, 0.0, kCameraCX, 0.0, kCameraFY, kCameraCY, 0.0, 0.0, 1.0);
    cam_intrinsic_eigen << kCameraFX, 0.0, kCameraCX,0.0, 0.0, kCameraFY, kCameraCY,0.0, 0.0, 0.0, 1.0,0.0,0.0,0.0,0.0,0.0;
    proj_mat = cuda_renderer::compute_proj(cam_intrinsic, width, height);

    cam_to_world_.matrix() << 7.65560111e-01, -8.88656873e-04,  6.43363760e-01, -9.00264040e-01,
                              6.80320522e-04,  9.99999605e-01,  5.71729852e-04 , 2.28142375e-03,
                             -6.43364014e-01, -1.72637553e-16,  7.65560413e-01,  7.37090700e-01,
                              0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00;
    Eigen::Isometry3d cam_z_front;
    Eigen::Isometry3d cam_to_body;
    cam_to_body.matrix() << 0, 0, 1, 0,
                    -1, 0, 0, 0,
                    0, -1, 0, 0,
                    0, 0, 0, 1;
    cam_z_front = cam_to_world_ * cam_to_body;
    cam_matrix =cam_z_front.matrix().inverse();
    background_image = cv::imread("/home/jessy/cv/background_OBJ4_2.png", cv::IMREAD_COLOR);
    x_min = -0.44;
    x_max = 0.5;
    y_min = -0.7;
    y_max = 0.7;
    // x_min = 0.17;
    // x_max = 0.19;
    // y_min = -0.26;
    // y_max = -0.24;
    res = 0.01;
    theta_res =0.1;
    table_height =  1.1634223;

    std::ifstream file("/home/jessy/testb.txt");
    std::string   line;
    while(std::getline(file, line))
    {
        std::stringstream   linestream(line);
        std::string         data;
        float x,y,theta;
        float h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13;
        std::getline(linestream, data, '\t');
        linestream >> x >> y>> theta >> h1>> h2 >> h3>> h4 >> h5>> h6 >> h7>> h8>> h9 >> h10>> h11 >> h12>> h13;
        hist_total.push_back(h1); hist_total.push_back(h2); hist_total.push_back(h3); hist_total.push_back(h4); hist_total.push_back(h5); hist_total.push_back(h6);hist_total.push_back(h7);  
        hist_total.push_back(h8); hist_total.push_back(h9); hist_total.push_back(h10); hist_total.push_back(h11); hist_total.push_back(h12); hist_total.push_back(h13);      
    }
    
    
    gpu_bb.push_back(models[0].bbox_min.x);gpu_bb.push_back(models[0].bbox_min.y);gpu_bb.push_back(models[0].bbox_min.z);
    gpu_bb.push_back(models[0].bbox_max.x);gpu_bb.push_back(models[0].bbox_min.y);gpu_bb.push_back(models[0].bbox_min.z);
    gpu_bb.push_back(models[0].bbox_min.x);gpu_bb.push_back(models[0].bbox_max.y);gpu_bb.push_back(models[0].bbox_min.z);
    gpu_bb.push_back(models[0].bbox_max.x);gpu_bb.push_back(models[0].bbox_max.y);gpu_bb.push_back(models[0].bbox_min.z);
    gpu_bb.push_back(models[0].bbox_min.x);gpu_bb.push_back(models[0].bbox_min.y);gpu_bb.push_back(models[0].bbox_max.z);
    gpu_bb.push_back(models[0].bbox_max.x);gpu_bb.push_back(models[0].bbox_min.y);gpu_bb.push_back(models[0].bbox_max.z);
    gpu_bb.push_back(models[0].bbox_min.x);gpu_bb.push_back(models[0].bbox_max.y);gpu_bb.push_back(models[0].bbox_max.z);
    gpu_bb.push_back(models[0].bbox_max.x);gpu_bb.push_back(models[0].bbox_max.y);gpu_bb.push_back(models[0].bbox_max.z);
    std::vector<float> cam_r1;
    std::vector<float> cam_r2;
    std::vector<float> cam_r3;
    Eigen::Matrix4d gpu_cam = cam_intrinsic_eigen*cam_matrix;
    cam_r1.push_back(gpu_cam(0,0));cam_r1.push_back(gpu_cam(0,1));cam_r1.push_back(gpu_cam(0,2));cam_r1.push_back(gpu_cam(0,3));
    cam_r2.push_back(gpu_cam(1,0));cam_r2.push_back(gpu_cam(1,1));cam_r2.push_back(gpu_cam(1,2));cam_r2.push_back(gpu_cam(1,3));
    cam_r3.push_back(gpu_cam(2,0));cam_r3.push_back(gpu_cam(2,1));cam_r3.push_back(gpu_cam(2,2));cam_r3.push_back(gpu_cam(2,3));
    
    gpu_cam_m.push_back(cam_r1);
    gpu_cam_m.push_back(cam_r2);
    gpu_cam_m.push_back(cam_r3);
    std::cout<<"SetInput Finished!!!"<<std::endl;
}


void color_only::imageCallback(const sensor_msgs::ImagePtr& msg)
{
  try
  {
    // std::vector<cuda_renderer::Model::mat4x4> result = test.Predict(image);
    // cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
    cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
    // cv::waitKey(30);
    auto start_l = std::chrono::steady_clock::now();
    // background subtraction
    origin_image.release();
    cv_input_color_image.release();
    trans_mat.clear();
    estimate_score.clear();
    Pose_list.clear();
    origin_image = image;
    cv::Mat mask;
    cv::absdiff(background_image,origin_image,mask);
    cv::cvtColor(mask,mask,CV_BGR2GRAY);
    cv::threshold(mask,mask,15,255,CV_THRESH_BINARY);
    cv::bitwise_or(origin_image, origin_image,cv_input_color_image, mask=mask);
    // find colored bounding box
    cv::Mat gray_input;
    cv::cvtColor(cv_input_color_image,gray_input,CV_BGR2GRAY);
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( gray_input, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );
    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    std::vector<int> boundRect;
    int bound_count = 0;
    for( size_t i = 0; i < contours.size(); i++ )
    {
        // might need to denoise !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        cv::approxPolyDP( contours[i], contours_poly[i], 3, true );
        cv::Rect cur_rect = boundingRect( contours_poly[i] );
        if(cur_rect.width>10 && cur_rect.height>10){
            boundRect.push_back(cur_rect.x);
            boundRect.push_back(cur_rect.y);
            boundRect.push_back(cur_rect.width);
            boundRect.push_back(cur_rect.height);
            bound_count++;
            // std::cout<<cur_rect.x<<","<<cur_rect.y<<","<<cur_rect.width<<","<<cur_rect.height<<std::endl;
            // cv::rectangle(cv_input_color_image, cv::Point(cur_rect.x,cur_rect.y),cv::Point((cur_rect.x+cur_rect.width),(cur_rect.y+cur_rect.height) ),
          // cv::Scalar(0,0,255));
        }        
    }
    std::cout<<"number of color regions"<<bound_count<<std::endl;
    boundRect.insert(boundRect.begin(),bound_count);

    // cv::imshow("aaa",cv_input_color_image);
    // cv::waitKey(0);
    std::vector<uint8_t> h_v;
    std::vector<uint8_t> s_v;
    std::vector<uint8_t> v_v;
    std::vector<uint8_t> r_v;
    std::vector<uint8_t> g_v;
    std::vector<uint8_t> b_v;

    cv::Mat hsv_input;
    int non_zero =0;
    cv::cvtColor(cv_input_color_image,hsv_input,CV_BGR2HSV);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            cv::Vec3b elem = hsv_input.at<cv::Vec3b>(y, x);
            cv::Vec3b elem_rgb = cv_input_color_image.at<cv::Vec3b>(y, x);
            r_v.push_back(elem_rgb[2]);
            g_v.push_back(elem_rgb[1]);
            b_v.push_back(elem_rgb[0]);
            int h = elem[0];
            int s = elem[1];
            int v = elem[2];
            h_v.push_back(h);
            s_v.push_back(s);
            v_v.push_back(v);
            if(h!=0 || s!=0 || v!=0){
              non_zero +=1;
            }
            
        }
    }

    std::vector<std::vector<uint8_t>> observed_hsv;
    observed_hsv.push_back(h_v);
    observed_hsv.push_back(s_v);
    observed_hsv.push_back(v_v);
    std::vector<std::vector<uint8_t>> observed_rgb;
    observed_rgb.push_back(r_v);
    observed_rgb.push_back(g_v);
    observed_rgb.push_back(b_v);
   
    std::cout<<"total pixel number"<< non_zero<<std::endl;
    //   start = chrono::steady_clock::now();
    hist_prioritize::s_pose his_result =  hist_prioritize::compare_hist(width,height,
                                                                    x_min,x_max,y_min,y_max,
                                                                    0.0,2 * M_PI,
                                                                    res,theta_res,non_zero,
                                                                    observed_hsv,gpu_cam_m,gpu_bb,hist_total,boundRect);
    

    estimate_score = his_result.score;

    for(int i =0; i < his_result.ps.size(); i ++){
        Pose cur = Pose(his_result.ps[i].x, his_result.ps[i].y, mode_trans[2], 0.0, 0.0, his_result.ps[i].theta);
        Pose_list.push_back(cur);
        Eigen::Matrix4d transform;
        transform = cur.Pose::GetTransform().matrix().cast<double>();
        Eigen::Matrix4d pose_in_cam = cam_matrix*transform;
        // std::cout<<cur.x_<<","<<cur.y_<<","<<cur.yaw_<<std::endl;
        // if(i==0){
        //     std::cout<<transform<<std::endl;
        //     std::cout<<pose_in_cam<<std::endl;
        // }
        cuda_renderer::Model::mat4x4 mat4;
      //multiply 100 to change scale, data scale is in meter can be write as a class function
        mat4.a0 = pose_in_cam(0,0)*100;
        mat4.a1 = pose_in_cam(0,1)*100;
        mat4.a2 = pose_in_cam(0,2)*100;
        mat4.a3 = pose_in_cam(0,3)*100;
        mat4.b0 = pose_in_cam(1,0)*100;
        mat4.b1 = pose_in_cam(1,1)*100;
        mat4.b2 = pose_in_cam(1,2)*100;
        mat4.b3 = pose_in_cam(1,3)*100;
        mat4.c0 = pose_in_cam(2,0)*100;
        mat4.c1 = pose_in_cam(2,1)*100;
        mat4.c2 = pose_in_cam(2,2)*100;
        mat4.c3 = pose_in_cam(2,3)*100;
        mat4.d0 = pose_in_cam(3,0);
        mat4.d1 = pose_in_cam(3,1);
        mat4.d2 = pose_in_cam(3,2);
        mat4.d3 = pose_in_cam(3,3);
        trans_mat.push_back(mat4);
        
    }
    int render_size = 200;
    int total_render_num = Pose_list.size();
    int num_render = (total_render_num-1)/render_size+1;
    int min_score = 99999999999;
    Pose best_pose(0,0,0,0,0,0);
    std::vector<cuda_renderer::Model::mat4x4> best_pose_in_cam(1);
    std::vector<int> total_result_cost;
    int count_p = 0;
    std::vector<cuda_renderer::Model::mat4x4> cur_transform;
    // std::ofstream myfile;
    // myfile.open ("/home/jessy/testb.txt",std::ios_base::app);
    for(int i =0; i <num_render; i ++){

        auto last = std::min(total_render_num, i*render_size + render_size);
        // std::vector<cuda_renderer::Model::mat4x4>::const_iterator start = trans_mat.begin() + i*render_size;
        // std::vector<cuda_renderer::Model::mat4x4>::const_iterator finish = trans_mat.begin() + last;
        // std::vector<cuda_renderer::Model::mat4x4> cur_transform(start,finish);

        std::vector<Pose>::const_iterator start = Pose_list.begin() + i*render_size;
        std::vector<Pose>::const_iterator finish = Pose_list.begin() + last;
        std::vector<Pose> cur_Pose_list(start,finish);

        for(int n = 0; n <cur_Pose_list.size();n++){
            Pose cur = cur_Pose_list[n];
            Eigen::Matrix4d transform;
            transform = cur.Pose::GetTransform().matrix().cast<double>();
            Eigen::Matrix4d pose_in_cam = cam_matrix*transform;
            cuda_renderer::Model::mat4x4 mat4;
          //multiply 100 to change scale, data scale is in meter can be write as a class function
            mat4.a0 = pose_in_cam(0,0)*100;
            mat4.a1 = pose_in_cam(0,1)*100;
            mat4.a2 = pose_in_cam(0,2)*100;
            mat4.a3 = pose_in_cam(0,3)*100;
            mat4.b0 = pose_in_cam(1,0)*100;
            mat4.b1 = pose_in_cam(1,1)*100;
            mat4.b2 = pose_in_cam(1,2)*100;
            mat4.b3 = pose_in_cam(1,3)*100;
            mat4.c0 = pose_in_cam(2,0)*100;
            mat4.c1 = pose_in_cam(2,1)*100;
            mat4.c2 = pose_in_cam(2,2)*100;
            mat4.c3 = pose_in_cam(2,3)*100;
            mat4.d0 = pose_in_cam(3,0);
            mat4.d1 = pose_in_cam(3,1);
            mat4.d2 = pose_in_cam(3,2);
            mat4.d3 = pose_in_cam(3,3);
            cur_transform.push_back(mat4);
        }
        // std::vector<std::vector<uint8_t>> result_gpu = cuda_renderer::render_cuda(models[0].tris,cur_transform,
        //                                                                             width, height,proj_mat);
        // std::vector<uint8_t> r_v;
        // std::vector<uint8_t> g_v;
        // std::vector<uint8_t> b_v;
        // for (int y = 0; y < env_params_.height; y++) {
        //   for (int x = 0; x < env_params_.width; x++) {
        //       cv::Vec3b elem = cv_input_color_image.at<cv::Vec3b>(y, x);
        //       r_v.push_back(elem[2]);
        //       g_v.push_back(elem[1]);
        //       b_v.push_back(elem[0]);
              
        //   }
        // }
        // std::vector<std::vector<uint8_t>> observed;
        // observed.push_back(r_v);
        // observed.push_back(g_v);
        // observed.push_back(b_v);
    // render histogram (used in offline)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // int height = 540;
        // int width = 960;
        // cv::Mat cur_mat = cv::Mat(height,width,CV_8UC3);
        // // // // // cv::imshow("gpu_mask1", cur_mat); 
        // // // // // cv::waitKey(0); 
        
        // for(int n = 0; n <cur_transform.size(); n ++){
        //     std::cout<<contposes[count_p].x()<<","<<contposes[count_p].y()<<","<<contposes[count_p].yaw()<<std::endl;
        //     for(int i = 0; i < height; i ++){
        //         for(int j = 0; j <width; j ++){
        //             int index = n*width*height+(i*width+j);
        //             int red = result_gpu[0][index];
        //             int green = result_gpu[1][index];
        //             int blue = result_gpu[2][index];
        //             cur_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green,red);
        //             // if(red == 0&& green ==0&&blue==0){
        //             //   cur_mat.at<cv::Vec3b>(i, j) = cv_input_color_image.at<cv::Vec3b>(i, j);
        //             // }else{
        //             //   cur_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green,red);
        //             // }
                    
        //             // std::cout<<red<<","<<green<<","<<blue<<std::endl;
                    
        //         }
        //     }

        //     cv::imshow("aaa",cur_mat);
        //     cv::waitKey(0);
        //     count_p = count_p+1;
        // }
        //     float min_x=10000;
        //     float max_x=-10000;
        //     float min_y=10000;
        //     float max_y=-10000;
        //     for(int m=0; m <8;m++){
        //         float x = bounding_boxes[count_p][m][0];
        //         float y = bounding_boxes[count_p][m][1];
        //         if(x<min_x) min_x = x;
        //         if(x>max_x) max_x = x;
        //         if(y<min_y) min_y = y;
        //         if(y>max_y) max_y = y;
        //         // cur_mat.at<Vec3b>(y, x) = Vec3b(0, 255,0);
        //     }
        //     // std::cout<<min_x<<","<<max_x<<","<<min_y<<","<<max_y<<std::endl;
        //     cv::Mat ob_mat = cv::Mat::zeros(height,width,CV_8UC3);
        //     int a = 0;
        //     std::vector<int> his(13);
        //     if(min_x>=0 && min_x<width&&max_x>=0 && max_x<width&&
        //         min_y>=0 && min_y<height&&max_y>=0 && max_y<height){
        //         // for(int cur_x = min_x;cur_x<=max_x;cur_x++){
        //         //     for(int cur_y=min_y;cur_y<=max_y;cur_y++){
        //         //       int cur_ind = cur_y*960+cur_x;
        //         //       ob_mat.at<cv::Vec3b>(cur_y, cur_x) = cv::Vec3b(b_v[cur_ind], g_v[cur_ind],r_v[cur_ind]);
        //         //       a+=b_v[cur_ind];
        //         //       a+=g_v[cur_ind];
        //         //       a+=r_v[cur_ind];
        //         //       // ob_mat.at<cv::Vec3b>(cur_y, cur_x) = cv::Vec3b(255, 0,0);

        //         //     }
        //         // }
        //         cv::Rect roi1(  min_x,min_y, max_x-min_x+1, max_y-min_y+1 );
        //         cv::Mat sub1( cur_mat, roi1 );
        //         cv::Mat sub_hsv;
        //         cv::cvtColor(sub1, sub_hsv, CV_BGR2HSV);
        //         // cv::Mat sub2( ob_mat, roi1 );
        //         // cv::Mat observed_hsv;
        //         // cv::cvtColor(cv_input_color_image, observed_hsv, CV_BGR2HSV);
        //         // double s = cv::sum(sub1)[0]+ cv::sum(sub1)[1]+ cv::sum(sub1)[2];
        //         // double s1 = cv::sum(sub2)[0]+ cv::sum(sub2)[1]+ cv::sum(sub2)[2];
        //         // std::cout<<test[0][n]<<std::endl;
              
                
        //         std::vector<cv::Mat> bgr_planes;
        //         cv::split( sub_hsv, bgr_planes );
               
        //         // std::vector<int> his(7);
        //         cv::Size s = sub_hsv.size();
        //         for(int i =0; i <s.height; i ++){
        //           for(int j = 0; j <s.width; j ++){
        //             cv::Vec3b hsv=sub_hsv.at<cv::Vec3b>(i,j);
        //             if(hsv.val[1]!=0||hsv.val[2]!=0){
        //               int h_value = hsv.val[0];
        //               int index = h_value/15;
        //               his[index] +=1;
        //               // sub_hsv.at<cv::Vec3b>(i, j) = cv::Vec3b(index*20, 100,100);
        //             }else{
        //               his[12]+=1;
        //             }
        //           }
        //         }
        //         // cv::imshow("gpu_mask1", sub1); 
        //         // cv::imshow("aaa",sub_hsv);
        //         // cv::waitKey(0); 
        //     }
        //     // else{
        //     //   his[0]=0; his[1]=0; his[2]=0; his[3]=0; his[4]=0; his[5]=0; his[6]=0;
        //     // }
            
            
        //     // std::cout<<his[0]<<","<<his[1]<<","<<his[2]<<","<<his[3]<<","<<his[4]<<","<<his[5]<<std::endl;
        //     // std::cout<<his[0]+his[1]+his[2]+his[3]+his[4]+his[5]<<std::endl;
        //     // !!!!Most recent file save!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //     myfile<<"\t"<<contposes[count_p].x()<<"\t"<<contposes[count_p].y()<<"\t"<<contposes[count_p].yaw()<<"\t"
        //     <<his[0]<<"\t"<<his[1]<<"\t"<<his[2]<<"\t"<<his[3]<<"\t"<<his[4]<<"\t"<<his[5]<<"\t"<<his[6]<<"\t"
        //     <<his[7]<<"\t"<<his[8]<<"\t"<<his[9]<<"\t"<<his[10]<<"\t"<<his[11]<<"\t"<<his[12]<<std::endl;
           
        //     // cv::imshow("aaa",sub_hsv);
        //     // cv::imshow("aaab",observed_hsv);
        //     // cv::waitKey(0); 
        //     count_p = count_p+1;
        // }
        // std::cout<<count_p<<",";
       //commented when rendering
        // std::vector<int> result_cost = cuda_renderer::compute_cost(result_gpu,observed_rgb,height,width,cur_transform.size());

        std::vector<int> result_cost = cuda_renderer::render_cuda_cal_cost(models[0].tris,cur_transform,
                                                      width, height, proj_mat,observed_rgb);
    
        int min_in_batch = *std::min_element(result_cost.begin(),result_cost.end());
        if (min_in_batch< min_score){
            min_score = min_in_batch;
            int min_ind = std::min_element(result_cost.begin(),result_cost.end()) - result_cost.begin();
            best_pose = cur_Pose_list[min_ind];
            best_pose_in_cam[0] = cur_transform[min_ind];
        }
 
        int total_last = i*render_size+result_cost.size()-1;
        std::cout<<min_score<<"!!!!!!!!!!"<<estimate_score[total_last]<<std::endl;
    
        if(estimate_score[total_last]>min_score){
            // std::cout<<"aaaaaaaaaaaaa";
            break;
        }
        cur_transform.clear();
    }
    std::cout<<best_pose.x_<<","<<best_pose.y_<<","<<best_pose.yaw_<<std::endl;
    std::vector<std::vector<uint8_t>> final_result = cuda_renderer::render_cuda(models[0].tris,best_pose_in_cam,
                                                                                    width,height,proj_mat);
    cv::Mat cur_mat = cv::Mat(height,width,CV_8UC3);
    for(int i = 0; i < height; i ++){
        for(int j = 0; j <width; j ++){
            int index = 0*width*height+(i*width+j);
            int red = final_result[0][index];
            int green = final_result[1][index];
            int blue = final_result[2][index];
            // cur_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green,red);
            if(red == 0&& green ==0&&blue==0){
              cur_mat.at<cv::Vec3b>(i, j) = cv_input_color_image.at<cv::Vec3b>(i, j);
            }else{
              cur_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green,red);
            }
            
            // std::cout<<red<<","<<green<<","<<blue<<std::endl;
            
        }
    }
    
    imwrite("test"+msg->header.frame_id+".jpg", cur_mat);

    std::cout<<best_pose_in_cam[0].a3<<","<<best_pose_in_cam[0].b3<<","<<best_pose_in_cam[0].c3;
    auto end_l = std::chrono::steady_clock::now();
    std::cout<< "Color only computing time!!!: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_l - start_l).count() << std::endl;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "listener");
    color_only test;
    ros::spin();
    
    return 0;
}
