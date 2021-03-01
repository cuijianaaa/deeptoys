#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "pose.h"

using namespace dt;

int main(int argc, char* argv[]) {
  const char* image_path = argv[1];
  cv::Mat image = cv::imread(image_path, 1);
  if (image.empty()) {
    return -1;
  }
  cv::Mat input_image;
  cv::resize(image, input_image, cv::Size(192, 256));

  dt::PoseNet pose_net("pose");
  std::cout << "1" << std::endl;
  pose_net.GiveInput(input_image);
  std::cout << "2" << std::endl;
  pose_net.RunInference();
  std::cout << "3" << std::endl;
  pose_net.Draw(&input_image);
  std::cout << "4" << std::endl;
  cv::imwrite("images/pose_result.jpg", input_image);
  cv::imshow("image", input_image);
  cv::waitKey(0);

  return 0;
}