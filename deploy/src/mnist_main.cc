#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "glog/logging.h"

#include "mnist.h"

using namespace dt;

int main(int argc, char* argv[]) {
  const char* image_path = argv[1];
  cv::Mat image = cv::imread(image_path, -1);
  if (image.empty()) {
    return -1;
  }
  dt::MnistNet mnist_net("mnist");
  mnist_net.GiveInput(image);
  mnist_net.RunInference();

  return 0;
}