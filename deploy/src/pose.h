#pragma once

#include "inference_engine.h"

namespace dt {

struct KeyPoint {
  cv::Point2f p;
  float prob;
};

class PoseNet : public InferenceEngine {
 public:
  PoseNet(const std::string& name) : InferenceEngine(name) {
    RegisterPlugin("PostProcess", std::bind(&PoseNet::PostProcess, this));
  }
  
  void Draw(cv::Mat* input_image);

 private:
  void PostProcess();

  std::vector<KeyPoint> keypoints_;

};

}  // namespace dt

