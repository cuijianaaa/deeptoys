#include "src/pose.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

namespace dt {

void PoseNet::PostProcess() {
  keypoints_.clear();
  const ncnn::Mat& out = GetOutput("conv3_fwd");
  for (int p = 0; p < out.c; ++p) {
    const ncnn::Mat m = out.channel(p);
    float max_prob = 0.f;
    int max_x = 0;
    int max_y = 0;
    for (int y = 0; y < out.h; ++y) {
      const float* ptr = m.row(y);
      for (int x = 0; x < out.w; ++x) {
        float prob = ptr[x];
        if (prob > max_prob) {
          max_prob = prob;
          max_x = x;
          max_y = y;
        }
      }
    }
    KeyPoint keypoint;
    keypoint.p = cv::Point2f(max_x * GetInput("data").w / static_cast<float>(out.w),
                             max_y * GetInput("data").h / static_cast<float>(out.h));
    keypoint.prob = max_prob;
    keypoints_.push_back(keypoint);
  }
}

void PoseNet::Draw(cv::Mat* image) {
  // Draw bone
  const int joint_pairs[16][2] = {
    {0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
    {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}
  };

  for (int i = 0; i < 16; ++i) {
    const KeyPoint& p1 = keypoints_[joint_pairs[i][0]];
    const KeyPoint& p2 = keypoints_[joint_pairs[i][1]];
    if (p1.prob < 0.2f || p2.prob < 0.2f) {
      continue;
    }
    cv::line(*image, p1.p, p2.p, cv::Scalar(255, 0, 0), 2);
  }

  // Draw joint
  for (int i = 0; i < keypoints_.size(); ++i) {
    const KeyPoint& keypoint = keypoints_[i];

    fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y, keypoint.prob);

    if (keypoint.prob < 0.2f) {
      continue;
    }
    cv::circle(*image, keypoint.p, 3, cv::Scalar(0, 255, 0), -1);
  }
}

}  // namespace dt