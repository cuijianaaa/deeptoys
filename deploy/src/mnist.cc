#include "src/mnist.h"

#include <vector>

#include "glog/logging.h"

namespace dt {

void MnistNet::PostProcess() {
  const ncnn::Mat& out = GetOutput("22");
  LOG(INFO) << out.c << " " << out.h << " " << out.w;
  float max_prob = ((float*)out.data)[0];
  float max_x = 0;
  for (int c = 0; c < out.c; ++c) {
    const ncnn::Mat m = out.channel(c);
    for (int y = 0; y < out.h; ++y) {
      const float* ptr = m.row(y);
      for (int x = 0; x < out.w; ++x) {
        const float prob = ptr[x];
        LOG(INFO) << prob;
        if (prob > max_prob) {
          max_prob = prob;
          max_x = x;
        }
      }
    }
  }
  LOG(INFO) << "The classification number is: " << max_x;
}

}  // namespace dt