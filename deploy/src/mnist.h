#pragma once

#include "inference_engine.h"

namespace dt {

class MnistNet : public InferenceEngine {
 public:
  MnistNet(const std::string& name) : InferenceEngine(name) {
    RegisterPlugin("PostProcess", std::bind(&MnistNet::PostProcess, this));
  }
  
 private:
  void PostProcess();

};

}  // namespace dt

