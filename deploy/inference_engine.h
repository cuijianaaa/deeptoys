#pragma once

#include <functional>
#include <map>
#include <opencv2/core/core.hpp>
#include <string>

#include "ncnn/net.h"

#include "common/defines.h"


namespace dt {

class InferenceEngine {
 public:
  InferenceEngine(const std::string& name);
  virtual ~InferenceEngine() = default;

  struct Processor {
    Processor(const std::string& type_in, const std::string& name_in)
        : type(type_in), name(name_in) {}
    std::string type;
    std::string name;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<int>> input_shapes;
    std::vector<std::vector<float>> input_means;
    std::vector<std::vector<float>> input_norms;
  };

  void RegisterPlugin(const std::string& name,
                      const std::function<void()>& func);
  void RunInference();
  void GiveInput(const cv::Mat& input, const int input_id = 0);
  void GiveInputs(const std::vector<cv::Mat>& inputs);
  const ncnn::Mat& GetInput(const std::string& input_name) const;
  const ncnn::Mat& GetOutput(const std::string& output_name) const; 

 private:
  void ParseProcessorsFromJson(const std::string& json_path);
  void CreateOperator(const std::vector<std::string>& input_names,
                      const std::vector<std::vector<int>> input_shapes,
                      const std::vector<std::string>& output_names,
                      const std::vector<std::vector<float>> input_means,
                      const std::vector<std::vector<float>> input_norms,
                      Processor* processor);
  void RunOperator(const Processor& processor);
  void RunPlugin(const Processor& processor);

  ncnn::Net net_;
  std::map<std::string, ncnn::Mat> inputs_;
  std::map<std::string, ncnn::Mat> outputs_;
  std::vector<Processor> processors_;
  std::map<std::string, std::function<void()>> plugins_;

  DISALLOW_COPY_MOVE_AND_ASSIGN(InferenceEngine);
};

}  // namespace dt