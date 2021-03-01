#include "inference_engine.h"

#include <string>

#include "glog/logging.h"
#include "json/json.h"

#include "common/io.h"

namespace dt {

InferenceEngine::InferenceEngine(const std::string& name) {
  const std::string model_path = std::string("models/" + name + "/" + name);
  LOG(INFO) << "Loading: " << model_path;
  LOG(INFO) << "Load param...";
  net_.load_param((model_path + ".param").c_str());
  LOG(INFO) << "Load model... ";
  net_.load_model((model_path + ".bin").c_str());
  LOG(INFO) << "Parse processors...";
  ParseProcessorsFromJson(model_path + ".json");
}

void InferenceEngine::RegisterPlugin(const std::string& name,
                                     const std::function<void()>& func) {
  LOG(INFO) << "Register plugin " << name;
  plugins_[name] = func;
}

void InferenceEngine::GiveInput(const cv::Mat& input, const int input_id) {
  CHECK(!processors_.empty());
  CHECK_GE(input_id, 0);
  CHECK_LT(input_id, processors_.front().input_names.size());
  const std::string& input_name = processors_.front().input_names[input_id];
  CHECK(inputs_.find(input_name) != inputs_.end());
  CHECK(input.channels() == 1 || input.channels() == 3) <<
      "GiveInput by cv::Mat only support channel 1 or 3, but got: " << input.channels();
  if (input.channels() == 1) {
    // For Grey image
    inputs_[input_name] = ncnn::Mat::from_pixels(
        input.data, ncnn::Mat::PIXEL_GRAY, input.cols, input.rows);
  }
  if (input.channels() == 3) {
    // For RGB image, The default cv::Mat is BGR, the default ncnn::Mat is RGB
    inputs_[input_name] = ncnn::Mat::from_pixels(
        input.data, ncnn::Mat::PIXEL_BGR2RGB, input.cols, input.rows);
  }
  if (!processors_.front().input_means[input_id].empty() &&
      !processors_.front().input_norms[input_id].empty()) {
    inputs_[input_name].substract_mean_normalize(processors_.front().input_means[input_id].data(),
                                                 processors_.front().input_norms[input_id].data());
  }
}

void InferenceEngine::GiveInputs(const std::vector<cv::Mat>& inputs) {
  CHECK(!processors_.empty());
  CHECK_EQ(inputs.size(), processors_.front().input_names.size());
  for (int i = 0; i < inputs.size(); ++i) {
    GiveInput(inputs[i], i);
  }
}

void InferenceEngine::RunInference() {
  for (const Processor& processor : processors_) {
    if (processor.type == "operator") {
      RunOperator(processor);
    } else if (processor.type == "plugin") {
      RunPlugin(processor);
    }
  }
}

void InferenceEngine::RunOperator(const Processor& processor) {
  ncnn::Extractor ex = net_.create_extractor();
  for (const std::string& input_name : processor.input_names) {
    ex.input(input_name.c_str(), inputs_[input_name]);
  }
  for (const std::string& output_name : processor.output_names) {
    ex.extract(output_name.c_str(), outputs_[output_name]);
  }
}

void InferenceEngine::RunPlugin(const Processor& processor) {
  CHECK(plugins_.find(processor.name) != plugins_.end())
      << "Plugin " << processor.name << " not registered";
  const auto& plugin = plugins_[processor.name];
  plugin();
}

void InferenceEngine::CreateOperator(const std::vector<std::string>& input_names,
                                     const std::vector<std::vector<int>> input_shapes,
                                     const std::vector<std::string>& output_names,
                                     const std::vector<std::vector<float>> input_means,
                                     const std::vector<std::vector<float>> input_norms,
                                     Processor* processor) {
  CHECK(input_names.size() == input_shapes.size());
  CHECK(input_names.size() > 0);
  CHECK(input_shapes[0].size() == 3);
  processor->input_names = input_names;
  processor->output_names = output_names;
  processor->input_means = input_means;
  processor->input_norms = input_norms;
  for (int i = 0; i < input_names.size(); ++i) {
    const std::vector<int>& input_shape = input_shapes[i];
    const ncnn::Mat mat(input_shape[1], input_shape[0], input_shape[2]);
    inputs_[input_names[i]] = mat;
  }
  for (const auto& output_name : output_names) {
    const ncnn::Mat mat;
    outputs_[output_name] = mat;
  }
}

void InferenceEngine::ParseProcessorsFromJson(const std::string& json_path) {
  Json::Value json_root;
  CHECK(io::ReadJsonFileToJsonValue(json_path, &json_root));
  for (const auto& json_value : json_root) {
    const Json::Value type_value = json_value["type"];
    CHECK(!type_value.isNull()) << "Json processor define must have type!";
    const std::string processor_type = type_value.asString();
    const Json::Value name_value = json_value["name"];
    CHECK(!name_value.isNull()) << "Json processor define must have name!";
    const std::string processor_name = name_value.asString();
    CHECK(processor_type == "operator" || processor_type == "plugin")
        << "Json processor define only support 'operator' or 'plugin' type!";
    LOG(INFO) << processor_type << ": " << processor_name;
    processors_.emplace_back(processor_type, processor_name);
    if (processor_type == "operator") {
      const Json::Value inputs = json_value["inputs"];
      CHECK(!inputs.isNull() && inputs.size() > 0) << "Json operator must have input!";
      std::vector<std::string> input_names;
      std::vector<std::vector<int>> input_shapes;
      std::vector<std::vector<float>> input_means;
      std::vector<std::vector<float>> input_norms;
      for (const Json::Value& input : inputs) {
        const Json::Value input_name_value = input["name"];
        CHECK(!input_name_value.isNull()) << "Json operator input must have name!";
        input_names.push_back(input_name_value.asString());
        const Json::Value input_shape_value = input["shape"];
        CHECK(!input_shape_value.isNull() and input_shape_value.size() > 0)
            << "Json operator input must have shape!";
        std::vector<int> input_shape;
        for (const auto& shape_dim : input_shape_value) {
          input_shape.push_back(shape_dim.asInt());
        }
        input_shapes.push_back(input_shape);
        const Json::Value input_mean_value = input["mean"];
        const Json::Value input_norm_value = input["norm"];
        std::vector<float> input_mean;
        std::vector<float> input_norm;
        if (!input_mean_value.isNull() || !input_norm_value.isNull()) {
          CHECK(!input_mean_value.isNull() && !input_norm_value.isNull())
              << "There must be mean and norm at the same time!";
          CHECK_EQ(input_shape.back(), input_mean_value.size())
              << "Dim of input mean must equal to input channels!";
          CHECK_EQ(input_shape.back(), input_norm_value.size())
              << "Dim of input norm must equal to input channels!";
          for (const auto& input_mean_dim : input_mean_value) {
            input_mean.push_back(input_mean_dim.asFloat());
          }
          for (const auto& input_norm_dim : input_norm_value) {
            input_norm.push_back(input_norm_dim.asFloat());
          }
        }
        input_means.push_back(input_mean);
        input_norms.push_back(input_norm);
      }
      const Json::Value outputs = json_value["outputs"];
      CHECK(!outputs.isNull() && outputs.size() > 0) << "Json operator must have outputs!";
      std::vector<std::string> output_names;
      for (const auto& output : outputs) {
        output_names.push_back(output.asString());
      }
      CreateOperator(input_names, input_shapes, output_names,
                     input_means, input_norms, &processors_.back());
    }
  } 
}

const ncnn::Mat& InferenceEngine::GetInput(const std::string& input_name) const {
  CHECK(inputs_.find(input_name) != inputs_.end())
      << "Input " << input_name << " not found";
  return inputs_.at(input_name); 
}

const ncnn::Mat& InferenceEngine::GetOutput(const std::string& output_name) const {
  CHECK(outputs_.find(output_name) != outputs_.end())
      << "Output " << output_name << " not found";
  return outputs_.at(output_name); 
}

}  // namespace dt