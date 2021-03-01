#include "common/io.h"

#include <fstream>

#include "glog/logging.h"

namespace io {

bool ReadJsonFileToJsonValue(const std::string& json_file, Json::Value* json_object) {
  std::ifstream input(json_file);
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  std::string errors;
  bool result = Json::parseFromStream(builder, input, json_object, &errors);
  if (!result) {
    LOG(ERROR) << errors;
  }
  return result;
}
}  // namespace io
