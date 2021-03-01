#pragma once

#include <string>

#include "json/json.h"


namespace io {

bool ReadJsonFileToJsonValue(const std::string& json_file, Json::Value* json_object);

}  // namespace io
