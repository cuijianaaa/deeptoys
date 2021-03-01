package(default_visibility = ["//visibility:public"])

cc_library(
    name = "inference_engine",
    srcs = [
        "inference_engine.cc",
    ],
    hdrs = [
        "inference_engine.h",
    ],
    deps = [
        "@ncnn",
        "@opencv",
        "@jsoncpp", 
        "//common:defines",
        "//common:io",
    ],
)
