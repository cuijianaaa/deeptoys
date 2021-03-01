package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pose",
    srcs = ["pose.cc"],
    hdrs = ["pose.h"],
    deps = [
        "@ncnn",
        "@opencv//:opencv",
        "//:inference_engine"
    ],
)

cc_binary(
    name = "pose_main",
    srcs = ["pose_main.cc"],
    deps = [
        ":pose",
    ],
)

cc_library(
    name = "mnist",
    srcs = ["mnist.cc"],
    hdrs = ["mnist.h"],
    deps = [
        "@ncnn",
        "//:inference_engine"
    ],
)

cc_binary(
    name = "mnist_main",
    srcs = ["mnist_main.cc"],
    deps = [
        ":mnist",
    ],
)
