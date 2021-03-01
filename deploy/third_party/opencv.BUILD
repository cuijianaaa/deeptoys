package(default_visibility = ["//visibility:public"])

cc_library(
    name = "opencv",
    srcs = glob([
        "lib/x86_64-linux-gnu/libopencv_*.so",
    ]),
    hdrs = glob([
        "include/opencv2/*/*.h",
        "include/opencv2/*/*.hpp",
    ]),
    linkopts = [
        "-lfreetype",
        "-lharfbuzz",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"], 
    linkstatic = 1,
)

