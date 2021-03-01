package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ncnn",
    srcs = ["lib/libncnn.a"],
    hdrs = glob([
        "include/ncnn/*.h",
    ]),
    linkopts = [
        "-fopenmp",
        "-lm",
    ],
    includes = ["include"],
    linkstatic = 1,
)

