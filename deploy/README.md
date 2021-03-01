# Deploy

DeepToys attaches great importance to deployment, focuses on embedded devices. For simple deployment, we simply encapsulate a class based on [ncnn](https://github.com/Tencent/ncnn)

## Install bazel
We use bazel to build the deploy code. go to [Bazel releases page on GitHub](https://github.com/bazelbuild/bazel/releases), and download **bazel-\<version\>-installer-linux-x86_64.sh**
```
# Install
sudo chmod +x bazel-<version>-installer-linux-x86_64.sh
sudo ./bazel-<version>-installer-linux-x86_64.sh
```
## Build & Run
```
# Build
cd deploy/
bazel build //src:pose_main

# Run
./bazel-bin/src/pose_main images/pose.jpg

```
the output:

![](images/pose_result.jpg)
