%% *每个标题叙述的是完成案例的过程，代码均由Gemni生成*%%
## 1.选择Github收藏量最高的案例(Python)

> [!最好找年份较近的代码]
>   代码是2017年的，时间久远，当时没注意看时间，只想着这个项目思路清晰，比较容易读懂，也就没管它放到现在是否跑得通，后来跑的时候问题频出，也注意到了这个问题。虽然跑代码的过程复杂了点，但我认为可以在修改代码的时候顺便熟悉一下代码。
> 

## 2.VS Code、WSL、虚拟环境配置

## 3.使用Gemni修改旧代码
## 4.矩池云服务器跑通train.py

使用VS Code租矩池云服务器的官方教程：https://matpool.com/supports/doc-vscode-connect-matpool/

> [!NOTE] 不要点右上角那个播放键运行代码
> 一定要在终端运行python train.py，选对目录，否则就是不断报错，血泪教训ing


## 5.本地环境预测效果展示

### 5.1.图片预测

先将手写的照片处理为张量形式，即CSV格式，具体代码如下：
```
import cv2

import numpy as np

import pandas as pd

import os

  

def process_multi_digit_to_csv(image_path, output_csv):

    """

    将包含多个手写数字的单张图片，切割并转化为 Kaggle 标准的 CSV 格式

    """

    if not os.path.exists(image_path):

        print(f"❌ 找不到图片：{image_path}")

        return

  

    print(f"正在处理图片：{image_path} ...")

    # 1. 读取并二值化

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. 寻找轮廓

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. 提取边界框，并按照 x 坐标从左到右排序（保证提取顺序与书写顺序一致）

    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    all_flattened_digits = []

    # 4. 循环切割每一个数字

    for i, (x, y, w, h) in enumerate(bounding_boxes):

        # 过滤掉太小的噪点

        if w < 5 or h < 5:

            continue

        # 留出一点内边距(Padding)，防止数字紧贴边缘

        pad = 10

        y_start = max(0, y - pad)

        y_end = min(thresh.shape[0], y + h + pad)

        x_start = max(0, x - pad)

        x_end = min(thresh.shape[1], x + w + pad)

        # 裁剪出独立的数字小图

        digit_roi = thresh[y_start:y_end, x_start:x_end]

        # 强制缩放成 28x28

        digit_resized = cv2.resize(digit_roi, (28, 28))

        # 【关键点】：展平为一维数组 (长度 784)。

        # 注意：这里保留 0-255 的整数，除以 255.0 的操作留给 test.py 去做

        digit_flattened = digit_resized.flatten()

        all_flattened_digits.append(digit_flattened)

        print(f"✂️ 成功提取第 {len(all_flattened_digits)} 个数字！")

  

    if not all_flattened_digits:

        print("⚠️ 未检测到任何数字，请检查图片背景是否干净。")

        return

  

    # 5. 将提取出的所有数字转化为 DataFrame 并保存为 CSV

    data_matrix = np.array(all_flattened_digits)

    columns = [f"pixel{i}" for i in range(784)]

    df = pd.DataFrame(data_matrix, columns=columns)

    df.to_csv(output_csv, index=False)

    print("\n=============================================")

    print(f"✅ 处理完成！成功生成: {output_csv}")

    print(f"📊 数据维度: {df.shape} (共 {df.shape[0]} 个数字，每个 784 像素)")

    print("=============================================")

  

if __name__ == '__main__':

    # 假设你上传了一张包含比如 "2026" 四个数字的照片

    target_image = 'image_path2.jpg'

    output_csv_file = 'my_multi_test.csv'

    process_multi_digit_to_csv(target_image, output_csv_file)
```


> [!NOTE] 应提醒Gemni处理手写多个数字的照片
> 初次生成的时候，没写这点，导致代码运行之后CSV文件里只有一行数值

下载权重文件model_weights.h5，运行test.py，将保存的prediction.csv传入模型，即运行display_random.py.我一共预测了两张手写照片，下面是它们的对应效果：

![[image_path.jpg|308]]👉![[prediction_results 1.png|314]]
![[image_path2.jpg|310]]👉![[prediction_results.png|311]]

### 5.2.视频预测

> [!NOTE] 一定要检查每个步骤所需工具，提高工作效率
> 没预料USB摄像头坏掉，重新买了一个，导致视频预测搁置几天，效率极低



## 6.c++推理

Gemni提供的小白推理步骤：
### 6.1.将模型转换为 ONNX 格式 (在目前的 Python 环境中完成)

您手头现在有的是 `modern_mnist_model.h5`。我们需要用工具把它转成 `.onnx`。

1. 在您的 `cv_env` 虚拟环境中，安装转换工具：
    
    ```
    pip install tf2onnx onnx
    
    ```
    
2. 运行转换命令，将您的 Keras 模型导出为 ONNX：
    
    ```
    python -m tf2onnx.convert --keras modern_mnist_model.h5 --output mnist.onnx
    ```
    
    _运行成功后，您会得到一个 `mnist.onnx` 文件。以后在 C++ 中，我们只需要这个文件，彻底和 TensorFlow 说再见。_
    

### 6.2.搭建 C++ 开发环境

在 Windows 宿主机或者 WSL 中配置 C++ 环境：

1. **安装 C++ 编译器：** Windows 下推荐安装 Visual Studio (勾选 C++ 桌面开发)；Linux/WSL 下安装 `g++` 和 `cmake`。
    opencv在命令提示符里安装了很多次都失败，下面是我安装成功的一次：
    
2. **下载并配置 OpenCV (C++版)：** 这是为了替代 Python 中的 `import cv2`，用来调用摄像头和处理图像。
    
3. **下载并配置 ONNX Runtime (C++版)：** 去 ORT 的 Github Release 页面下载编译好的 C++ 动态库（`.dll` / `.so` 以及头文件）。
    

### 6.3.用 C++ 编写预测代码 (替换掉 python 脚本)

这一步是将您在 Python 里写的逻辑“直译”为 C++ 代码。以下是 C++ 代码的核心逻辑骨架（供您参考和学习）：

```
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

int main() {
    // 1. 初始化 ORT 环境和加载模型
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MNIST_Inference");
    Ort::SessionOptions session_options;
    // 加载我们第一步转出来的 onnx 模型
    Ort::Session session(env, "mnist.onnx", session_options);

    // 2. 打开 OpenCV 摄像头 (对应 Python: cap = cv2.VideoCapture(0))
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    cv::Mat frame;
    while (cap.read(frame)) {
        // --- 图像预处理部分 (必须和 Python 保持 100% 一致) ---
        cv::Mat gray, blurred, thresh;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
        cv::adaptiveThreshold(blurred, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 15, 5);
        
        // 假设我们这里只取画面中间的一个 28x28 ROI 区域进行演示
        cv::Mat roi = thresh(cv::Rect(frame.cols/2 - 14, frame.rows/2 - 14, 28, 28));
        
        // 归一化 (类似 Python: roi.astype('float32') / 255.0)
        cv::Mat float_roi;
        roi.convertTo(float_roi, CV_32FC1, 1.0f / 255.0f);

        // 3. 将 OpenCV 的 Mat 数据塞入 ORT Tensor 中
        std::vector<float> input_tensor_values(float_roi.begin<float>(), float_roi.end<float>());
        std::vector<int64_t> input_shape = {1, 28, 28, 1}; // NHWC format matching your TF model
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

        // 4. 执行推理 (对应 Python: model.predict)
        const char* input_names[] = {"conv2d_input"}; // 请根据实际 ONNX 模型输入节点名称修改
        const char* output_names[] = {"dense_1"};     // 请根据实际 ONNX 模型输出节点名称修改
        
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        // 5. 解析输出结果 (寻找最大概率的索引，对应 Python: np.argmax)
        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        int predicted_digit = std::distance(floatarr, std::max_element(floatarr, floatarr + 10));

        // 6. 在画面上画框并显示 (对应 Python: cv2.putText, cv2.imshow)
        cv::putText(frame, std::to_string(predicted_digit), cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2);
        cv::imshow("C++ ORT Inference", frame);

        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}

```

