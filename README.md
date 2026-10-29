# 流式 ASR 助听字幕 Demo

基于 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 2.0.1 的 Android 流式语音识别 Demo，验证量化模型在低端真机上的精度 / 速度 / 体积折中。

- 模型：流式 zipformer transducer，中英双语（bilingual），混合精度量化（enc int8 + dec fp32 + join int8）
- 实测（Huawei MRD-AL00 / Helio P35，8×A53）：CER 3.77%，RTF 0.635，首字 ~950ms，APK 250 MiB

详细技术方案与优化路线见 [docs/技术方案与优化路线.md](docs/技术方案与优化路线.md)。

## 目录结构

```
ai_asr/
├── android/AsrCaptionDemo/   Android 工程（纯 Java，无 AndroidX）
│   ├── app/src/main/java/com/example/asrdemo/  Demo 代码
│   ├── app/src/main/java/com/k2fsa/sherpa/    sherpa-onnx Java API
│   └── app/src/main/assets/model/             内嵌模型（onnx 不上传）
├── scripts/                   PC 端评测脚本（CER / RTF / 首字延迟）
├── data/                      评测结果 JSON（不含音频）
├── docs/                      技术文档
└── res/                       UI 设计稿
```

## 模型与运行时下载

模型权重与 native .so 未随仓库上传，请按需下载：

### ASR 模型（双语中英，当前使用）

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xjf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
```

- `encoder-epoch-99-avg-1.int8.onnx`（必须 int8）
- `decoder-epoch-99-avg-1.onnx`（**保持 fp32**，int8 会触发重复循环）
- `joiner-epoch-99-avg-1.int8.onnx`（int8 安全，省 9 MB）
- `tokens.txt`

拷入 `android/AsrCaptionDemo/app/src/main/assets/model/`（保留 `LABEL.txt`）。

### sherpa-onnx Android 运行时

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v2.0.1/sherpa-onnx-v2.0.1-android.tar.bz2
```

将 `jniLibs/{arm64-v8a,armeabi-v7a}/` 下的 `libonnxruntime.so`、`libsherpa-onnx-jni.so`
拷入 `android/AsrCaptionDemo/app/src/main/jniLibs/`。

### Java API 源码

sherpa-onnx 2.0.1 未发布 AAR，从源码包提取 `java-api` 下的 `*.java` 到
`app/src/main/java/com/k2fsa/sherpa/onnx/`：

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v2.0.1/sherpa-onnx-v2.0.1.tar.bz2
```

## 构建

```bash
cd android/AsrCaptionDemo
# local.properties 设置 sdk.dir
./gradlew assembleDebug
```

## 评测

```bash
python scripts/eval_cer.py --manifest data/manifest_aishell3.jsonl \
  --enc-int8 --join-int8
```

## 许可证

Apache-2.0
