# Third-party notices

AI-Video Studio 自身的 Python 源码使用 MIT License。以下组件按各自许可证使用，不因本项目的 MIT License 而改变。

## CPython

- Release baseline: Python 3.12.10 x64
- Source: https://www.python.org/downloads/release/python-31210/
- License: Python Software Foundation License

发行包可携带官方安装介质，但不会复制一个不可移动的现成虚拟环境；首次启动会按当前用户安装后重建 venv。

## FFmpeg

- Release baseline: FFmpeg 7.1, BtbN `win64-gpl` build
- Build source: https://github.com/BtbN/FFmpeg-Builds
- Legal and license information: https://ffmpeg.org/legal.html
- License for selected build: GPL-3.0-or-later

FFmpeg 作为独立可执行组件分发。发行包必须附 GPL 许可证、构建来源、对应源码获取方式和 SHA-256，不把 FFmpeg 代码合并进本项目 Python 源码。

## MoneyPrinterTurbo

- Version: 1.3.3
- Pinned commit: `b4218dd66851acf2e19d4aa5f10252b08380f742`
- Source: https://github.com/harry0703/MoneyPrinterTurbo
- License: MIT

源码和隔离 Python 环境保存在工作区 `runtime/`。公开源码仓库和默认轻量发行包不包含该运行时。

## pypinyin

- Version: 0.55.0
- Source: https://github.com/mozillazg/python-pinyin
- Package: https://pypi.org/project/pypinyin/0.55.0/
- License: MIT

主环境使用 pypinyin 对不超过四个汉字的 Whisper 同音替换做无声调拼音校验。对应 wheel 包随离线 wheelhouse 分发，许可证不因本项目的 MIT License 而改变。

## ComfyUI

- Source: https://github.com/comfyanonymous/ComfyUI
- License: GPL-3.0

AI-Video Studio 只通过本机 HTTP API 连接用户自行安装的 ComfyUI。ComfyUI、自定义节点和模型不随项目分发。

## Image-generation and language models

Z-Image-Turbo、Whisper、DeepSeek、OpenAI 及用户导入的其他模型各自受模型卡、服务条款和地区限制约束。它们不随公开源码仓库分发。用户必须自行确认模型许可、API 费用和发布平台标识要求。

## Jianying fonts

剪映字体和缓存文件不随项目分发。`social_pink` 只有在本机字体文件和剪映字体映射都通过检查时才能同时用于 MP4 和可编辑草稿。

## Source Han Sans / 思源黑体

- Version: 2.005R
- Asset: `09_SourceHanSansSC.zip`, Heavy weight used at runtime
- Source: https://github.com/adobe-fonts/source-han-sans
- License: SIL Open Font License 1.1
- License text: https://github.com/adobe-fonts/source-han-sans/blob/2.005R/LICENSE.txt

The official archive is verified against its published SHA-256 before the font is used.

## NVIDIA CUDA runtime libraries

- Packages: `nvidia-cublas-cu12==12.9.2.10`, `nvidia-cudnn-cu12==9.11.0.98`
- Source: https://pypi.org/project/nvidia-cublas-cu12/ and https://pypi.org/project/nvidia-cudnn-cu12/
- Publisher: NVIDIA
- License: NVIDIA proprietary package license

These runtime DLLs are installed only inside `.runtime/mpt-venv` for local GPU inference and are not redistributed with the source project.
