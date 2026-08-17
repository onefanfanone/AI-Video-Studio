# AI-Video Studio

AI-Video Studio 是一个面向 Windows 小白用户的本机历史短视频粗剪工具。双击 `run.bat` 后，可在浏览器控制台完成脚本、文本模型、画面来源、ComfyUI 工作流、Edge 音色、字幕样式和剪映输出配置；锁稿后再进入现有 16 阶段流水线。

程序不会自动发布视频。所有任务的 `publish_ready` 都保持 `false`，历史事实、素材授权和平台 AI 标识仍需人工终审。

## 第一次运行

双击 `run.bat`。首次向导会检查 Windows、Python、FFmpeg、剪映、NVIDIA/CUDA、Whisper、MoneyPrinterTurbo、字体和本机 ComfyUI，并建议把用户数据放在：

```text
%USERPROFILE%\Videos\AI-Video-Studio
```

程序代码与用户数据分开：

```text
%LOCALAPPDATA%\AI-Video-Studio\
  settings.json
  secrets.dat
  bootstrap\
  state\

<工作区>\
  projects\ outputs\ raw\ cache\ runtime\
  workflows\ profiles\ fonts\ exports\
```

API Key 在页面中填写后由 Windows 当前用户 DPAPI 加密到 `secrets.dat`，页面只显示“已配置”，不会回显明文。现有 `.env.local` 可以在首次向导中导入；往返解密验证成功后，才可选择删除旧明文文件。

Whisper、CUDA、ComfyUI、大模型和剪映字体不会静默下载。环境页会显示缺失项、预计体积和手动准备位置。

## 四步创建视频

1. **脚本**：`direct` 逐字使用成稿；`review` 保留原稿并生成问题与建议稿；`topic` 根据选题和可选资料生成初稿。目标时长必须从 30、45、60、90 秒中选择。
2. **AI 与画面**：选择文本模型、馆藏＋AI / 纯 AI / 本地 raw、生图服务、ComfyUI 工作流和每镜 1–8 张 AI 候选。纯 AI 会跳过馆藏网络搜索。
3. **声音与字幕**：选择或试听 Edge TTS 音色；调整语速、音高、字幕模板、字体、字号、颜色、描边、阴影、位置、单行字数和淡入淡出。9:16 画布实时展示 8 字、短句和关键词效果。
4. **输出与汇总**：检查镜头与生图数量估算，选择是否生成剪映草稿和 AI 片尾说明，然后锁稿。

普通界面只展示推荐设置。Base URL、模型角色分配、调用上限和细节参数放在高级折叠区。锁稿时会生成：

- `script.txt`：最终人工确认脚本。
- `project.yaml`：现有流水线继续读取的 version 1 配置。
- `profile_snapshot.json`：全部非密钥配置、版本与哈希；全局默认值后来改变也不会影响旧任务。
- `script_manifest.json` 与 `script_history/`：模式、版本、风险和人工修改记录。

## 配置档

控制台管理五类配置档：

- `llm`：OpenAI Chat Completions 兼容协议（DeepSeek 及同协议服务）和原生 OpenAI Responses；需通过短 JSON 测试。
- `image`：本机 ComfyUI 或 OpenAI Images 兼容服务。外部服务必须设置固定单次上限，测试生成前会再次提示可能计费。
- `comfyui_workflow`：导入 API 格式 JSON，扫描 prompt、seed、尺寸、输出、Checkpoint、UNet、CLIP、VAE、LoRA 和采样器；只允许 `127.0.0.1`、`localhost` 或 `::1`。
- `voice`：Edge TTS voice、rate 和 pitch。内置六个试听组合，可在线读取并缓存音色列表。
- `subtitle`：内置 `history_clean`、`history_keyword`、`history_hook`、`social_pink`，并可另存自定义样式。

配置包导出会排除 API Key、密钥引用、本机绝对路径、模型、输出和缓存。导入后必须重新绑定密钥并重新测试。

## 构建与续跑

锁稿后的任务依次执行：

```text
preflight → voice → alignment → captions → visual_plan → asset_search
→ asset_semantic_review → ai_fallback → asset_review → asset_download
→ license_audit → storyboard → clips → preview → draft → validation
```

`asset_review: waiting_for_review` 是正常暂停。每个镜头选择画面后才会下载第三方原图。剪映运行时会停在 `draft`；关闭剪映后从控制台继续即可。

续跑同时校验项目、脚本、构建选项、`profile_snapshot.json` 和产物完整性。只有哈希一致且产物有效的成功阶段会复用；配置改变时会安全创建新任务，不会从错误阶段继续。

底层命令仍可用于维护旧项目：

```powershell
.\.venv\Scripts\python.exe -m app studio
.\.venv\Scripts\python.exe -m app build --project <project-id> --resume auto
.\.venv\Scripts\python.exe -m app build --project <project-id> --visual-mode local
.\.venv\Scripts\python.exe -m app assets review --resume latest
.\.venv\Scripts\python.exe -m app migrate --target D:\Your-AI-Video-Data
.\.venv\Scripts\python.exe -m app migrate --target D:\Your-AI-Video-Data --apply
```

迁移命令默认只做 dry-run。`--apply` 会先复制、逐文件 SHA-256 校验，再将旧 `.runtime`、`.cache`、`raw`、`outputs` 替换为 Windows junction；`projects` 只复制，不替换 Git 跟踪目录。如果安全扫描器长期占用旧 `outputs`，程序会保留该目录作为既有剪映草稿的兼容锚点，同时让新任务使用新工作区。验证通过后仍保留 `pre-studio-backup-*` 回滚目录，待用户确认剪映草稿和续跑正常后再删除。

## 画面、字幕与配音

scene-plan-v3 按每个话题动态生成时代、地区、必需和禁止元素，不维护文明或年代硬编码词表。DeepSeek 只负责镜头规划和候选相关性，不是史实来源。

馆藏素材只接受 The Met、Smithsonian、Wikimedia Commons 可核验的 PD/CC0；Openverse 只作发现入口。授权硬拒绝不可覆盖。AI 图标记为 `provider_terms`，并在片尾显示“部分画面为 AI 历史重构”。

默认 `social_pink` 是单行最多 8 字的粉色粗书体，带约 150ms 淡入淡出。MP4 使用 ASS，剪映使用可编辑富文本；只有通过字体文件和剪映映射检查的样式才能同时生成两种输出。

Whisper 对齐保留 98% 的有效覆盖率门槛。程序先按锁定原稿做精确字符匹配，再只对双方等长、每侧不超过 4 个汉字且逐字无声调拼音完全相同的替换作保守容错；例如识别成同音字时沿用 Whisper 时间，但成片字幕仍显示锁定原稿。数字、英文、漏句、额外插话、长片段和非同音替换不会被该规则放行。

`transcript.json` 和 `alignment_diagnostics.json` 会分别记录 `exact_coverage`（精确覆盖率）、`phonetic_coverage`/`coverage`（同音校正后的有效覆盖率）、真正缺失与额外识别比例，以及每个差异片段。即使质量判定失败，也会先保存诊断文件，便于判断是同音误识别还是真正漏读。

## 输出与剪映

每次运行写入 `<工作区>/outputs/<project>-<run-id>`，包括 MP4、旁白、transcript、`alignment_diagnostics.json`、SRT/ASS、视觉计划、候选与选择、授权台账、storyboard v2、task、validation 和构建报告。

剪映草稿默认发现：

```text
%LOCALAPPDATA%\JianyingPro\User Data\Projects\com.lveditor.draft
```

剪映草稿直接引用输出目录的旁白和预渲染镜头。仍需编辑草稿时，不要移动或删除对应输出目录。

## 公开发行

项目 Python 代码采用 MIT License。公开发行包设计为携带 Python 3.12.10 x64 安装介质、固定 wheelhouse 和独立 FFmpeg 7.1 `win64-gpl` 组件。构建工具会在校验值未固定或资产缺失时拒绝打包：

```powershell
.\tools\build_release.ps1 -AssetsDirectory D:\verified-assets
```

生成干净公开源码树：

```powershell
.\.venv\Scripts\python.exe tools\build_public_export.py
```

导出采用白名单，并在检测到密钥、个人绝对路径、个人项目、模型或输出时停止。第三方边界见 `THIRD_PARTY_NOTICES.md`。

## 故障排查与测试

- 双击一闪而过：运行 `run.bat --self-test`。
- TTS 无法连接微软语音：恢复网络后重试，其他有效阶段会复用。
- 夜间不想联网搜索馆藏：选择“纯 AI”。
- ComfyUI 不可用：确认本机服务地址和工作流测试通过。
- 单个馆藏接口临时失败：选择“本机 ComfyUI＋所有镜头候选”时会保留故障记录并继续本机生图；按缺图触发或外部付费生图仍会停止，避免把服务故障误判为零结果并产生费用。
- Whisper 模型缺失：把完整 faster-whisper `large-v3` 放到 `<工作区>/runtime/models/whisper-large-v3/`。
- Whisper 对齐未通过：查看任务目录中的 `alignment_diagnostics.json`。同音短替换会自动校正并记录；真正漏读、额外插话、数字或英文变化仍需重新配音或修改读音提示后再运行。
- 剪映草稿失败：关闭剪映后从首页继续任务。

离线测试不会访问真实 DeepSeek、馆藏 API 或 ComfyUI，也不产生费用：

```powershell
.\.venv\Scripts\python.exe -m compileall -q app tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
cmd.exe /d /c "call run.bat --self-test"
```
