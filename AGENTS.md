# AGENTS.md

本文件适用于 `AI-Video` 及其全部子目录。接手项目的 AI 或工程师修改代码前必须完整阅读；面向用户的操作说明维护在 `README.md`，不要再把阶段日志或临时输出编号追加到本文件。

## 1. 目标与边界

本项目是 Windows 本地“奇怪世界历史趣闻”短视频粗剪流水线。当前能力包括脚本工作台、MoneyPrinterTurbo/Edge TTS 旁白、faster-whisper 逐词对齐、ASS/剪映富文本字幕、DeepSeek 镜头规划、馆藏检索、ComfyUI AI 候选、人工素材审核、授权台账、竖屏 MP4 和剪映 9.9 草稿。

明确不包含：史实检索与引用核验、自动发布、平台上传、背景音乐、新版剪映自动点击导出。DeepSeek 不是史实来源，授权台账也不等于事实核查；`publish_ready` 必须保持 `false`。

## 2. 入口与目录

根目录只保留一个面向普通用户的双击入口 `run.bat`。它打开统一 Studio 控制台，并在锁稿后使用 `--resume auto` 构建。不要重新增加 `resume_last.bat`、`review_assets.bat` 等根目录入口；低频诊断入口统一放在 `tools/`。

主要模块：

```text
app/__main__.py             CLI 和退出码
app/pipeline.py             16 阶段编排、渲染、草稿和验收
app/script_workbench.py     direct/review/topic、版本、锁稿和本机页面
app/studio_console.py       首次向导、首页、四步配置与配置档 UI
app/studio_settings.py      settings、工作区路径和 Windows DPAPI 密钥
app/studio_profiles.py      配置档、ComfyUI 扫描、快照和配置包
app/studio_providers.py     LLM/生图/语音显式测试
app/studio_migrate.py       可回滚的数据迁移和 junction
app/task_state.py           原子 task.json 与阶段续跑
app/mpt_runtime.py          MPT、模型、字体和隔离环境管理
app/mpt_worker.py           Edge TTS 与 faster-whisper 子进程
app/transcript.py           原文对齐、字幕分块和强调词
app/scene_plan_schema.py    scene-plan-v3 结构和动态冲突校验
app/visual_semantics.py     馆藏候选文字元数据语义复核
app/jianying_text.py        剪映富文本适配层
config/                     项目模板、音色和字幕 preset
projects/<id>/              锁稿项目或人工维护项目
tests/fixtures/             离线全链路 fixture
raw/                        用户原始素材，只读
outputs/<run-id>/           每次构建的独立输出
.cache/                     TTS、对齐、视觉与素材缓存
.runtime/                   MPT 隔离环境、Whisper 模型和字体
```

代码根目录不是用户数据目录。新用户数据位于 settings 指定工作区，默认 `%USERPROFILE%\Videos\AI-Video-Studio`；旧项目在未初始化时保持仓库内兼容模式。使用 Python 3.12，系统 Python 3.13 不是验证基线。源码、YAML、JSON 和字幕统一使用 UTF-8；`.bat` 必须保持纯 ASCII。

## 3. 脚本工作台与项目配置

统一控制台运行在 16 阶段任务之前，草稿保存在工作区 `projects/_drafts/<draft-id>`。只有人工锁稿后才能原子创建 `projects/history-<时间戳>` 并调用视频流水线。

- `direct` 不得调用 DeepSeek 或改写用户脚本。
- `review` 必须保留原稿；API 失败仍允许人工锁稿。
- `topic` 未成功生成初稿时不得创建项目。
- 风险提示不是事实核查，也不能阻止锁稿。
- 页面只绑定随机 `127.0.0.1` 端口，保留 CSRF、CSP、HTML 转义、`no-store` 和请求大小限制。
- 密钥只存 Windows 当前用户 DPAPI `secrets.dat`；旧 `.env.local` 仅为迁移兼容。密钥不得进入页面、日志、草稿、项目、任务、报告、配置包或 provenance。
- 配音列表必须动态读取 `config/voice_profiles.yaml`，试听文件只能通过 profile 白名单和试听 manifest 暴露。

`project.yaml` 目前固定为 `version: 1`。新增字段必须提供安全默认值或明确校验，并保持现有 version 1 项目兼容。相对路径按项目目录解析，不能依赖终端当前目录。用户路径可能包含空格和中文，外部命令必须使用参数数组。

项目模板不得继承某个旧选题的读音、强调词、专名或素材覆盖。默认配音为 `yunyang_soft`，默认字幕为 `social_pink`，默认画面为 sourced 模式下的 `museum_and_ai`；工作台可以写入 `ai_only`。

配置档分为 `llm`、`image`、`comfyui_workflow`、`voice`、`subtitle`。锁稿必须写 `profile_snapshot.json`，且任务输入哈希必须包含完整快照；修改全局默认值不能改变已锁稿任务。外部 LLM 只允许 Chat Completions 兼容协议或原生 OpenAI Responses；外部生图只允许 Images 兼容协议并使用固定整数上限。禁止开放任意请求头、任意 HTTP 模板或任意本地代码。

## 4. 构建、缓存与续跑

`build_project()` 的阶段顺序为：

```text
preflight → voice → alignment → captions → visual_plan → asset_search
→ asset_semantic_review → ai_fallback → asset_review → asset_download
→ license_audit → storyboard → clips → preview → draft → validation
```

每个阶段开始、成功、等待或失败都必须原子更新 `task.json`。`asset_review: waiting_for_review` 是正常暂停。续跑只能复用满足以下全部条件的阶段：

- 项目、构建选项和完整输入哈希一致。
- 阶段状态为成功。
- 声明的产物仍存在并通过阶段校验。
- 缓存键包含会影响结果的 provider、模型、版本、配置和内容。

配置或脚本变化时，不能从不兼容的旧任务继续；可复用独立验证过的 TTS、Whisper、搜索和 AI 缓存。搜索、生图和下载应在每个成功项后更新检查点，中断后只重试缺失或无效项，避免重复 API 费用和本机生图。

`storyboard.json` 使用 schema v2。画面供应模块必须通过 storyboard 接回渲染器，不得绕过预渲染、草稿或验收层。每个 shot 保留时长、整数帧、本地源文件、裁切/运动、字幕、intent/asset/provenance、rights、AI 标识、审核和语义复核字段。

预渲染镜头不能随意移除：MP4 和剪映必须复用同一批统一分辨率、帧率和时长的镜头文件。镜头时长先换算为整数帧再计算时间；30fps 下总画面与旁白误差应接近一帧且不超过 0.2 秒。

## 5. 视觉 provider 与授权边界

scene-plan-v3 的 `time_context`、`must_include`、`avoid` 和搜索词都由当前话题动态生成。Python 只校验结构、年份顺序、搜索锚点、动态必需/禁止元素冲突和否定语义；禁止重新加入古罗马、中世纪、明朝或“现代物品”等文明/年代硬编码词表。

局部计划失败时只修订有错误的 shot ID，再按 ID 合并回完整计划。修订响应缺镜头、重复 ID 或 ID 不匹配时不能覆盖旧计划。现有配置允许 1–10 轮局部结构修订；完整短语冲突不能拆成普通单词比较。结构合格但 DeepSeek 语义审校不可用时可以进入人工审核，但必须记录 `unavailable` 且不自动勾选候选。

馆藏 provider 使用关闭式授权策略：

- The Met：只接受 `isPublicDomain=true` 且有 HTTPS 原图。
- Smithsonian：只接受媒体层 `usage.access=CC0`；缺少 key 时跳过。
- Wikimedia Commons：必须同时有作者、来源页和可归一化为 CC0 1.0/PDM 1.0 的授权。
- Openverse：只作发现，未经已实现上游 provider 再核验时不可选择。

CC BY、CC BY-SA、NC、ND、授权不明、来源不全、HTTP 原图、尺寸不足或文件解码不完整都是不可覆盖的硬拒绝。搜索服务故障必须与零结果分开记录，不能因为 provider 故障直接触发 AI 生图。

DeepSeek 候选复核只读取必要文字元数据，拍摄/数字化年份不能单独成为拒绝理由。语义拒绝允许用户确认风险后覆盖，授权拒绝永远不可覆盖。

`comfyui_local` 只调用配置的 `127.0.0.1` 服务。流水线不得自动安装或启动 ComfyUI；工作流必须是 API 格式且包含唯一 prompt marker。`candidate_policy: all_shots` 可给每镜头生成多张候选；外部付费 provider 必须使用明确正整数上限，不能使用无限或按全部镜头动态放大的额度。

审核服务只监听 `127.0.0.1`。提交的候选 ID 必须属于当前任务，每个镜头必须明确选择，硬拒绝不可覆盖。审核前只缓存缩略图和 AI 候选；审核后下载第三方原图，并再次验证授权、HTTPS、MIME、魔数、完整像素解码、尺寸、来源页和 SHA-256。

第三方原图进入 `.cache/assets/<sha256>`，运行目录使用独立副本。AI 图的 `rights_code` 只能是 `provider_terms`，必须记录 provider、模型、提示词、seed/请求 ID、尺寸、生成时间和 SHA-256。OpenAI moderation 拒绝应按安全失败处理，不得改写请求绕过。

## 6. 配音、字幕、画面与剪映约束

- MoneyPrinterTurbo 固定为 1.3.3、提交 `b4218dd66851acf2e19d4aa5f10252b08380f742`，只能安装在工作区 `runtime/mpt-venv`。主项目不得导入 MPT 的顶层 `app` 包，只通过 JSON 和子进程调用 worker。
- Whisper 正式项目必须使用本地 `large-v3` 对齐，原文覆盖率低于 98% 时失败，不能静默退回估算时间轴。
- Edge TTS 7.2.8 的 metadata 是 JSONL，offset/duration 单位为 100ns；SRT 必须由项目自己生成。
- `transcript.json` 是字幕唯一时间轴来源；SRT 是无样式备份，ASS 用于 MP4，剪映富文本从 transcript 生成。
- 通用字幕可使用双行 12 字 preset；账号默认 `social_pink` 是庞门正道粗书体、单行最多 8 字、粉色描边和约 150ms 渐显渐隐。账号样式只能放配置，不得硬编码进重排器。
- pyJianYingDraft 轨道使用连续整数微秒游标，禁止用独立浮点秒计算相邻起点。AI 披露和正文字幕必须分轨，避免重叠。
- 剪映 9.9 只生成可编辑草稿，由用户手动导出。写草稿前检查 `JianyingPro.exe`；失败时清理本次半成品草稿。
- 剪映草稿直接引用 `outputs/<run-id>/working/clips` 和旁白。用户仍需编辑草稿时，不得移动或删除对应输出目录。
- 静态图运动默认在 2 倍内部画布上使用余弦缓入缓出，再降采样到 1080×1920。不要退回最终分辨率上的线性整数像素平移。

## 7. 数据与安全约束

- `raw/` 是用户只读素材，不得移动、改名、覆盖或删除。
- 每次构建创建独立时间戳输出，不覆盖旧结果。
- 不自动删除输出、剪映草稿、模型或缓存；清理前解析正式剪映草稿引用并取得用户同意。
- `secrets.dat`、`.env.local`、raw、runtime、`.venv`、cache、outputs 和脚本草稿不提交 Git。GitHub 不是这些本机数据的备份。
- 工作区迁移必须先复制和 SHA-256 校验，再把旧数据目录改为 junction；Git 跟踪的 `projects` 不得改为 junction。Windows 安全扫描器若持续占用旧 `outputs`，允许在报告中显式保留原目录作草稿兼容锚点，新任务仍切换到新工作区。迁移保留回滚备份，只有用户验证剪映草稿和续跑后才可删除。
- 公开导出必须使用白名单。发现 API Key、用户名绝对路径、个人项目、输出、模型或旧 Git 历史时必须停止发布。
- 外部命令使用参数数组，禁止将用户输入拼进 shell 命令。
- 馆藏 URL 必须统一规范化：安全百分号编码路径/查询，保留已有 `%xx`，拒绝 CR/LF/C0 控制字符和带空白主机名。
- 面向用户的可恢复错误使用中文 `BuildError`，同时保留必要原始错误；失败写入 `task.json` 和 `build_report.json`。
- 不得自动发布、批量上传、抓取授权不明站点或把来源不明素材描述为可商用。

## 8. 测试与完成标准

所有影响生成结果的修改至少执行：

```powershell
.\.venv\Scripts\python.exe -m compileall -q app tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
cmd.exe /d /c "call run.bat --self-test"
```

普通测试必须离线运行，不访问 DeepSeek、馆藏 API 或真实 ComfyUI，也不产生费用。真实网络或显卡 smoke test 必须显式执行。

涉及 FFmpeg、TTS、字幕、素材、镜头、剪映或配置的修改，还要使用 `tests/fixtures/roman-edge-smoke` 和隔离的 `.test-drafts` 完成一次全链路构建。测试结束清理测试输出和字节码，不得污染正式剪映目录。

完成标准：

- MP4 为 1080×1920、30fps、H.264/AAC、`yuv420p`，时长与旁白误差不超过 0.2 秒。
- 字幕时间单调、不重叠，并符合当前 preset 的行数、字数、时长、字体和安全区。
- Whisper 覆盖率不低于 98%，强调词时间位于对应字幕区间。
- storyboard 的源文件、授权、审核状态和 manifest 可互相追溯。
- 剪映草稿包含 video、narration、subtitles；使用 AI 画面时另有 `ai_disclosure` 文字轨道。
- `validation.json` 全部通过，`raw` 的文件名、大小和修改时间不变。
- README、示例配置和实际 CLI 行为一致；不要在文档中保存阶段性输出编号或过期回归基线。
