# Video Factory

面向 IT、软件和 AI 技术内容的可追溯短视频生产系统。自动工厂只生成待审核包，由本地 Dashboard 逐条人工确认后发布到视频号。Bilibili 自动路由目前暂停，历史发布代码保留但不会出现在自动队列或 Dashboard 中。

## 多渠道资源发现与自动成片

统一 discovery 将 `x / github / projects / robotics / autonomous_driving / news / news_zh / official / official_zh / paper / youtube / openrouter` 作为独立搜索来源。先用 `quality_score` 判断来源是否可信、材料是否完整且可视化，再用独立的 `adoption_score` 判断“受众为什么今天应该关心”。普通内容 75 分成片，机器人/自动驾驶/硬件 82 分，有实质变化的 LLM 内容 +6 分；无每日条数上限，同一事件跨来源只保留 adoption score 最高的一条。YouTube 独立池 80 分门槛。调度：X/新闻/官网/YouTube 每 2 小时，projects/robotics/AD 每 4 小时，论文每 24 小时，GitHub 每 48 小时；门槛与加成在 `examples/resource_discovery.json` 的 `adoption_policy` 中调整。

`projects` 只收录 Reuters、Bloomberg、FT、TechCrunch、Sifted 等高信号来源且带可核验 breakout 指标（用户/下载/Stars/收入/融资/增长）的事件。`robotics`/`autonomous_driving` 分别过滤实体机器人与道路车辆，7 天去重。`official_zh` 覆盖 DeepSeek、GLM、Kimi、Qwen、豆包、混元等厂商；`news_zh` 覆盖财新、财联社、36Kr、澎湃等媒体。

```bash
# 到期渠道搜索完成后会自动调用现有 generate 流程生成成片
video-factory --workspace workspace discover \
  --config examples/resource_discovery.json --provider deepseek

# 只强制运行指定渠道；--channel 可重复
video-factory --workspace workspace discover \
  --config examples/resource_discovery.json --channel x --channel official --force

video-factory --workspace workspace discovery-status
video-factory --workspace workspace discovery-status --channel github

# 同一资源连续三次生产失败后保持 blocked，后续周期继续重试它；
# 确认资源本身不可制作时才能人工释放渠道。
video-factory --workspace workspace adopt <candidate-id> \
  --config examples/resource_discovery.json --provider deepseek
video-factory --workspace workspace discovery-skip <candidate-id> \
  --reason 'source page is no longer available'
```

候选池由可信账号、组织、媒体、官网种子、垂直 RSS 与开放主题查询组成。候选必须在标题/摘要/正文明确命中 IT、软件、AI、计算硬件或 AI 相关机器人/自动驾驶主题。每个候选记录 quality/adoption 分、分项、门槛、类别与未采用原因；候选、淘汰理由、去重、生产尝试和阻塞状态写入 SQLite 与 `workspace/discovery/` 审计 JSON，不会绕过人工发布审批。重试先复用可修复的 manifest；若确定性校验明确报告该 manifest 无法重渲染，则丢弃旧 manifest，并在下一次有界尝试中重新调用策划生成，避免反复渲染同一份不合格内容。

### 发现 → 生成 → 发布队列

`pipeline` 在 discovery 完成后，把通过质量门的结果转成幂等的待审核发布批次（目前仅视频号，Bilibili 暂停）。YouTube 两条允许路径：技术讲座按 3–6 分钟短课拆分；人物对谈只取一个 60–180 秒高光。政治门禁只检查最终片段及其标题/人物标签/钩子，原始长视频其余部分不追责。

```bash
video-factory --workspace workspace pipeline \
  --config examples/resource_discovery.json \
  --publish-config examples/pipeline_publish.json \
  --provider auto

# pipeline 只创建 ready_for_review；审核后才真正上传
video-factory --workspace workspace dashboard --actor claire
# 浏览器打开 http://127.0.0.1:8765，预览后点击每张卡片的“确认并发布”
```

同一 manifest 重复 `pipeline` 复用现有批次；YouTube 合集需先完成复用依据审核（否则 `blocked`）。Dashboard 只绑 loopback（http://127.0.0.1:8765），逐条预览真实 MP4、填北京时间定时发布（至少提前 2 小时）并单独确认，不会连带提交同批其他视频。

### macOS 持续运行

三个 LaunchAgent：discovery 每 10 分钟心跳（实际搜索频率由持久化 `next_run_at` 控制）、dashboard 常驻 8765、self-audit 每天 03:15（与生产共用锁，每晚最多 5 个去重问题、最多 $1，只复用归档 trace，不联网不登录不发布；失败按 1/3/7 天退避；Python 修复只落在 `agent-fix/<problem-id>` 隔离分支并带回归测试，永不自动 merge/push）。

首次运行进入 7 天监督期：每轮写 `workspace/automation/runs/<run-id>.json|md` 与 `latest.*`，问题追加 `problems.jsonl`，费用追加 `llm-costs.jsonl`；第 7 天标记 `remote_deployment_ready`，人工发布闸门不自动解除。

已确认 Bug 登记在 `regressions.json`：`fixed` 条目必须绑定 `tests/` 中真实存在的测试方法（`tests/test_regressions.py` 校验）。

```bash
mkdir -p workspace/logs ~/Library/LaunchAgents
chmod 700 deploy/macos/run-discovery.zsh
cp deploy/macos/com.clairehou.video-factory.discovery.plist \
  ~/Library/LaunchAgents/
chmod 700 deploy/macos/run-dashboard.zsh
cp deploy/macos/com.clairehou.video-factory.dashboard.plist \
  ~/Library/LaunchAgents/
chmod 700 deploy/macos/run-self-audit.zsh
cp deploy/macos/com.clairehou.video-factory.self-audit.plist \
  ~/Library/LaunchAgents/
launchctl bootstrap "gui/$(id -u)" \
  ~/Library/LaunchAgents/com.clairehou.video-factory.discovery.plist
launchctl bootstrap "gui/$(id -u)" \
  ~/Library/LaunchAgents/com.clairehou.video-factory.dashboard.plist
launchctl bootstrap "gui/$(id -u)" \
  ~/Library/LaunchAgents/com.clairehou.video-factory.self-audit.plist

# 状态、最近日志与手动立即触发
launchctl print "gui/$(id -u)/com.clairehou.video-factory.discovery"
tail -n 200 workspace/logs/discovery-"$(date '+%Y-%m-%d')".log
launchctl kickstart -k "gui/$(id -u)/com.clairehou.video-factory.discovery"
open http://127.0.0.1:8765
video-factory --workspace workspace automation-status
video-factory --workspace workspace automation-status --notify  # 重发通知
video-factory --workspace workspace self-audit status

# 收到通知后：在 Dashboard 完整预览，再逐条点击发布
video-factory --workspace workspace dashboard --actor claire

# 停用
launchctl bootout "gui/$(id -u)/com.clairehou.video-factory.discovery"
launchctl bootout "gui/$(id -u)/com.clairehou.video-factory.dashboard"
launchctl bootout "gui/$(id -u)/com.clairehou.video-factory.self-audit"
```

本机 `--notify` 用 macOS 通知中心；远程部署设置 `VIDEO_FACTORY_AUDIT_WEBHOOK_URL` 可收到不含凭据的审核摘要。

### 异步自我改进

发现故事角度、导演选择、翻译、排版、采集、调度或发布边界问题时，只记录问题，不在生产路径里增加一次 LLM 审稿：

```bash
video-factory --workspace workspace problem-note \
  --stage generation --category story_axis --severity high \
  --job <job-id> \
  --expected 'Google 保持为观众第一眼能识别的事件主体' \
  --observed '次要技术机制取代了人才流失主线'

video-factory --workspace workspace self-audit status
video-factory --workspace workspace self-audit replay <problem-id>
video-factory --workspace workspace self-audit run --max-issues 5 --max-cost-usd 1
video-factory --workspace workspace self-audit rollback <policy-version>
```

`result.json.artifact_identity` 保存 manifest/成片/代码 revision/policy 哈希；文件被替换会登记 `artifact_drift`。policy 默认 shadow 模式，校准后才设 `VIDEO_FACTORY_SELF_AUDIT_PROMOTION=1` 或 `--allow-policy-promotion`。

### 本地 Arize Phoenix

本地 JSONL 是审计事实源；Phoenix 只提供 trace、annotation、dataset 和 experiment 视图，服务不可用时不会阻断生产：

```bash
python -m pip install -e '.[observability]'
docker compose -f deploy/phoenix/compose.yaml up -d
export VIDEO_FACTORY_PHOENIX_ENABLED=1
open http://127.0.0.1:6006
```

Phoenix 固定 `arizephoenix/phoenix:13.12.0`，只监听 `127.0.0.1:6006/4317`，数据在 `workspace/phoenix`；凭据字段统一脱敏。详见 `deploy/phoenix/README.md`。

不需要再引入一套 LLM framework。所有文本、审稿、视觉、source-video 选段和 nightly self-audit 请求都经过同一个 transport：每次真实 HTTP POST（包括重试和模型返回错误）追加到 `workspace/observability/llm-calls.jsonl`，记录 job/candidate/stage、请求与实际模型、token、OpenRouter 返回的 cost、延迟和错误；只保存 prompt 哈希，不保存 prompt 或密钥。相同事件同时进入 Phoenix。

Static/GitHub 生产路径直接使用低价模型 plan/write/repair + 最多两次 OpenRouter semantic-review POST（review、verification），不再从头启动 Gemini fallback。默认 transport 限额分别为 12 requests / $0.25 和 12 requests / $0.30，可用 `VIDEO_FACTORY_STATIC_LLM_MAX_REQUESTS`、`VIDEO_FACTORY_STATIC_LLM_MAX_COST_USD`、`VIDEO_FACTORY_GITHUB_LLM_MAX_REQUESTS`、`VIDEO_FACTORY_GITHUB_LLM_MAX_COST_USD` 调整。最终 verification 仍失败时进入人工审核，不降低发布门槛。上线前可完全离线复核最近的已发布 corpus：

所有 job（包括 YouTube）另有 200 requests / $5 的外层紧急上限，通过 `VIDEO_FACTORY_JOB_LLM_MAX_REQUESTS` / `VIDEO_FACTORY_JOB_LLM_MAX_COST_USD` 调整；Static/GitHub 仍由上面的更严格内层限额约束。

```bash
python tools/check_llm_pipeline_corpus.py --workspace workspace --limit 30
```

## YouTube 中文精选合集

YouTube 是一等来源：每 2 小时从科技人物、AI 工程、startup、机器人/自动驾驶池搜索，每轮最多选 1 条 ≥70 分候选，无合格候选只记 `no_selection`。来源权威分只看实际发布频道；二次搬运硬淘汰。翻译用 `translate / preserve / bilingual_once` 三态术语表，产品/API/代码词保留英文。

```bash
# 首次使用：安装固定版本 yt-dlp、EJS、stable-ts 和本地 bgutil PO-token provider
video-factory youtube-runtime setup
video-factory youtube-runtime status

video-factory --workspace workspace discover-youtube \
  --config examples/youtube_discovery.json --no-render

video-factory --workspace workspace discovery-status

video-factory --workspace workspace generate \
  'https://www.youtube.com/watch?v=zCJtYuqwm7E' \
  --youtube-subtitles workspace/imports/zCJtYuqwm7E.en-orig.json3 \
  --provider deepseek
```

媒体下载固定 `mweb + EJS + PO token provider`（yt-dlp 2026.08.19 固定版本），可用 `VIDEO_FACTORY_YOUTUBE_PO_TOKEN` / `VIDEO_FACTORY_YOUTUBE_COOKIES_FROM_BROWSER` 显式配置。

源视频硬性 1080p 门禁（ffprobe 验证，低于即记 `source_below_1080` 并停止）；重建样片不传 `--youtube-media`，新清单经 `supersedes_collection_id` 审计关联。

技术讲座只生成 3–6 分钟竖屏 mini lesson，长视频最多拆 24 条；人物对谈只取 1–3 分钟高光。相关度评分先剔除赞助文案；片段需可独立理解，顶部常驻“人物身份 + 强观点”。远程源先取 metadata/transcript 选段，再 `yt-dlp --download-sections` 只下载所选区间（含 2 秒余量），字幕/范围/hook 保留原片起止秒。

微信成片 1080×1920，中间为向上移动的源画面舞台，构图支持讲者居中、全屏投影片和 `split` 左右布局。每条视频保存三份可追溯到字幕 cue 的候选钩子，质量门拒绝空泛/点击诱饵/无来源结论。

YouTube 成片保留原英文字幕并叠加更大的中文翻译，同时输出 `.en.srt`、`.zh-Hans.srt`、`.bilingual.srt`。完成字幕和复用依据审核后，再创建合集发布批次：

```bash
video-factory --workspace workspace validate-collection workspace/jobs/<job>/collection-manifest.json
video-factory --workspace workspace review-collection-rights workspace/jobs/<job>/collection-manifest.json \
  --actor editor@example.com --status reviewed --basis educational_noncommercial
video-factory --workspace workspace publish-collection-create \
  workspace/jobs/<job>/collection-manifest.json \
  --spec examples/youtube_collection_publish.json
```

Bilibili 自动发布暂停，历史实现保留用于审计；pipeline、示例配置和 Dashboard 不调用它。

## 首期范围

- 统一 `Candidate → Evidence → Scene → RenderManifest` 协议锁定事实/脚本/镜头关系；素材归档并哈希，`proof`/`explanation` 镜头必须指回证据。
- 版式为深蓝黑固定上栏（事件钩子）+ 中间真实来源录制 + 固定下栏（结论）的无旁白式；固定栏不展示来源 URL。
- 单 URL 采集覆盖 X、网页、官方公告、论文 PDF 与 GitHub；X 首版经 OpenCLI 复用已登录浏览器，Cookie/密码绝不归档。
- 多平台提交必须过质量门 + 人工批次审批；文件/文案/账号/平台参数变化必须重新审批。
- 工厂先渲染原生 9:16 MP4 视觉轨；MPT 只做背景音乐与最终 H.264/AAC 编码。

## 八类叙事合同

`practice_post`、`github_project`、`tool_sdk_agent`、`model_or_product`、`company_or_team`、`research_or_benchmark`、`official_announcement`、`linked_external_source` 八类各有必答问题。实践帖必须呈现主张/证据上下文/适用边界；论文/Benchmark 必须呈现实验条件与适用范围；帖中外链按对应模板进入来源队列。

## 有限循环内容 Agent

生产入口不是一次大 Prompt，也不是任意行动的通用 Agent。`BoundedContentAgent` 是明确的低成本循环，不是任意行动 Agent：

1. 便宜模型先做调研规划，只能用已有证据与候选外链。
2. 证据工具最多三个来源；外链网页先归档哈希才能被分镜引用。
3. 便宜模型生成结构化方案（GitHub 用 `github_brief`，其余用 `EditorialOpportunity + ContextGraph + DirectorBrief + evidence_shots`）；不能返回底层 kind/Scene/时长/浏览器动作/渲染参数。
4. 校验失败只修复失败的可见语义字段，不整包重喂旧文案。
5. 正常“规划 + 写作”两次，语义补丁三次，启用备用模型后总调用 ≤4 次硬上限。

调用、Token、研究动作与升级原因写入 manifest 的 `content_agent` trace；浏览器动作、标注、合成、质量检查与发布边界由确定性工具负责。

## Storyboard Director

`StoryboardDirector` 接收核验过的 `Evidence` 与结构化编辑方案，输出 `RenderManifest`（每幕含文案、视觉动作、时长、录屏 cue、素材角色）；不编造事实，超时只能换更长格式或重组镜头。`WebScrollVideoAdapter` 把 `CaptureCue` 编译成 web-scroll-video cue sheet 并输出静音 H.264 视觉轨（1080×1920/H.264/yuv420p）。`StoryWriterPacket` 是 LLM/人工编辑唯一写作入口：只暴露必答问题与已归档证据，返回可被导演再次校验的 JSON；正式运行不依赖 Codex。

GitHub 钩子是结构化字段（`subject_name/action/consequence`、`hook_opening/reveal/verdict`、稳定 `project_title` 等），导演编排成事件 → 能力揭示 → 观点/影响三段冷开场；背景范围严格限于 README 及 README 直接链接的仓库文档。

正式运行不依赖 Codex。X、GitHub、工具/SDK、模型/产品、公司/团队、论文/Benchmark、官方公告共用单 URL 生产入口：

```bash
video-factory --workspace workspace generate \
  https://github.com/harry0703/MoneyPrinterTurbo

video-factory --workspace workspace generate \
  https://x.com/JeffDean/status/2085034604172603724

video-factory --workspace workspace generate \
  https://arxiv.org/pdf/2501.12948 \
  --topic research_or_benchmark --format deep_dive

# 实验性 Radar V2：classic 仍是默认，可随时回退
video-factory --workspace workspace generate \
  https://x.com/JeffDean/status/2085034604172603724 \
  --render-profile radar_v2

# 录屏、FFmpeg 或 MPT 偶发失败时复用清单，不重新采集、不调用 LLM
video-factory --workspace workspace rerender workspace/jobs/<job-id>/manifest.json
```

`radar_v2` 是实验性渲染：快讯保留上下栏，GitHub/论文/架构移除底栏，启用证据聚光灯与局部放大；复杂效果失败自动退回落简化渲染并写 `*.render-fallback.json`。旧清单与未指定 profile 的任务用 `classic`。已有视觉轨可做无 LLM 的 A/B 构图测试：

```bash
video-factory --workspace workspace frame-video \
  workspace/jobs/<job-id>/manifest.json \
  workspace/jobs/<job-id>/evidence-browser.mp4 \
  --render-profile radar_v2 --out workspace/jobs/<job-id>/radar-v2-visual.mp4
```

每次运行在 `workspace/jobs/<job-id>/result.json` 持久化阶段、路由、模型、清单、成片与质量门结果。快报目标 10–15 秒（可扩展到 24 秒）。`--no-render` 只采集与内容 Agent，`--research off` 关上下文扩展，`--refresh` 跳采集+生成缓存，`--refresh-prices` 强制查价；Schema 变化后旧归档证据仍复用，不重新打开登录浏览器。

非 GitHub 内容先出 `EditorialOpportunity + ContextGraph`，再出 `AttentionStrategy + DirectorBrief + subjects/context_events/evidence_shots`；只有改变事件理解的上下文节点进入成片。执行层会确定性移除被舍弃的上下文引用、合并同屏重复中文并按最终素材类型重算阅读时间。观众文案由独立低成本 critic 逐字段审稿（人物—动作—对象—接收者、因果强度、中文自然度、无旁白可读性）：事实、关系与来源问题最多做一次字段级修复，纯中文风格和节奏建议记录为 advisory，不触发昂贵修复循环。模型/产品视频标题必须保留具体模型/产品名；面向 vibe coder 的短片每条最多解释一个真正影响结论的指标。

外链官网的合影、产品截图、架构图与 Benchmark 图归档为一等 `web:source_image` 证据；清单保存 `editorial_evidence_coverage`（已用与舍弃的证据），黄色只用于真实来源的重点框与相邻翻译。文档页不能升级成“正式发布”，除非原文明确 release/launch/announcement。Flash 门禁拒绝近似重复、机械直译、无来源推断、员工账号冒充官方、浪费结尾等；外文 X 帖保持一张卡并附 40–120 字中文释义，底栏超 62 字符渲染前做低 Token 字段压缩。

模型层支持 OpenAI 兼容接口：DeepSeek 默认 `deepseek-chat`（`DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL` 覆盖）；Kimi Code 月付用 `KIMI_CODE_API`、稳定别名 `kimi/kimi3`（映射官方 k3，不按 Token 计费）；旧 `KIMI_API_KEY` / `MOONSHOT_API_KEY` 仍可用。

```bash
video-factory generate-story examples/moneyprinterturbo_story_packet.json \
  --provider deepseek --out workspace/manifests/mpt-story.json

video-factory generate <youtube-url> --provider auto --model kimi/kimi3 --no-render
```

自动打开帖中外链才加 `--allow-linked-fetch`；备用模型显式配置（`--fallback-provider/--fallback-model`），默认关闭。

GitHub 背景采集按 README 内生规则：只打开 README 明确链接的 `vendor-notes`、background/reference 文档，X 帖子研究范围更宽，两类不发散共用。

设置 `OPENROUTER_API_KEY` 后，`generate --provider auto` 的故事写作默认固定使用低成本 `google/gemini-3.7-flash`，可用 `VIDEO_FACTORY_STORY_MODEL` 或 `--model` 覆盖；将 `VIDEO_FACTORY_STORY_MODEL` 显式设为空时，才按每日 Models API 快照选择最便宜合格模型。视觉和翻译仍使用每日价格/能力路由；可用 `OPENROUTER_TEXT_MODELS` / `OPENROUTER_VISION_MODELS`（准入名单）、`OPENROUTER_MIN_INTELLIGENCE`（故事/审稿 55、视觉 45、翻译 30）、`OPENROUTER_DATA_COLLECTION=deny` / `OPENROUTER_ZDR=1`。OpenRouter 也是独立 2 小时渠道，只有价格异常（便宜 50%、折扣 ≥75% 或新模型 5 折）且 `temptation_score >= 70` 才成片；DeepSeek 比价以官方价格页为主证据。阈值见 `examples/resource_discovery.json` 的 `openrouter.settings`。

GitHub README 只有 Architecture/Benchmark/Performance/Workflow 等段落中的图才进多模态分析（badge/赞助图/Star History 排除）；结果只用于编辑理解，不作唯一事实来源。无 `OPENROUTER_API_KEY` 回退 DeepSeek 文本路径。GitHub 成片浏览器路径固定：仓库首页与文件树 → README 顶部 → 两个有价值的真实模块；完整遍历另存为审核证据。代码示例用代码上方的真实说明句画黄框（文字外侧 8px outline，释义最长 44 字贴原文下方）。

安全研究、漏洞、绕过类证据进入人工审核：只报道披露/影响/修复/防护，不自动生成复现步骤。背景音乐 `music_license_status` 必须 `verified / licensed / original / royalty_free_verified` 且写 `license_records` 才过发布门。

发布由固定版本、隔离安装的 `social-auto-upload` CLI 执行四个平台提交；任一账号预检失败整批不提交，提交后结果不明确标记 `uncertain` 并禁止自动重试。

## 多平台发布

发布后端独立安装（固定版本）；安装器优先复用本机 Chrome，否则装隔离 Chromium：

```bash
video-factory --workspace workspace publisher setup
video-factory --workspace workspace publisher login tencent --account main
video-factory --workspace workspace publisher login douyin --account main
video-factory --workspace workspace publisher login xiaohongshu --account main
video-factory --workspace workspace publisher login bilibili --account main
```

默认隔离目录 `~/.video-factory/social-auto-upload`（`VIDEO_FACTORY_SAU_HOME` 可覆盖）；Cookie/浏览器状态不写入工作区。需要时 `VIDEO_FACTORY_INSTALL_MANAGED_CHROMIUM=1` 强制隔离 Chromium。

用 [examples/publish_targets.json](examples/publish_targets.json) 创建批次后，先审核视频/文案/账号/发布时间/平台参数，再显式批准执行：

```bash
video-factory --workspace workspace publish-create workspace/manifests/story.json --spec examples/publish_targets.json
video-factory --workspace workspace publish-approve <batch-id> --actor editor@example.com
video-factory --workspace workspace publish-run <batch-id>
video-factory --workspace workspace publish-status <batch-id>
```

`publish-retry` 只接受 `failed_pre_submit`；`submitted`/`uncertain` 永不自动重试。上线前先用测试账号逐平台灰度。

## Oracle 单机部署

首期生产环境采用 Oracle A1 Ubuntu 单机：生成、SQLite、工作区、浏览器登录态与发布后端都保存在同一台持久化 VM，不引入 Cloudflare、远程队列或第二套数据库。部署脚本、ARM64 冒烟测试、SSH-only 登录桌面、systemd 定时任务与不包含 Cookie 的备份流程见 [deploy/oci/README.md](deploy/oci/README.md)。第三方项目的固定版本与许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 本地使用

```bash
python3 -m pip install -e .
video-factory --workspace workspace init
video-factory --workspace workspace archive-asset path/to/source.png --category tweet
video-factory --workspace workspace publish-policy
video-factory --workspace workspace validate examples/flash_manifest.json
video-factory inspect-video output/final.mp4 --max-duration 10
video-factory inspect-video output/visual-track.mp4 --visual-track
video-factory --workspace workspace ingest-twitter /tmp/twitter-capture.json
video-factory --workspace workspace ingest-github /tmp/repo.json /tmp/README.md
video-factory --workspace workspace ingest-web https://example.com /tmp/page.md --title 'Official page' --parent-candidate tweet-123
video-factory --workspace workspace create-story-packet --candidate tweet-123 --include web-example-com --topic company_or_team --format explainer --duration 40 --out /tmp/story-packet.json
video-factory frame-video workspace/manifests/story.json output/browser.mp4 --out output/framed.mp4
PYTHONPATH=src python3 -m unittest discover -s tests
```

`workspace/` 是运行时资产库，默认不进入版本控制。
