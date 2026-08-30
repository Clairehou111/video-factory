# Video Factory

面向技术内容的可追溯短视频生产系统。自动工厂只生成待审核包，由本地 Dashboard 逐条人工确认后发布到视频号。Bilibili 自动路由目前暂停，历史发布代码保留但不会出现在自动队列或 Dashboard 中。

## 多渠道资源发现与自动成片

统一 discovery 将 `x / github / news / news_zh / official / official_zh / paper / youtube / openrouter` 视为并列渠道，而不是带优先级的来源列表。`news`、`official` 是英文渠道；`news_zh`、`official_zh` 是中文渠道。四个新闻/官网渠道分别拥有搜索预算、质量门、运行状态和成片选择名额，不混池也不轮流分配名额。每个到期渠道独立搜索并最多选择一条达到自身质量门的资源；同一事件跨渠道只制作一次，重复渠道会顺延到自己的下一条合格资源。X、两种语言的新闻与官网每 2 小时运行，论文每 24 小时运行，GitHub 与 YouTube 每 48 小时运行。

`official_zh` 覆盖 DeepSeek、GLM/智谱、Kimi、Qwen、豆包、混元、文心/千帆、MiniMax、阶跃星辰、百川、零一万物、商汤日日新、讯飞星火和华为盘古；`news_zh` 覆盖财新、财联社、36Kr、界面、第一财经、澎湃及主要 AI 技术媒体。重点事件包括新模型/产品、显著调价或价格战、厂商对标与争议、开源权重/许可证、API/上下文/多模态升级、下线迁移、重要基准和重大故障/安全事件。低于 20% 的普通折扣默认不过视频门槛。厂商模型卡、开发文档或公告命中时归入 `official_zh`；媒体报道同一事件仍属于 `news_zh`，只作为独立候选或交叉证据，不会把官网事件标成新闻来源。

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

候选池由可信账号、组织、媒体和官网种子与开放主题查询共同组成。搜索摘要不能直接通过质量门：系统还会检查可核验的身份和日期、足够的事实与叙事材料、真实视觉路径及各渠道特有条件。热度只参与软评分。候选、淘汰理由、跨渠道事件匹配、生产尝试和阻塞状态同时写入 SQLite 与 `workspace/discovery/` 的审计 JSON；发现并生成视频不会绕过现有人工发布审批。

### 发现 → 生成 → 发布队列

`pipeline` 在统一 discovery 完成后，立即把通过成片质量门的结果转换成幂等的待审核发布批次。所有自动来源目前只进入视频号，Bilibili 暂停。YouTube 有两条允许路径：AI 工程等技术讲座按顺序拆成 3–6 分钟短课并覆盖完整技术内容；知名科技人物对谈只选一个 60–180 秒、可独立理解且最有传播力的高光。普通生活访谈、融资闲聊和不知名人物对谈不进入候选。

政治门禁只检查最终入选片段及其标题、人物标签和三条钩子；原始长视频其他部分含政治话题不会导致整片淘汰。入选片段禁止政治、选举、政府、战争、地缘政治和政治人物内容，但不同国家之间的技术、研究、工程、人才、学校或教育比较明确允许。

```bash
video-factory --workspace workspace pipeline \
  --config examples/resource_discovery.json \
  --publish-config examples/pipeline_publish.json \
  --provider auto

# pipeline 只创建 ready_for_review；审核后才真正上传
video-factory --workspace workspace dashboard --actor claire
# 浏览器打开 http://127.0.0.1:8765，预览后点击每张卡片的“确认并发布”
```

同一 manifest 重复执行 `pipeline` 会复用现有批次，不会重复投稿。YouTube 合集仍须先完成复用依据审核；未审核时批次保持 `blocked`，审核完成后再次运行 pipeline 才会生成可审批批次。

Dashboard 只绑定 loopback 地址，按 manifest 去重，仅显示视频号未发布项。每张卡片可预览真实 MP4；按钮会在点击时绑定成片哈希、标题、文案和目标，完成账号预检后只发布这一条，不会连带提交同批其他视频。缺失成片、质量门失败或提交结果不确定时按钮保持禁用。

### macOS 持续运行

`deploy/macos/com.clairehou.video-factory.discovery.plist` 是自动工厂 LaunchAgent：登录后立即运行，并每 10 分钟调用一次 `pipeline`；各渠道仍由持久化的 `next_run_at` 控制实际搜索频率。`deploy/macos/com.clairehou.video-factory.dashboard.plist` 常驻提供 `http://127.0.0.1:8765` 审核台。pipeline 会自动发现、生成、执行有界审稿/修复并创建待审核发布批次，但不会自行批准或公开投稿。

首次定时运行会在 `workspace/automation/state.json` 开始 7 天本地监督期。每一轮（包括编排层崩溃）都会写入 `workspace/automation/runs/<run-id>.json`、同名 Markdown、`latest.json` 和 `latest.md`；历史问题追加到 `problems.jsonl`，Token 与费用追加到 `llm-costs.jsonl`。审计会区分未解决问题、已自动修复问题、实际/估算 LLM 成本、模型选择依据和待审核批次。单次候选内部最多按 `retry_backoff_seconds` 有界修复；整轮失败后至少冷却 `blocked_retry_delay_hours`，最多自动重跑 `max_blocked_retry_runs` 轮，随后转入 `needs_human_candidates`，避免 launchd 每 10 分钟重复烧 Token。第 7 天后报告会把下一阶段标记为 `remote_deployment_ready`，但人工发布闸门不会自动解除。

已确认的产品 Bug 同时登记在 `regressions.json`。每个标记为 `fixed` 的条目必须绑定至少一个 `tests/test_*.py::TestClass::test_method`；`tests/test_regressions.py` 会检查引用的测试文件、测试类和测试方法确实存在。因此运行日志负责发现问题，回归清单负责确保同一问题不再出现；没有 test case 的条目不能标记为已修复。

```bash
mkdir -p workspace/logs ~/Library/LaunchAgents
chmod 700 deploy/macos/run-discovery.zsh
cp deploy/macos/com.clairehou.video-factory.discovery.plist \
  ~/Library/LaunchAgents/
chmod 700 deploy/macos/run-dashboard.zsh
cp deploy/macos/com.clairehou.video-factory.dashboard.plist \
  ~/Library/LaunchAgents/
launchctl bootstrap "gui/$(id -u)" \
  ~/Library/LaunchAgents/com.clairehou.video-factory.discovery.plist
launchctl bootstrap "gui/$(id -u)" \
  ~/Library/LaunchAgents/com.clairehou.video-factory.dashboard.plist

# 状态、最近日志与手动立即触发
launchctl print "gui/$(id -u)/com.clairehou.video-factory.discovery"
tail -n 200 workspace/logs/discovery-"$(date '+%Y-%m-%d')".log
launchctl kickstart -k "gui/$(id -u)/com.clairehou.video-factory.discovery"
open http://127.0.0.1:8765
video-factory --workspace workspace automation-status
video-factory --workspace workspace automation-status --notify  # 重发通知

# 收到通知后：在 Dashboard 完整预览，再逐条点击发布
video-factory --workspace workspace dashboard --actor claire

# 停用
launchctl bootout "gui/$(id -u)/com.clairehou.video-factory.discovery"
launchctl bootout "gui/$(id -u)/com.clairehou.video-factory.dashboard"
```

本机 `--notify` 使用 macOS 通知中心；远程部署时设置 `VIDEO_FACTORY_AUDIT_WEBHOOK_URL`，系统会向该地址 POST 不含凭据的审核摘要（run id、待审核 batch id、问题数和人工操作提示）。通知只报告待办，绝不会替代 `publish-approve` 或触发上传。

## YouTube 中文精选合集

YouTube 现在是一等来源，不再按普通网页处理。系统每 48 小时从 Perplexity/Aravind Srinivas、Andrej Karpathy、Y Combinator 和热门 AI 工程查询池中搜索一次，每轮最多选择一个达到 70 分质量门的源视频；没有合格候选时只记录 `no_selection`，不会生成内容。来源权威分只看实际发布频道，不会因为标题里出现 Karpathy 等名字就当作官方源；可信原始频道和可靠访谈频道优先，明确标注 `Source: @原频道`、re-upload/repost 的二次搬运会被硬性淘汰。

翻译采用 `translate / preserve / bilingual_once` 三态术语表。没有成熟中文译法、翻译后更难理解或属于产品/API/代码的词保留英文；例如 `Harness` 首次可显示为 `Harness（Agent 的执行与反馈框架）`，随后继续使用 `Harness`，禁止为了全中文而写成“挽具”。

```bash
# 首次使用：安装固定版本 yt-dlp、EJS 和本地 bgutil PO-token provider
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

媒体下载固定使用 `mweb + EJS + PO token provider`，不会静默伪装成 `android_vr`。运行时固定为 `yt-dlp 2026.08.19`、`yt-dlp-ejs 0.8.0` 和 `bgutil-ytdlp-pot-provider 1.3.2` 的 script provider 模式。也可显式配置 `VIDEO_FACTORY_YOUTUBE_PO_TOKEN` 或 `VIDEO_FACTORY_YOUTUBE_COOKIES_FROM_BROWSER`；`--youtube-media` 和 `--youtube-subtitles` 仍可提供已获准使用的本地文件。

源视频有硬性 `1920×1080` 门禁：本地文件和下载文件都会用 `ffprobe` 验证，低于 1080p 就记录 `source_below_1080` 并停止，不做 AI 放大，也不渲染模糊的 PC 成片。需要重建旧样片时，不传 `--youtube-media` 即可重新获取高码率源；新清单会通过 `supersedes_collection_id` 与旧清单建立可审计的替代关系，旧文件不会被删除。

技术讲座只生成微信竖屏 mini lesson：每条约 3–6 分钟，按原始顺序覆盖至少 90% 的完整技术故事，长视频最多拆成 24 条，不会因为超过 45 分钟就静默丢掉后半段。知名科技人物对谈只生成一个约 1–3 分钟高光；片段必须有独立观点和回报，顶部整段常驻“人物身份 + 强观点”，人物画面居中，双语字幕保留原声。身份标签必须可由来源核实，不能凭空制造头衔。远程对谈会先仅获取 metadata 和 transcript，完成非政治片段选择与校验后，再用 `yt-dlp --download-sections` 只下载所选区间及两秒边界余量；渲染时间轴改为片段本地秒数，同时在字幕、范围和 hook 中保留原视频起止秒数。技术讲座仍下载完整源视频，本地媒体仍使用完整源时间轴。

微信保持 1080×1920，中间使用向上移动的源画面舞台。构图可以在镜头范围内切换：讲者画面使用居中放大；真正的全屏投影片完整适配；“左侧讲者＋右侧投影片”使用 `split` 构图，避免裁掉讲者或文字。每条视频保存三份可追溯到字幕 cue、语义完整且不截断英文术语的候选钩子；问候语、空泛问题、泛化点击诱饵和没有来源支持的夸张结论会被质量门拒绝。

所有 YouTube 成片保留原始英文字幕，并在其下方叠加字号更大的自然中文翻译；同时输出独立 `.en.srt`、`.zh-Hans.srt` 和 `.bilingual.srt`。没有自然中文对应词的 AI/软件术语可以保留英文。完成字幕和复用依据审核后，再创建绑定全部文件、标题和顺序的合集发布批次：

```bash
video-factory --workspace workspace validate-collection workspace/jobs/<job>/collection-manifest.json
video-factory --workspace workspace review-collection-rights workspace/jobs/<job>/collection-manifest.json \
  --actor editor@example.com --status reviewed --basis educational_noncommercial
video-factory --workspace workspace publish-collection-create \
  workspace/jobs/<job>/collection-manifest.json \
  --spec examples/youtube_collection_publish.json
```

Bilibili 自动发布目前暂停。历史上传与合集关联实现仍保留用于审计和未来重新设计，但 pipeline、示例发布配置和 Dashboard 都不会调用它。

## 首期范围

- 用统一的 `Candidate → Evidence → Scene → RenderManifest` 协议锁定事实、脚本和镜头的关系。
- 原始截图、录屏和公共素材都归档、哈希；任何 `proof` / `explanation` 镜头必须指回证据。
- 画面采用深蓝黑固定上栏（事件钩子）+ 中间真实来源录制 + 固定下栏（结论）的无旁白版式；分镜文本是编辑/审核元数据，不能以大卡片遮住来源画面。外文只在真实高亮文本旁附简短中文释义。
- 固定栏不展示来源 URL；来源 URL 与采集时间留在证据包中。
- 已接入 X、普通网页、官方公告、论文 PDF 与 GitHub 的单 URL 采集，随后自动进入内容 Agent、真实页面录制、MPT 合成和 `ffprobe` 验收。
- X 首版通过 OpenCLI 复用已登录浏览器读取结构化原帖；成片先展示一张完整原帖卡，再进入原帖指向的官网证据。Cookie、密码和本地存储绝不归档。
- 多平台提交必须先通过质量门并绑定一次人工批次审批；审批后的文件、文案、账号或平台参数发生变化时必须重新审批。
- 工厂侧先把视觉轨渲染为原生 9:16 MP4 和真 crossfade；MPT 不接收静态图片、不负责镜头调度，只做背景音乐与最终 H.264/AAC 编码。

## 八类叙事合同

`practice_post`、`github_project`、`tool_sdk_agent`、`model_or_product`、`company_or_team`、`research_or_benchmark`、`official_announcement` 和 `linked_external_source` 都有自己的必答问题。短快报只检查该类型的最小事实/边界集合；20 秒以上讲解则必须完整回答该类型的叙事问题。

特别约束：实践帖子必须呈现作者的主张、证据上下文与适用边界；论文/Benchmark 必须呈现实验条件与适用范围；帖子中的外链会进入原始来源队列，按 GitHub、论文、公告或工具页的对应模板处理。

## 有限循环内容 Agent

生产入口不是一次大 Prompt，也不是任意行动的通用 Agent。`BoundedContentAgent` 使用明确的低成本循环：

1. 便宜模型先做一次调研规划，只能选择已有证据和候选中已经发现的外链。
2. 证据工具按最多三个来源执行；开启外链采集时，网页必须先归档并哈希，之后才能被分镜引用。
3. 便宜模型生成严格结构化的内容方案；GitHub 使用 `github_brief`，其余类型使用 `EditorialOpportunity + ContextGraph + DirectorBrief + evidence_shots`。模型只决定选题理由、观点、上下文取舍和语义视觉家族，不能返回底层 `kind`、Scene、时长表、浏览器动作或渲染参数。
4. 确定性导演与质量门校验失败时，只修复失败的可见语义字段；旧错误文案不会再次整包喂给模型，证据 ID、URL、页面锚点、镜头顺序与素材指令保持不变。
5. 只有显式配置备用模型且主模型仍失败时，才允许升级模型。正常是“规划 + 写作”两次；需要语义补丁时为三次，启用备用模型后的总调用仍受四次硬上限约束。

Agent 负责选角度、证据、重点与观点；浏览器动作、黄色标注、固定上下栏、MPT 合成、质量检查和发布边界仍由确定性工具负责。所有调用、Token 用量、研究动作和升级原因都会写入 manifest 的 `content_agent` trace。

## Storyboard Director

`StoryboardDirector` 是采集和合成之间的独立服务。它接收经核验的 `Evidence`、内容类型、每个叙事问题的回答，以及模型返回的结构化编辑方案；输出一份 `RenderManifest`。每幕都有屏幕文案、视觉动作、时长、录屏 cue、指向目标和素材角色；`narration` 只保留为内部编辑注释，首版不合成旁白。

它不自由编造事实，也不允许通过删掉因果关系硬塞进 10 秒：超过时长时必须选择更长的内容格式或重新组织镜头。MPT 只消费导演输出的原生视觉轨。

`WebScrollVideoAdapter` 将导演的 `CaptureCue` 编译成可编辑的 web-scroll-video cue sheet，再输出静音的 H.264 视觉轨。视觉轨只验收 1080×1920 / H.264 / yuv420p；交给 MPT 后，最终成片再额外验收 AAC。

`StoryWriterPacket` 是给 LLM 或人工编辑的唯一写作入口：它只暴露当前主题必答问题和已归档证据，并要求返回能被 `StoryboardDirector` 再次校验的 JSON。模型必须提供固定钩子、固定结论、逐幕事实/翻译与逐幕解释；正式运行不依赖 Codex。帖子外链可在工作区中作为 `parent_candidate_id` 关联的原始来源追加到同一证据包，避免脱离原帖单独讲网页。

GitHub 钩子不是一段不可审计的自由文案。模型分别返回 `subject_name/action/consequence`、可选的 `background_actor/action/consequence/evidence_ids`、`hook_opening/reveal/verdict`、`hook_evidence_ids` 和稳定的 `project_title`。导演把它编排成三段独立冷开场：事件 → 能力揭示 → 观点/影响。GitHub 的背景范围严格限制在 README 及 README 直接链接的仓库文档；没有厂商因果背景时不做外部行业发散。存在背景时第一屏点名厂商动作、第二屏点名项目回应；不存在背景时开场直接点名项目。安全/隐私项目还会校验动词强度，清理或检测不得被写成绕过官方机制。

正式运行不依赖 Codex。X、GitHub、工具/SDK/API/Agent、模型/产品、公司/团队、论文/Benchmark 和官方公告已经共用真正的单 URL 生产入口；采集、内容 Agent、真实浏览器录制、固定栏/冷开场合成、MPT 音乐母版与格式验收都由程序内部编排：

```bash
video-factory --workspace workspace generate \
  https://github.com/harry0703/MoneyPrinterTurbo

video-factory --workspace workspace generate \
  https://x.com/JeffDean/status/2085034604172603724

video-factory --workspace workspace generate \
  https://arxiv.org/pdf/2501.12948 \
  --topic research_or_benchmark --format deep_dive

# 录屏、FFmpeg 或 MPT 偶发失败时复用清单，不重新采集、不调用 LLM
video-factory --workspace workspace rerender workspace/jobs/<job-id>/manifest.json
```

每次运行会在 `workspace/jobs/<job-id>/result.json` 持久化阶段、路由结果、模型选择、清单、成片和质量门结果。自动路由后仍可用 `--topic`、`--format` 和 `--duration` 覆盖；信息快报会被硬限制在 15 秒内。`--no-render` 只运行采集与内容 Agent，`--research off` 关闭上下文扩展，`--refresh` 同时跳过采集与生成缓存，`--refresh-prices` 强制重新查价。采集缓存与生成缓存相互独立：内容 Schema 变化导致旧清单失效时，已经归档的 X/网页原始证据仍会复用，不会再次打开登录浏览器；相同 URL 与配置的有效清单则直接渲染，不再研究和调用模型。

非 GitHub 内容先生成 `EditorialOpportunity`（为何现在、为何观众在意、选中理由、待验证扩展维度）和 `ContextGraph`，再生成 `AttentionStrategy + DirectorBrief + subjects/context_events/evidence_shots`。导演简报明确观点、观众悬念、情绪强度和事件→背景→证据→影响→回报；上下文必须调查，但只有改变事件理解的节点才会进入成片。X 快报还会在作者最近时间线中按主题重合度找一条更早的 setup；人物流动则强制增加一次原组织历史变动查询，并按发布时间过滤同日重复报道，避免把三家媒体报道同一件事误写成趋势。

观众文案采用独立的低成本语义审稿模型，不用针对单条新闻维护术语替换或内容正则。Writer 完稿后，critic 必须逐字段抽取“人物—动作—对象—接收者”和确定性，并对照证据检查实体关系、因果强度、技术名词具体度、中文自然度与无旁白可读性；失败项以结构化 issue 返回给 writer 做一次字段级修复。确定性程序只校验 Schema、证据 ID、镜头顺序、时长和素材可渲染性。配置允许时 critic 与 writer 使用不同模型，避免模型自审直接放行。

模型/产品视频的固定标题必须保留具体模型或产品名；排版层会通过缩小字号、增加行数和抬高标题栏适配长标题，不允许为“塞得下”而删除主语。面向 vibe coder 的短视频每条最多解释一个真正难懂、且会影响结论的专业指标：标题中的术语优先，否则从全片选理解门槛最高的一项。上下文窗口、吞吐量、API、Token 价格等开发者常用词不占解释位。

外链官网中的团队合影、产品截图、架构图和 Benchmark 图会被下载并归档为一等 `web:source_image` 证据；与故事主体直接相关的高价值图片优先进入对应镜头，同一张图片在快报中不会被重复当作背景。每个清单还保存 `editorial_evidence_coverage`：列出已研究且进入成片的证据，以及因重复背景或时长预算被舍弃的证据，避免“导演知道、观众没看见”。派生文字卡不绘制无语义的黄色装饰线；黄色只用于真实来源中的重点框与相邻翻译。

`trial`、`boundary` 等编辑语义与底层 Scene 素材类型在 Schema 中物理分离；非 GitHub 模型甚至不再返回 `kind` 和时长。确定性编译器根据引用证据和 `visual_family/retention_job` 生成完整原帖、引用帖、真实官网/图表/代码或 evidence-backed 节奏卡，并统一安排 Flash 的 1.3–2.8 秒节奏。文档页只能证明产品可用及其能力，除非原文明确写出 release/launch/announcement，否则不能被升级成“正式发布”。

Flash 门禁会拒绝：相邻事实近似重复、把同一页换标签冒充视觉变化、品牌功能名机械直译、无来源的 API/成本/配额/无限制推断、把员工账号写成公司官方账号、引用帖未说明此前/随后关系、人物流动有可靠历史背景却未在画面使用，以及用“机制未知/关注后续”浪费结尾。完整外文 X 帖保持一张卡，并在原文旁显示 40–120 字的技术中文释义；固定底栏超过 62 个中文视觉字符时会在渲染前做一次低 Token 的字段压缩，禁止省略号截断。

模型层支持 OpenAI 兼容接口：DeepSeek 默认使用便宜的 `deepseek-chat`（可用 `DEEPSEEK_MODEL` 覆盖），读取 `DEEPSEEK_API_KEY`；Kimi 读取 `KIMI_API_KEY` 或 `MOONSHOT_API_KEY`，并要求显式传入 `--model`（或设置 `KIMI_MODEL`）。例如：

```bash
video-factory generate-story examples/moneyprinterturbo_story_packet.json \
  --provider deepseek --out workspace/manifests/mpt-story.json
```

只有需要自动打开帖子中已有的外链时才加 `--allow-linked-fetch`。备用模型不会默认启用；需要时显式配置，例如 `--fallback-provider kimi --fallback-model <account-model>`。普通故事正常仍是“规划 + 生成”两次；单 URL GitHub 工厂允许一次或两次完整修复，并额外保留一次低 Token 的可见字段修复。

GitHub 背景采集采用 README 内生规则：程序只打开 README 明确链接的 `vendor-notes`、background/reference 等仓库文档，再把其中标记为官方来源的链接交给有限研究工具。官方帮助中心若拒绝普通 HTTP 客户端，会通过只读文本代理抓取，但证据 URL 始终保留原始官网。X 帖子的研究范围更宽，可以围绕帖子中的人物或公司查证近期相关事件；这两类内容不会共用同一套发散策略。

设置 `OPENROUTER_API_KEY` 后，`generate --provider auto` 会在每天第一条任务前同时读取 OpenRouter 的官方 Models API 和 [`discount=true` 模型页](https://openrouter.ai/models?discount=true)，并缓存当天快照。折扣页只用于发现值得报道的价格事件，不再影响生产模型路由；故事、独立审稿、翻译和视觉任务分别通过输入模态、上下文、结构化输出和 Artificial Analysis intelligence 门槛后，始终按一次典型任务的有效成本选择最便宜的合格模型。更强模型只在便宜模型耗尽有界修复后升级。请求中的 OpenRouter provider router 仍优先选择低价 endpoint。可用环境变量：

- `OPENROUTER_TEXT_MODELS`、`OPENROUTER_VISION_MODELS`：逗号分隔的人工准入名单；
- `OPENROUTER_MIN_INTELLIGENCE`：未设置名单时的最低 intelligence index；故事/审稿默认 55、视觉默认 45、翻译默认 30；
- `OPENROUTER_DATA_COLLECTION=deny`、`OPENROUTER_ZDR=1`：供应商数据策略。

资源发现还把 OpenRouter 当作独立的两小时渠道，但它不是“折扣播报”。程序先从折扣页取得候选，再读取每个模型的 endpoint API，以最近 30 分钟可用率不低于 99% 的实际供应商价格计算一次典型开发任务（18K 输入 + 4K 输出）。只有以下价格异常之一成立、且 `temptation_score >= 70` 才会进入视频：可靠线路比模型原厂谷时价至少便宜 50%；折扣至少 75%；或发布 21 天内的新模型折扣至少 50%。10%–20% 的日常促销只保存候选并标记 `promotion_not_compelling`，不会自动生成。

DeepSeek 比价会同时保存其[官方价格页](https://api-docs.deepseek.com/quick_start/pricing)作为主证据，并分别计算原厂谷时、峰时和 OpenRouter 指定 endpoint 的成本；标题和首镜必须显示供应商、抓取时间与工作负载假设，不能把低价偷换成能力推荐。阈值均可在 `examples/resource_discovery.json` 的 `openrouter.settings` 中调整。

GitHub README 中只有位于 Architecture、Benchmark、Performance、Workflow 等相关段落的图才会进入多模态分析；badge、赞助图和 Star History 会被排除。程序把 README 文本和最多三张高价值图片一起交给通过同一价格/质量门的 OpenRouter 多模态模型，输出图表中文释义、它能证明什么以及不能证明什么。该结果只用于编辑理解，不能成为浏览器锚点或钩子的唯一事实来源。没有 `OPENROUTER_API_KEY` 时自动回退 DeepSeek 文本路径，并在 `result.json` 记录原因。

技术翻译提示已固定为 AI、软件工程、安全与 GitHub 上下文；`hygiene` 不再机械译成“卫生”，`honestly certify/guarantee` 不再译成“诚实保证”。原文本身含中文时，执行层不会重复贴一遍中文“翻译”。

GitHub 成片的真实浏览器路径固定为：仓库首页与文件树 → README 顶部（先讲项目是什么）→ 两个有价值的真实模块。不会把 README 从头滚到尾当成成片；完整遍历另存为审核证据资产，供编辑复查。

GitHub 重点同时保留内容证据 `target` 和录制锚点 `browser_target`。普通文字二者相同；代码示例则用代码上方的真实说明句画黄色框，让命令完整留在框旁，避免 GitHub 的复制按钮被误标为重点。录制后的视觉门按单个黄色连通框验收，零散黄色像素不能再让图标误判通过。

黄色框使用文字外侧 8px 的 outline，不覆盖字形。录制质检会把每个框的真实坐标写入 capture sidecar；中文释义再按该坐标贴到原文下方，底部空间不足时自动移到原文上方。释义最长 44 字，避免遮住主要证据画面。

安全研究、漏洞、私密数据和绕过类证据会进入人工审核：可以报道披露、影响、修复与防护，不能自动生成复现步骤、载荷或实操演示。

背景音乐的 `music_license_status` 只有在 `verified / licensed / original / royalty_free_verified` 且同时写入 `license_records` 时才通过发布门。MPT 可以为内部样片合成未核验音乐，但这类成片不能进入发布准备。

旧的 `WeChatVideoAccountUploader` 仍保留“停在最终发布按钮前”的单平台能力。新的批次发布层将质量门、审批摘要、幂等状态和审计记录留在 Video Factory 内部，并通过固定版本、隔离安装的 `social-auto-upload` CLI 执行四个平台提交。任一账号预检失败时整批不会开始提交；提交进程已经启动后若结果不明确，会标记为 `uncertain`，禁止自动重试。

## 多平台发布

发布后端不会作为本项目 Python 依赖直接导入。先显式安装固定版本；安装器优先复用本机 Chrome，没有可用 Chrome 时才安装隔离的 Chromium 运行时：

```bash
video-factory --workspace workspace publisher setup
video-factory --workspace workspace publisher login tencent --account main
video-factory --workspace workspace publisher login douyin --account main
video-factory --workspace workspace publisher login xiaohongshu --account main
video-factory --workspace workspace publisher login bilibili --account main
```

默认隔离目录为 `~/.video-factory/social-auto-upload`，可通过 `VIDEO_FACTORY_SAU_HOME` 覆盖。Cookie、浏览器状态和 Bilibili 运行时不会写入项目工作区。

如需为忽略 `LOCAL_CHROME_PATH` 的上游平台强制准备隔离 Chromium，可设置 `VIDEO_FACTORY_INSTALL_MANAGED_CHROMIUM=1` 后重新运行 `publisher setup`。上游旧 uploader 的未声明 `playwright` 导入会在固定源码副本中确定性映射到已声明的 `patchright`，补丁文件清单写入安装元数据。

使用 [examples/publish_targets.json](examples/publish_targets.json) 创建批次后，先审核输出中的视频、文案、账号、发布时间和平台参数，再由审核人显式批准并执行：

```bash
video-factory --workspace workspace publish-create workspace/manifests/story.json --spec examples/publish_targets.json
video-factory --workspace workspace publish-approve <batch-id> --actor editor@example.com
video-factory --workspace workspace publish-run <batch-id>
video-factory --workspace workspace publish-status <batch-id>
```

`publish-retry` 只接受 `failed_pre_submit` 平台；`submitted` 和 `uncertain` 永远不会自动重试。实际账号上线前，应先用测试账号完成逐平台灰度验证。

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
