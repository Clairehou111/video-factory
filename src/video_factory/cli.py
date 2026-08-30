from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from .media import probe_video, validate_wechat_mp4
from .quality import is_publishable, validate_manifest
from .serde import load_manifest
from .storage import Workspace
from .ingest import GitHubIngestor, TwitterCliIngestor, WebPageIngestor
from .links import ExternalLinkResolver
from .director import StoryboardDirector
from .llm import OpenAICompatibleStoryWriter, LLMSettings, StoryDraftError, packet_from_json
from .agent import (
    AgentBudget, ArchivedEvidenceTool, BoundedContentAgent, ContentAgentError,
    LinkedSourceResearchTool,
)
from .models import ContentType, InformationRenderProfile, TopicType, now_iso
from .serde import load_collection_manifest
from .youtube import (
    DiscoveryConfig, YouTubeCollectionRenderer, YouTubeDiscoveryService, validate_collection,
)
from .youtube_runtime import ManagedYouTubeRuntime
from .writer import StoryWriterPacket
from .webcapture import WebScrollVideoAdapter, WebScrollVideoSettings
from .compositor import compose_information_frame, overlay_fixed_footer, render_fixed_footer
from .mpt import MPTAssemblyAdapter, MPTSettings
from .publish import (
    PublishBatchService, PublishPlatform, SocialAutoUploadBackend,
    create_publish_batch, targets_from_spec,
)
from .collection_publish import (
    CollectionPublishBatch, CollectionPublishBatchService, create_collection_publish_batch,
)
from .factory import GenerateOptions, VideoFactory
from .discovery import ChannelConfig, DiscoveryChannel, ResourceDiscoveryConfig, ResourceDiscoveryService
from .automation import (
    AutomationAuditService, AutomationPolicy, DiscoveryPublishBridge,
    PipelinePublishConfig, is_pipeline_lock_collision, pipeline_lock,
)
from .dashboard import serve_dashboard


def _workspace(path: str) -> Workspace:
    workspace = Workspace(Path(path).resolve())
    workspace.initialize()
    return workspace


def main() -> None:
    parser = argparse.ArgumentParser(prog="video-factory")
    parser.add_argument("--workspace", default="workspace", help="本地资产与清单目录")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("init", help="初始化本地工作区")
    archive = subcommands.add_parser("archive-asset", help="归档并哈希一份原始资产")
    archive.add_argument("file")
    archive.add_argument("--category", required=True)
    archive.add_argument("--name")
    validate = subcommands.add_parser("validate", help="验证一个 RenderManifest JSON")
    validate.add_argument("manifest")
    validate_collection_command = subcommands.add_parser("validate-collection", help="验证一个 YouTube VideoCollectionManifest JSON")
    validate_collection_command.add_argument("manifest")
    rights_review = subcommands.add_parser("review-collection-rights", help="记录第三方源视频的人工复用依据审核")
    rights_review.add_argument("manifest")
    rights_review.add_argument("--actor", required=True)
    rights_review.add_argument("--status", required=True, choices=["permission_granted", "licensed", "educational_excerpt", "reviewed"])
    rights_review.add_argument("--basis", default="educational_noncommercial")
    rights_review.add_argument("--notes", default="")
    repair_collection_audio = subcommands.add_parser(
        "repair-collection-audio",
        help="只重渲染音轨缺失或实际静音的 YouTube 合集成片",
    )
    repair_collection_audio.add_argument("manifest")
    inspect_video = subcommands.add_parser("inspect-video", help="用 ffprobe 验收最终 MP4")
    inspect_video.add_argument("file")
    inspect_video.add_argument("--max-duration", type=float)
    inspect_video.add_argument("--visual-track", action="store_true", help="静音视觉轨：不要求 AAC，最终成片不要使用此选项")
    tweet = subcommands.add_parser("ingest-twitter", help="导入 twitter-cli 结构化采集 JSON")
    tweet.add_argument("capture")
    tweet.add_argument("--resolve-links", action="store_true", help="只跟随外链重定向以路由到原始来源")
    github = subcommands.add_parser("ingest-github", help="导入 GitHub API 元数据和 README")
    github.add_argument("repo_json")
    github.add_argument("readme")
    web = subcommands.add_parser("ingest-web", help="导入已采集的官网或原始网页，作为帖子外链的延伸证据")
    web.add_argument("url")
    web.add_argument("content")
    web.add_argument("--title", required=True)
    web.add_argument("--parent-candidate")
    generate_story = subcommands.add_parser("generate-story", help="调用 Kimi/DeepSeek 生成受证据约束的分镜 JSON")
    generate_story.add_argument("packet")
    generate_story.add_argument("--provider", choices=["deepseek", "kimi", "openrouter"], required=True)
    generate_story.add_argument("--model")
    generate_story.add_argument("--out", required=True)
    generate_story.add_argument("--max-llm-calls", type=int, default=4)
    generate_story.add_argument("--max-research-sources", type=int, default=3)
    generate_story.add_argument("--allow-linked-fetch", action="store_true", help="允许 Agent 打开候选中已有的外链并先归档为证据")
    generate_story.add_argument("--fallback-provider", choices=["deepseek", "kimi", "openrouter"], help="仅在便宜模型与一次修复均失败后启用")
    generate_story.add_argument("--fallback-model")
    packet = subcommands.add_parser("create-story-packet", help="从已归档候选与证据创建模型写作输入")
    packet.add_argument("--candidate", required=True)
    packet.add_argument("--topic", choices=[item.value for item in TopicType], required=True)
    packet.add_argument("--format", choices=[item.value for item in ContentType], required=True)
    packet.add_argument("--duration", type=float, required=True)
    packet.add_argument("--out", required=True)
    packet.add_argument("--include", action="append", default=[], help="以当前候选为入口，追加一个已归档的外链候选证据")
    capture_github = subcommands.add_parser("capture-github", help="录制 GitHub 成片：仓库首页、README 顶部与两个关键模块")
    capture_github.add_argument("manifest")
    capture_github.add_argument("--out", required=True)
    capture_github.add_argument("--frames", required=True)
    capture_github_audit = subcommands.add_parser("capture-github-audit", help="归档完整 README 连续滚动，不作为最终成片主画面")
    capture_github_audit.add_argument("manifest")
    capture_github_audit.add_argument("--out", required=True)
    capture_github_audit.add_argument("--frames", required=True)
    capture_linked = subcommands.add_parser("capture-linked-web", help="从 X 原帖开始录制，再打开其指向的原始官网")
    capture_linked.add_argument("manifest")
    capture_linked.add_argument("--primary-url", required=True)
    capture_linked.add_argument("--anchor", action="append", required=True, help="官网中真实可见的标题/锚点；可重复传入")
    capture_linked.add_argument("--out", required=True)
    capture_linked.add_argument("--frames", required=True)
    footer = subcommands.add_parser("apply-footer", help="兼容命令：仅将固定结论叠加到原生视觉轨")
    footer.add_argument("manifest")
    footer.add_argument("visual_track")
    footer.add_argument("--out", required=True)
    frame = subcommands.add_parser("frame-video", help="渲染固定上下栏，保持中部真实证据画面可读")
    frame.add_argument("manifest")
    frame.add_argument("visual_track")
    frame.add_argument("--out", required=True)
    frame.add_argument("--render-profile", choices=[item.value for item in InformationRenderProfile])
    assemble = subcommands.add_parser("assemble-mpt", help="用 MPT 将无旁白视觉轨与背景音乐编码为最终 MP4")
    assemble.add_argument("manifest")
    assemble.add_argument("visual_track")
    assemble.add_argument("--out", required=True)
    assemble.add_argument("--task-id")
    publisher = subcommands.add_parser("publisher", help="管理隔离的多平台发布后端与账号")
    publisher_actions = publisher.add_subparsers(dest="publisher_action", required=True)
    publisher_actions.add_parser("setup", help="安装固定版本的 social-auto-upload 与 Chromium")
    publisher_login = publisher_actions.add_parser("login", help="交互式登录一个发布账号")
    publisher_login.add_argument("platform", choices=[item.value for item in PublishPlatform])
    publisher_login.add_argument("--account", required=True)
    publisher_login.add_argument("--headless", action="store_true", help="不显示浏览器；首次登录通常不建议")
    publisher_check = publisher_actions.add_parser("check", help="检查一个账号的登录状态")
    publisher_check.add_argument("platform", choices=[item.value for item in PublishPlatform])
    publisher_check.add_argument("--account", required=True)
    publish_create = subcommands.add_parser("publish-create", help="通过质量门后创建一次多平台待审批批次")
    publish_create.add_argument("manifest")
    publish_create.add_argument("--spec", required=True, help="多平台标题、账号与平台参数 JSON")
    publish_collection_create = subcommands.add_parser("publish-collection-create", help="为 YouTube 多成片合集创建一次绑定全部文件和顺序的待审批批次")
    publish_collection_create.add_argument("manifest")
    publish_collection_create.add_argument("--spec", required=True)
    publish_approve = subcommands.add_parser("publish-approve", help="对批次中的视频、文案和目标做一次人工审批")
    publish_approve.add_argument("batch")
    publish_approve.add_argument("--actor", required=True)
    publish_run = subcommands.add_parser("publish-run", help="预检账号并顺序提交一个已审批批次")
    publish_run.add_argument("batch")
    publish_status = subcommands.add_parser("publish-status", help="查看批次及各平台状态")
    publish_status.add_argument("batch")
    publish_retry = subcommands.add_parser("publish-retry", help="仅重试明确未进入提交阶段的平台")
    publish_retry.add_argument("batch")
    publish_retry.add_argument("--platform", required=True, choices=[item.value for item in PublishPlatform])
    publish_reconcile = subcommands.add_parser(
        "publish-confirm-pre-submit-failure",
        help="凭明确的上传前鉴权拒绝证据，将不确定状态恢复为可安全重试",
    )
    publish_reconcile.add_argument("batch")
    publish_reconcile.add_argument("--platform", required=True, choices=[item.value for item in PublishPlatform])
    publish_reconcile.add_argument("--actor", required=True)
    publish_collection_retry = subcommands.add_parser("publish-collection-retry-link", help="只重试已上传 Bilibili 视频的合集关联，绝不重复上传")
    publish_collection_retry.add_argument("batch")
    publish_collection_retry.add_argument("--item", required=True)
    publish_collection_reconcile = subcommands.add_parser(
        "publish-collection-confirm-pre-submit-failure",
        help="将有明确 Bilibili -101 证据的不确定项确认成可安全重试的上传前失败",
    )
    publish_collection_reconcile.add_argument("batch")
    publish_collection_reconcile.add_argument("--item", required=True)
    publish_collection_reconcile.add_argument("--actor", required=True)
    subcommands.add_parser("publish-policy", help="显示发布安全边界")
    youtube_runtime = subcommands.add_parser(
        "youtube-runtime",
        help="管理固定版本的 1080p YouTube 下载运行时（mweb + EJS + PO token provider）",
    )
    youtube_runtime_actions = youtube_runtime.add_subparsers(dest="youtube_runtime_action", required=True)
    youtube_runtime_actions.add_parser("setup", help="安装或更新固定版本运行时")
    youtube_runtime_actions.add_parser("status", help="检查运行时和本地 PO token provider")
    discover = subcommands.add_parser("discover", help="搜索所有到期来源渠道，每渠道最多自动生成一条合格视频")
    discover.add_argument("--config", help="多渠道搜索池、节奏和质量门配置 JSON")
    discover.add_argument("--channel", action="append", choices=[item.value for item in DiscoveryChannel], help="只运行指定渠道；可重复")
    discover.add_argument("--force", action="store_true", help="忽略所选渠道的 next_run_at")
    discover.add_argument("--provider", choices=["auto", "deepseek", "kimi", "openrouter"], default="auto")
    discover.add_argument("--model")
    pipeline = subcommands.add_parser(
        "pipeline", help="发现资源、生成成片，并按渠道规则创建待审核发布批次",
    )
    pipeline.add_argument("--config", help="多渠道搜索池、节奏和质量门配置 JSON")
    pipeline.add_argument("--publish-config", help="视频号与 Bilibili 账号、分区和标签配置 JSON")
    pipeline.add_argument("--channel", action="append", choices=[item.value for item in DiscoveryChannel], help="只运行指定渠道；可重复")
    pipeline.add_argument("--force", action="store_true", help="忽略所选渠道的 next_run_at")
    pipeline.add_argument("--provider", choices=["auto", "deepseek", "kimi", "openrouter"], default="auto")
    pipeline.add_argument("--model")
    pipeline.add_argument("--trial-days", type=int, default=7, help="本地监督运行期；默认 7 天")
    pipeline.add_argument("--notify", action="store_true", help="运行后发送 macOS 待审核通知")
    automation_status = subcommands.add_parser(
        "automation-status", help="查看最近一次（或指定一次）自动工厂运行审计",
    )
    automation_status.add_argument("--run", help="discovery run id；默认读取 latest")
    automation_status.add_argument("--notify", action="store_true", help="重发这次审计的 macOS 通知")
    dashboard = subcommands.add_parser("dashboard", help="启动本地成片审核与逐条发布 Dashboard")
    dashboard.add_argument("--host", default="127.0.0.1", help="仅允许 loopback 地址")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--actor", default="claire", help="写入审批记录的审核人")
    discover_youtube = subcommands.add_parser("discover-youtube", help="按 48 小时节奏发现并最多生产一个高价值 YouTube 源视频")
    discover_youtube.add_argument("--config", help="YouTube 搜索池和质量门配置 JSON")
    discover_youtube.add_argument("--force", action="store_true", help="忽略 next_run_at，立即执行一次搜索")
    discover_youtube.add_argument("--select-only", action="store_true", help="只选择并记录候选，不启动生成")
    discover_youtube.add_argument("--provider", choices=["auto", "deepseek", "kimi", "openrouter"], default="auto")
    discover_youtube.add_argument("--model")
    discover_youtube.add_argument("--no-render", action="store_true")
    discovery_status = subcommands.add_parser("discovery-status", help="显示各来源渠道的搜索周期、历史和阻塞状态")
    discovery_status.add_argument("--channel", choices=[item.value for item in DiscoveryChannel])
    adopt = subcommands.add_parser("adopt", help="恢复或重试一个已通过来源质量门的候选")
    adopt.add_argument("candidate")
    adopt.add_argument("--config", help="多渠道配置 JSON")
    adopt.add_argument("--provider", choices=["auto", "deepseek", "kimi", "openrouter"], default="auto")
    adopt.add_argument("--model")
    discovery_skip = subcommands.add_parser("discovery-skip", help="人工跳过一个阻塞候选并释放该渠道")
    discovery_skip.add_argument("candidate")
    discovery_skip.add_argument("--reason", required=True)
    generate = subcommands.add_parser("generate", help="输入一个 URL，程序自动完成采集、策划、录屏、合成和验收")
    generate.add_argument("url")
    generate.add_argument("--provider", choices=["auto", "deepseek", "kimi", "openrouter"], default="auto")
    generate.add_argument("--model")
    generate.add_argument("--topic", choices=["auto", *[item.value for item in TopicType]], default="auto")
    generate.add_argument("--format", dest="content_format", choices=["auto", *[item.value for item in ContentType]], default="auto")
    generate.add_argument("--duration", "--duration-max", dest="duration", type=float)
    generate.add_argument("--research", choices=["on", "off"], default="on")
    generate.add_argument("--live-capture", action=argparse.BooleanOptionalAction, default=True)
    generate.add_argument("--no-render", action="store_true", help="只生成可验收的内容清单，不录屏和合成")
    generate.add_argument("--refresh-prices", action="store_true", help="忽略当日 OpenRouter 缓存并重新查价")
    generate.add_argument("--refresh", action="store_true", help="忽略该 URL 已缓存的采集与内容清单")
    generate.add_argument(
        "--render-profile", choices=[item.value for item in InformationRenderProfile],
        default=InformationRenderProfile.CLASSIC.value,
        help="classic 保持稳定样式；radar_v2 启用分层视口、聚光灯与微动态",
    )
    generate.add_argument("--youtube-media", help="YouTube web 媒体不可取时，显式提供已获准使用的本地源视频")
    generate.add_argument(
        "--youtube-subtitles",
        help="显式提供本地 YouTube json3 时间轴字幕，避免再次下载字幕",
    )
    generate.add_argument(
        "--youtube-translation-plan",
        help="复用已完成的 YouTube translation-plan.json，只修复规划并渲染",
    )
    generate.add_argument(
        "--youtube-editorial-mode",
        choices=["auto", "technical_coverage", "known_tech_interview_clip", "study"],
        default="auto",
        help="覆盖 YouTube 自动分类；正常自动工厂保持 auto",
    )
    rerender = subcommands.add_parser("rerender", help="复用现有内容清单，只重试确定性录屏、合成和验收")
    rerender.add_argument("manifest")
    args = parser.parse_args()
    workspace = _workspace(args.workspace)

    if args.command == "youtube-runtime":
        runtime = ManagedYouTubeRuntime()
        result = runtime.setup() if args.youtube_runtime_action == "setup" else runtime.status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.youtube_runtime_action == "status" and not (
            result.get("installed") and result.get("provider_ready")
            and result.get("version_pins_valid")
        ):
            sys.exit(1)
    elif args.command == "init":
        print(f"initialized {workspace.root}")
    elif args.command == "dashboard":
        serve_dashboard(workspace, args.host, args.port, args.actor)
    elif args.command == "discover":
        config = ResourceDiscoveryConfig.from_path(Path(args.config).resolve() if args.config else None)
        channels = [DiscoveryChannel(item) for item in args.channel] if args.channel else None
        result = ResourceDiscoveryService(workspace).run(
            config, scheduled=not args.force, channels=channels,
            provider=args.provider, model=args.model,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        if result.status == "failed":
            sys.exit(1)
    elif args.command == "pipeline":
        config = ResourceDiscoveryConfig.from_path(Path(args.config).resolve() if args.config else None)
        publish_config = PipelinePublishConfig.from_path(
            Path(args.publish_config).resolve() if args.publish_config else None,
        )
        channels = [DiscoveryChannel(item) for item in args.channel] if args.channel else None
        auditor = AutomationAuditService(workspace, AutomationPolicy(trial_days=args.trial_days))
        try:
            with pipeline_lock(workspace.root):
                result = ResourceDiscoveryService(workspace).run(
                    config, scheduled=not args.force, channels=channels,
                    provider=args.provider, model=args.model,
                )
                prepared = DiscoveryPublishBridge(workspace, publish_config).prepare(result)
                audit = auditor.record(result, prepared, args.provider, args.model)
                if args.notify:
                    auditor.notify(audit)
        except Exception as error:
            if is_pipeline_lock_collision(error):
                print(json.dumps({
                    "status": "already_running",
                    "detail": str(error),
                    "publication_boundary": "no second run started and nothing was published",
                }, ensure_ascii=False, indent=2))
                return
            audit = auditor.record_crash(error, args.provider, args.model)
            if args.notify:
                auditor.notify(audit)
            raise
        print(json.dumps({
            "discovery": result.to_dict(),
            "publish_queue": [item.to_dict() for item in prepared],
            "automation_audit": audit,
            "publication_boundary": "review and approve each batch before publish-run",
        }, ensure_ascii=False, indent=2))
        if result.status == "failed":
            sys.exit(1)
    elif args.command == "automation-status":
        auditor = AutomationAuditService(workspace)
        report = auditor.load(args.run)
        if args.notify:
            auditor.notify(report, force=True)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "discover-youtube":
        config = DiscoveryConfig.from_path(Path(args.config).resolve() if args.config else None)
        if args.select_only or args.no_render:
            service = YouTubeDiscoveryService(workspace)
            callback = None
            if not args.select_only:
                callback = lambda selected: VideoFactory(workspace).generate(selected.url, GenerateOptions(
                    provider=args.provider, model=args.model, render=False,
                ))
            result = service.run(config, scheduled=not args.force, on_selected=callback)
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            if result.status in {"configuration_blocked", "search_failed", "generation_failed"}:
                sys.exit(1)
        else:
            unified = ResourceDiscoveryConfig()
            for channel in DiscoveryChannel:
                unified.channels[channel].enabled = channel == DiscoveryChannel.YOUTUBE
            unified.channels[DiscoveryChannel.YOUTUBE] = ChannelConfig(
                enabled=True, cadence_hours=config.cadence_hours,
                lookback_hours=config.lookback_days * 24, minimum_score=config.minimum_score,
                max_candidates=max(config.results_per_query, config.metadata_probe_limit),
                probe_limit=config.metadata_probe_limit, queries=[], settings={
                    "minimum_duration_seconds": config.minimum_duration_seconds,
                    "maximum_duration_seconds": config.maximum_duration_seconds,
                    "results_per_query": config.results_per_query,
                    "metadata_probe_limit": config.metadata_probe_limit,
                    "timezone": config.timezone, "query_pools": config.query_pools,
                },
            )
            result = ResourceDiscoveryService(workspace).run(
                unified, scheduled=not args.force, channels=[DiscoveryChannel.YOUTUBE],
                provider=args.provider, model=args.model,
            )
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            if result.status == "failed":
                sys.exit(1)
    elif args.command == "discovery-status":
        channel = DiscoveryChannel(args.channel) if args.channel else None
        print(json.dumps(ResourceDiscoveryService(workspace).status(channel), ensure_ascii=False, indent=2))
    elif args.command == "adopt":
        config = ResourceDiscoveryConfig.from_path(Path(args.config).resolve() if args.config else None)
        result = ResourceDiscoveryService(workspace).adopt_candidate(
            args.candidate, config, provider=args.provider, model=args.model,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "generated":
            sys.exit(1)
    elif args.command == "discovery-skip":
        result = ResourceDiscoveryService(workspace).skip(args.candidate, args.reason)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "generate":
        result = VideoFactory(workspace).generate(args.url, GenerateOptions(
            provider=args.provider, model=args.model, duration=args.duration,
            render=not args.no_render, refresh_prices=args.refresh_prices,
            topic=None if args.topic == "auto" else TopicType(args.topic),
            content_type=None if args.content_format == "auto" else ContentType(args.content_format),
            research=args.research == "on", live_capture=args.live_capture,
            refresh=args.refresh,
            youtube_media=args.youtube_media,
            youtube_subtitles=args.youtube_subtitles,
            youtube_translation_plan=args.youtube_translation_plan,
            youtube_editorial_mode=args.youtube_editorial_mode,
            render_profile=args.render_profile,
        ))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "rerender":
        result = VideoFactory(workspace).rerender(Path(args.manifest))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "archive-asset":
        path, digest = workspace.archive_asset(Path(args.file), args.category, args.name)
        print(json.dumps({"path": path, "sha256": digest}, ensure_ascii=False))
    elif args.command == "validate":
        checks = validate_manifest(load_manifest(Path(args.manifest)), workspace.root)
        print(json.dumps([check.to_dict() for check in checks], ensure_ascii=False, indent=2))
        if not is_publishable(checks):
            sys.exit(1)
    elif args.command == "validate-collection":
        checks = validate_collection(load_collection_manifest(Path(args.manifest)), workspace.root)
        print(json.dumps([check.to_dict() for check in checks], ensure_ascii=False, indent=2))
        if not all(check.passed for check in checks):
            sys.exit(1)
    elif args.command == "review-collection-rights":
        path = Path(args.manifest).resolve()
        manifest = load_collection_manifest(path)
        manifest.rights_review.status = args.status
        manifest.rights_review.basis = args.basis
        manifest.rights_review.reviewed_by = args.actor.strip()
        manifest.rights_review.reviewed_at = now_iso()
        manifest.rights_review.notes = args.notes
        manifest.quality_checks = [
            check.to_dict() for check in validate_collection(manifest, workspace.root)
        ]
        workspace.save_collection_manifest(manifest)
        path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result_path = path.parent / "result.json"
        if result_path.is_file():
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            result_payload["checks"] = manifest.quality_checks
            result_payload["publishable"] = all(item["passed"] for item in manifest.quality_checks)
            result_path.write_text(
                json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
        print(json.dumps(manifest.rights_review.__dict__ if hasattr(manifest.rights_review, "__dict__") else {
            "status": manifest.rights_review.status, "basis": manifest.rights_review.basis,
            "reviewed_by": manifest.rights_review.reviewed_by, "reviewed_at": manifest.rights_review.reviewed_at,
            "notes": manifest.rights_review.notes,
        }, ensure_ascii=False, indent=2))
    elif args.command == "repair-collection-audio":
        path = Path(args.manifest).resolve()
        manifest = load_collection_manifest(path)
        repaired = YouTubeCollectionRenderer(workspace).repair_silent_audio(manifest)
        manifest.quality_checks = [
            check.to_dict() for check in validate_collection(manifest, workspace.root)
        ]
        workspace.save_collection_manifest(manifest)
        path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result_path = path.parent / "result.json"
        if result_path.is_file():
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            result_payload["checks"] = manifest.quality_checks
            result_payload["publishable"] = all(item["passed"] for item in manifest.quality_checks)
            result_path.write_text(
                json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
        failed = [item for item in manifest.quality_checks if not item["passed"]]
        print(json.dumps({
            "repaired": repaired, "failed_checks": failed,
        }, ensure_ascii=False, indent=2))
        if failed:
            sys.exit(1)
    elif args.command == "inspect-video":
        checks = validate_wechat_mp4(probe_video(Path(args.file)), args.max_duration, require_audio=not args.visual_track)
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        if not all(check["passed"] for check in checks):
            sys.exit(1)
    elif args.command == "ingest-twitter":
        resolver = ExternalLinkResolver().resolve if args.resolve_links else None
        result = TwitterCliIngestor().ingest(Path(args.capture), workspace, (lambda url: resolver(url).resolved_url) if resolver else None)
        print(json.dumps({"candidate": result.candidate.id, "evidence": [item.id for item in result.evidence], "linked_candidates": [item.id for item in result.linked_candidates]}, ensure_ascii=False, indent=2))
    elif args.command == "ingest-github":
        result = GitHubIngestor().ingest(Path(args.repo_json), Path(args.readme), workspace)
        print(json.dumps({"candidate": result.candidate.id, "evidence": [item.id for item in result.evidence]}, ensure_ascii=False, indent=2))
    elif args.command == "ingest-web":
        result = WebPageIngestor().ingest(args.url, Path(args.content), workspace, args.title, args.parent_candidate)
        print(json.dumps({"candidate": result.candidate.id, "evidence": [item.id for item in result.evidence]}, ensure_ascii=False, indent=2))
    elif args.command == "generate-story":
        packet = packet_from_json(Path(args.packet))
        writer = OpenAICompatibleStoryWriter(LLMSettings.from_environment(args.provider, args.model))
        fallback = None
        if args.fallback_provider:
            fallback = OpenAICompatibleStoryWriter(LLMSettings.from_environment(args.fallback_provider, args.fallback_model))
        research_tool = LinkedSourceResearchTool(workspace) if args.allow_linked_fetch else ArchivedEvidenceTool()
        agent = BoundedContentAgent(
            writer, research_tool=research_tool, escalation=fallback,
            budget=AgentBudget(
                max_llm_calls=args.max_llm_calls,
                max_research_sources=args.max_research_sources,
                max_repairs=1,
                max_escalations=1 if fallback else 0,
            ),
        )
        try:
            result = agent.run(packet)
            manifest = result.manifest
        except ContentAgentError as error:
            error_path = Path(args.out).with_suffix(".agent-error.json")
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(json.dumps({"error": str(error), "trace": error.trace}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"content agent failed; trace written to {error_path}") from error
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(output)
    elif args.command == "create-story-packet":
        candidate = workspace.load_candidate(args.candidate)
        included = [workspace.load_candidate(identifier) for identifier in args.include]
        invalid_parents = [item.id for item in included if item.parent_candidate_id != candidate.id]
        if invalid_parents:
            raise ValueError(f"included candidates must be linked from {candidate.id}: {', '.join(invalid_parents)}")
        combined_urls = list(dict.fromkeys([*candidate.linked_sources, *(item.source_url for item in included)]))
        packet_candidate = replace(candidate, linked_sources=combined_urls)
        packet = StoryWriterPacket(
            packet_candidate, workspace.evidence_for_candidates([candidate.id, *(item.id for item in included)]),
            TopicType(args.topic), ContentType(args.format), args.duration,
        )
        output = Path(args.out)
        output.write_text(json.dumps({
            "candidate": packet.candidate.__dict__ if hasattr(packet.candidate, "__dict__") else {
                "id": packet.candidate.id, "source_type": packet.candidate.source_type, "source_url": packet.candidate.source_url,
                "title": packet.candidate.title, "author": packet.candidate.author, "linked_sources": packet.candidate.linked_sources,
                "published_at": packet.candidate.published_at, "dedupe_key": packet.candidate.dedupe_key,
                "parent_candidate_id": packet.candidate.parent_candidate_id, "metadata": packet.candidate.metadata, "captured_at": packet.candidate.captured_at,
            },
            "evidence": [item.__dict__ if hasattr(item, "__dict__") else {
                "id": item.id, "candidate_id": item.candidate_id, "url": item.url, "quote": item.quote,
                "source_kind": item.source_kind, "captured_asset": item.captured_asset, "captured_at": item.captured_at,
                "sha256": item.sha256, "notes": item.notes,
            } for item in packet.evidence],
            "topic_type": packet.topic_type, "content_type": packet.content_type, "target_duration": packet.target_duration,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(output)
    elif args.command in {"capture-github", "capture-github-audit"}:
        manifest = load_manifest(Path(args.manifest))
        if manifest.topic_type != TopicType.GITHUB_PROJECT:
            raise ValueError("capture-github requires a github_project manifest")
        repo_url = next((url for url in manifest.source_urls if "github.com/" in url), None)
        if not repo_url:
            raise ValueError("GitHub manifest has no repository URL")
        if args.command == "capture-github":
            if not manifest.github_brief:
                raise ValueError("capture-github requires the evidence-ranked github_brief; regenerate legacy manifests")
            request = WebScrollVideoAdapter.github_story_request(
                repo_url, manifest.github_brief, Path(args.out), Path(args.frames), manifest.duration,
            )
        else:
            if not manifest.github_walkthrough:
                raise ValueError("capture-github-audit requires the legacy audit traversal indexes")
            request = WebScrollVideoAdapter.github_audit_request(
                repo_url, manifest.github_walkthrough, Path(args.out), Path(args.frames), manifest.duration,
            )
        print(WebScrollVideoAdapter(WebScrollVideoSettings.from_environment()).capture(request))
    elif args.command == "capture-linked-web":
        manifest = load_manifest(Path(args.manifest))
        post_url = next((url for url in manifest.source_urls if "x.com/" in url), None)
        if not post_url:
            raise ValueError("capture-linked-web requires an X post as the root source")
        request = WebScrollVideoAdapter.linked_post_request(
            post_url, args.primary_url, args.anchor, Path(args.out), Path(args.frames), manifest.duration,
        )
        print(WebScrollVideoAdapter(WebScrollVideoSettings.from_environment()).capture(request))
    elif args.command == "apply-footer":
        manifest = load_manifest(Path(args.manifest))
        output = Path(args.out)
        footer_png = output.with_suffix(".footer.png")
        render_fixed_footer(manifest.fixed_footer or "", footer_png)
        print(overlay_fixed_footer(Path(args.visual_track), footer_png, output))
    elif args.command == "frame-video":
        manifest = load_manifest(Path(args.manifest))
        if args.render_profile:
            manifest.render_profile = args.render_profile
        print(compose_information_frame(manifest, Path(args.visual_track), Path(args.out)))
    elif args.command == "assemble-mpt":
        manifest = load_manifest(Path(args.manifest))
        result = MPTAssemblyAdapter(MPTSettings.from_environment()).assemble(manifest, Path(args.visual_track), args.task_id)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result, output)
        print(output)
    elif args.command == "publisher":
        backend = SocialAutoUploadBackend()
        if args.publisher_action == "setup":
            print(json.dumps(backend.setup(), ensure_ascii=False, indent=2))
        elif args.publisher_action == "login":
            result = backend.login_account(PublishPlatform(args.platform), args.account, headless=args.headless)
            if not result.succeeded:
                raise RuntimeError(result.stderr or result.stdout or "publisher login failed")
            print(json.dumps({"platform": args.platform, "account": args.account, "status": "login_completed"}, ensure_ascii=False))
        elif args.publisher_action == "check":
            result = backend.check_login(PublishPlatform(args.platform), args.account)
            print(json.dumps({"platform": args.platform, "account": args.account, "valid": result.succeeded}, ensure_ascii=False))
            if not result.succeeded:
                sys.exit(1)
    elif args.command == "publish-create":
        manifest = load_manifest(Path(args.manifest))
        spec_path = Path(args.spec).resolve()
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        targets = targets_from_spec(spec, spec_path.parent)
        batch = create_publish_batch(manifest, targets, workspace.root)
        workspace.save_publish_batch(batch)
        print(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2))
        if batch.state.value == "blocked":
            sys.exit(1)
    elif args.command == "publish-collection-create":
        manifest = load_collection_manifest(Path(args.manifest))
        spec_path = Path(args.spec).resolve()
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        batch = create_collection_publish_batch(manifest, spec, workspace.root)
        workspace.save_publish_batch(batch)
        print(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2))
        if batch.state.value == "blocked":
            sys.exit(1)
    elif args.command == "publish-approve":
        batch = workspace.load_publish_batch(args.batch)
        service = CollectionPublishBatchService(workspace, SocialAutoUploadBackend()) if isinstance(batch, CollectionPublishBatch) else PublishBatchService(workspace, SocialAutoUploadBackend())
        service.approve(batch, args.actor)
        print(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "publish-run":
        batch = workspace.load_publish_batch(args.batch)
        service = CollectionPublishBatchService(workspace, SocialAutoUploadBackend()) if isinstance(batch, CollectionPublishBatch) else PublishBatchService(workspace, SocialAutoUploadBackend())
        service.run(batch)
        print(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2))
        if batch.state.value not in {"succeeded", "partial_success"}:
            sys.exit(1)
    elif args.command == "publish-status":
        print(json.dumps(workspace.load_publish_batch(args.batch).to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "publish-retry":
        batch = workspace.load_publish_batch(args.batch)
        if isinstance(batch, CollectionPublishBatch):
            raise ValueError("collection batches retry only collection association via publish-collection-retry-link")
        PublishBatchService(workspace, SocialAutoUploadBackend()).retry(batch, PublishPlatform(args.platform))
        print(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2))
        if batch.state.value not in {"succeeded", "partial_success"}:
            sys.exit(1)
    elif args.command == "publish-confirm-pre-submit-failure":
        batch = workspace.load_publish_batch(args.batch)
        if isinstance(batch, CollectionPublishBatch):
            raise ValueError("collection batches use publish-collection-confirm-pre-submit-failure")
        PublishBatchService(workspace, SocialAutoUploadBackend()).confirm_pre_submit_failure(
            batch, PublishPlatform(args.platform), args.actor,
        )
        print(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "publish-collection-retry-link":
        batch = workspace.load_publish_batch(args.batch)
        if not isinstance(batch, CollectionPublishBatch):
            raise ValueError("publish-collection-retry-link requires a collection batch")
        CollectionPublishBatchService(workspace, SocialAutoUploadBackend()).retry_collection_link(batch, args.item)
        print(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2))
        if batch.state.value not in {"succeeded", "partial_success"}:
            sys.exit(1)
    elif args.command == "publish-collection-confirm-pre-submit-failure":
        batch = workspace.load_publish_batch(args.batch)
        if not isinstance(batch, CollectionPublishBatch):
            raise ValueError("publish-collection-confirm-pre-submit-failure requires a collection batch")
        CollectionPublishBatchService(workspace, SocialAutoUploadBackend()).confirm_pre_submit_auth_rejection(
            batch, args.item, args.actor,
        )
        print(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "publish-policy":
        print(json.dumps({
            "quality_gate_required": True,
            "batch_approval_required": True,
            "automatic_submit_after_batch_approval": True,
            "supported_platforms": [item.value for item in PublishPlatform],
            "automatic_retry_after_submit_started": False,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
