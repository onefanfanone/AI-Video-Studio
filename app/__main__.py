from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import (
    BuildError,
    build_project,
    create_voice_audition,
    review_asset_task,
    validate_run,
)
from .script_workbench import ScriptWorkbenchError, run_workbench_server
from .studio_console import StudioConsoleError, run_studio_console
from .studio_profiles import ProfileError
from .studio_settings import StudioSettingsError
from .studio_migrate import MigrationError, migrate_workspace


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI 视频自动粗剪")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="构建一个实验项目")
    build.add_argument("--project", required=True, help="项目名称或项目目录")
    build.add_argument("--draft-root", help="覆盖剪映草稿目录，主要用于隔离测试")
    build.add_argument("--skip-draft", action="store_true", help="只生成 MP4，不生成剪映草稿")
    build.add_argument("--open-output", action="store_true", help="成功后打开输出目录")
    build.add_argument(
        "--resume",
        help="续跑 task id；latest 续跑最近任务，auto 自动续跑兼容任务或新建",
    )
    build.add_argument(
        "--visual-mode",
        choices=("sourced", "local"),
        help="覆盖项目的画面供应模式",
    )

    assets = subparsers.add_parser("assets", help="素材供应工具")
    asset_commands = assets.add_subparsers(dest="asset_command", required=True)
    review = asset_commands.add_parser("review", help="重新打开等待中的素材审核")
    review.add_argument("--resume", default="latest", help="task id 或 latest")

    voices = subparsers.add_parser("voices", help="配音工具")
    voice_commands = voices.add_subparsers(dest="voice_command", required=True)
    audition = voice_commands.add_parser("audition", help="生成六种免费音色试听")
    audition.add_argument("--project", required=True, help="项目名称或项目目录")
    audition.add_argument("--open-output", action="store_true", help="成功后打开试听目录")

    scripts = subparsers.add_parser("scripts", help="脚本工作台")
    script_commands = scripts.add_subparsers(dest="script_command", required=True)
    workbench = script_commands.add_parser("workbench", help="打开本机脚本工作台")
    workbench.add_argument(
        "--resume",
        choices=("latest",),
        help="直接恢复最近一个未完成脚本草稿",
    )
    workbench.add_argument("--open-output", action="store_true", help="构建完成后打开输出目录")

    studio = subparsers.add_parser("studio", help="打开 AI-Video Studio 本机控制台")
    studio.add_argument("--open-output", action="store_true", help="构建完成后打开输出目录")
    studio.add_argument("--no-browser", action="store_true", help="仅启动服务，不自动打开浏览器")

    migrate = subparsers.add_parser("migrate", help="迁移用户数据到独立工作区")
    migrate.add_argument("--target", type=Path, required=True)
    migrate.add_argument("--apply", action="store_true", help="执行迁移；默认只显示 dry-run")

    validate = subparsers.add_parser("validate", help="重新验证一个已有输出目录")
    validate.add_argument("run_dir", type=Path)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    try:
        if args.command == "build":
            result = build_project(
                args.project,
                draft_root_override=args.draft_root,
                skip_draft=args.skip_draft,
                open_output=args.open_output,
                resume=args.resume,
                visual_mode=args.visual_mode,
            )
            print(f"\n构建完成：{result}")
            return 0
        if args.command == "voices":
            result = create_voice_audition(args.project, open_output=args.open_output)
            print(f"\n试听生成完成：{result}")
            return 0
        if args.command == "assets":
            result = review_asset_task(args.resume)
            print(f"\n素材审核及续跑完成：{result}")
            return 0
        if args.command == "scripts":
            result = run_workbench_server(resume_latest=args.resume == "latest")
            if result.get("status") == "locked":
                project_dir = Path(str(result["project"]))
                output = build_project(
                    project_dir.name,
                    resume="auto",
                    visual_mode="sourced",
                    open_output=args.open_output,
                )
                print(f"\n脚本锁定并完成构建：{output}")
                return 0
            if result.get("status") == "resume_video":
                task = result["resume_task"]
                options = task.get("options", {})
                output = build_project(
                    str(task["project_id"]),
                    draft_root_override=options.get("draft_root"),
                    skip_draft=bool(options.get("skip_draft", False)),
                    # Auto-resume only selects a task whose full build input hash
                    # still matches. If the project settings changed in the
                    # workbench, the pipeline creates a new task and reuses only
                    # independently validated caches.
                    resume="auto",
                    visual_mode=str(options.get("visual_mode", "sourced")),
                    open_output=args.open_output,
                )
                print(f"\n视频任务续跑完成：{output}")
                return 0
            print("\n脚本工作台已关闭，未开始视频构建。")
            return 0
        if args.command == "studio":
            result = run_studio_console(open_browser=not args.no_browser)
            if result.get("status") == "locked":
                project_dir = Path(str(result["project"]))
                options = dict(result.get("options") or {})
                output = build_project(
                    project_dir.name,
                    resume="auto",
                    visual_mode=str(options.get("visual_mode", "sourced")),
                    skip_draft=bool(options.get("skip_draft", False)),
                    open_output=args.open_output,
                )
                print(f"\n项目已锁定并完成构建：{output}")
                return 0
            if result.get("status") == "resume_video":
                task = result.get("resume_task") or {}
                options = task.get("options", {})
                output = build_project(
                    str(task["project_id"]),
                    draft_root_override=options.get("draft_root"),
                    skip_draft=bool(options.get("skip_draft", False)),
                    resume="auto",
                    visual_mode=str(options.get("visual_mode", "sourced")),
                    open_output=args.open_output,
                )
                print(f"\n视频任务续跑完成：{output}")
                return 0
            print("\nAI-Video Studio 已关闭，未开始新构建。")
            return 0
        if args.command == "migrate":
            report = migrate_workspace(args.target, apply=args.apply)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        report = validate_run(args.run_dir)
        print(report)
        return 0 if report["success"] else 1
    except (BuildError, ScriptWorkbenchError, StudioConsoleError, ProfileError, StudioSettingsError, MigrationError) as exc:
        print(f"\n[构建失败] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n用户取消构建。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
