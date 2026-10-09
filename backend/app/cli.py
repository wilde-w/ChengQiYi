"""运维 CLI：python -m app.cli <命令>

存在的理由：这个系统有四个基础设施依赖，任何一个没起来，
表现都是「分析跑不出结果」而不是一个明确的报错。
`check` 让这类问题在 3 秒内定位。
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import sys
from pathlib import Path

import typer

from app.clients.neo4j_client import close_neo4j, init_neo4j
from app.clients.qdrant_client import close_qdrant, init_qdrant
from app.clients.readiness import probe_all
from app.clients.redis_client import close_redis, init_redis
from app.config import ConfigurationError, get_settings
from app.db.session import dispose_engine, init_engine
from app.logging_conf import configure_logging

cli = typer.Typer(
    add_completion=False,
    help="观心 — 运维命令",
    no_args_is_help=True,
)

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _mark(ok: bool) -> str:
    return f"{_GREEN}✓{_RESET}" if ok else f"{_RED}✗{_RESET}"


def _bootstrap() -> None:
    """让 CLI 命令也能用上应用级的配置与连接。"""
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL, json_output=False)
    init_engine()
    init_redis()
    init_qdrant()
    init_neo4j()


async def _shutdown() -> None:
    await close_neo4j()
    await close_qdrant()
    await close_redis()
    await dispose_engine()


@cli.command("douyin-probe")
def douyin_probe(
    link: str = typer.Argument("", help="视频链接 / 分享口令 / aweme_id；留空则用内置样本"),
    # 默认 3 页 = 60 条，刚好覆盖内置样本的全部 52 条。
    # 少了会让人误以为过滤规则没生效（广告点赞低，按热度排在末页）。
    pages: int = typer.Option(3, "--pages", "-p", min=1, max=10, help="抓取页数"),
    sort: str = typer.Option("hot", "--sort", help="hot | time"),
) -> None:
    """验证抖音数据源：解析链接、取视频、翻页抓评论、跑一遍清洗规则。

    在接真实 MCP 之前先用它确认适配器层是通的——
    换数据源最容易出问题的地方是字段映射，而不是网络。
    """

    async def run() -> int:
        configure_logging("WARNING", json_output=False)

        from app.douyin.factory import get_douyin_provider
        from app.douyin.mock_provider import fixture_ids
        from app.services.comment_service import clean_comments

        provider = get_douyin_provider()

        print()
        print("  观心 — 抖音数据源自检")
        print(f"  {_DIM}{'─' * 56}{_RESET}")
        print(f"    provider   {provider.name}  (is_mock={provider.is_mock})")

        health = await provider.health()
        print(f"    health     {'✓' if health.get('ok') else '✗'} {_DIM}{health}{_RESET}")

        # 1) 解析
        raw = link or f"https://www.douyin.com/video/{fixture_ids()[0]}"
        try:
            ref = await provider.resolve_link(raw)
        except Exception as exc:
            print(f"  {_RED}✗ 链接解析失败：{exc}{_RESET}\n")
            return 1
        print(f"    resolve    aweme_id={ref.aweme_id}  via={ref.resolved_via}")

        # 2) 视频
        # 元数据失败不算致命：主料是评论。第三方 MCP（如 hhy5562877）的详情接口
        # 被抖音 Argus 风控长期压制（403），但评论接口不受影响——所以这里降级为
        # 警告继续跑，真正决定「数据源能不能用」的是评论链路。
        video = None
        try:
            video = await provider.get_video(ref.aweme_id)
        except Exception as exc:
            print(f"  {_YELLOW}⚠ 取视频失败（继续查评论）：{exc}{_RESET}")
        if video is not None:
            print(f"    video      {video.author_name}｜{(video.title or '')[:34]}")
            print(f"    stats      {_DIM}{video.stats}{_RESET}")

        # 3) 分页评论
        print(f"  {_DIM}{'─' * 56}{_RESET}")
        cursor: str | None = None
        all_items = []
        try:
            for page in range(pages):
                chunk = await provider.get_comments(
                    ref.aweme_id, cursor=cursor, count=20, sort=sort
                )
                all_items.extend(chunk.items)
                print(
                    f"    第 {page + 1} 页    {len(chunk.items):>3} 条   "
                    f"total={chunk.total}  has_more={chunk.has_more}"
                )
                cursor = chunk.next_cursor
                if not chunk.has_more:
                    break
        except Exception as exc:
            print(f"  {_RED}✗ 取评论失败：{exc}{_RESET}\n")
            return 1

        # 4) 清洗规则
        cleaned, stats = clean_comments(all_items)
        print(f"  {_DIM}{'─' * 56}{_RESET}")
        print(f"    抓到        {stats.total} 条")
        print(f"    保留        {stats.kept} 条")
        print(
            f"    过滤        {stats.ads} 广告 · {stats.spam} 灌水 · "
            f"{stats.empty} 空内容 · {stats.duplicates} 重复"
        )
        for item in cleaned:
            if item.filter_reason:
                reason = f"{_YELLOW}{item.filter_reason}{_RESET}"
                print(f"      {reason:<20} {item.comment_id}  {item.text[:34]}")

        print(f"  {_DIM}{'─' * 56}{_RESET}")
        visible = [c for c in cleaned if c.is_visible_by_default]
        top = sorted(visible, key=lambda c: -c.like_count)[:3]
        for item in top:
            print(f"    {_GREEN}▲{_RESET} {item.like_count:>6}  {item.text[:44]}")
        print(f"  {_GREEN}数据源可用。{_RESET}\n")
        return 0

    sys.exit(asyncio.run(run()))


@cli.command()
def check(
    providers: bool = typer.Option(True, "--providers/--no-providers", help="同时打印 provider 模式"),
) -> None:
    """探测四个基础设施依赖与 provider 解析结果。"""

    async def run() -> int:
        settings = get_settings()
        configure_logging(settings.LOG_LEVEL, json_output=False)
        init_engine()
        init_redis()
        init_qdrant()
        init_neo4j()

        print()
        print("  观心 — 依赖自检")
        print(f"  {_DIM}{'─' * 52}{_RESET}")

        results = await probe_all()
        ok_all = True
        for name, res in results.items():
            ok_all &= res.ok
            line = f"  {_mark(res.ok)} {name:<10} {res.latency_ms:>7.1f} ms"
            if res.extra:
                detail = "  ".join(f"{k}={v}" for k, v in res.extra.items())
                line += f"  {_DIM}{detail}{_RESET}"
            if not res.ok:
                line += f"\n      {_RED}{res.detail}{_RESET}"
            print(line)

        if providers:
            print(f"  {_DIM}{'─' * 52}{_RESET}")
            modes = settings.provider_summary()
            for key in ("llm", "embedding", "douyin"):
                mode = modes[key]
                is_mock = mode == "mock"
                mark = f"{_YELLOW}○{_RESET}" if is_mock else _mark(True)
                suffix = f"  {_DIM}(mock){_RESET}" if is_mock else ""
                print(f"  {mark} {key:<10} {mode}{suffix}")
            print(f"  {_DIM}  MOCK_MODE={modes['mock_mode']}{_RESET}")
            if modes["is_demo"]:
                print(
                    f"  {_YELLOW}!{_RESET} 当前处于演示模式：部分数据为 mock。"
                    f" 在 .env 里配置对应 API key 即自动切换。"
                )

        print(f"  {_DIM}{'─' * 52}{_RESET}")
        if ok_all:
            print(f"  {_GREEN}全部依赖就绪。{_RESET}\n")
        else:
            print(f"  {_RED}存在不可用依赖，先执行 `docker compose up -d`。{_RESET}\n")

        await _shutdown()
        return 0 if ok_all else 1

    sys.exit(asyncio.run(run()))


@cli.command("init-db")
def init_db() -> None:
    """按 ORM 定义建表（开发用；生产走 alembic）。"""

    async def run() -> int:
        _bootstrap()
        from app.db.init import create_all

        tables = await create_all()
        print(f"  已创建 {len(tables)} 张表：")
        for t in tables:
            print(f"    {_DIM}·{_RESET} {t}")
        await _shutdown()
        return 0

    sys.exit(asyncio.run(run()))


@cli.command("ingest-kb")
def ingest_kb(
    reset: bool = typer.Option(False, "--reset", help="重建向量集合与图谱（换 embedding 模型后必须用）"),
    force: bool = typer.Option(False, "--force", help="忽略哈希差分，全部重新嵌入"),
    library: str = typer.Option("", "--library", "-l", help="只处理一个库：psychology|literature|poetry"),
) -> None:
    """把 corpus 里的语料灌进 Qdrant + Neo4j，并登记 content_hash。"""

    async def run() -> int:
        _bootstrap()
        from app.constants import Library
        from app.kb.ingest import EmptyLibraryError, ingest
        from app.kb.loader import CorpusError
        from app.providers.factory import get_embedding_model

        libs: list[Library] | None = None
        if library:
            try:
                libs = [Library(library)]
            except ValueError:
                print(f"  {_RED}未知的库「{library}」{_RESET}，可选：psychology / literature / poetry\n")
                return 2

        model = get_embedding_model()
        print()
        print("  观心 — 知识库摄取")
        print(f"  {_DIM}{'─' * 56}{_RESET}")
        print(f"    embedding  {model.name}  dim={model.dim}  is_mock={model.is_mock}")

        if model.is_mock:
            print(
                f"    {_YELLOW}!{_RESET} 当前是 mock embedding：向量可复现但语义粗糙，"
                f"检索只能作为离线演示。"
            )

        if reset:
            # `--reset` 的语义是「从零重建」，导入的内容必然一起没掉。
            # 这个行为本身不改，但必须在执行前说清楚——否则用户只会在某天
            # 发现导入的书不见了，而那天离事故已经很远。
            from app.kb.ingest import imported_chunk_count
            from app.services.kb_import_service import IMPORT_DIR

            imported = await imported_chunk_count(libs)
            if imported:
                print(
                    f"\n    {_YELLOW}!{_RESET} --reset 会一并清除"
                    f" {_YELLOW}{imported}{_RESET} 条导入的内容（Qdrant + Neo4j + 登记表）。\n"
                    f"      文件仍留在 {_DIM}{IMPORT_DIR}{_RESET}，可以重新导入。"
                )

        def progress(done: int, total: int) -> None:
            bar_len = 28
            filled = int(bar_len * done / total) if total else bar_len
            bar = "█" * filled + "·" * (bar_len - filled)
            print(f"\r    嵌入  {bar}  {done}/{total}", end="", flush=True)

        try:
            report = await ingest(model, libraries=libs, reset=reset, force=force, on_progress=progress)
        except CorpusError as exc:
            # 语料格式错误必须原样报出来——这里的消息带文件名与行号，
            # 被任何前缀包裹都会让人多花一分钟去找。
            print(f"\n  {_RED}✗ 语料错误{_RESET}：{exc}\n")
            return 1
        except EmptyLibraryError as exc:
            print(f"\n  {_RED}✗ {exc}{_RESET}\n")
            return 1

        print()
        print(f"  {_DIM}{'─' * 56}{_RESET}")
        print(f"    集合       {report.collection}  点数={report.qdrant_points}")
        print(f"    图谱       {report.graph_edges} 条聚合边")
        for lib, stat in report.stats.items():
            mark = f"{_GREEN}✓{_RESET}" if stat.total else f"{_RED}✗{_RESET}"
            print(
                f"  {mark} {lib!s:<12} 共 {stat.total:>3}"
                f"  {_DIM}新增 {stat.added} · 更新 {stat.updated} · 跳过 {stat.skipped}{_RESET}"
            )
        if report.removed:
            print(f"    {_YELLOW}清理{_RESET}     已从语料移除的 {report.removed} 条（Qdrant + Neo4j + 登记表）")
        print(f"  {_DIM}{'─' * 56}{_RESET}")
        print(
            f"    added={report.added} updated={report.updated} "
            f"skipped={report.skipped} total={report.total}"
        )
        print(f"  {_GREEN}知识库摄取完成。{_RESET}\n")
        await _shutdown()
        return 0

    sys.exit(asyncio.run(run()))


@cli.command("import-kb")
def import_kb(
    path: Path = typer.Argument(..., help="要导入的文件：纯文本（UTF-8 / GBK 均可）或 epub 电子书"),
    library: str = typer.Option("psychology", "--library", "-l", help="归入哪个库"),
    title: str = typer.Option("", "--title", "-t", help="书名 / 篇名；留空则用文件名"),
    author: str = typer.Option("", "--author", "-a"),
    discipline: str = typer.Option("", "--discipline", "-d", help="领域，仅 psychology 库使用"),
    tag: bool = typer.Option(True, "--tag/--no-tag", help="用 AI 抽取标签；--no-tag 只做规则抽取"),
    allow_partial: bool = typer.Option(
        False, "--allow-partial", help="打标大面积失败时仍然入库（默认失败，什么都不写）"
    ),
) -> None:
    """把本机的一个文本文件导进知识库。

    与 UI 走**同一个 service**，唯一的区别是不做确认环节——命令行本身就是
    显式动作。默认归入 psychology：那套模板渲染「领域｜概念｜作者｜书名｜关键词」，
    天生适合抽象论述；文学/诗词模板只渲染意象与情感，哲学文本产不出稳定意象，
    等于把可达性押在「它恰好提到光或深渊」上。
    """

    async def run() -> int:
        _bootstrap()
        from app.constants import ImportStage, Library, RunStatus
        from app.services import kb_import_service as svc

        try:
            lib = Library(library)
        except ValueError:
            print(f"\n  {_RED}未知的库「{library}」{_RESET}，可选：psychology / literature / poetry\n")
            return 2

        if not path.is_file():
            print(f"\n  {_RED}✗ 找不到文件{_RESET}：{path}\n")
            return 2

        print()
        print(f"  观心 — 导入知识库   {_DIM}{path.name}{_RESET}")
        print(f"  {_DIM}{'─' * 60}{_RESET}")

        try:
            result = await svc.create_import(
                path.read_bytes(),
                filename=path.name,
                library=lib,
                title=title,
                author=author,
                discipline=discipline,
                use_llm=tag,
                allow_partial=allow_partial,
            )
        except svc.ImportRejected as exc:
            print(f"\n  {_RED}✗ {exc}{_RESET}\n")
            return 1

        job = result.job
        print(f"    文件      {job.file_size / 1024:.0f} KB → 切出 {job.chunk_total} 段")
        print(f"    归入      {lib.label}")

        if not result.auto_started:
            # CLI 不做确认环节：敲下这行命令就是确认。
            print(
                f"    {_YELLOW}!{_RESET} {job.chunk_total} 段超过确认阈值，直接开跑"
                f"（命令行本身就是确认）"
            )
            await svc.start_import(job.id)
        for warning in result.warnings:
            print(f"    {_YELLOW}!{_RESET} {warning}")

        bar_len = 28
        last_line = ""
        while True:
            await asyncio.sleep(0.5)
            job = await svc.get_import(job.id)
            try:
                stage = ImportStage(job.stage).label if job.stage else "排队中"
            except ValueError:
                stage = str(job.stage)
            filled = int(bar_len * int(job.progress or 0) / 100)
            bar = "█" * filled + "·" * (bar_len - filled)
            line = (
                f"\r    {bar} {int(job.progress or 0):>3}%  {stage}"
                f"  {_DIM}{job.message or ''}{_RESET}"
            )
            if line != last_line:
                print(line, end="", flush=True)
                last_line = line
            if _is_terminal(job.status):
                break
        print()

        print(f"  {_DIM}{'─' * 60}{_RESET}")
        print(f"    切分 {job.chunk_total} 段 · 打标 {job.chunk_tagged} 段 · 入库 {job.chunk_indexed} 条")
        for warning in job.warnings or []:
            print(f"    {_YELLOW}!{_RESET} {warning}")

        status = RunStatus(job.status)
        if status is RunStatus.SUCCEEDED:
            print(f"  {_GREEN}导入完成。{_RESET}")
            print(f"  {_DIM}验证：python -m app.cli kb-search «关键词»{_RESET}\n")
            code = 0
        elif status is RunStatus.CANCELLED:
            print(f"  {_YELLOW}已取消，未写入任何内容。{_RESET} 文件仍在 {svc.IMPORT_DIR}\n")
            code = 1
        else:
            print(f"  {_RED}✗ {job.message or '导入失败'}{_RESET}")
            if job.error:
                print(f"    {_DIM}{job.error[:400]}{_RESET}")
            print()
            code = 1

        await _shutdown()
        return code

    sys.exit(asyncio.run(run()))


def _is_terminal(status: str) -> bool:
    from app.constants import RunStatus

    try:
        return RunStatus(status).is_terminal
    except ValueError:
        # 认不出的状态一律当终态：让循环停下来总好过转到天荒地老。
        return True


@cli.command("kb-search")
def kb_search(
    query: str = typer.Argument(..., help="检索词，如「落花 哀伤」"),
    limit: int = typer.Option(6, "--limit", "-n", min=1, max=30, help="每路返回条数"),
    graph: bool = typer.Option(True, "--graph/--no-graph", help="同时跑图谱路径"),
    library: str = typer.Option("", "--library", "-l", help="限定单个库"),
) -> None:
    """检索知识库，同时打印向量路径与图谱路径的结果。

    **两条路径并排打印是刻意的**：向量检索不出东西时（mock embedding 下很常见），
    能一眼看出是语料没灌进去还是图谱断了，而不是笼统地"搜不到"。
    """

    async def run() -> int:
        _bootstrap()
        from app.constants import Library
        from app.kb import neo4j_index, qdrant_index
        from app.providers.factory import get_embedding_model

        lib = Library(library) if library else None
        model = get_embedding_model()

        print()
        print(f"  观心 — 知识库检索   {_DIM}「{query}」{_RESET}")
        print(f"  {_DIM}{'─' * 62}{_RESET}")

        vector = (await model.embed([query]))[0]
        hits = await qdrant_index.search(vector, library=lib, limit=limit)
        print(f"  {_GREEN}向量路径{_RESET}  {_DIM}（{model.name}）{_RESET}")
        if not hits:
            print(f"    {_YELLOW}无结果{_RESET} —— 先跑 `python -m app.cli ingest-kb`")
        for hit in hits:
            payload = hit.payload or {}
            source = payload.get("work") or payload.get("chunk_id")
            extra = payload.get("author") or payload.get("character") or ""
            print(f"    {hit.score:.3f}  {payload.get('library', '')!s:<10} {source}  {_DIM}{extra}{_RESET}")
            print(f"           {_DIM}{(payload.get('text') or '')[:52]}{_RESET}")

        if not graph:
            print()
            await _shutdown()
            return 0

        # 图谱路径：查询词按空白/顿号切开，每个词分别当情绪与意象各试一次。
        # 命中不了就说明这个词不在图里，静默跳过——图查询没有"模糊匹配"，
        # 这正是它比向量精确的地方，也是它的局限。
        tokens = [t for t in re.split(r"[\s、,，/]+", query) if t]
        print(f"  {_GREEN}图谱路径{_RESET}  {_DIM}（Neo4j：情绪→意象→语料）{_RESET}")
        found = 0
        for token in tokens:
            rows = await neo4j_index.chunks_by_emotion(token, limit=limit)
            via = "情绪"
            if not rows:
                rows = await neo4j_index.chunks_by_imagery(token, limit=limit)
                via = "意象"
            if not rows:
                continue
            found += len(rows)
            print(f"    {_DIM}「{token}」作为{via}命中 {len(rows)} 条{_RESET}")
            for row in rows[:limit]:
                print(
                    f"      {float(row.get('weight') or 0):.2f}  "
                    f"{row.get('library', '')!s:<10} {row.get('chunk_id')}"
                )
                print(f"           {_DIM}{row.get('snippet')}{_RESET}")
        if not found:
            print(f"    {_YELLOW}无结果{_RESET} —— 查询词不在图词汇表里，或图未构建")

        print(f"  {_DIM}{'─' * 62}{_RESET}\n")
        await _shutdown()
        return 0

    sys.exit(asyncio.run(run()))


@cli.command("expand-poetry")
def expand_poetry(
    count: int = typer.Option(200, "--count", "-n", min=1, max=5000, help="本次最多追加多少首"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印将要追加的内容，不写文件"),
    source: str = typer.Option("", "--source", help="自定义 chinese-poetry JSON 地址"),
) -> None:
    """从 chinese-poetry 扩充诗词语料。

    **这个命令的价值在于"扩充"而不是"灌数据"**：上游只有诗句，没有意象与
    情感标注，而检索完全依赖这些标注（见 kb/schema.py 的 text_for_embedding）。
    所以脚本做的是按一份小词表给诗打标，**只收下意象与情感都认得出的那些**——
    宁可少收，也不要往库里灌一批检索不到的原文。
    """

    async def run() -> int:
        from app.kb.expand import ExpansionError
        from app.kb.expand import expand_poetry as do_expand

        print()
        print("  观心 — 诗词语料扩充")
        print(f"  {_DIM}{'─' * 56}{_RESET}")
        try:
            result = await do_expand(count=count, source=source or None, dry_run=dry_run)
        except ExpansionError as exc:
            print(f"  {_RED}✗ {exc}{_RESET}\n")
            return 1

        print(f"    抓取       {result.scanned} 首")
        print(f"    标注通过   {result.accepted} 首")
        print(f"    已存在跳过 {result.duplicate} 首")
        for failure in result.failed:
            print(f"    {_YELLOW}!{_RESET} 跳过数据源 {failure[:110]}")
        if dry_run:
            print(f"  {_DIM}{'─' * 56}{_RESET}")
            for row in result.samples[:8]:
                print(f"    {row}")
            print(f"  {_YELLOW}--dry-run：没有写任何文件。{_RESET}\n")
            return 0
        print(f"  {_DIM}{'─' * 56}{_RESET}")
        print(f"    写入       {result.written} 首 → {result.path}")
        print(f"  {_GREEN}完成。跑 `python -m app.cli ingest-kb --library poetry` 让它进入检索。{_RESET}\n")
        return 0

    sys.exit(asyncio.run(run()))


def _force_utf8_console() -> None:
    """把标准输出切成 UTF-8，并让 Windows 控制台按 UTF-8 解码。

    中文 Windows 的默认输出编码是 GBK，而这里打印的全是中文加 `✓ ✗ █`。
    不处理的话，**摄取成功之后打印报表这一步会抛 UnicodeEncodeError**——
    活儿干完了，命令却以异常退出，看起来像失败了。这比不打印更糟。

    `errors="replace"` 是第二道保险：真到了编码不了的终端上，
    宁可显示 `?` 也不要中断一条本来已经跑完的命令。
    """
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)  # type: ignore[attr-defined]
            ctypes.windll.kernel32.SetConsoleCP(65001)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - 非 Windows 终端或权限受限
            pass
    for stream in (sys.stdout, sys.stderr):
        # 已被重定向为非文本流时 reconfigure 不存在，忽略即可
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def main() -> None:
    _force_utf8_console()
    try:
        cli()
    except ConfigurationError as exc:
        print(f"\n  {_RED}配置错误{_RESET}：{exc}\n", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
