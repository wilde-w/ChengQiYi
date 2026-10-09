"""知识库导入：从字节到「查得到」。

这个文件里有两种测试，边界很清楚：

- **纯函数**（解码、文件名清洗、输入拒绝）谁来都能跑；
- **集成**（真的写进 PG + Qdrant + Neo4j）需要 `docker compose up -d`。
  集成部分挂在一个显式探测的 fixture 上——起不来就跳过并说明原因，
  而不是让断言在一种「其实什么都没测」的状态下变绿。

集成部分自己清理：导入的正文、登记行、图节点、作业行全部按本次的
`origin` 收回，`IMPORT_DIR` 被 monkeypatch 到 pytest 的临时目录，
所以跑完不会在仓库里留文件，也不会污染开发库里的知识库。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import delete

from app.clients.qdrant_client import collection_count
from app.constants import Library, RunStatus
from app.db.session import session_scope
from app.kb import qdrant_index
from app.kb.chunker import decode_bytes, make_chunk_id, split_document
from app.kb.ingest import (
    corpus_chunk_ids,
    corpus_source_files,
    imported_chunk_count,
    index_chunks,
)
from app.kb.lexical_tags import build_frequency_table
from app.kb.tagging import tag_pieces
from app.models.library import KbImportJob
from app.providers.factory import get_chat_model, get_embedding_model

ORIGIN = "测试导入-哲学.txt"
WORK = "测试导论"
AUTHOR = "测试"
DISCIPLINE = "存在主义哲学"
DOC = """\
操心是此在存在的整体结构。它先行于自身，并且已经在世界之中寓于他物。

此在的时间性构成了操心的意义。没有时间性，操心就无法被理解为一个整体。

畏揭示了虚无。虚无并不是某个存在者的缺席，而是存在本身的遮蔽方式。

语言是存在之家。人在语言中居住，并以此方式回应存在的呼唤与遮蔽。
"""


def _bytes(text: str = DOC, encoding: str = "utf-8") -> bytes:
    return text.encode(encoding)


EPUB_ORIGIN = "测试导入-哲学.epub"
EPUB_TITLE = "电子书自带书名"


def _mini_epub() -> bytes:
    """一本最小的 epub：正文就是 DOC 那四段（切出来应同为 4 段）。

    前后各带一页真书里必然有的东西——封面（按名字跳过）与一整页只有
    `<img>` 的插图页（抽不出字）。两者都不影响切分结果，但它们是导入
    提示的来源，得有东西可断言。

    组装逻辑借 `test_epub.py` 的夹具——同一套 zip 结构不该有两份写法。
    """
    from tests.test_epub import build

    body = "".join(f"<p>{p.strip()}</p>" for p in DOC.split("\n\n") if p.strip())
    return build(
        [
            ("cover", "cover.xhtml", "<img src='cover.jpg'/>"),
            ("c1", "c1.xhtml", body),
            ("pic", "pic.xhtml", "<div><img src='plate.jpg' alt=''/></div>"),
        ],
        title=EPUB_TITLE,
        creator="电子书作者",
    )


def _plain_zip() -> bytes:
    """一个没有 container.xml 的普通压缩包：测「不是 epub」与体积尺子用。"""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "x" * 200)
    return buf.getvalue()


async def _wait(job_id: str, timeout: float = 60.0) -> str:
    """轮询到终态。**刻意不用内部 task 对象**——那正是前端要走的路。"""
    from app.services import kb_import_service as svc

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        job = await svc.get_import(job_id)
        if job.status in (RunStatus.SUCCEEDED.value, RunStatus.FAILED.value,
                          RunStatus.CANCELLED.value):
            return job.status
        await asyncio.sleep(0.05)
    raise AssertionError(f"导入作业 {job_id} 在 {timeout}s 内没有结束")


async def _drop_job_rows(*job_ids: str) -> None:
    async with session_scope() as session:
        await session.execute(delete(KbImportJob).where(KbImportJob.id.in_(list(job_ids))))


@pytest.fixture
async def stack() -> None:
    """集成测试的地基。**跳过时必须说清楚为什么**，否则「全绿」是假的。"""
    try:
        await corpus_chunk_ids([Library.PSYCHOLOGY])
        await collection_count()
    except Exception as exc:
        pytest.skip(f"需要 docker compose 起的 PG 与 Qdrant：{type(exc).__name__}: {exc}")


@pytest.fixture
async def imported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stack: None) -> AsyncIterator[str]:
    """真跑一次导入，结束后按 `origin` 完整回滚。"""
    from app.services import kb_import_service as svc

    monkeypatch.setattr(svc, "IMPORT_DIR", tmp_path)
    result = await svc.create_import(
        _bytes(), filename=ORIGIN, library=Library.PSYCHOLOGY,
        title=WORK, author=AUTHOR, discipline=DISCIPLINE,
    )
    job_id = result.job.id
    if not result.auto_started:
        await svc.start_import(job_id)
    status = await _wait(job_id)
    assert status == RunStatus.SUCCEEDED.value, (await svc.get_import(job_id)).error

    yield job_id

    await svc.delete_source(ORIGIN)
    await _drop_job_rows(job_id)


# ----------------------------------------------------------------------
# 纯函数
# ----------------------------------------------------------------------


class TestDecodingStability:
    """编码兼容的全部意义在这一条上：**同一个文件导入两次，必须完全相同。**"""

    def test_gbk_and_utf8_produce_the_same_chunk_ids(self) -> None:
        ids = []
        for encoding in ("utf-8", "gb18030", "utf-8-sig"):
            report = split_document(decode_bytes(_bytes(encoding=encoding)))
            ids.append([make_chunk_id(Library.PSYCHOLOGY, WORK, p.body) for p in report.pieces])
        assert ids[0] == ids[1] == ids[2]
        assert len(ids[0]) == 4

    def test_body_text_is_byte_identical_across_encodings(self) -> None:
        a = split_document(decode_bytes(_bytes(encoding="utf-8")))
        b = split_document(decode_bytes(_bytes(encoding="gb18030")))
        assert [p.body for p in a.pieces] == [p.body for p in b.pieces]


class TestSafeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (r"C:\Users\x\哲学导论.txt", "哲学导论.txt"),
            ("/home/x/book.md", "book.md"),
            ("../../etc/passwd", "passwd"),
            ("带\x00控制\x1f字符.txt", "带控制字符.txt"),
            ("a/b\\c:d*e?f.txt", "cdef.txt"),
            ("", "未命名.txt"),
            ("   ", "未命名.txt"),
        ],
    )
    def test_sanitized(self, raw: str, expected: str) -> None:
        # 用户提供的字符**绝不拼进路径**：落盘名永远是 <job_id>.txt，
        # 这个值只用于展示、写进 origin、以及当「删掉这一次导入」的键。
        from app.services.kb_import_service import safe_filename

        assert safe_filename(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "a\x00b.txt", "\\\\?\\C:\\x.txt", "..", ".........", "／全角斜杠.txt",
            "tab\there.txt", "\u202e反转.txt", "a" * 500 + ".txt", "🐍.txt",
        ],
    )
    def test_hostile_names_never_escape(self, raw: str) -> None:
        """枚举不完的，就断言性质：结果非空、不含分隔符与控制字符、长度有界。"""
        from app.services.kb_import_service import safe_filename

        name = safe_filename(raw)
        assert name
        assert len(name) <= 200
        assert not set(name) & set("/\\\x00\x01\x1f\r\n\t:*?\"<>|")
        assert Path(name).name == name


class TestInputRejection:
    """这些必须在**建作业之前**就报错，且消息要能直接展示给用户。"""

    async def test_empty_file(self) -> None:
        from app.services.kb_import_service import ImportRejected, create_import

        with pytest.raises(ImportRejected, match="空的"):
            await create_import(b"", filename="空.txt", library=Library.PSYCHOLOGY)

    async def test_blank_file(self) -> None:
        from app.services.kb_import_service import ImportRejected, create_import

        with pytest.raises(ImportRejected):
            await create_import(b" \n\t\n", filename="空白.txt", library=Library.PSYCHOLOGY)

    async def test_oversize_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from app.config import get_settings
        from app.services import kb_import_service as svc

        monkeypatch.setattr(svc, "IMPORT_DIR", tmp_path)
        monkeypatch.setattr(get_settings(), "KB_IMPORT_MAX_BYTES", 100)
        with pytest.raises(svc.ImportRejected, match="上限"):
            await svc.create_import(b"x" * 200, filename="大.txt", library=Library.PSYCHOLOGY)

    async def test_too_many_chunks(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # 不静默截断：截断一本书然后显示「成功」比失败更糟。
        from app.config import get_settings
        from app.services import kb_import_service as svc

        monkeypatch.setattr(svc, "IMPORT_DIR", tmp_path)
        monkeypatch.setattr(get_settings(), "KB_IMPORT_MAX_CHUNKS", 3)
        with pytest.raises(svc.ImportRejected, match="段"):
            await svc.create_import(_bytes(), filename="长.txt", library=Library.PSYCHOLOGY)

    async def test_zip_that_is_not_epub(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """压缩包要报出「是什么问题」，而且是在建作业之前。"""
        from app.services import kb_import_service as svc

        monkeypatch.setattr(svc, "IMPORT_DIR", tmp_path)
        with pytest.raises(svc.ImportRejected, match="不是 epub"):
            await svc.create_import(_plain_zip(), filename="包.zip", library=Library.PSYCHOLOGY)

    async def test_epub_用的是另一把体积尺子(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """纯文本 5MB 那条闸门不该拿来量电子书——它大半体积是图片。"""
        from app.config import get_settings
        from app.services import kb_import_service as svc

        monkeypatch.setattr(svc, "IMPORT_DIR", tmp_path)
        settings = get_settings()
        monkeypatch.setattr(settings, "KB_IMPORT_MAX_BYTES", 100)
        monkeypatch.setattr(settings, "KB_IMPORT_EPUB_MAX_BYTES", 10_000)
        blob = _plain_zip()
        # 用一个 100 字节的纯文本尺子量它，报的必须是「不是 epub」而不是体积
        with pytest.raises(svc.ImportRejected, match="不是 epub"):
            await svc.create_import(blob, filename="包.zip", library=Library.PSYCHOLOGY)

        # 反过来：epub 上限压到比文件还小，就必须报体积
        monkeypatch.setattr(settings, "KB_IMPORT_EPUB_MAX_BYTES", 50)
        with pytest.raises(svc.ImportRejected, match="上限"):
            await svc.create_import(blob, filename="包.zip", library=Library.PSYCHOLOGY)


class TestUploadReading:
    """`_read_upload`：字节 → 正文的那一道岔路口。纯函数，不碰数据库。"""

    def test_纯文本走解码_path(self) -> None:
        from app.services import kb_import_service as svc

        text, title, author, notes = svc._read_upload(_bytes(), title="表单书名", author="")
        assert "操心是此在存在的整体结构" in text
        assert (title, author, notes) == ("表单书名", "", [])

    def test_epub_在表单留空时补书名作者(self) -> None:
        from app.services import kb_import_service as svc

        text, title, author, notes = svc._read_upload(_mini_epub(), title="", author="")
        assert "操心是此在存在的整体结构" in text
        assert (title, author) == (EPUB_TITLE, "电子书作者")
        assert len(notes) == 2  # 跳过封面页 + 一页纯图片

    def test_表单填了就以表单为准(self) -> None:
        from app.services import kb_import_service as svc

        _, title, author, _ = svc._read_upload(_mini_epub(), title="我定的名", author="我定的作者")
        assert (title, author) == ("我定的名", "我定的作者")

    def test_跳过的封面页会变成一条提示(self) -> None:
        from app.services import kb_import_service as svc
        from tests.test_epub import build

        blob = build(
            [("cover", "cover.xhtml", "<img src='c.jpg'/>"), ("c1", "c1.xhtml", "<p>正文</p>")]
        )
        _, _, _, notes = svc._read_upload(blob, title="", author="")
        assert len(notes) == 1
        assert "cover.xhtml" in notes[0]

    def test_PK_开头的纯文本落回解码(self) -> None:
        """头两个字节是 PK 的文本不该被当成坏掉的压缩包拒掉。"""
        from app.services import kb_import_service as svc

        text, *_ = svc._read_upload("PK 开头的两行\n第二行".encode(), title="", author="")
        assert text.strip() == "PK 开头的两行\n第二行"


# ----------------------------------------------------------------------
# 集成：真的进库、真的查得到
# ----------------------------------------------------------------------


class TestEndToEnd:
    async def test_import_then_search(self, imported: str) -> None:
        """**整个功能的验收断言。** 前面的都只是过程，这一条才是目的。"""
        from app.services import kb_import_service as svc

        job = await svc.get_import(imported)
        assert job.status == RunStatus.SUCCEEDED.value
        assert job.chunk_indexed == 4
        assert job.chunk_tagged == 4
        assert job.progress == 100

        model = get_embedding_model()
        hits = await qdrant_index.search(
            (await model.embed(["操心 此在 时间性"]))[0], library=Library.PSYCHOLOGY, limit=5
        )
        assert any(h.payload.get("origin") == ORIGIN for h in hits), (
            "导入的内容检索不到——它只是被写进了库，没有真正可用"
        )

    async def test_source_listing_and_preview(self, imported: str) -> None:
        from app.services import kb_import_service as svc

        sources = {s.source_file: s for s in await svc.list_sources()}
        assert ORIGIN in sources
        assert sources[ORIGIN].chunks == 4

        docs = await svc.source_documents(ORIGIN)
        assert len(docs) == 4
        assert all(d.text_preview for d in docs)
        # 语料文件不该出现在这张表里——列出来会让人以为可以删，
        # 而删掉之后下一次 ingest-kb 又会把它们装回来。
        assert not set(corpus_source_files()) & set(sources)

    async def test_reimport_skips_every_embedding(self, imported: str) -> None:
        """重导同一个文件：正文一个字没变 → **一次嵌入都不做**。

        用同一份正文按 service 的同一套参数重建 chunk，再走一次入库——
        这是唯一能直接看到 `skipped` 的角度：作业表只记「入库了几条」，
        而它在重导时和首次导入是一模一样的数字。
        """
        text = decode_bytes(_bytes())
        report = split_document(text)
        chunks, tag_report = await tag_pieces(
            report.pieces,
            library=Library.PSYCHOLOGY,
            model=get_chat_model(),
            work=WORK,
            author=AUTHOR,
            discipline=DISCIPLINE,
            origin=ORIGIN,
            freq=build_frequency_table(text),
        )
        assert tag_report.fallback == 0

        again = await index_chunks(chunks, get_embedding_model())
        assert (again.added, again.updated, again.skipped) == (0, 0, 4)

    async def test_prune_orphans_keeps_imported_content(self, imported: str) -> None:
        """**本次改动里最重要的一条回归。**

        `_prune_orphans` 的语义是「语料文件是唯一权威」。导入的内容不属于
        任何语料文件，旧实现会把整本书连同向量、图节点一起删掉，屏幕上
        只留一行 `kb_pruned count=312`——用户看到的是「我明明导入过，
        怎么查不到了」，而且没有任何线索指向这里。
        """
        from app.services import kb_import_service as svc

        known = await corpus_chunk_ids([Library.PSYCHOLOGY])
        imported_ids = {d.chunk_id for d in await svc.source_documents(ORIGIN)}
        assert imported_ids, "前提不成立：这次导入没有写进登记表"

        # 判据就是这一条：导入的 chunk 不在「语料」那一侧。于是孤儿判定
        # `known - corpus_ids` 不可能选中它们，无论跑了几个库、语料多少条。
        assert not imported_ids & known
        assert await imported_chunk_count([Library.PSYCHOLOGY]) >= 4

    async def test_delete_source_rolls_the_import_back(self, tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch,
                                                        stack: None) -> None:
        from app.services import kb_import_service as svc

        monkeypatch.setattr(svc, "IMPORT_DIR", tmp_path)
        result = await svc.create_import(
            _bytes(), filename=ORIGIN, library=Library.PSYCHOLOGY, title=WORK
        )
        job_id = result.job.id
        if not result.auto_started:
            await svc.start_import(job_id)
        assert await _wait(job_id) == RunStatus.SUCCEEDED.value

        try:
            model = get_embedding_model()
            query = (await model.embed(["操心 此在 时间性"]))[0]
            before = await qdrant_index.search(query, library=Library.PSYCHOLOGY, limit=5)
            assert any(h.payload.get("origin") == ORIGIN for h in before)

            removed = await svc.delete_source(ORIGIN)
            assert removed == 4

            after = await qdrant_index.search(query, library=Library.PSYCHOLOGY, limit=5)
            assert not any(h.payload.get("origin") == ORIGIN for h in after)
            assert await svc.source_documents(ORIGIN) == []
            # 幂等：再删一次不该报错，也不该删到别的东西。
            assert await svc.delete_source(ORIGIN) == 0
        finally:
            await _drop_job_rows(job_id)

    async def test_corpus_contents_survive_an_import(self, imported: str) -> None:
        # 导入是加法。库里原有的 190 条语料必须一条不动。
        known = await corpus_chunk_ids([Library.PSYCHOLOGY])
        assert len(known) >= 50
        assert await collection_count() >= 190


class TestSourceText:
    """「我导入的到底是什么」——原文全文，不是那条 120 字的切块预览。

    这条路走的是**落盘文件**而不是拼 Qdrant chunk，所以断言是逐字相等：
    只要中间任何一层做了规范化（折行、去空白、重排），用户看到的就不再是
    他上传的那本书，而这里会立刻红。
    """

    async def test_原文与上传的正文逐字一致(self, imported: str) -> None:
        from app.services import kb_import_service as svc

        got = await svc.source_text(ORIGIN)

        assert got.text == DOC
        assert got.char_count == len(DOC)
        assert got.truncated is False
        # 书名/作者取的是导入时冻结在 options 里的那份，不是重新解析文件
        assert (got.title, got.author) == (WORK, AUTHOR)
        assert got.library == Library.PSYCHOLOGY.value

    async def test_按字符分页(self, imported: str) -> None:
        """整本红楼梦上百万字，一次性塞进 DOM 会卡——分页按**字符**而不是行。"""
        from app.services import kb_import_service as svc

        head = await svc.source_text(ORIGIN, offset=0, limit=10)
        tail = await svc.source_text(ORIGIN, offset=10, limit=10)

        assert head.text == DOC[:10]
        assert tail.text == DOC[10:20]
        assert tail.offset == 10
        # char_count 是**全文**长度：前端要靠它算「还有多少没加载」
        assert head.char_count == len(DOC)
        assert head.truncated is True

    async def test_读完最后一页不再说还有更多(self, imported: str) -> None:
        from app.services import kb_import_service as svc

        got = await svc.source_text(ORIGIN, offset=len(DOC) - 5)

        assert got.text == DOC[-5:]
        assert got.truncated is False

    async def test_不是从界面导入的来源取不到原文(self, imported: str) -> None:
        """随仓库交付的语料没有落盘正文。

        这句话必须与「记录还在但文件被清理了」分开——前者用户无法可施，
        后者重新导入一次就好，混成一句会让人去瞎找文件。
        """
        from app.services import kb_import_service as svc

        with pytest.raises(svc.ImportNotFound) as exc:
            await svc.source_text("随仓库交付的语料.txt")

        assert "不是从界面导入" in str(exc.value)


class TestEpubEndToEnd:
    """epub 走完整导入：**落盘的是正文，入库的是正文切出来的段**。

    与纯文本那套共用同一个 service 断言口径（4 段、4 条入库），差别只在
    入口多了一层解包——这正是要证明的：epub 到了切分那一步就与 txt 合流。
    """

    async def test_epub_imports_like_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stack: None
    ) -> None:
        from app.services import kb_import_service as svc

        monkeypatch.setattr(svc, "IMPORT_DIR", tmp_path)
        blob = _mini_epub()
        result = await svc.create_import(blob, filename=EPUB_ORIGIN, library=Library.PSYCHOLOGY)
        job_id = result.job.id
        try:
            # 建作业时就已解包：盘上是正文，不是那包 zip
            stored = svc._stored_path(job_id).read_bytes()
            assert not stored.startswith(b"PK")
            text = stored.decode("utf-8")
            assert "操心是此在存在的整体结构" in text
            assert "<p>" not in text and "<?xml" not in text

            if not result.auto_started:
                await svc.start_import(job_id)
            status = await _wait(job_id)
            assert status == RunStatus.SUCCEEDED.value, (await svc.get_import(job_id)).error

            job = await svc.get_import(job_id)
            assert (job.chunk_total, job.chunk_indexed) == (4, 4)
            # 溯源与体积仍按**原始上传**记录，不是抽取后的正文
            assert job.filename == EPUB_ORIGIN
            assert job.file_size == len(blob)
            # 表单里书名留空 → 用电子书自带的
            assert job.options["title"] == EPUB_TITLE
            # 建作业时算出的提示必须活到终态那一行：执行侧会用切分报告重写
            # warnings，这里防的就是它把「跳过了封面页」吃掉——实测吃过一次，
            # 界面上那一栏就再也没有 epub 专属的提示了。
            assert any("cover.xhtml" in w for w in job.warnings), job.warnings
        finally:
            await svc.delete_source(EPUB_ORIGIN)
            await _drop_job_rows(job_id)
