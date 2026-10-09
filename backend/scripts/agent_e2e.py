"""故事工坊的端到端验收（Playwright / msedge headless）。

断言的是**用户看得见的那几件事**，不是接口通不通（接口那层 agent_api_check.py
已经跑过了）：
  1. 顶栏「✍ 写故事」能打开面板——它与流水线并列，分析在跑时也点得开
  2. 粘一段材料 →「开始写」→ 中栏长出工具卡（带耗时）、右栏长出正文
  3. 底部说一句「再暗一点」→ 第二版与第一版不同
  4. 刷新页面 → 对话、工具卡、正文都自己回来（这是「刷新后能接着聊」那一条）

跑法（两个服务都要在跑：后端 8000、前端 5173）：

    cd backend && .venv/Scripts/python.exe -u scripts/agent_e2e.py

截图落在 `E2E_OUT`（默认 `D:/tmp`）——**刻意放在仓库外**：它们是给人看的
证据，不是要提交的东西，留在仓库里只会让每次验收都多出几个待提交文件。
"""
import os
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

OUT = Path(os.environ.get("E2E_OUT", "D:/tmp"))
URL = "http://localhost:5173"

# 一段有情绪、有具体物件的材料——工具调用才会真的发生
INPUT = (
    "我妈昨天打电话来，说家里那盆养了十年的君子兰终于开花了，她讲得很高兴，"
    "讲了快十分钟。我一边嗯嗯地应着，一边在想明天要交的报表。挂掉之后我才"
    "反应过来——她其实是想我了，从过年到现在我一次都没回去过。"
)

fail = []


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fail.append(name)


def wait_turn(page, timeout_s=180):
    """等这一轮真正跑完。

    **必须先等它开始，再等它结束。** 只判断「停止按钮不可见」的话，在按下的
    那一瞬它是真的不可见（请求还没发出、状态还没翻成 running），循环立刻
    退出，读到的还是上一版正文——一个只在测试里出现、看起来像产品坏了的假象。
    """
    stop = page.get_by_role("button", name="停止")
    for _ in range(40):
        if stop.is_visible():
            break
        page.wait_for_timeout(500)
    for _ in range(timeout_s * 2):
        if not stop.is_visible():
            break
        page.wait_for_timeout(500)


def shot(page, name):
    page.screenshot(path=str(OUT / name))
    print(f"     截图 {OUT / name}")


def panes(page):
    """三栏各自的定位器，按结构取而不是按类名猜。

    **别用 `div.overflow-y-auto` 的 first/last**：材料栏、对话栏、正文栏都挂
    着这个类，而「第一个」是材料栏。上一版就是这么把「刷新后对话重建」读到了
    材料上去的——它当然不含「再暗一点」，于是测试报了一个产品其实没有的错。
    面板的结构是定的：body 的三个直接子元素就是三栏。
    """
    body = page.locator("div.fixed.inset-0 div.flex.min-h-0.overflow-hidden").last
    cols = body.locator(":scope > div")
    material = cols.nth(0).locator("div.overflow-y-auto").first
    transcript = cols.nth(1).locator(":scope > div").first
    story = cols.nth(2).locator("div.overflow-y-auto").last
    return material, transcript, story


def card_geometry(page):
    """每张工具卡的几何。**这是唯一能抓住「参数被挤成一行一个字」的检查。**

    那个 bug 躲过了单测（它们看的是数据）和 HTTP 走查（它们看的是事件）：
    卡片在数据上完全正确，只是被自己的排版压成了 700px 高的一条竖线，
    参数正好被卡片的 `overflow-hidden` 切掉。
    """
    return page.evaluate(
        """() => {
      const btns = [...document.querySelectorAll(
        'div.fixed.inset-0 button[title="展开参数与结果"]')];
      return btns.map(b => {
        const row = b.children[0];
        const r = b.getBoundingClientRect();
        return {h: Math.round(r.height),
                mid: Math.round(row.children[1].getBoundingClientRect().width),
                over: b.scrollWidth > b.clientWidth + 1};
      });
    }"""
    )


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1600, "height": 950})
    page.goto(URL)
    page.wait_for_load_state("networkidle")

    # ---- 1. 入口 --------------------------------------------------------
    page.get_by_role("button", name="✍ 写故事").click()
    dialog = page.locator("div.fixed.inset-0").last
    expect(page.get_by_role("button", name="开始写")).to_be_visible(timeout=8000)
    check("顶栏入口能打开面板", True)
    shot(page, "agent_1_empty.png")

    # ---- 1b. 面板能挪 ----------------------------------------------------
    # 挪走再挪回来，后面的几何断言仍旧按居中位置读。这一条守的是
    # `animate-rise`（`animation: … both`）与内联 transform 的层叠关系：
    # 动画的填充值盖得住内联样式，类不摘掉，面板纹丝不动而代码看着全对。
    panel = page.locator("div[role='dialog']").last
    p0 = panel.bounding_box()
    page.mouse.move(p0["x"] + p0["width"] / 2, p0["y"] + 22)
    page.mouse.down()
    for i in range(1, 13):
        page.mouse.move(p0["x"] + p0["width"] / 2 + 120 * i / 12, p0["y"] + 22 + 80 * i / 12)
    page.mouse.up()
    page.wait_for_timeout(150)
    p1 = panel.bounding_box()
    check("按住标题栏能把面板挪走", abs(p1["x"] - p0["x"] - 120) < 3 and abs(p1["y"] - p0["y"] - 80) < 3,
          f"走了 ({p1['x'] - p0['x']:.0f}, {p1['y'] - p0['y']:.0f})")
    page.mouse.move(p1["x"] + 120, p1["y"] + 22)
    page.mouse.down()
    for i in range(1, 13):
        page.mouse.move(p1["x"] + 120 - 120 * i / 12, p1["y"] + 22 - 80 * i / 12)
    page.mouse.up()
    page.wait_for_timeout(150)
    p2 = panel.bounding_box()
    check("还能挪回原位（后续断言按居中位置读）",
          abs(p2["x"] - p0["x"]) < 3 and abs(p2["y"] - p0["y"]) < 3,
          f"({p2['x']:.0f}, {p2['y']:.0f}) vs ({p0['x']:.0f}, {p0['y']:.0f})")

    # ---- 2. 开写 --------------------------------------------------------
    page.get_by_placeholder("把一段网友评论粘进来").fill(INPUT)
    page.get_by_role("button", name="开始写").click()

    # 工具卡是「agent 自己决定了什么」唯一可核查的证据，等它出现
    cards = page.locator("button[title='展开参数与结果'], button[title='收起']")
    try:
        expect(cards.first).to_be_visible(timeout=90000)
    except AssertionError:
        check("中栏出现工具卡", False, "90s 内一张都没出现")
    page.wait_for_timeout(1500)

    panel = page.locator("div.fixed.inset-0").filter(has=page.get_by_text("正文", exact=True)).last
    _, _, story_box = panes(page)
    # 等正文开始长出来（右侧栏那一段）
    grew = False
    for _ in range(120):
        if len(story_box.inner_text().strip()) > 200:
            grew = True
            break
        page.wait_for_timeout(1000)
    check("右栏出现正文", grew)

    # 跑到这一轮结束：底部提示换成「写完了。可以继续说要求…」
    wait_turn(page)
    n_cards = cards.count()
    check("工具卡 ≥ 2 张", n_cards >= 2, f"实际 {n_cards} 张")

    header = page.locator("div.flex.items-baseline.justify-between").filter(has_text="正文").first
    meta = header.inner_text().replace("\n", " ")
    check("右栏头部常驻轮数与工具次数", "轮" in meta and "次工具" in meta, meta)

    if n_cards == 0:
        # 模型这一轮一次都没查。**这本身是要报出来的事**（计划里的风险 3：它
        # 会安静地退化成一个不查资料的写作模型，而日志上一切正常）。但也不能
        # 让它把后面几项检查用一个 30 秒的超时盖掉——上一版就是这样，报错栈
        # 里只剩 Playwright 的 TimeoutError，看不出到底哪一项没过。
        print("     ！模型这一轮没有调用任何工具，卡片相关的两项没有对象")
    else:
        # 卡片没被自己的排版压扁（见 card_geometry 的说明）
        geo = card_geometry(page)
        squashed = [g for g in geo if g["mid"] < 60 or g["h"] > 140 or g["over"]]
        check("工具卡没有被挤扁", not squashed, f"异常 {len(squashed)}/{len(geo)} 张 {squashed[:2]}")

        # 卡片上那行字里必须带耗时（「6 条 · 320ms」）
        first_card_text = cards.first.inner_text()
        check("工具卡带耗时", "ms" in first_card_text or "s" in first_card_text,
              first_card_text.replace("\n", " ")[:60])

    v1 = story_box.inner_text().strip()
    shot(page, "agent_2_written.png")
    print(f"     第一版 {len(v1)} 字")

    # ---- 3. 改稿 --------------------------------------------------------
    box = page.get_by_placeholder("接着说要求，比如「再暗一点，短一些」")
    box.fill("再暗一点，短一些")
    shot(page, "agent_3_instruction.png")
    box.press("Enter")
    wait_turn(page)
    page.wait_for_timeout(1200)
    v2 = story_box.inner_text().strip()
    check("第二版与第一版不同", v1 != v2 and len(v2) > 100, f"{len(v1)} 字 → {len(v2)} 字")
    # 改稿轮查不查资料由模型定，**不设成硬检查**：「再暗一点，短一些」不需要
    # 新素材时它不查才是对的（实测两轮改稿一次都没查）。这里只把数字记下来，
    # 好知道它有没有在悄悄退化成「不查资料的写作模型」。
    print(f"     观察：改稿轮的工具卡 {n_cards} → {cards.count()} 张")
    shot(page, "agent_4_revised.png")

    # ---- 4. 刷新重建 -----------------------------------------------------
    page.reload()
    page.wait_for_load_state("networkidle")
    page.get_by_role("button", name="✍ 写故事").click()
    # 接回上一场是**异步**的（先 GET 快照再回放事件），固定 sleep 一个 3 秒
    # 是不够的——它只会让检查时快时慢。等它真的长出来。
    _, transcript_box, story_box = panes(page)
    v3 = ""
    for _ in range(60):
        v3 = story_box.inner_text().strip()
        if len(v3) > 100:
            break
        page.wait_for_timeout(500)
    check("刷新后正文重建", len(v3) > 100 and v2[:60] in v3, f"{len(v3)} 字")
    check("刷新后工具卡也在", cards.count() >= 2, f"{cards.count()} 张")
    transcript = transcript_box.inner_text()
    # 对话里应当能看到上一轮说的话（用户气泡 + 上一版正文）
    check("刷新后对话重建", "再暗一点" in transcript, transcript[:60].replace("\n", " "))
    shot(page, "agent_5_reloaded.png")

    browser.close()

print("\n失败项：" + ("无" if not fail else str(fail)))
sys.exit(1 if fail else 0)
