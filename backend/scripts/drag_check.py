"""故事工坊面板的拖动验收（Playwright / msedge headless）。

看的是**面板有没有真的跟着鼠标走**，以及几条边界：
  1. 按住标题栏拖 → 位移与鼠标一致（这是最容易被 CSS 吃掉的一条：
     `animate-rise` 是 `animation: … both`，动画的填充值盖得住内联 transform，
     类不摘掉的话面板纹丝不动，而代码看上去完全正确）
  2. 在任何位置都不许「拖丢」——面板至少留一块在视口里
  3. 拖到一半不许把面板拖没了（点遮罩关闭那条路不能被拖动带出来）
  4. 关掉再打开回到居中（位置不持久化）

跑法（只需前端在跑；面板空着也能拖，不用等模型）：

    cd backend && .venv/Scripts/python.exe -u scripts/drag_check.py

截图落在 `E2E_OUT`（默认 `D:/tmp`），与 agent_e2e.py 同一处。
"""
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

OUT = Path(os.environ.get("E2E_OUT", "D:/tmp"))
URL = "http://localhost:5173"

fail = []


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fail.append(name)


def drag(page, x, y, dx, dy, steps=12):
    """按住 (x, y) 拖动。**分步移动**：一步到位的话不会产生 pointermove
    中间态，拖动逻辑看起来工作，实际上是「一次点击 + 一次跳变」。"""
    page.mouse.move(x, y)
    page.mouse.down()
    for i in range(1, steps + 1):
        page.mouse.move(x + dx * i / steps, y + dy * i / steps)
    page.mouse.up()
    page.wait_for_timeout(120)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.on("console", lambda m: print(f"     [console.{m.type}] {m.text}") if m.type == "error" else None)
    page.goto(URL)
    page.wait_for_load_state("networkidle")

    page.get_by_role("button", name="✍ 写故事").click()
    dialog = page.locator("div[role='dialog']").filter(has_text="粘一段评论").last
    expect(dialog).to_be_visible(timeout=8000)
    page.wait_for_timeout(400)          # 让进场动画播完
    box = dialog.bounding_box()
    print(f"     初始位置 x={box['x']:.0f} y={box['y']:.0f} w={box['width']:.0f} h={box['height']:.0f}")

    # ---- 1. 标题栏拖动 ---------------------------------------------------
    grip_x, grip_y = box["x"] + box["width"] / 2, box["y"] + 22
    drag(page, grip_x, grip_y, 180, 100)
    box2 = dialog.bounding_box()
    dx, dy = box2["x"] - box["x"], box2["y"] - box["y"]
    check("按住标题栏能拖动", abs(dx - 180) < 3 and abs(dy - 100) < 3,
          f"鼠标走了 (180, 100)，面板走了 ({dx:.0f}, {dy:.0f})")
    check("拖动没有关掉面板", dialog.is_visible())
    page.screenshot(path=str(OUT / "drag_1_moved.png"))

    # ---- 2. 正文区不是抓取区 ---------------------------------------------
    before = dialog.bounding_box()
    drag(page, before["x"] + before["width"] / 2, before["y"] + before["height"] * 0.7, -120, -60)
    after = dialog.bounding_box()
    check("中栏/正文区不能被拖动（那里要选字、要滚动）",
          abs(after["x"] - before["x"]) < 1 and abs(after["y"] - before["y"]) < 1,
          f"位移 ({after['x'] - before['x']:.0f}, {after['y'] - before['y']:.0f})")

    # ---- 3. 拖不丢 -------------------------------------------------------
    # **抓取点必须落在视口里**，否则鼠标根本没按到标题栏上，断言会「空过」：
    # 第一版就是按 `x + width/2` 取点，而面板被夹到右边之后那个点已经在
    # 屏幕外了——三条边界检查里有两条测的是「没发生拖动」。
    vw = page.evaluate("() => window.innerWidth")

    def grip_at(box, frac_y=22):
        """标题栏**露在视口里的那一段**的中点。

        不是面板中心——面板被拖到边上时中心已经在屏幕外了，鼠标按下去
        什么也抓不到，于是「拖不回来」这类检查会在什么都没发生的情况下判过。
        """
        lo = max(box["x"], 0)
        hi = min(box["x"] + box["width"], vw)
        return (lo + hi) / 2, max(box["y"], 0) + frac_y

    gx, gy = grip_at(after)
    drag(page, gx, gy, 3000, 0)
    far = dialog.bounding_box()
    check("往右拖到底：停在「还留 120px」这条线上",
          abs(far["x"] - (vw - 120)) < 3, f"左边界 {far['x']:.0f}，期望 {vw - 120}")
    check("往右拖到底：仍有一块在视口里", 0 < far["x"] < vw, f"左边界 {far['x']:.0f}")

    gx, gy = grip_at(far)
    drag(page, gx, gy, 0, -3000)
    up = dialog.bounding_box()
    check("往上拖到底：顶边停在 0，标题栏不会被顶出去",
          abs(up["y"]) < 3, f"顶边 {up['y']:.0f}")

    gx, gy = grip_at(up)
    drag(page, gx, gy, -3000, 0)
    left = dialog.bounding_box()
    check("往左拖到底：右边界停在 120",
          abs(left["x"] + left["width"] - 120) < 3,
          f"右边界 {left['x'] + left['width']:.0f}")
    page.screenshot(path=str(OUT / "drag_2_corner.png"))

    # ---- 3b. 拖到边上，还得能拖回来 ---------------------------------------
    gx, gy = grip_at(left)
    drag(page, gx, gy, 420, 60)
    back = dialog.bounding_box()
    check("拖到边上还能用露出来的那一条把它拖回来",
          abs(back["x"] - (left["x"] + 420)) < 3 and abs(back["y"] - (left["y"] + 60)) < 3,
          f"({back['x']:.0f}, {back['y']:.0f}) vs 期望 ({left['x'] + 420:.0f}, {left['y'] + 60:.0f})")

    # ---- 4. 拖远了还能点回来 ---------------------------------------------
    page.mouse.click(back["x"] + back["width"] / 2, back["y"] + 200)
    check("拖动之后面板内部照常可点", dialog.is_visible())

    # ---- 5. 关掉再打开回到居中 -------------------------------------------
    page.get_by_role("button", name="关闭").click()
    page.wait_for_timeout(200)
    page.get_by_role("button", name="✍ 写故事").click()
    expect(dialog).to_be_visible(timeout=8000)
    page.wait_for_timeout(400)
    box3 = dialog.bounding_box()
    check("关掉再打开回到居中", abs(box3["x"] - box["x"]) < 2 and abs(box3["y"] - box["y"]) < 2,
          f"({box3['x']:.0f}, {box3['y']:.0f}) vs 初始 ({box['x']:.0f}, {box['y']:.0f})")
    page.screenshot(path=str(OUT / "drag_3_reopened.png"))

    # ---- 6. 别的小对话框不受影响 -----------------------------------------
    page.get_by_role("button", name="关闭").click()
    page.wait_for_timeout(150)
    page.get_by_role("button", name="📖 原文").click()
    page.wait_for_timeout(400)
    small = page.locator("div[role='dialog']").last
    expect(small).to_be_visible(timeout=8000)
    sb = small.bounding_box()
    drag(page, sb["x"] + sb["width"] / 2, sb["y"] + 20, 150, 80)
    sb2 = small.bounding_box()
    check("未开启 draggable 的对话框纹丝不动",
          abs(sb2["x"] - sb["x"]) < 1 and abs(sb2["y"] - sb["y"]) < 1,
          f"位移 ({sb2['x'] - sb['x']:.0f}, {sb2['y'] - sb['y']:.0f})")
    check("未开启 draggable 的对话框标题栏也没有抓手光标",
          page.evaluate(
              """() => {
            const d = document.querySelectorAll('[role=dialog]');
            const h = d[d.length - 1].children[0];
            return getComputedStyle(h).cursor;
          }"""
          ) == "auto",
          page.evaluate(
              """() => {
            const d = document.querySelectorAll('[role=dialog]');
            return getComputedStyle(d[d.length - 1].children[0]).cursor;
          }"""
          ))

    browser.close()

print()
print("失败项：", fail or "无")
raise SystemExit(1 if fail else 0)
