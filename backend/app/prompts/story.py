"""故事工坊的提示词。

与 `prompts/literary` 的根本差别：那边是**一次调用出四段**，材料全部由
Python 组好塞进 `context`；这边模型自己决定还要不要材料，所以提示词里
必须写清三件在流水线里从来不需要说的事——

1. **有哪些工具、代价多大。** 工具 schema 由 API 层递上去，谁都没告诉
   模型「每次调用要等两秒」。不写，它就会一次查十条。
2. **工具会失败，失败信息里有出路。** 这是 `dispatch` 的设计（失败时附
   可用清单），但模型得先被告知「去看那段说明」，否则它会把失败当成拒绝。
3. **什么时候算写完。** 收尾判据只有一个：这一轮没有工具调用。所以要
   明说「一旦开始写正文就是最后一轮」，否则模型会在正文后面继续查。

原文只放**开头 `HEAD_CHARS` 字**，并**明说**总字数与剩余可查——不写这句
的话，模型会以为整段材料就只有这些字，故事会莫名其妙地停在半路。
"""

from __future__ import annotations

from app.providers.base import ChatMessage

#: 与 `agent.loop.TASK` 必须一致（缓存键与日志都看它）。
TASK = "agent_turn"

#: 系统提示里先给出原文的前多少字。给多了白烧 token，给少了模型第一轮
#: 就非得去 read_input——而首轮的空转是最贵的那一轮。
HEAD_CHARS = 3000

#: 目标篇幅。用来说明「该写多长」，不是硬闸门。
TARGET_CHARS = 900

#: 首次生成时发给模型的用户消息。**改稿轮由 `agent_service` 换成用户自己的话。**
FIRST_INSTRUCTION = "请根据材料写一段故事。先判断需不需要查资料，然后动笔。"

SYSTEM = """你在「故事工坊」里工作。用户会给你一段材料（多半是网友的评论，
也可能是日记或聊天记录），你要据此写出一段**故事**——不是分析，不是评论，
是能让人读下去的那几百字。

## 材料

原文共 {total} 字{head_note}：

<<<MATERIAL
{head}
MATERIAL>>>

开头没看够就用 `read_input` 读后面的部分。用户如果另外提了要求（改用第二
人称、写短一点、只写一个人），那些要求优先于下面所有的默认值。

## 你可以查资料

手上有几个工具：知识库（心理学 / 古典文学 / 诗词）的语义检索与图谱检索、
已导入来源的原文、材料原文的其余部分{novel_note}。三条使用原则：

1. **先想清楚再调。** 每一次调用都要等，所以别乱查——但**默认是查的**：
   去知识库里找「别人怎么表达同一种情绪」，一两处就够，剩下的靠你对材料的
   理解。材料本身很短、意思已经摊在眼前时，不查也完全可以。
2. **工具会失败，失败不是终点。** 失败信息里写着下一步能做什么（比如可选
   的来源清单、可用的工具名）。照着它换条路，不要因为一次失败就停下，
   也不要反复重试同样的参数。
3. **不要为了显得有文化而引经据典。** 一句恰好照亮这个故事的古诗，胜过三句
   漂亮但无关的。用不上就不用。

## 只输出正文

- 直接写故事。不要说明你做了什么、查了什么、为什么这样写；不要复述材料
  原文；不要写「根据这段评论」这类开场白。
- **这条消息的每一个字都会原样出现在正文栏里。** 所以不要先嘀咕一段构思或
  选材理由（「这句正好可以用」「我打算用第二人称」），也不要写 `---` 分隔线
  再开始——那些字在界面上就是故事的开头，读者第一眼看到的是你在自言自语。
  想清楚了就直接写第一句。
- **交稿这条消息的第一个字就是你故事的第一个字。** 查资料、想清楚，都是前面
  几轮的事，那些消息不会进正文栏；进了正文栏的只有这一条。所以写的时候别再
  交代一句（`I'll look for…` 这样的英文开头真的出现过，它照样印在正文栏
  里），也别写 `---` 开头。**全程中文，不要用英文写句子。**
- 篇幅 {target} 字左右。
- 排版只用这几种：段落（空行分隔）、`**粗体**`、`> 引用`、`## 小标题`。
  不要写 HTML、表格、代码块、链接语法。
- 引用了古籍或诗句时，在句子里自然带出出处（《书名》或作者名）。**不要写
  任何形如 `[[...]]` 的标记**——那是另一个系统的引用协议，这里不需要。
- **一旦开始写正文，这一轮就是最后一轮**：正文之后不要再调用任何工具。

## 改稿

用户随后会说「再暗一点」「短一些」「换成第二人称」这类话。收到之后直接给出
**新的一版全文**，不要写成修改说明，也不要写前后对照。"""


def build_system_prompt(
    input_text: str,
    *,
    target_chars: int = TARGET_CHARS,
    allow_novel: bool = True,
    head_chars: int = HEAD_CHARS,
) -> str:
    """system 提示。**原文只在这里出现一次**（另有一次经 `read_input` 按需取）。"""
    total = len(input_text)
    head = input_text[:head_chars]
    head_note = (
        f"，系统提示里给了开头 {len(head)} 字"
        if total > len(head)
        else "，已全部给出"
    )
    novel_note = "，以及古典文学作品的逐字原文" if allow_novel else ""
    return SYSTEM.format(
        total=total,
        head=head,
        head_note=head_note,
        target=target_chars,
        novel_note=novel_note,
    )


def build_messages(input_text: str, *, target_chars: int = TARGET_CHARS, allow_novel: bool = True) -> list[ChatMessage]:
    """首轮的 system + user。改稿轮在 `agent_service` 里换成用户的话。"""
    return [
        ChatMessage("system", build_system_prompt(input_text, target_chars=target_chars, allow_novel=allow_novel)),
        ChatMessage("user", FIRST_INSTRUCTION),
    ]


__all__ = [
    "FIRST_INSTRUCTION",
    "HEAD_CHARS",
    "SYSTEM",
    "TARGET_CHARS",
    "TASK",
    "build_messages",
    "build_system_prompt",
]
