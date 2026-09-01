"""Turning source files and natural-language tasks into comparable terms.

The repo map indexes identifiers; a task is written in prose. Ranking one
against the other only works to the extent that the two vocabularies overlap,
and everything in this module exists to widen that overlap.

For English the overlap is accidental but frequent: a task about pagination
tends to mention words that appear in the names of the code that paginates. Two
things still leak — ``paginate`` is not ``pagination``, and ``OrderService`` is
not ``order``. Splitting identifiers on case and underscores fixes the second;
a shared-prefix match at the call site fixes the first.

For Chinese the overlap is zero. Paths and symbols in a Chinese codebase are
almost always English, so a Chinese term cannot match one no matter how it is
tokenised, and a ranking built on such matches degrades to a constant — every
Chinese task gets the same map. Two mechanisms restore the signal:

* **Bigrams.** Chinese is not space-delimited, so a run of characters has to be
  segmented before it can be matched at all. Overlapping bigrams are the
  standard segmenter-free approximation, and they are what makes the *comments*
  of a Chinese project searchable.
* **Translation.** A small domain dictionary maps Chinese words onto the English
  words that are actually spelled in the identifiers. This is the mechanism that
  connects 分页 to ``paginate``; nothing else in the pipeline can.

Comments and docstrings are indexed for the same reason. A file's identifiers
say what it is called, not what it is for, and in a Chinese codebase the only
Chinese in the file is in its comments.
"""

from __future__ import annotations

import re

__all__ = [
    "QUERY_STOPWORDS",
    "cjk_bigrams",
    "expand_cjk",
    "identifier_groups",
    "is_stopword",
    "prose_text",
    "split_identifier",
]

QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "change",
        "fix",
        "for",
        "implement",
        "in",
        "of",
        "or",
        "please",
        "the",
        "to",
        "update",
    }
)
"""English words a task uses to describe *doing* something rather than to
describe *what*. They match nothing useful and would rank test files first."""

_IDENTIFIER = re.compile(r"[A-Za-z0-9_$]+")
_IDENTIFIER_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z][a-z0-9]*|[0-9]+")
_CJK_RUN = re.compile(r"[㐀-鿿豈-﫿]+")

_MAX_PROSE_CHARS = 600
"""How much comment text one file contributes. Enough to carry what the file is
for; not enough to make the index a second copy of the project."""

_COMMENT_LINE = re.compile(
    r"""^\s*
    (?: //+ | \#+ | --+ | ;+ | /\*+ | \*+ (?!/) | <!-- | \"\"\" | ''' )
    \s?(.*?)\s*
    (?: \*/ | --> | \"\"\" | ''' )?\s*$""",
    re.VERBOSE,
)

_CJK_STOP_CHARS = frozenset(
    "的了是在和与我你他她它这那个们就都把被给对从很之其及并且吧呢啊嗯请帮有没不也"
    "还会能可让使于里但而或如若因所由向着得地一些多少大小已再又呀咯嘛麻烦下上中为以"
)
"""Characters that carry no retrieval signal on their own. A bigram is dropped
only when *both* of its characters are in here, which keeps 用户 and 中间 while
discarding 帮我 and 一下."""

_CJK_KEYWORDS: dict[str, tuple[str, ...]] = {
    # identity and access
    "用户": ("user",),
    "账号": ("account",),
    "账户": ("account",),
    "登录": ("login", "signin", "auth"),
    "登出": ("logout",),
    "注销": ("logout",),
    "注册": ("register", "signup"),
    "密码": ("password",),
    "鉴权": ("auth", "authorize"),
    "认证": ("auth", "authenticate"),
    "授权": ("auth", "authorize"),
    "权限": ("permission", "auth"),
    "角色": ("role",),
    "令牌": ("token",),
    "会话": ("session",),
    # commerce
    "订单": ("order",),
    "商品": ("product", "item"),
    "库存": ("stock", "inventory"),
    "支付": ("payment", "pay"),
    "付款": ("payment", "pay"),
    "退款": ("refund",),
    "购物车": ("cart",),
    "价格": ("price",),
    "优惠": ("discount", "coupon"),
    "折扣": ("discount",),
    "发票": ("invoice",),
    "物流": ("shipping", "logistics"),
    "配送": ("delivery", "shipping"),
    "地址": ("address",),
    # data access
    "分页": ("page", "paginate", "pagination", "offset", "cursor", "limit"),
    "排序": ("sort", "order"),
    "过滤": ("filter",),
    "筛选": ("filter",),
    "搜索": ("search",),
    "检索": ("search", "retrieve"),
    "查询": ("query", "search"),
    "数据库": ("database", "db"),
    "字段": ("field", "column"),
    "迁移": ("migration", "migrate"),
    "索引": ("index",),
    "事务": ("transaction",),
    "缓存": ("cache",),
    "队列": ("queue",),
    # structure
    "接口": ("api", "interface", "endpoint"),
    "路由": ("route", "router", "routing"),
    "中间件": ("middleware",),
    "控制器": ("controller",),
    "服务": ("service",),
    "模型": ("model",),
    "视图": ("view",),
    "仓库": ("repository", "repo"),
    "模块": ("module",),
    "组件": ("component",),
    "客户端": ("client",),
    "服务器": ("server",),
    "服务端": ("server", "backend"),
    "网关": ("gateway",),
    "代理": ("proxy",),
    "回调": ("callback", "webhook"),
    "钩子": ("hook",),
    "事件": ("event",),
    "消息": ("message",),
    "通知": ("notification", "notify"),
    "任务": ("task", "job"),
    "调度": ("scheduler", "schedule"),
    "定时": ("cron", "schedule", "timer"),
    # runtime
    "异步": ("async",),
    "并发": ("concurrent", "concurrency"),
    "线程": ("thread",),
    "进程": ("process",),
    "沙箱": ("sandbox",),
    "容器": ("container",),
    "镜像": ("image",),
    "快照": ("snapshot",),
    "引擎": ("engine",),
    "循环": ("loop",),
    "终端": ("terminal", "tui", "console"),
    "界面": ("ui", "interface"),
    "配置": ("config", "settings", "configuration"),
    "环境": ("env", "environment"),
    "参数": ("param", "argument", "option"),
    "常量": ("constant",),
    "变量": ("variable",),
    "日志": ("log", "logging", "logger"),
    "监控": ("monitor", "metrics"),
    "指标": ("metric", "metrics"),
    "告警": ("alert",),
    "追踪": ("trace", "tracing"),
    "错误": ("error",),
    "异常": ("exception", "error"),
    "报错": ("error",),
    "重试": ("retry",),
    "超时": ("timeout",),
    "中断": ("interrupt", "abort"),
    "取消": ("cancel",),
    "限流": ("ratelimit", "throttle"),
    "熔断": ("breaker", "circuit"),
    "降级": ("fallback",),
    "性能": ("performance", "perf"),
    "内存": ("memory",),
    "状态": ("state", "status"),
    "上下文": ("context",),
    "审批": ("approval", "approve"),
    "策略": ("policy", "strategy"),
    "注册表": ("registry",),
    "提示词": ("prompt",),
    "补全": ("completion", "complete"),
    "流式": ("stream", "streaming"),
    "截断": ("truncate", "truncation"),
    "摘要": ("summary", "summarize"),
    "压缩": ("compact", "compaction", "compress"),
    "预算": ("budget",),
    "端点": ("endpoint",),
    "密钥": ("key", "apikey"),
    "凭证": ("credential",),
    "请求": ("request",),
    "响应": ("response",),
    "编辑": ("edit",),
    "替换": ("replace",),
    "匹配": ("match", "matching"),
    "差异": ("diff",),
    "补丁": ("patch",),
    # transformation
    "校验": ("validate", "validation"),
    "验证": ("validate", "verify"),
    "序列化": ("serialize", "serialization"),
    "解析": ("parse", "parser"),
    "编码": ("encode", "encoding"),
    "解码": ("decode",),
    "加密": ("encrypt", "crypto"),
    "解密": ("decrypt",),
    "签名": ("sign", "signature"),
    "哈希": ("hash",),
    "上传": ("upload",),
    "下载": ("download",),
    "文件": ("file",),
    "目录": ("directory", "dir"),
    "路径": ("path",),
    # ui
    "页面": ("page",),
    "模板": ("template",),
    "表单": ("form",),
    "按钮": ("button",),
    "弹窗": ("dialog", "modal"),
    "对话框": ("dialog", "modal"),
    "列表": ("list",),
    "详情": ("detail",),
    "首页": ("home", "index"),
    "样式": ("style", "css"),
    "主题": ("theme",),
    "国际化": ("i18n", "locale"),
    # lifecycle
    "测试": ("test",),
    "用例": ("case", "testcase"),
    "覆盖率": ("coverage",),
    "断言": ("assert", "assertion"),
    "构建": ("build",),
    "打包": ("bundle", "package"),
    "部署": ("deploy", "deployment"),
    "发布": ("release", "publish"),
    "版本": ("version",),
    "依赖": ("dependency", "require"),
    "安装": ("install",),
    "脚本": ("script",),
    "命令": ("command", "cmd"),
    "工具": ("tool", "util"),
    "文档": ("doc", "docs"),
    "注释": ("comment",),
    "类型": ("type",),
    "函数": ("function", "func"),
    "方法": ("method", "func"),
    "初始化": ("init", "initialize"),
    "入口": ("entry", "main"),
    "启动": ("start", "startup", "boot"),
    "关闭": ("close", "shutdown"),
    "重启": ("restart",),
    "重构": ("refactor",),
    "删除": ("delete", "remove"),
    "添加": ("add", "create"),
    "新增": ("add", "create"),
    "创建": ("create", "new"),
    "生成": ("generate",),
    "读取": ("read",),
    "写入": ("write",),
    "导入": ("import",),
    "导出": ("export",),
    "修改": ("update", "modify"),
    "更新": ("update",),
    "修复": ("fix",),
}
"""Chinese domain terms mapped onto the English words a Chinese project actually
spells in its code. Deliberately hand-written and small: a general-purpose
dictionary would map 服务 onto a dozen senses, and every extra sense is a file
promoted for the wrong reason. Entries whose translation is a query stopword
(``fix``, ``update``) are kept for completeness and filtered by the caller."""

_KEYWORDS_BY_LENGTH: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    sorted(_CJK_KEYWORDS.items(), key=lambda item: (-len(item[0]), item[0]))
)
"""Longest first, so 中间件 is matched before 中间 could be."""


def identifier_groups(text: str) -> list[set[str]]:
    """One word set per identifier in ``text``, rather than one flat set.

    ``OrderService`` is a single thing the task named, not three; keeping its
    parts together lets the caller pay for it once instead of three times.
    """
    return [split_identifier(chunk) for chunk in _IDENTIFIER.findall(text)]


def split_identifier(text: str) -> set[str]:
    """Lowercase words from identifiers, paths, and prose.

    ``src/api/order_service.ts`` and ``cancelProviderStream`` both yield their
    parts *and* their whole, because a task may name either one.
    """
    words: set[str] = set()
    for chunk in _IDENTIFIER.findall(text):
        lowered = chunk.lower()
        if len(lowered) >= 2:
            words.add(lowered)
        for part in _IDENTIFIER_WORD.findall(chunk):
            if len(part) >= 2 or part.isdigit():
                words.add(part.lower())
    return words


def cjk_bigrams(text: str) -> set[str]:
    """Overlapping character bigrams for every run of Chinese in ``text``.

    Segmentation without a segmenter: 订单接口 becomes 订单/单接/接口, one of
    which is the word the reader meant. Bigrams made entirely of function
    characters are dropped — they match everywhere and mean nothing.
    """
    grams: set[str] = set()
    for run in _CJK_RUN.findall(text):
        for index in range(len(run) - 1):
            gram = run[index : index + 2]
            if gram[0] in _CJK_STOP_CHARS and gram[1] in _CJK_STOP_CHARS:
                continue
            grams.add(gram)
    return grams


def expand_cjk(text: str) -> list[tuple[str, ...]]:
    """English identifier words implied by the Chinese in ``text``, per word.

    This is the step that lets a Chinese task rank an English codebase. Matching
    is by substring against each Chinese run, longest key first, because the
    runs are unsegmented — 订单分页错误 has to yield 订单 and 分页 and 错误.

    One tuple per Chinese word, not one flat set. The senses within a tuple are
    alternatives for the same concept and must be scored as one; the tuples are
    different concepts and must be scored separately, or a Chinese task would
    be paid once where the English phrasing of it is paid three times.
    """
    expanded: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for run in _CJK_RUN.findall(text):
        remaining = run
        for chinese, english in _KEYWORDS_BY_LENGTH:
            if chinese in remaining and chinese not in seen:
                seen.add(chinese)
                expanded.append(english)
                remaining = remaining.replace(chinese, "　")
    return expanded


def prose_text(source: str, docstrings: tuple[str, ...] = ()) -> str:
    """The natural language inside a source file, capped and flattened.

    Comments and docstrings are where a file says what it is *for*, which is the
    only thing in it phrased the way a task is phrased. Chinese literals are
    picked up too — an error message reading 订单不存在 identifies the file that
    handles orders as reliably as a comment would.

    Args:
        source: The file's text.
        docstrings: Already-extracted docstrings, when the language has a parser
            that can produce them more accurately than a line scan.
    """
    collected: list[str] = list(docstrings)
    budget = _MAX_PROSE_CHARS - sum(len(item) for item in collected)

    for line in source.splitlines():
        if budget <= 0:
            break
        match = _COMMENT_LINE.match(line)
        if match is None:
            # Not a comment, but a Chinese string literal is prose all the same.
            for run in _CJK_RUN.findall(line):
                if len(run) >= 2:
                    collected.append(run)
                    budget -= len(run)
            continue
        body = match.group(1).strip()
        if not body:
            continue
        collected.append(body[:budget])
        budget -= len(body)

    return "\n".join(collected)[:_MAX_PROSE_CHARS]


def is_stopword(term: str) -> bool:
    """Whether a term is too common in task phrasing to rank anything."""
    return term in QUERY_STOPWORDS
