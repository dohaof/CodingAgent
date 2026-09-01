# Cagent

`cagent` 是一个面向代码任务的轻量TUI智能体。

它可以查看项目文件、搜索代码、编辑文件、运行测试和命令，并根据执行结果继续
工作，直到任务完成或触发限制。

## 功能概览

- 支持一次性任务和全屏交互式会话，另有可选的 Electron 桌面界面（`cagent --gui`）。
- 提供文件读取、精确编辑、多处编辑、目录浏览、文件名匹配和正则搜索。
- 可以执行项目自己的测试、构建和脚本，并处理超时、失败和输出截断。
- 自动生成项目结构索引（Repo Map），帮助模型快速定位相关文件。
- 自动读取工作区根目录的 `AGENTS.md`，将项目约定加入 system prompt；切换到 Docker
  沙箱后读取沙箱副本中的同一文件，且不会因此放宽工具权限或路径边界。
- 对较长任务自动压缩上下文，同时保留原始任务和最近操作。
- 只读查询工具可以并行执行，写入和命令执行保持有序。
- 文件修改显示 diff；命令和高风险操作可按审批策略确认。
- 可选 Docker 临时沙箱、JSONL 调试记录、会话恢复、撤销上下文和费用统计。

## 环境要求

- Python 3.11 或更高版本。
- Node.js 18 或更高版本，仅在使用桌面界面（`cagent --gui`）时需要。
- 一个可访问的模型接口，支持 OpenAI Chat Completions 格式或 Anthropic
  Messages 格式。
- Docker 仅在需要隔离执行命令时使用，无Docker只能无限制执行命令或者只执行只读命令。
- Windows、Linux 和 macOS 均可运行。Linux/macOS 优先使用 Bash；Windows
  会优先使用 Git Bash，没有可用 Bash 时使用系统 shell。

## 安装

### 一键安装（推荐）

仓库根目录提供安装脚本，一次装好命令行和桌面界面：

- **Windows**：双击 `install.cmd`
- **macOS / Linux**：`./install.sh`

脚本会依次检查 Python 与 Node.js 版本、用 `pipx` 安装 `cagent`、构建 Electron
客户端，最后把客户端路径写入 `~/.cagent.toml` 的 `desktop_path`。装完在任意
目录即可使用：

```bash
cagent           # 终端界面
cagent --gui     # 桌面窗口
```

每一步都是幂等的，中途失败（缺 Node、下载中断）直接重跑即可。可用参数：

| 参数 | 说明 |
| --- | --- |
| `-y`、`--yes` | 所有确认都回答 yes，适合无人值守 |
| `--skip-cli` | 不动已安装的 `cagent`，只准备桌面客户端 |
| `--skip-desktop` | 只装命令行，不构建桌面客户端 |

写入配置时会先备份原文件为 `.cagent.toml.bak`，并校验改写后的文档仍能解析、
且原有每一项设置都保持原值 —— 任一条不满足就放弃写入。这个文件通常存着
API key，注释和顺序也都会保留。

### 手动全局安装

使用 `pipx` 可以把 `cagent` 安装到独立环境，并让它在任意项目目录中可用。

Windows PowerShell：

```powershell
py -m pip install --user pipx
py -m pipx ensurepath
py -m pipx install C:\path\to\CodingAgent
```

Linux 或 macOS：

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
python3 -m pipx install /path/to/CodingAgent
```

执行 `ensurepath` 后重新打开终端，然后检查：

```bash
cagent --version
```

从源码更新后重新安装：

```bash
pipx reinstall coding-agent
```

卸载：

```bash
pipx uninstall coding-agent
```

### 开发安装

在仓库目录中创建虚拟环境并以可编辑模式安装：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell 使用：.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

开发安装会提供测试、类型检查和代码检查工具。使用项目虚拟环境时，先激活
它再运行 `cagent`，这样智能体执行的 `python`、`pytest` 等命令会使用项目
自己的依赖。

## 模型配置

配置文件只有一个 `[cagent]` 表。配置加载顺序如下，后者覆盖前者：

```text
内置默认值 → ~/.cagent.toml → <工作区>/.cagent.toml → CLI 参数
```

复制仓库中的 `.cagent.example.toml` 为 `.cagent.toml`，或直接创建：

```toml
[cagent]
base_url = "https://api.example.com/v1"
model = "你的模型名称"
api_key = "你的 API Key"
wire = "openai"
# reasoning_effort = "high"
```

`base_url` 会按原值使用，并在末尾追加请求路径：OpenAI 格式追加
`/chat/completions`，Anthropic 格式追加 `/messages`。请按照服务商要求保留
版本路径（通常是 `/v1`）。使用 Anthropic Messages API 时设置：

```toml
wire = "anthropic"
```

可设置 `reasoning_effort` 控制单次请求的推理强度。OpenAI-compatible wire 支持
`none`、`minimal`、`low`、`medium`、`high`、`xhigh` 或 `max`，请求字段为
`reasoning_effort`；Anthropic wire 支持 `low`、`medium`、`high`、`xhigh` 或 `max`，
请求字段为 `output_config.effort`。不同模型支持的档位可能不同；不设置该项时
不会发送 effort 字段，由模型或网关采用默认值。

本地无需密钥的服务（例如 Ollama 或 llama.cpp）可以设置：

```toml
requires_key = false
```

密钥只应放在未提交的 `.cagent.toml` 或用户目录配置中，不支持通过 CLI 传入。
运行 `cagent --show-config` 可以查看最终生效的地址、模型、工作区、审批和
沙箱设置；

### 费用配置

项目不内置价格表。若希望在 `/cost` 和会话总结中显示费用，在配置中添加
每百万 token 的美元价格：

```toml
[cagent.prices."你的模型名称"]
input_per_m = 0.27
output_per_m = 1.10
cached_input_per_m = 0.07
```

模型名按最长前缀匹配，因此同一厂商的日期版本也可以复用配置。未配置价格时, 仍会显示 token 数量

### 常用配置项

除接口信息外，以下设置也可以直接写入 `[cagent]`：

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `max_output_tokens` | `8192` | 单次模型回复的最大 token 数 |
| `reasoning_effort` | 未设置 | 请求级推理强度；OpenAI 支持 `none` 到 `max`，Anthropic 支持 `low`、`medium`、`high`、`xhigh`、`max` |
| `request_timeout` | `120` | 模型请求超时时间（秒） |
| `max_retries` | `4` | 请求失败时的最大重试次数 |
| `context_window` | `128000` | 模型上下文窗口大小 |
| `compact_threshold` | `0.75` | 上下文达到窗口比例后开始压缩 |
| `keep_recent_turns` | `6` | 压缩时保留的最近步骤数 |
| `bash_timeout` | `120` | shell 命令默认超时时间（秒） |
| `approval_mode` | `auto-edit` | `suggest`、`auto-edit` 或 `full-auto` |
| `repo_map_enabled` | `true` | 是否生成 Repo Map |
| `repo_map_token_budget` | `1600` | 结构索引的 token 预算；放在系统提示词里，逐轮不变以便命中 provider 的前缀缓存 |
| `repo_map_focus_token_budget` | `1200` | 任务相关文件清单的 token 预算；随每轮用户输入发送，因此重排不会作废缓存 |
| `prompt_caching` | `true` | 为 Anthropic 请求标记缓存断点（工具 + 系统提示词、以及会话前缀）。OpenAI 兼容端点自动缓存，忽略此项 |
| `trace_dir` | 工作区下 `.cagent/traces` | JSONL trace 目录；用 `--no-trace` 关闭 |

## 快速开始

在目标项目目录执行：

```bash
cagent "运行测试并修复失败"
```

不带任务参数时进入交互会话：

```bash
cagent
```

默认工作区是启动命令时的当前目录。也可以从其他目录指定项目：

```bash
cagent --workspace /path/to/project "检查并修复测试"
# Windows 示例
cagent --workspace D:\Projects\MyProject "运行测试并修复失败"
```

查看版本、配置和工具：

```bash
cagent --version
cagent --show-config
cagent --list-tools
```

## 交互界面

连接到终端时，`cagent` 使用全屏界面显示对话、模型输出、工具调用、审批提示、
diff 和实时状态。通过管道或重定向输入时，会自动切换为适合脚本和日志的逐行模式。

工具需要确认时，在输入框输入：

- `y`：允许本次操作（直接回车也表示允许）；
- `n`：拒绝本次操作，并把拒绝原因返回给模型；
- `a`：在允许时记住同类操作；
- `q`：停止当前任务。

常用快捷键：

| 快捷键 | 作用 |
| --- | --- |
| `Ctrl+C` | 有选中文本时复制；有输入时清空输入；任务运行中中断任务；空闲且输入为空时退出 |
| `Ctrl+Q` | 退出交互会话 |
| `Ctrl+R` | 打开会话恢复列表 |
| `F1` | 打开帮助 |

## 桌面界面（可选）

除终端界面外，仓库的 `desktop/` 目录提供一个 Electron 桌面客户端。它不是另一
个智能体：窗口只负责渲染，上下文管理、工具执行、审批和沙箱仍全部由同一个
Python 后端持有，两者之间通过 stdin/stdout 上的 JSONL 协议通信
（`cagent.gui.bridge`）。

### 构建

用一键安装脚本（`install.cmd` / `install.sh`）会自动完成构建和路径登记。手动
构建需要 Node.js 18 或更高版本：

```bash
cd desktop
npm install       # 会自动执行一次 npm run build
```

### 启动

```bash
cagent --gui            # 打开桌面窗口，工作区为当前目录
cagent --gui ~/project  # 指定工作区
```

使用 `--gui` 时，任务参数的位置改为接收工作区目录 —— 桌面客户端有自己的输入
框，命令行不再需要传入任务。窗口的设置由 `.cagent.toml` 决定，`--model` 等
命令行覆盖不会转发给它（传了会给出提示）。

### 告诉 cagent 桌面客户端在哪里

Electron 运行时约 500 MB，无法随 Python 包一起分发，因此全局安装的 `cagent`
需要知道仓库位置。查找顺序为：

1. 环境变量 `CAGENT_DESKTOP`；
2. 配置项 `desktop_path`；
3. 与已安装包相邻的源码检出（源码运行或可编辑安装时自动命中）。

用 `pipx` 全局安装时前两条都没有设置 —— 一键安装脚本做的就是替你填好第 2 条。
手动设置则在 `~/.cagent.toml` 中加上：

```toml
[cagent]
desktop_path = "/path/to/CodingAgent/desktop"
```

启动器会把当前解释器（`sys.executable`）通过 `CAGENT_PYTHON` 传给桌面进程，
这样 `pipx` 独立环境中的 Python 才能被用来跑后端 —— 否则桌面进程只能退回到
`PATH` 上的裸 `python`，而那个解释器里并没有安装 cagent。

### 桌面界面操作

- 左侧会话列表的每张卡片右上角有 `⋯` 菜单：打开会话、复制 trace 路径、删除
  会话（两步确认；当前正在进行的会话不可删除，因为它的 trace 仍在写入）。
- 工具结果卡片（如 `read_file`）、恢复历史中的调用记录和斜杠命令输出，点击
  卡片头部即可折叠或展开。折叠后标题仍显示行数。
- 右下角的步数是**整段对话**的模型请求数，包含从已保存会话恢复的部分。

## 交互命令

在交互会话中输入以下命令。输入 `/help <命令>` 可以查看对应帮助。

| 命令 | 作用 |
| --- | --- |
| `/help [主题]` | 显示帮助或指定主题的详细说明 |
| `/tools` | 列出当前启用的工具及参数结构 |
| `/cost` | 显示 prompt、completion、缓存 token 和步骤数 |
| `/context` | 显示当前上下文占用、消息数和压缩次数 |
| `/effort` | 查看推理强度；使用 `/effort high` 等命令调整后续请求，`/effort default` 恢复模型默认值 |
| `/approve` | 查看当前审批模式 |
| `/approve suggest` | 后续每次文件修改和命令都请求确认 |
| `/approve auto-edit` | 工作区内文件编辑自动执行，命令仍请求确认 |
| `/approve full-auto` | 仅危险操作请求确认 |
| `/resume [编号、ID 或路径]` | 恢复已保存的会话，不带参数时打开选择列表 |
| `/undo` | 从上下文移除最近一轮对话 |
| `/clear` | 清空当前对话上下文 |
| `/sandbox ...` | 查看或控制 Docker 沙箱 |
| `/exit`、`/quit` | 退出会话 |

`/undo` 只改变模型上下文，不会撤销已经写入的文件、执行过的命令、安装的
依赖或其他副作用。需要还原文件时请使用版本控制工具或沙箱的
`/sandbox rollback`。

## CLI 参数

任务可以放在所有选项之后，也可以直接用引号包起来。完整帮助可通过
`cagent --help` 查看。

### 模型接口

| 参数 | 说明 |
| --- | --- |
| `--base-url URL` | 覆盖 API 地址 |
| `--model NAME` | 覆盖模型名称 |
| `--wire openai 或 anthropic` | 选择请求格式 |
| `--no-key` | 声明接口不需要 API Key |
| `--temperature FLOAT` | 覆盖采样温度 |
| `--reasoning-effort LEVEL` | 设置请求级推理强度；Anthropic 不支持 `none` 和 `minimal` |

### 限制与工作区

| 参数 | 说明 |
| --- | --- |
| `--token-budget INT` | 本次任务允许消耗的 token 上限 |
| `--context-window INT` | 模型上下文窗口大小，用于上下文压缩 |
| `--bash-timeout FLOAT` | 单条命令的超时时间（秒） |
| `--workspace PATH` | 指定工作区，默认是当前目录 |
| `--allow-outside-workspace` | 允许文件工具访问工作区之外的路径 |

### 审批与沙箱

| 参数 | 说明 |
| --- | --- |
| `--approval suggest、auto-edit 或 full-auto` | 设置审批策略 |
| `-y`、`--yes` | `--approval full-auto` 的简写 |
| `--sandbox auto、off 或 docker` | 自动选择、关闭或强制使用 Docker |
| `--sandbox-sync never、ask 或 always` | 沙箱结束时丢弃、询问或自动同步修改 |
| `--sandbox-image IMAGE` | 指定本地 Docker 镜像 |
| `--sandbox-network` | 允许容器使用 Docker 默认 bridge 网络；默认断网 |
| `--sandbox-memory-mb INT` | 沙箱内存上限，单位 MiB |
| `--sandbox-cpus FLOAT` | 沙箱 CPU 上限 |
| `--sandbox-pids INT` | 沙箱进程数上限 |
| `--sandbox-workspace-mb INT` | 沙箱工作区普通文件大小上限 |

### 输出与诊断

| 参数 | 说明 |
| --- | --- |
| `--no-repo-map` | 不生成项目结构索引 |
| `--no-thinking` | 隐藏模型的思考过程显示 |
| `--quiet` | 只显示警告和最终总结 |
| `--trace-dir PATH` | 将 JSONL trace 写入指定目录 |
| `--no-trace` | 关闭 trace 记录 |
| `--show-config` | 显示解析后的配置并退出 |
| `--list-tools` | 显示工具及 schema 并退出 |
| `--version` | 显示版本并退出 |

### 界面

| 参数 | 说明 |
| --- | --- |
| `--gui [工作区]` | 打开 Electron 桌面客户端，替代终端界面 |

## Docker 沙箱

沙箱用于把命令和文件操作放在项目临时副本中执行。默认配置为：

```text
sandbox_mode = "auto"
sandbox_sync = "ask"
sandbox_image = "python:3.12-slim"
sandbox_network = false
```

`auto` 只有在 Docker daemon 和镜像都已存在时才启用隔离；条件不满足会给出
警告并回退到宿主机。镜像不会自动拉取。要强制隔离，可使用：

```bash
cagent --sandbox docker --sandbox-image my-project-agent:latest "运行测试"
```

在交互会话中也可以控制：

```text
/sandbox                         查看状态
/sandbox image my-project-agent:latest
/sandbox sync ask
/sandbox on
/sandbox apply                   立即把当前修改同步到真实项目
/sandbox rollback                丢弃尚未同步的修改
/sandbox off                     按同步策略退出沙箱
```

沙箱使用一次临时快照和一个会话级容器，容器结束后会被删除。默认网络关闭；如确需在容器内下载依赖，
可在未提交的 `.cagent.toml` 的 `[cagent]` 表中设置 `sandbox_network = true`（使用 Docker 默认 bridge 网络），
或本次运行传入 `--sandbox-network`。开启网络会扩大命令的外部访问面，请仅在可信项目和必要时使用。
根文件系统只读，并限制内存、CPU、进程数和工作区大小。`never` 会丢弃修改，
`ask` 退出时展示受限 diff 并询问，`always` 自动同步。宿主机模式下，shell
进程本身不受工作区边界限制；审批和路径检查不能替代进程隔离。

如果项目依赖 Node、Java 或其他工具，建议预先构建本地镜像：

```dockerfile
FROM node:22-bookworm
RUN npm install -g pnpm
```

```bash
docker build -f Dockerfile.agent -t my-project-agent:latest .
```

请不要把密钥或 `.cagent.toml` 放入 Docker 构建上下文。

## Agent 工具

模型当前可以调用以下 8 个工具：

| 工具 | 用途 | 主要参数 |
| --- | --- | --- |
| `read_file` | 分页读取文本文件并显示行号 | `path`, `offset`, `limit` |
| `write_file` | 创建文件或完整覆盖文件 | `path`, `content` |
| `edit_file` | 替换文件中的唯一文本片段并生成 diff | `path`, `old_string`, `new_string`, `replace_all` |
| `multi_edit` | 按顺序原子执行同一文件的多处替换 | `path`, `edits` |
| `list_dir` | 查看有限深度的目录树 | `path`, `depth`, `max_entries` |
| `glob_files` | 按 glob 模式查找文件 | `pattern`, `path` |
| `grep_search` | 按正则表达式搜索文件内容 | `pattern`, `path`, `glob`, `case_sensitive`, `context`, `max_results` |
| `run_bash` | 执行测试、构建和其他 shell 命令 | `command`, `timeout`, `description` |

只读查询工具可以并行运行；文件写入和命令执行会按顺序进行。`edit_file`
会依次尝试精确、空白归一化和模糊匹配，匹配不唯一时拒绝修改；
`multi_edit` 任一步失败都会使整个批次保持不变。

## Repo Map 与上下文

启动任务时，Agent 可以生成项目结构索引，包含源文件路径、声明和导入关系，
并按任务相关性排序。它是导航摘要，不会替代 `read_file`，也不会自动下载语法
解析器。

长任务接近 `context_window` 时会自动压缩上下文：优先省略旧工具输出，必要时
总结旧步骤，最后才移除更早步骤。最初的用户任务和最近操作会保留。可用
`--context-window` 调整窗口，用 `/context` 查看当前压力。

## Trace 与会话恢复

默认 trace 位于：

```text
<workspace>/.cagent/traces/*.jsonl
```

每行是一个事件，便于在任务中断后检查过程。也可以用 `--trace-dir PATH` 指定
目录，或用 `--no-trace` 完全关闭记录。

恢复方式：

```text
cagent
/resume                         选择最近会话
/resume 1                       按列表编号恢复
/resume 20260831-abcdef         按会话 ID 或前缀恢复
/resume path/to/session.jsonl   从指定文件恢复
```

恢复只加载对话历史，继续使用当前工作区、配置、审批状态和沙箱；不会恢复旧
容器或未同步的文件。会话选择器默认只显示包含有效用户请求的记录。

## 常见问题

**提示缺少 `base_url` 或 `model`。** 在工作区或用户目录创建 `.cagent.toml`，
然后运行 `cagent --show-config` 检查最终配置。

**首次请求返回 404。** 检查 `base_url` 是否包含服务商要求的版本路径，并确认
`wire` 与接口格式匹配。

**输入 `cagent` 提示找不到命令。** 重新打开终端，确认 `pipx ensurepath` 添加
的目录在 `PATH` 中；也可以在已激活的项目虚拟环境中重新执行开发安装。

**Docker 没有启用。** `auto` 模式要求 Docker daemon 正在运行且镜像已在本地。
用 `cagent --show-config` 查看状态；需要严格隔离时使用 `--sandbox docker`，
缺少条件时它会直接失败而不会回退到宿主机。

**模型改错了文件。** 查看工具输出的 diff；`/undo` 只能撤销对话上下文，文件
本身请使用 Git、编辑器历史或沙箱的 `/sandbox rollback` 恢复。

## 开发检查

```bash
python -m pytest
python -m ruff check .
python -m mypy src
```
