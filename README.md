# PaddleOCR MCP Server

将 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)(PP-OCRv5)封装为标准的
[MCP (Model Context Protocol)](https://modelcontextprotocol.io) 服务,支持
**中文、英文、日文、韩文**的文本识别,可被 Claude Desktop、Cursor、Codex 等
任意 MCP 客户端调用,也可作为 Docker 镜像通过网络服务对外提供 OCR 能力。

提供两种运行方式,均支持 HTTP(`streamable-http`)传输:

- **Docker**(推荐):镜像内置四种语言模型,构建后可完全离线运行。
- **Python 本地**:首次运行自动下载模型,之后同样可离线。

## 功能特性

- 基于官方 MCP Python SDK(FastMCP),标准 `stdio` 与 `streamable-http` 双传输。
- 每种语言独立缓存 OCR 引擎实例,首次调用懒加载、后续秒级响应。
- 图片输入支持本地路径、`http(s)` URL、`data:` URI、裸 base64 字符串。
- 返回每行文本、置信度和文本框坐标。
- Docker 镜像内置全部四种语言模型,**构建后可完全离线运行**。

## 支持的语言

| 代码 | 语言 | 说明 |
|------|------|------|
| `ch` | 中文 + 英文 | 默认值,PP-OCRv5,适合中英混排 |
| `en` | 英文 | 拉丁文字 |
| `japan` | 日文 | 日语 |
| `korean` | 韩文 | 韩语 |

同时接受常用别名:`zh` / `zh-cn` / `chinese` / `cn`(中文),`english`(英文),
`ja` / `japanese`(日文),`ko` / `kr` / `hangul`(韩文)。

## Quickstart

两种方式都默认以 HTTP 服务启动,监听 `0.0.0.0:8000`,MCP 端点为
`http://<主机IP>:8000/mcp`。

### 方式一:Docker(推荐)

镜像构建时会预下载四种语言模型,容器启动即可调用、无需联网。

#### 从 Docker Hub 拉取预构建镜像

仓库提供手动触发的 [构建 workflow](.github/workflows/dockerhub-publish.yml),会在原生 runner 上并行构建 amd64 与 arm64 双架构镜像,合并为单个多架构 manifest 后发布到 Docker Hub。直接拉取即可,免去本地构建:

```bash
# 拉取镜像(自动匹配当前主机的 amd64 / arm64 架构)
docker pull pewee-live/ocr-mcp:latest

# 启动服务(仅本机访问)
docker run -d --name ocr-mcp -p 8000:8000 pewee-live/ocr-mcp:latest
```

> `pewee-live` 为示例命名空间,请替换为你的 Docker Hub 用户名(需与仓库 Secrets 中的 `DOCKERHUB_USERNAME` 一致)。
> 触发方式:GitHub 仓库 → Actions → 选择 **Build & Push to Docker Hub** → Run workflow,按需填写镜像名、标签与是否同步打 `:latest`。首次使用前需在 Secrets 中配置 `DOCKERHUB_USERNAME` 与 `DOCKERHUB_TOKEN`(Access Token,非密码)。

#### 本地构建镜像

```bash
# 构建镜像(首次会下载 paddlepaddle 与四种语言模型,约需数分钟)
docker build -t ocr-mcp:latest .

# 启动服务(仅本机访问)
docker run -d --name ocr-mcp -p 8000:8000 ocr-mcp:latest

# 如果要让局域网/其他机器通过 IP 访问,需要放开 Host 白名单:
docker run -d --name ocr-mcp -p 8000:8000 \
  -e MCP_ALLOWED_HOSTS="192.168.7.49:*,localhost:*" \
  -e MCP_ALLOWED_ORIGINS="http://192.168.7.49:*,http://localhost:*" \
  ocr-mcp:latest

# 查看日志
docker logs -f ocr-mcp
```

启动后,MCP 端点为 `http://<容器所在主机IP>:8000/mcp`。

### 方式二:Python 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 以 HTTP 服务启动(仅本机访问)
MCP_TRANSPORT=streamable-http python -m ocr_mcp

# 监听所有网卡,供局域网访问(并把本机 IP 加入白名单)
MCP_TRANSPORT=streamable-http \
MCP_HOST=0.0.0.0 \
MCP_PORT=8000 \
MCP_ALLOWED_HOSTS="192.168.7.49:*,localhost:*" \
MCP_ALLOWED_ORIGINS="http://192.168.7.49:*,http://localhost:*" \
python -m ocr_mcp
```

> Windows PowerShell 设置环境变量用 `$env:MCP_TRANSPORT="streamable-http"` 等,
> 或写成一行:`$env:MCP_TRANSPORT='streamable-http'; python -m ocr_mcp`。

首次运行时会自动从 Hugging Face 下载 PP-OCRv5 模型到本地缓存目录
(`~/.paddlex/official_models`),之后可离线使用。

### 切换模型:server(高精度)vs mobile(快速)

本项目内置两套 PP-OCRv5 模型,Docker 镜像构建时已**同时预下载**,运行时通过环境变量切换,无需重新构建镜像或联网下载:

| 变体 | 检测模型 | 识别模型 | 特点 |
|------|---------|---------|------|
| mobile(默认) | `PP-OCRv5_mobile_det` | `PP-OCRv5_mobile_rec` | 快 3-5 倍,适合运单等清晰印刷体 |
| server | `PP-OCRv5_server_det` | `PP-OCRv5_server_rec` | 精度最高、最慢,适合复杂场景 |

> 也可混搭,例如 mobile 检测(快)+ server 识别(准),只设一个、留空另一个即可。

#### Docker 方式切换

镜像里两套模型都预装了,启动时通过 `-e` 指定即可,切换后**首次调用需将模型加载进内存(数秒)**:

```bash
# 默认(mobile,快)— 镜像内置,直接 run
docker run -d -p 8000:8000 ocr-mcp:latest

# 切到 server(精度最高,慢)
docker run -d -p 8000:8000 \
  -e OCR_DET_MODEL=PP-OCRv5_server_det \
  -e OCR_REC_MODEL=PP-OCRv5_server_rec \
  ocr-mcp:latest

# 混搭:mobile 检测(快)+ server 识别(准)
docker run -d -p 8000:8000 \
  -e OCR_DET_MODEL=PP-OCRv5_mobile_det \
  -e OCR_REC_MODEL=PP-OCRv5_server_rec \
  ocr-mcp:latest
```

#### Python 方式切换

Python 方式同样用 `OCR_DET_MODEL` / `OCR_REC_MODEL`,首次切换会联网下载对应模型(之后缓存):

```bash
# 默认 server(Python 不像 Docker 有预设,需手动指定,否则按 PP-OCRv5 自动选 server)
MCP_TRANSPORT=streamable-http python -m ocr_mcp

# 切到 mobile(快)
MCP_TRANSPORT=streamable-http \
OCR_DET_MODEL=PP-OCRv5_mobile_det \
OCR_REC_MODEL=PP-OCRv5_mobile_rec \
python -m ocr_mcp

# 切到 server(精度最高)
MCP_TRANSPORT=streamable-http \
OCR_DET_MODEL=PP-OCRv5_server_det \
OCR_REC_MODEL=PP-OCRv5_server_rec \
python -m ocr_mcp
```

> Windows PowerShell:
> `$env:OCR_DET_MODEL='PP-OCRv5_mobile_det'; $env:OCR_REC_MODEL='PP-OCRv5_mobile_rec'; python -m ocr_mcp`

## 重要:跨机访问与 Host 白名单

MCP SDK 默认开启 **DNS 重绑定防护**,只放行本机地址:
`127.0.0.1`、`localhost`、`[::1]`。

因此,**只要你是通过 IP、域名或另一台机器访问**(例如
`http://192.168.7.49:8000/mcp`),就**必须**把该访问地址加入白名单,
否则请求会被拒绝并返回:

```
Invalid Host header   (HTTP 421)
```

这种情况下 MCP 客户端连不上服务,工具也不会出现在客户端里(表现为客户端
改用自带模型去"识别",而不是调用本 MCP)。

配置方法(逗号分隔,`*` 表示端口通配):

| 环境变量 | 示例 | 说明 |
|----------|------|------|
| `MCP_ALLOWED_HOSTS` | `192.168.7.49:*,localhost:*` | 允许的 Host 头 |
| `MCP_ALLOWED_ORIGINS` | `http://192.168.7.49:*,http://localhost:*` | 允许的 Origin 头(浏览器/带 Origin 的客户端) |

> 只在本机访问(用 `localhost`)时无需配置。

## 验证服务是否正常

MCP 采用 `streamable-http` 传输,是有状态的协议:必须先 `initialize` 握手拿到会话 ID,再发 `notifications/initialized`,之后才能调用工具。下面提供两种验证方式。

### 方式一:curl 快速握手

用 curl 做 MCP 握手,确认服务可达:

```bash
# 成功:返回 200 + 一段 JSON/SSE 流,包含 serverInfo
curl -i -X POST http://192.168.7.49:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl","version":"1.0"}}}'

# 如果返回 "Invalid Host header"(421),说明该访问地址没加入白名单
```

### 方式二:Postman 端到端(base64)

仓库提供了可直接导入的 Postman Collection:[tests/ocr-mcp.postman_collection.json](tests/ocr-mcp.postman_collection.json),内含三个按顺序的请求,会话 ID 自动传递,无需手动复制。

**第 0 步:准备 base64 图片**

Postman 无法直接读取本地文件,需先把图片转成 data URI。任选一种:

```powershell
# PowerShell(Windows):输出一整行 data URI,复制备用
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("sample_data\zh.png"))
"data:image/png;base64,$b64"
```

```bash
# Python(跨平台)
python -c "import base64; b=base64.b64encode(open('sample_data/zh.png','rb').read()).decode(); print('data:image/png;base64,'+b)"
```

**第 1 步:导入并配置**

1. Postman → Import → 选择 `tests/ocr-mcp.postman_collection.json`。
2. 点击 Collection 名 → Variables,设置:
   - `base_url`:你的服务地址(如 `http://192.168.137.94:8000`)
   - `base64_image`:上一步复制的 data URI
   - `session_id`:留空(会自动填充)

**第 2 步:按顺序发送三个请求**

| 顺序 | 请求 | 作用 | 预期 |
|------|------|------|------|
| 1 | `1. Initialize` | 握手,获取会话 ID | `200`,响应头含 `mcp-session-id`,脚本自动存入变量 |
| 2 | `2. Notifications/initialized` | 通知初始化完成 | `202` |
| 3 | `3. recognize_text (base64)` | 用 base64 图片做 OCR | `200`,Body(SSE)里含识别文本 |

> 三个请求都是 `POST {{base_url}}/mcp`,必须带这两个头:
> `Content-Type: application/json`、`Accept: application/json, text/event-stream`。
> 第 2、3 个请求还要带 `mcp-session-id: {{session_id}}`。
> Collection 已配好这些,直接 Send 即可。

**第 3 步:查看结果**

第 3 个请求的响应是 SSE 流,Postman 会以 `EventStream` 形式展示,其中 `data:` 后的 JSON 结构为:

```json
{
  "jsonrpc": "2.0", "id": 2,
  "result": {
    "content": [{"type": "text", "text": "{\"language\":\"ch\",\"count\":2,\"text\":\"你好世界\nHello OCR\"}"}],
    "isError": false
  }
}
```

即 OCR 成功。若 `isError` 为 `true`,看 `content` 里的错误信息(例如跨机访问时本机路径无效,需改用 data URI 或服务端可达的 URL)。

### 其他:stdio 端到端测试

```bash
python tests/test_mcp_client.py
```

## 接入 MCP 客户端

### Codex

编辑 Codex 配置文件(Windows:`~/.codex/config.toml`,即
`%USERPROFILE%\.codex\config.toml`):

```toml
[mcp_servers.ocr]
enabled = true
url = "http://192.168.7.49:8000/mcp"
```

保存后重启 Codex,即可在对话中让其"识别这张图片里的文字"。

### Claude Desktop

编辑配置文件(macOS:`~/Library/Application Support/Claude/claude_desktop_config.json`,
Windows:`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "paddleocr": {
      "url": "http://192.168.7.49:8000/mcp",
      "type": "http"
    }
  }
}
```

### Cursor

在 Cursor 设置 → MCP 中添加 HTTP 类型的 MCP Server,URL 填
`http://192.168.7.49:8000/mcp`。

## 提供的 MCP 工具

### `recognize_text`

对图片执行 OCR 文本识别。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `image` | string | 必填 | 本地文件路径、`http(s)` URL、`data:image/...;base64,...` 或裸 base64 |
| `language` | string | `"ch"` | 语言代码:`ch` / `en` / `japan` / `korean` |
| `detail` | bool | `false` | 为 `true` 时返回每行的文本框 `box` 与置信度 `confidence` |
| `min_confidence` | float | `0.0` | 过滤低于该置信度的行(0-1,0 表示保留全部)|

返回示例:

```json
{
  "language": "ch",
  "count": 2,
  "text": "你好世界\nHello OCR",
  "lines": [
    {"text": "你好世界", "confidence": 0.98, "box": [[25, 58], [300, 58], [300, 114], [25, 114]]},
    {"text": "Hello OCR", "confidence": 0.96, "box": null}
  ]
}
```

> `lines` 仅在 `detail=true` 时返回。

### `list_supported_languages`

返回本服务支持的语言代码与说明,无需参数。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MCP_TRANSPORT` | `stdio` | 传输方式:`stdio` 或 `streamable-http` |
| `MCP_HOST` | `0.0.0.0` | HTTP 模式监听地址 |
| `MCP_PORT` | `8000` | HTTP 模式监听端口 |
| `MCP_ALLOWED_HOSTS` | (仅 loopback) | 允许的 Host 头(逗号分隔,跨机访问必填) |
| `MCP_ALLOWED_ORIGINS` | (仅 loopback) | 允许的 Origin 头(逗号分隔) |

OCR 模型与推理相关:

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OCR_VERSION` | `PP-OCRv5` | PaddleOCR 模型版本。paddleocr >=3.7 默认 v6 会在 CPU 段错误,故锁回 v5。可选 `PP-OCRv4`/`PP-OCRv6` |
| `OCR_PIR` | `0` | 设为 `1` 时不关闭 paddle 的 PIR 执行器(默认关闭以规避 CPU 段错误) |
| `OCR_ENGINE` | 自动 | 推理引擎。aarch64(树莓派等)自动用 `onnxruntime` 规避 native 段错误;x86 用 `paddle`。可强制 `onnxruntime`/`paddle` |
| `OCR_DEVICE` | `cpu` | 推理设备:`cpu` / `gpu` / `gpu:0`。用 `gpu` 需 GPU 版 paddlepaddle + NVIDIA 驱动 |
| `OCR_DET_MODEL` | 镜像默认 mobile | 检测模型名,如 `PP-OCRv5_mobile_det`(快)/ `PP-OCRv5_server_det`(准)。留空则按 `ocr_version` 自动选 |
| `OCR_REC_MODEL` | 镜像默认 mobile | 识别模型名,如 `PP-OCRv5_mobile_rec`(快)/ `PP-OCRv5_server_rec`(准)。留空则按 `ocr_version` 自动选 |
| `OCR_MAX_IMAGE_SIDE` | `2880` | OCR 前将图片长边缩放到此值以下(像素)。调小可显著加速,但过小会丢失小字 |

Docker 镜像默认设置 `MCP_TRANSPORT=streamable-http`。

## 本地 stdio 模式

适用于 Claude Desktop、Cursor 等通过子进程接入的客户端。

```bash
# 不设置 MCP_TRANSPORT 即为 stdio
python -m ocr_mcp
```

Claude Desktop 配置(stdio):

```json
{
  "mcpServers": {
    "paddleocr": {
      "command": "python",
      "args": ["-m", "ocr_mcp"],
      "cwd": "C:/develop/pythonws/ocr-server",
      "env": { "MCP_TRANSPORT": "stdio" }
    }
  }
}
```

## 构建与测试

```bash
# 生成四种语言的测试图片(需要本机有对应字体)
python tests/make_test_image.py

# 在 Docker 容器内直接验证 OCR 引擎(中/英/日/韩)
docker run --rm \
  -v "${PWD}/tests:/work/tests" \
  -v "${PWD}/sample_data:/work/sample_data" \
  -e PYTHONPATH=/app -w /app \
  --entrypoint python ocr-mcp:latest /work/tests/test_engine.py

# 端到端 MCP 客户端测试(stdio)
python tests/test_mcp_client.py
```

## 项目结构

```text
ocr-server/
├── ocr_mcp/
│   ├── __init__.py        # 包入口
│   ├── __main__.py        # python -m ocr_mcp 入口
│   ├── ocr_engine.py      # PaddleOCR 引擎封装与结果归一化
│   ├── server.py          # FastMCP 服务与工具定义
│   └── warmup.py          # 模型预下载脚本(Docker 构建期调用)
├── tests/
│   ├── make_test_image.py # 生成四语言测试图
│   ├── test_engine.py     # OCR 引擎直跑验证
│   └── test_mcp_client.py # stdio MCP 端到端验证
├── Dockerfile             # 内置模型的 Docker 镜像
├── requirements.txt       # 运行时依赖
└── pyproject.toml         # 包元数据与入口命令
```

## 实现说明

- **模型版本锁定为 PP-OCRv5**:paddleocr >=3.7 在只传 `lang`、不传 `ocr_version` 时会
  默认选用 **PP-OCRv6**(`PP-OCRv6_medium_det`/`rec`),而 PP-OCRv6 在 paddlepaddle 3.x
  CPU 上会触发原生段错误(`ConvertPirAttribute2RuntimeAttribute`),连 `enable_mkldnn=False`
  也无法避免,直接导致进程崩溃、服务停止。本项目通过 `ocr_version="PP-OCRv5"` 锁定到验证可用的
  v5 模型(中/英/日/韩均有对应模型)。可用环境变量 `OCR_VERSION` 覆盖(如 `PP-OCRv4`)。
- **关闭 PIR 执行器**:在导入 paddle 前,默认设置 `FLAGS_enable_pir_in_executor=0`、
  `FLAGS_enable_pir_api=0`,作为 CPU 段错误的兜底防御。需要恢复新 IR 执行器时设 `OCR_PIR=1`。
- **CPU 推理**:`device="cpu"` + `enable_mkldnn=False`,规避 OneDNN 相关问题。如需 GPU,
  可在 [ocr_engine.py](ocr_mcp/ocr_engine.py) 中将 `device="cpu"` 改为 `"gpu"` 并移除
  `enable_mkldnn=False`,同时使用 GPU 版 paddlepaddle。
- **模型变体(server / mobile 两套预热)**:镜像构建时通过 warmup 同时预下载 **server**
  (高精度、慢)与 **mobile**(快 3-5 倍,适合运单等清晰印刷体)两套模型,运行时用
  `OCR_DET_MODEL` / `OCR_REC_MODEL` 切换,无需重新 build 或联网下载。镜像默认 mobile。
- **GPU 加速**:设置 `OCR_DEVICE=gpu` 即可走 GPU 推理(需 GPU 版 `paddlepaddle-gpu` +
  NVIDIA 驱动 + `nvidia-container-toolkit`,`docker run --gpus all`)。aarch64(树莓派等)无
  NVIDIA GPU,只能 CPU。
- **ARM64(aarch64)用 ONNX Runtime**:paddlepaddle 3.x 的 aarch64 预编译包在推理时存在
  原生空指针段错误(native kernel,无环境变量可绕)。本项目在检测到 aarch64(树莓派、Graviton、
  Apple Silicon Docker 等)时自动改用 `engine='onnxruntime'`,完全绕开 paddle native kernel。
  需要安装 `onnxruntime` 与 `paddle2onnx`(已加入 requirements)。可用 `OCR_ENGINE` 覆盖。
- **模型缓存**:镜像构建时通过 `python -m ocr_mcp.warmup` 预初始化四种语言的引擎,
  模型权重写入 `/home/app/.paddlex/official_models`,使运行期不再依赖网络。
- **线程安全**:OCR 推理通过 `anyio.to_thread.run_sync` 在工作线程中执行,避免阻塞
  MCP 事件循环;引擎实例以语言为键缓存,带双重检查锁。

## 依赖版本

- paddleocr `3.7.0` + paddlepaddle `3.2.2`(3.2.2 是最高同时有 amd64/aarch64 wheel 的版本)
 onnxruntime + paddle2onnx(ORT 引擎 / aarch64 推理所需)
 pydantic `2.13.4` + mcp `1.28.1`(版本锁定,避免 pip 回溯解析失败)
 Python `>= 3.10`(镜像基于 `python:3.11-slim`)

## License

MIT
