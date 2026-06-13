# PaperLens

[English](README.md) | [中文](README.zh-CN.md)

[![Windows Installer][ci-badge]][ci-workflow]
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](LICENSE)

PaperLens 是一个桌面端论文阅读 agent。它不是 PDF reader，不是传统文献管理器，也不是普通论文总结器。它的核心目标是把论文变成可审计的知识状态：带稳定 source ID 的 PaperDOM、ClaimGraph、派生 PaperMemory 视图、有证据边界的 QA，以及一个只包含已读论文的本地图 library。

## 它做什么

- 读取 PDF 的文本、版面、页面图像、图表线索和论文内确定性工具结果。
- 只建一次 PaperDOM：章节、source ID、图表、公式和关键文本块。
- 基于 PaperDOM source ID 生成确定性的 Reading Plan，并只记录 append-only observation。
- 以 ClaimGraph 作为事实源；PaperMemory 只是派生的产品视图，不是模型可随意改写的状态文件。
- 用确定性审计检查图节点、边、source ID 和数值定位。
- 报告是 ClaimGraph 的可读视图，不能新增事实。
- QA 直接从 ClaimGraph 节点、PaperDOM source evidence 和 library graph record 回答，不从渲染后的报告或页码引用里二次总结。
- 维护一个只包含 PaperLens 已经处理过论文的本地图 library。
- 作为 Windows Tauri 桌面 App 发布，核心引擎是 Python `paperlens-core` sidecar。

## 当前产品形态

桌面端主要有三个区域：

- **Library**：已读论文、标题、等级、概念和一句话简介。
- **Capsule**：Markdown 渲染的论文知识胶囊。
- **Chat**：可以问当前论文，也可以问整个本地 library。

核心引擎也可以通过 `paperlens-core` 直接使用。

## 输出结构

```text
output/
  PaperLens.md
  papers/
    <paper_id>_<short_title>.md
  .paperlens/
    workspace.json
    state.sqlite
    cache/
    recovery/
    library/
      library_records.jsonl
      index/
        search_index.json
    pages/
    figures/
    data/
      run.json
      events.jsonl
      migrations.jsonl
      core/
        v2/
          <paper_id>/
            paper_dom.v2.json
            reading_plan.v2.json
            observation_log.v2.json
            claim_graph.v2.json
            relation_candidate_log.v2.json
            audit_findings.v2.json
            quality_metrics.v2.json
            paper_memory_view.v2.json
            report_draft.v2.json
            report_audit_findings.v2.json
            core_manifest.v2.json
```

如果直接看输出文件，先打开 `PaperLens.md`。桌面 App 读取同一个输出目录，并展示 library、capsule、evidence 和 chat。

## 本地 Workspace 存储

输出目录现在是带版本的 PaperLens workspace。`.paperlens/workspace.json` 声明
storage schema 和目录布局。PaperLens 在读论文、重建 library、回答问题前，会先
bootstrap 或迁移当前 workspace。

产品级维护命令：

```powershell
uv run python -m paperlens_core.main workspace doctor --output-dir output
uv run python -m paperlens_core.main workspace doctor --output-dir output --repair
uv run python -m paperlens_core.main workspace export --output-dir output --archive PaperLens-backup.zip
uv run python -m paperlens_core.main workspace import --output-dir output --archive PaperLens-backup.zip
uv run python -m paperlens_core.main workspace cleanup-cache --output-dir output --max-age-days 30
```

关键 JSON 产物、报告、library index、QA cache 和 typed core artifacts 都会通过
原子替换写入。修复时如果发现关键 JSON 损坏，PaperLens 会把它移到
`.paperlens/recovery/`，而不是静默覆盖。

## 安装和使用

发布版本从 GitHub Actions artifacts 或带 `v*` tag 的 GitHub Releases 下载 Windows NSIS 安装包。

安装包是 per-user 模式：

- 默认不需要管理员权限；
- 安装到当前用户；
- 用户选择的论文库和输出目录不放在程序安装目录里；
- 卸载时会询问是否同时删除本机设置和 WebView 缓存。

App 不会内置模型 key、模型名称或 provider URL。你需要在 UI 里输入会话所需配置。API key 不会持久化保存到本机设置。

大版本升级前，建议用 `paperlens-core workspace export` 导出重要 workspace。
启动或打开 workspace 时，PaperLens 会原地迁移所选 workspace。如果需要覆盖式导入，
旧的受管理 workspace 文件会先移动到 workspace 旁边的带时间戳备份目录。

## 开发

需要：

- Node.js 24
- Python 3.12+
- Rust stable
- `uv`
- Windows 上构建安装包需要 NSIS

```powershell
npm ci
uv run --extra dev ruff check .
npm run lint
npm run build
```

开发模式启动桌面端：

```powershell
npm run tauri:dev
```

构建 Python sidecar 和 Windows 安装包：

```powershell
npm run core:build
npm run tauri:build
```

构建 Windows 绿色版目录和 zip：

```powershell
npm run portable:build
```

一次性推进并同步所有发布版本号：

```powershell
npm run version:patch
```

较大的版本使用 `npm run version:minor` 或 `npm run version:major`。CI 会执行 `npm run version:check`，避免 `package.json`、Python metadata、Tauri config、Rust metadata 和锁文件版本号静默漂移。

安装包输出到：

```text
src-tauri/target/release/bundle/nsis/
```

绿色版输出到：

```text
build/PaperLens/
build/PaperLens-<version>-windows-x64-portable.zip
build/PaperLens-<version>-windows-x64-portable.zip.sha256
build/PaperLens-<version>-windows-x64-portable.json
```

## CI

Windows 安装包 workflow 在 [.github/workflows/windows-installer.yml](.github/workflows/windows-installer.yml)。

它会执行：

- Linux preflight：版本同步检查、Python lint、前端 lint、前端 build；
- 已提交密钥扫描；
- Python sidecar 构建；
- Tauri NSIS 安装包构建；
- Windows 绿色版 zip 打包；
- 安装包 artifact 上传；
- 推送 `v*` tag、`main` 上 `package.json` 版本号变化，或手动运行并选择 `publish_release=true` 时发布 GitHub Release；
- 配置 updater 签名 secret 后，同时发布应用内自动升级所需的签名元数据。

CI 不需要模型 API key。

应用内自动升级只在签名 release 构建里启用。发布自更新版本前，需要配置这些 repository secrets：

- `PAPERLENS_UPDATER_PUBKEY`：Tauri signer 生成的公钥；
- `TAURI_SIGNING_PRIVATE_KEY`：只在 GitHub Actions 中使用的私钥；
- `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`：可选的私钥密码。

workflow 会把 NSIS 安装包、绿色版 zip、绿色版 hash/metadata、签名文件和 `latest.json` 一起上传到 GitHub Release。默认检查地址是 `https://github.com/<owner>/<repo>/releases/latest/download/latest.json`；如需使用自己的更新源，可以设置 repository variable `PAPERLENS_UPDATER_ENDPOINT`。

Release notes 来自 [RELEASE_NOTES.md](RELEASE_NOTES.md)，并复用于 GitHub Release
和签名 updater manifest。

## 安全和隐私默认值

PaperLens 的默认设计是避免把 key 写进仓库和输出产物。

UI 不会把 API key 写入 `localStorage`。桌面端访问本地 Core 服务时使用 HTTP header 鉴权，不再把 token 放进 URL query，实时进度流和报告图片也是如此。Markdown 报告会经过清洗，并且不会渲染 raw HTML。Git 会忽略生成的论文库、报告、sidecar、本地缓存、虚拟环境、构建产物和 `.env` 文件。本地模型调用账本只记录请求大小、阶段名、状态和 provider usage metadata，不记录 prompt 正文或 API key。

## 文档

- [English README](README.md)

## License

PaperLens 使用 [Unlicense](LICENSE) 发布。

[ci-badge]: https://github.com/chenty2333/PaperLens/actions/workflows/windows-installer.yml/badge.svg
[ci-workflow]: https://github.com/chenty2333/PaperLens/actions/workflows/windows-installer.yml
