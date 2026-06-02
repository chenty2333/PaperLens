# PaperLens

[English](README.md) | [中文](README.zh-CN.md)

[![Windows Installer][ci-badge]][ci-workflow]
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](LICENSE)

PaperLens 是一个桌面端论文阅读 agent。它不是 PDF reader，不是传统文献管理器，也不是普通论文总结器。它的核心目标是把论文变成可审计的知识状态：结构化 PaperMemory、可读知识胶囊、有证据边界的 QA，以及一个只包含已读论文的本地 library。

## 它做什么

- 读取 PDF 的文本、版面、页面图像、图表线索和论文内确定性工具结果。
- 只建一次论文地图：章节、页码、图表和关键文本块。
- 分块阅读论文并增量更新 PaperMemory；阅读阶段不写报告。
- 只做一次集中核对：检查高风险 claim 和局部原文证据，然后直接修正 memory。
- 按段生成知识胶囊，报告从 memory 和 evidence 写出，不让一次模型调用承担整篇长报告。
- QA 直接从 memory、evidence、局部页面和 library record 回答，不从渲染后的报告里二次总结。
- 维护一个只包含 PaperLens 已经处理过论文的本地 library。
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
    state.sqlite
    library/
      library_records.jsonl
      index/
        search_index.json
    pages/
    figures/
    data/
      run.json
      events.jsonl
      memory/
        v3/
          <paper_id>.paper_memory.v3.json
          <paper_id>.memory_patches.jsonl
```

如果直接看输出文件，先打开 `PaperLens.md`。桌面 App 读取同一个输出目录，并展示 library、capsule、evidence 和 chat。

## 安装和使用

发布版本从 GitHub Actions artifacts 或带 `v*` tag 的 GitHub Releases 下载 Windows NSIS 安装包。

安装包是 per-user 模式：

- 默认不需要管理员权限；
- 安装到当前用户；
- 用户选择的论文库和输出目录不放在程序安装目录里；
- 卸载时会询问是否同时删除本机设置和 WebView 缓存。

App 不会内置模型 key、模型名称或 provider URL。你需要在 UI 里输入会话所需配置。API key 不会持久化保存到本机设置。

## 开发

需要：

- Node.js 22
- Python 3.12+
- Rust stable
- `uv`
- Windows 上构建安装包需要 NSIS

```powershell
npm ci
uv run --extra dev ruff check .
uv run --extra dev pytest
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

安装包输出到：

```text
src-tauri/target/release/bundle/nsis/
```

## CI

Windows 安装包 workflow 在 [.github/workflows/windows-installer.yml](.github/workflows/windows-installer.yml)。

它会执行：

- 已提交密钥扫描；
- Python lint/tests；
- 前端 lint/build；
- Python sidecar 构建；
- Tauri NSIS 安装包构建；
- 安装包 artifact 上传；
- 推送 `v*` tag、`main` 上 `package.json` 版本号变化，或手动运行并选择 `publish_release=true` 时发布 GitHub Release。

CI 不需要模型 API key。

## 安全和隐私默认值

PaperLens 的默认设计是避免把 key 写进仓库和输出产物。

```text
OPENAI_AGENTS_DONT_LOG_MODEL_DATA=1
OPENAI_AGENTS_DONT_LOG_TOOL_DATA=1
OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA=0
```

UI 不会把 API key 写入 `localStorage`。Git 会忽略生成的论文库、报告、sidecar、本地缓存、虚拟环境、构建产物和 `.env` 文件。

## 文档

- [PaperLens Core v1 设计](docs/PaperLens_Core_v1.md)
- [English README](README.md)

## License

PaperLens 使用 [Unlicense](LICENSE) 发布。

[ci-badge]: https://github.com/chenty2333/PaperLens/actions/workflows/windows-installer.yml/badge.svg
[ci-workflow]: https://github.com/chenty2333/PaperLens/actions/workflows/windows-installer.yml
