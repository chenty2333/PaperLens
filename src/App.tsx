import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import { convertFileSrc, invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import { confirm, message, open, save } from '@tauri-apps/plugin-dialog'
import { openPath, openUrl } from '@tauri-apps/plugin-opener'
import { relaunch } from '@tauri-apps/plugin-process'
import { check, type DownloadEvent } from '@tauri-apps/plugin-updater'
import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import 'katex/dist/katex.min.css'
import {
  AlertTriangle,
  Archive,
  ArrowUp,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Database,
  FileText,
  FolderOpen,
  KeyRound,
  Library,
  Loader2,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Pause,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Settings2,
  ShieldCheck,
  Square,
  Trash2,
  Wrench,
  XCircle,
} from 'lucide-react'
import './App.css'

type ProviderKind = 'openai' | 'openai-compatible' | 'anthropic' | 'anthropic-compatible'
type AskScope = 'paper' | 'library'
type ResizePane = 'left' | 'right'

type ServiceInfo = {
  baseUrl: string
  token: string
}

type RunSettings = {
  inputDir: string
  outputDir: string
  providerKind: ProviderKind
  baseUrl: string
  apiKey: string
  model: string
  reasoningModel: string
  concurrency: string
  outputLanguage: 'zh' | 'en'
  readMode: 'standard'
}

type PaperSummary = {
  paper_id: string
  title: string
  grade?: string
  recommendation?: string
  brief?: string
  core_idea?: string
  concepts?: string[]
  tags?: string[]
  report_path?: string
  report_file?: string
  source?: {
    year?: number
    venue?: string
    pages?: number
    doi?: string
    original_path?: string
  }
  quality?: Record<string, unknown>
  memory?: {
    claim_count?: number
    evidence_count?: number
    core_v2_path?: string | null
  }
  qa?: {
    count?: number
    last_question?: string
    last_time?: string
  }
}

type Workspace = {
  output_dir: string
  status: string
  manifest?: Record<string, unknown>
  storage?: WorkspaceStorageHealth
  papers: PaperSummary[]
  paper_count: number
}

type ReportPayload = {
  paper: PaperSummary
  path: string
  base_dir: string
  markdown: string
}

type PaperLensEvent = {
  seq?: number
  type: string
  time?: string
  level?: 'info' | 'warning' | 'error' | 'critical'
  stage?: string | null
  message?: string | null
  progress?: number | null
  data?: Record<string, unknown>
}

type JobSummary = {
  job_id: string
  status: string
  current_stage?: string
  input_dir?: string
  output_dir?: string
  created_at?: string
  updated_at?: string
  completed_at?: string | null
  error?: string | null
  latest_event?: PaperLensEvent | null
  result?: Record<string, unknown> | null
  live?: boolean
}

type SourceAttribution = {
  paper_claims?: string[]
  paperlens_inferences?: string[]
  cross_paper_synthesis?: string[]
  background_context?: string[]
  evidence_limits?: string[]
}

type AnswerPayload = {
  answer_markdown?: string
  cited_source_ids?: string[]
  confidence?: 'high' | 'medium' | 'low'
  source_attribution?: SourceAttribution
  related_papers?: Array<{
    paper_id: string
    title: string
    report_path?: string
    why_related?: string
  }>
}

type AnswerSummary = {
  answer_id: string
  status: string
  scope: AskScope
  paper_id?: string | null
  question: string
  answer?: AnswerPayload | null
  error?: string | null
}

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  scope: AskScope
  content: string
  answer?: AnswerPayload | null
  pending?: boolean
  error?: string | null
}

type ChatThread = {
  id: string
  subjectKey: string
  scope: AskScope
  title: string
  createdAt: string
  updatedAt: string
  messages: ChatMessage[]
}

type ChatStore = {
  threads: Record<string, ChatThread>
  activeBySubject: Record<string, string>
}

type ImageCandidate = {
  src: string
  authenticated?: boolean
}

const EMPTY_CHAT_MESSAGES: ChatMessage[] = []
const ACTIVE_JOB_STATUSES = new Set(['queued', 'running', 'paused', 'cancelling'])
const RETRYABLE_JOB_STATUSES = new Set(['failed', 'cancelled'])

type CleanupReport = {
  removed: string[]
  missing: string[]
  errors: string[]
}

type WorkspaceStorageStats = {
  managed_bytes?: number
  cache_bytes?: number
  paper_reports?: number
}

type WorkspaceStorageHealth = {
  status?: string
  issues?: string[]
  repairs?: string[]
  stats?: WorkspaceStorageStats
  library?: {
    exists?: boolean
    records?: number
    invalid_lines?: number
  }
  state_db?: string
}

type WorkspaceMaintenanceResult = WorkspaceStorageHealth & {
  selected?: number
  bytes?: number
  removed?: string[]
  errors?: string[]
  archive?: string
  file_count?: number
  extracted?: number
  backup_dir?: string
}

const defaultSettings: RunSettings = {
  inputDir: '',
  outputDir: '',
  providerKind: 'openai-compatible',
  baseUrl: '',
  apiKey: '',
  model: '',
  reasoningModel: '',
  concurrency: '1',
  outputLanguage: 'zh',
  readMode: 'standard',
}

function modelSettingsPayload(settings: RunSettings) {
  return {
    provider_kind: settings.providerKind,
    base_url: settings.baseUrl || null,
    api_key: settings.apiKey,
    model: settings.model || null,
    reasoning_model: settings.reasoningModel || null,
  }
}

function readJobSettingsPayload(settings: RunSettings) {
  return {
    ...modelSettingsPayload(settings),
    concurrency: Number(settings.concurrency || '1') || 1,
    output_language: settings.outputLanguage,
    read_mode: settings.readMode,
  }
}

const markdownSanitizeSchema = {
  ...defaultSchema,
  tagNames: [...new Set([...(defaultSchema.tagNames ?? []), 'figure', 'figcaption'])],
  attributes: {
    ...defaultSchema.attributes,
    '*': [...(defaultSchema.attributes?.['*'] ?? []), 'className'],
    a: [...(defaultSchema.attributes?.a ?? []), 'href', 'title'],
    code: [...(defaultSchema.attributes?.code ?? []), 'className'],
    div: [...(defaultSchema.attributes?.div ?? []), 'className'],
    figcaption: ['className'],
    figure: ['className'],
    img: [
      ...(defaultSchema.attributes?.img ?? []),
      'src',
      'alt',
      'title',
      'width',
      'height',
      'loading',
    ],
    span: [...(defaultSchema.attributes?.span ?? []), 'className', 'aria-hidden'],
  },
  protocols: {
    ...defaultSchema.protocols,
    href: ['http', 'https', 'mailto'],
    src: ['http', 'https', 'data'],
  },
}

function loadSettings(): RunSettings {
  const raw = safeGetLocalStorage('paperLens.settings')
  if (!raw) return defaultSettings
  try {
    return { ...defaultSettings, ...JSON.parse(raw), apiKey: '' }
  } catch {
    return defaultSettings
  }
}

function serviceAuthHeaders(service: ServiceInfo) {
  return {
    Authorization: `Bearer ${service.token}`,
  }
}

function serviceJsonHeaders(service: ServiceInfo) {
  return {
    ...serviceAuthHeaders(service),
    'Content-Type': 'application/json',
  }
}

function serviceUrl(service: ServiceInfo, path: string) {
  return `${service.baseUrl}${path}`
}

async function apiGet<T>(service: ServiceInfo, path: string): Promise<T> {
  const response = await fetch(serviceUrl(service, path), {
    headers: serviceAuthHeaders(service),
  })
  if (!response.ok) throw new Error(await responseErrorMessage(response))
  return response.json() as Promise<T>
}

async function apiPost<T>(service: ServiceInfo, path: string, payload: unknown): Promise<T> {
  const response = await fetch(serviceUrl(service, path), {
    method: 'POST',
    headers: serviceJsonHeaders(service),
    body: JSON.stringify(payload),
  })
  if (!response.ok) throw new Error(await responseErrorMessage(response))
  return response.json() as Promise<T>
}

async function responseErrorMessage(response: Response) {
  const fallback = response.statusText || `HTTP ${response.status}`
  try {
    const contentType = response.headers.get('content-type') ?? ''
    if (contentType.includes('application/json')) {
      const payload = await response.json() as Record<string, unknown>
      const message = typeof payload.error === 'string' ? payload.error : JSON.stringify(payload)
      return message || fallback
    }
    const text = await response.text()
    return text.trim() || fallback
  } catch {
    return fallback
  }
}

function errorMessage(err: unknown) {
  return err instanceof Error ? err.message : String(err)
}

function parsePaperLensSseFrame(frame: string): PaperLensEvent | null {
  const dataLines: string[] = []
  for (const rawLine of frame.split(/\r?\n/)) {
    if (!rawLine || rawLine.startsWith(':')) continue
    const separator = rawLine.indexOf(':')
    const field = separator === -1 ? rawLine : rawLine.slice(0, separator)
    let value = separator === -1 ? '' : rawLine.slice(separator + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'data') dataLines.push(value)
  }
  if (!dataLines.length) return null
  try {
    return JSON.parse(dataLines.join('\n')) as PaperLensEvent
  } catch {
    return null
  }
}

function startPaperLensEventStream(
  service: ServiceInfo,
  path: string,
  onEvent: (event: PaperLensEvent) => void,
  onClose: () => void,
): AbortController {
  const controller = new AbortController()
  void (async () => {
    try {
      const response = await fetch(serviceUrl(service, path), {
        headers: serviceAuthHeaders(service),
        signal: controller.signal,
      })
      if (!response.ok) throw new Error(await responseErrorMessage(response))
      if (!response.body) throw new Error('PaperLens event stream is unavailable')
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        let boundary = buffer.indexOf('\n\n')
        while (boundary >= 0) {
          const frame = buffer.slice(0, boundary)
          buffer = buffer.slice(boundary + 2)
          const parsed = parsePaperLensSseFrame(frame)
          if (parsed) onEvent(parsed)
          boundary = buffer.indexOf('\n\n')
        }
      }
      buffer += decoder.decode()
      const parsed = parsePaperLensSseFrame(buffer)
      if (parsed) onEvent(parsed)
      if (!controller.signal.aborted) onClose()
    } catch {
      if (!controller.signal.aborted) onClose()
    }
  })()
  return controller
}

function encodeOutput(outputDir: string) {
  return encodeURIComponent(outputDir)
}

function normalizeLocalPath(path: string) {
  const slash = path.includes('\\') ? '\\' : '/'
  const isUncPath = /^[\\/]{2}[^\\/]/.test(path)
  const isRootedPath = !isUncPath && /^[\\/]/.test(path)
  const normalized = path.replace(/\//g, '\\')
  const parts: string[] = []
  for (const part of normalized.split('\\')) {
    if (!part || part === '.') continue
    if (part === '..') {
      parts.pop()
      continue
    }
    parts.push(part)
  }
  if (/^[A-Za-z]:$/.test(parts[0] ?? '')) {
    return `${parts[0]}\\${parts.slice(1).join('\\')}`
  }
  if (isUncPath) {
    return `\\\\${parts.join('\\')}`
  }
  if (isRootedPath) {
    return `${slash}${parts.join(slash)}`
  }
  return parts.join(slash)
}

const CHAT_STORE_KEY = 'paperLens.chatStore.v2'
const CHAT_STORE_LIMITS = {
  maxThreads: 60,
  maxMessagesPerThread: 40,
  maxMessageChars: 24000,
  maxSerializedChars: 1_500_000,
}
const CHAT_STORE_FALLBACK_LIMITS = {
  maxThreads: 20,
  maxMessagesPerThread: 20,
  maxMessageChars: 12000,
  maxSerializedChars: 900_000,
}

function safeGetLocalStorage(key: string) {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function safeSetLocalStorage(key: string, value: string) {
  try {
    localStorage.setItem(key, value)
    return true
  } catch {
    return false
  }
}

function safeRemoveLocalStorage(key: string) {
  try {
    localStorage.removeItem(key)
  } catch {
    // Local storage can be unavailable in restricted webview sessions.
  }
}

function loadChatStore(): ChatStore {
  const raw = safeGetLocalStorage(CHAT_STORE_KEY)
  if (!raw) return { threads: {}, activeBySubject: {} }
  try {
    const value = JSON.parse(raw) as unknown
    if (!value || typeof value !== 'object' || Array.isArray(value)) return { threads: {}, activeBySubject: {} }
    const rawStore = value as Partial<ChatStore>
    const threads: Record<string, ChatThread> = {}
    const rawThreads = rawStore.threads && typeof rawStore.threads === 'object' ? rawStore.threads : {}
    for (const [id, thread] of Object.entries(rawThreads)) {
      if (!thread || typeof thread !== 'object') continue
      const candidate = thread as Partial<ChatThread>
      if (!candidate.subjectKey || !candidate.title || !Array.isArray(candidate.messages)) continue
      const messages = candidate.messages
        .filter((message): message is ChatMessage => {
          if (!message || typeof message !== 'object') return false
          const item = message as ChatMessage
          return ['user', 'assistant'].includes(String(item.role)) && typeof item.content === 'string'
        })
        .slice(-CHAT_STORE_LIMITS.maxMessagesPerThread)
        .map((message) => compactStoredChatMessage(message, CHAT_STORE_LIMITS.maxMessageChars))
      threads[id] = {
        id,
        subjectKey: String(candidate.subjectKey),
        scope: candidate.scope === 'library' ? 'library' : 'paper',
        title: String(candidate.title),
        createdAt: String(candidate.createdAt || new Date().toISOString()),
        updatedAt: String(candidate.updatedAt || candidate.createdAt || new Date().toISOString()),
        messages,
      }
    }
    const activeBySubject = rawStore.activeBySubject && typeof rawStore.activeBySubject === 'object'
      ? Object.fromEntries(
          Object.entries(rawStore.activeBySubject).filter(([, threadId]) => typeof threadId === 'string' && threads[threadId]),
        ) as Record<string, string>
      : {}
    return { threads, activeBySubject }
  } catch {
    return { threads: {}, activeBySubject: {} }
  }
}

function compactStoredChatMessage(message: ChatMessage, maxChars: number): ChatMessage {
  return {
    ...message,
    content: truncateText(message.content, maxChars),
    error: message.error ? truncateText(message.error, 2000) : null,
    answer: compactStoredAnswer(message.answer),
  }
}

function compactStoredAnswer(answer?: AnswerPayload | null): AnswerPayload | undefined {
  if (!answer) return undefined
  return {
    cited_source_ids: answer.cited_source_ids?.slice(0, 80),
    confidence: answer.confidence,
    source_attribution: compactSourceAttribution(answer.source_attribution),
    related_papers: answer.related_papers?.slice(0, 12).map((paper) => ({
      paper_id: paper.paper_id,
      title: truncateText(paper.title, 300),
      report_path: paper.report_path,
      why_related: paper.why_related ? truncateText(paper.why_related, 800) : undefined,
    })),
  }
}

function compactSourceAttribution(source?: SourceAttribution): SourceAttribution | undefined {
  if (!source) return undefined
  return {
    paper_claims: source.paper_claims?.slice(0, 12).map((item) => truncateText(item, 800)),
    paperlens_inferences: source.paperlens_inferences?.slice(0, 12).map((item) => truncateText(item, 800)),
    cross_paper_synthesis: source.cross_paper_synthesis?.slice(0, 12).map((item) => truncateText(item, 800)),
    background_context: source.background_context?.slice(0, 12).map((item) => truncateText(item, 800)),
    evidence_limits: source.evidence_limits?.slice(0, 12).map((item) => truncateText(item, 800)),
  }
}

function pruneChatStore(
  store: ChatStore,
  limits = CHAT_STORE_LIMITS,
): ChatStore {
  const threads = Object.fromEntries(
    Object.values(store.threads)
      .map((thread) => ({
        ...thread,
        title: truncateText(thread.title, 120),
        messages: thread.messages
          .filter((message) => !message.pending)
          .slice(-limits.maxMessagesPerThread)
          .map((message) => compactStoredChatMessage(message, limits.maxMessageChars)),
      }))
      .filter((thread) => thread.messages.length)
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
      .slice(0, limits.maxThreads)
      .map((thread) => [thread.id, thread]),
  )
  const activeBySubject = Object.fromEntries(
    Object.entries(store.activeBySubject).filter(([, threadId]) => Boolean(threads[threadId])),
  )
  return { threads, activeBySubject }
}

function persistChatStore(store: ChatStore) {
  for (const limits of [CHAT_STORE_LIMITS, CHAT_STORE_FALLBACK_LIMITS]) {
    const pruned = pruneChatStore(store, limits)
    const payload = JSON.stringify(pruned)
    if (payload.length <= limits.maxSerializedChars && safeSetLocalStorage(CHAT_STORE_KEY, payload)) {
      return
    }
  }
  safeRemoveLocalStorage(CHAT_STORE_KEY)
}

function newChatThreadId() {
  return `thread_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`
}

function newChatMessageId() {
  return `msg_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`
}

function chatTitleFromQuestion(question: string) {
  const text = question.replace(/\s+/g, ' ').trim()
  return text.length > 24 ? `${text.slice(0, 24)}...` : text || '新对话'
}

function dirname(path: string) {
  const normalized = path.replace(/\\/g, '/')
  const index = normalized.lastIndexOf('/')
  return index >= 0 ? normalized.slice(0, index) : ''
}

function resolveRelativePath(baseDir: string, target: string) {
  const normalized = target.replace(/\\/g, '/')
  if (/^[A-Za-z]:[\\/]/.test(normalized) || normalized.startsWith('/')) return normalized
  const parts: string[] = []
  for (const part of `${baseDir}/${normalized}`.split('/')) {
    if (!part || part === '.') continue
    if (part === '..') {
      parts.pop()
    } else {
      parts.push(part)
    }
  }
  return parts.join('/')
}

function assetServiceSrc(service: ServiceInfo | null, outputDir: string, assetPath: string) {
  if (!service || !outputDir || !assetPath) return ''
  return serviceUrl(
    service,
    `/assets?output_dir=${encodeOutput(outputDir)}&path=${encodeURIComponent(assetPath)}`,
  )
}

function outputRootForReport(report: ReportPayload | null, outputDir: string) {
  if (outputDir) return outputDir
  if (!report?.base_dir) return ''
  const normalized = report.base_dir.replace(/\\/g, '/')
  return normalized.endsWith('/papers') ? normalized.slice(0, -'/papers'.length) : ''
}

function isAbsoluteLocalPath(path: string) {
  return /^[A-Za-z]:[\\/]/.test(path) || path.startsWith('/') || path.startsWith('\\')
}

function absoluteReportAssetPath(src: string, report: ReportPayload | null, outputDir: string) {
  const normalized = src.replace(/\\/g, '/')
  if (isAbsoluteLocalPath(normalized)) return normalizeLocalPath(normalized)
  const root = report?.base_dir && (normalized.startsWith('../') || normalized.startsWith('./'))
    ? report.base_dir
    : outputRootForReport(report, outputDir)
  if (!root) return ''
  return normalizeLocalPath(`${root}\\${normalized.replace(/\//g, '\\')}`)
}

function viteDevFileSrc(path: string) {
  if (!path || !import.meta.env.DEV) return ''
  return `/paperlens-media?path=${encodeURIComponent(path)}`
}

function uniqueImageCandidates(values: Array<ImageCandidate | null | undefined>) {
  const seen = new Set<string>()
  const result: ImageCandidate[] = []
  for (const value of values) {
    if (!value?.src || seen.has(value.src)) continue
    seen.add(value.src)
    result.push(value)
  }
  return result
}

function localAssetCandidates(src: string | undefined, report: ReportPayload | null, outputDir: string, service: ServiceInfo | null) {
  if (!src) return []
  if (/^https?:/i.test(src)) return []
  if (/^(data:|blob:)/i.test(src)) return [{ src }]
  const baseDir = report?.paper.report_path ? dirname(report.paper.report_path) : ''
  const assetPath = resolveRelativePath(baseDir, src)
  const outputRoot = outputRootForReport(report, outputDir)
  const absolutePath = absoluteReportAssetPath(src, report, outputDir)
  const serviceSrc = assetServiceSrc(service, outputRoot, assetPath)
  return uniqueImageCandidates([
    serviceSrc ? { src: serviceSrc, authenticated: true } : null,
    viteDevFileSrc(absolutePath) ? { src: viteDevFileSrc(absolutePath) } : null,
    absolutePath ? { src: convertFileSrc(absolutePath) } : null,
    { src },
  ])
}

function statusLabel(status?: string) {
  switch (status) {
    case 'running':
      return '运行中'
    case 'queued':
      return '排队中'
    case 'completed':
      return '完成'
    case 'failed':
      return '失败'
    case 'cancelled':
      return '已停止'
    case 'paused':
      return '暂停'
    case 'cancelling':
      return '取消中'
    default:
      return status || '就绪'
  }
}

function stageLabel(stage?: string | null) {
  const labels: Record<string, string> = {
    stage_00_ingest: '导入 PDF',
    stage_01_parse: '解析论文',
    stage_02_parse_verify: '检查解析质量',
    stage_03_skim: '建立论文地图',
    stage_07_normal_read: '阅读论文',
    stage_08_evidence_verify: '核对证据',
    stage_15_export: '生成报告',
    stage_17_manifest: '收尾保存',
  }
  return labels[String(stage ?? '')] ?? String(stage ?? '')
}

function answerText(answer?: AnswerPayload | null) {
  return answer?.answer_markdown || ''
}

function cleanupSummary(report: CleanupReport) {
  const pieces = [`清理 ${report.removed.length} 项`]
  if (report.errors.length) pieces.push(`${report.errors.length} 项失败`)
  return pieces.join('，')
}

function formatBytes(value?: number) {
  if (!value || value <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let size = value
  let unitIndex = 0
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex += 1
  }
  const digits = size >= 10 || unitIndex === 0 ? 0 : 1
  return `${size.toFixed(digits)} ${units[unitIndex]}`
}

function workspaceHealthSummary(storage?: WorkspaceStorageHealth) {
  if (!storage) return '未打开'
  const status = storage.status || 'UNKNOWN'
  const issues = storage.issues?.length ?? 0
  const repairs = storage.repairs?.length ?? 0
  const cache = formatBytes(storage.stats?.cache_bytes)
  const records = storage.library?.records ?? 0
  const pieces = [`${status}`, `${issues} 个问题`, `${records} 篇记录`, `缓存 ${cache}`]
  if (repairs) pieces.push(`${repairs} 项修复`)
  return pieces.join(' · ')
}

function workspaceStatusClass(status?: string) {
  const normalized = String(status || '').toLowerCase()
  if (normalized === 'pass' || normalized === 'repaired') return 'pass'
  if (normalized === 'warn') return 'warn'
  if (normalized === 'fail') return 'fail'
  return 'unknown'
}

function workspaceMaintenanceSummary(action: string, result: WorkspaceMaintenanceResult) {
  const status = result.status || 'PASS'
  if (action === 'cache') {
    const errors = result.errors?.length ?? 0
    const pieces = [`清理缓存 ${result.selected ?? result.removed?.length ?? 0} 项`, formatBytes(result.bytes)]
    if (errors) pieces.push(`${errors} 项失败`)
    return pieces.join('，')
  }
  if (action === 'export') {
    const archive = result.archive ? `，位置：${result.archive}` : ''
    return `已导出备份：${result.file_count ?? 0} 个文件，${formatBytes(result.bytes)}${archive}`
  }
  if (action === 'import') {
    const backup = result.backup_dir ? `，原库备份：${result.backup_dir}` : ''
    return `已导入备份：${result.extracted ?? 0} 个文件${backup}`
  }
  const issues = result.issues?.length ?? 0
  const repairs = result.repairs?.length ?? 0
  const prefix = action === 'repair' ? '修复完成' : '诊断完成'
  return `${prefix}：${status}，${issues} 个问题，${repairs} 项修复`
}

function defaultBackupPath(outputDir: string) {
  const trimmed = outputDir.replace(/[\\/]+$/, '')
  const separator = trimmed.includes('/') && !trimmed.includes('\\') ? '/' : '\\'
  const stamp = new Date().toISOString().slice(0, 10)
  return `${trimmed}${separator}PaperLens-workspace-${stamp}.zip`
}

function clearPaperLensLocalStorage() {
  try {
    for (const key of Object.keys(localStorage)) {
      if (key.startsWith('paperLens.')) safeRemoveLocalStorage(key)
    }
  } catch {
    for (const key of [
      'paperLens.settings',
      'paperLens.layout.leftWidth',
      'paperLens.layout.rightWidth',
      'paperLens.layout.leftOpen',
      'paperLens.layout.rightOpen',
      CHAT_STORE_KEY,
    ]) {
      safeRemoveLocalStorage(key)
    }
  }
}

function truncateText(value: string, maxLength: number) {
  if (value.length <= maxLength) return value
  return `${value.slice(0, maxLength - 1)}…`
}

function formatUpdateError(err: unknown) {
  const text = errorMessage(err)
  const lower = text.toLowerCase()
  if (lower.includes('endpoints') || lower.includes('endpoint') || lower.includes('configured') || lower.includes('pubkey')) {
    return '当前构建没有配置自动更新源。正式 GitHub Release 构建会使用签名的 latest.json 自动升级；开发版可从 Releases 手动下载安装包。'
  }
  if (lower.includes('signature')) {
    return '更新包签名验证失败。为避免安装被篡改的版本，PaperLens 已停止升级。'
  }
  if (lower.includes('timeout') || lower.includes('timed out')) {
    return '检查或下载更新超时。请稍后重试，或从 GitHub Releases 手动下载安装包。'
  }
  if (lower.includes('network') || lower.includes('failed to fetch') || lower.includes('dns')) {
    return '无法连接更新源。请检查网络后重试，或从 GitHub Releases 手动下载安装包。'
  }
  return `检查更新失败：${text}`
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function loadNumberPreference(key: string, fallback: number) {
  const raw = safeGetLocalStorage(key)
  if (!raw) return fallback
  const parsed = Number(raw)
  return Number.isFinite(parsed) ? parsed : fallback
}

function loadBooleanPreference(key: string, fallback: boolean) {
  const raw = safeGetLocalStorage(key)
  if (!raw) return fallback
  return raw === '1'
}

function App() {
  const [service, setService] = useState<ServiceInfo | null>(null)
  const [settings, setSettings] = useState<RunSettings>(loadSettings)
  const [workspace, setWorkspace] = useState<Workspace | null>(null)
  const [selectedPaperId, setSelectedPaperId] = useState('')
  const [report, setReport] = useState<ReportPayload | null>(null)
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [activeJobId, setActiveJobId] = useState('')
  const [jobEvents, setJobEvents] = useState<PaperLensEvent[]>([])
  const [chatScope, setChatScope] = useState<AskScope>('paper')
  const [question, setQuestion] = useState('')
  const [chatStore, setChatStore] = useState<ChatStore>(loadChatStore)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [evidenceOpen, setEvidenceOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [updateBusy, setUpdateBusy] = useState(false)
  const [updateStatus, setUpdateStatus] = useState('')
  const [maintenanceBusy, setMaintenanceBusy] = useState(false)
  const [maintenanceStatus, setMaintenanceStatus] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [leftWidth, setLeftWidth] = useState(() => loadNumberPreference('paperLens.layout.leftWidth', 300))
  const [rightWidth, setRightWidth] = useState(() => loadNumberPreference('paperLens.layout.rightWidth', 360))
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(() => loadBooleanPreference('paperLens.layout.leftOpen', true))
  const [rightSidebarOpen, setRightSidebarOpen] = useState(() => loadBooleanPreference('paperLens.layout.rightOpen', true))
  const [viewportWidth, setViewportWidth] = useState(() => window.innerWidth)
  const chatEndRef = useRef<HTMLDivElement>(null)
  const eventStreamsRef = useRef<Set<AbortController>>(new Set())

  const selectedPaper = useMemo(
    () => workspace?.papers.find((paper) => paper.paper_id === selectedPaperId) ?? workspace?.papers[0] ?? null,
    [workspace, selectedPaperId],
  )
  const selectedActiveJob = jobs.find(
    (job) => job.job_id === activeJobId && ACTIVE_JOB_STATUSES.has(job.status),
  )
  const activeJob = selectedActiveJob ?? jobs.find((job) => ACTIVE_JOB_STATUSES.has(job.status)) ?? null
  const visibleJob = activeJob ?? jobs[0] ?? null
  const visibleJobEvent = activeJob
    ? jobEvents[jobEvents.length - 1] ?? activeJob.latest_event ?? null
    : visibleJob?.latest_event ?? null
  const currentOutputDir = workspace?.output_dir || settings.outputDir
  const chatSubjectKey = useMemo(() => {
    const root = currentOutputDir || 'no-output'
    return chatScope === 'library'
      ? `${root}::library`
      : `${root}::paper::${selectedPaper?.paper_id || 'none'}`
  }, [chatScope, currentOutputDir, selectedPaper?.paper_id])
  const subjectThreads = useMemo(
    () => Object.values(chatStore.threads)
      .filter((thread) => thread.subjectKey === chatSubjectKey)
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)),
    [chatStore.threads, chatSubjectKey],
  )
  const activeChatThreadId = chatStore.activeBySubject[chatSubjectKey] || subjectThreads[0]?.id || ''
  const activeChatThread = activeChatThreadId ? chatStore.threads[activeChatThreadId] ?? null : null
  const chatMessages = activeChatThread?.messages ?? EMPTY_CHAT_MESSAGES
  const effectiveLeftSidebarOpen = leftSidebarOpen && viewportWidth >= 760
  const effectiveRightSidebarOpen = rightSidebarOpen && viewportWidth >= 1040
  const layoutLeftWidth = effectiveLeftSidebarOpen
    ? clamp(leftWidth, 240, Math.max(240, viewportWidth - 720))
    : 48
  const layoutRightWidth = effectiveRightSidebarOpen
    ? clamp(rightWidth, 300, Math.max(300, viewportWidth - layoutLeftWidth - 520))
    : 48
  const providerReady = settings.providerKind === 'openai'
    || settings.providerKind === 'anthropic'
    || Boolean(settings.baseUrl.trim())
  const canRun = Boolean(
    service &&
      settings.inputDir &&
      settings.outputDir &&
      settings.apiKey.trim() &&
      settings.model.trim() &&
      providerReady,
  )
  const canAsk = Boolean(
    question.trim()
      && service
      && currentOutputDir
      && settings.apiKey.trim()
      && settings.model.trim()
      && providerReady,
  )
  const canRetryJob = Boolean(service && settings.apiKey.trim() && settings.model.trim() && providerReady)

  function setChatThreadMessages(threadId: string, update: (current: ChatMessage[]) => ChatMessage[]) {
    setChatStore((current) => {
      const thread = current.threads[threadId]
      if (!thread) return current
      return {
        ...current,
        threads: {
          ...current.threads,
          [threadId]: {
            ...thread,
            updatedAt: new Date().toISOString(),
            messages: update(thread.messages).slice(-80),
          },
        },
      }
    })
  }

  function setCurrentChatMessages(update: (current: ChatMessage[]) => ChatMessage[]) {
    if (activeChatThreadId) setChatThreadMessages(activeChatThreadId, update)
  }

  function createChatThread(initialQuestion?: string) {
    const id = newChatThreadId()
    const now = new Date().toISOString()
    const thread: ChatThread = {
      id,
      subjectKey: chatSubjectKey,
      scope: chatScope,
      title: chatTitleFromQuestion(initialQuestion || '新对话'),
      createdAt: now,
      updatedAt: now,
      messages: [],
    }
    setChatStore((current) => ({
      threads: { ...current.threads, [id]: thread },
      activeBySubject: { ...current.activeBySubject, [chatSubjectKey]: id },
    }))
    return id
  }

  function selectChatThread(threadId: string) {
    if (!chatStore.threads[threadId]) return
    setChatStore((current) => ({
      ...current,
      activeBySubject: { ...current.activeBySubject, [chatSubjectKey]: threadId },
    }))
  }

  useEffect(() => {
    safeSetLocalStorage('paperLens.settings', JSON.stringify({ ...settings, apiKey: '' }))
  }, [settings])

  useEffect(() => {
    safeSetLocalStorage('paperLens.layout.leftWidth', String(leftWidth))
  }, [leftWidth])

  useEffect(() => {
    safeSetLocalStorage('paperLens.layout.rightWidth', String(rightWidth))
  }, [rightWidth])

  useEffect(() => {
    safeSetLocalStorage('paperLens.layout.leftOpen', leftSidebarOpen ? '1' : '0')
  }, [leftSidebarOpen])

  useEffect(() => {
    safeSetLocalStorage('paperLens.layout.rightOpen', rightSidebarOpen ? '1' : '0')
  }, [rightSidebarOpen])

  useEffect(() => {
    persistChatStore(chatStore)
  }, [chatStore])

  useEffect(() => {
    function onResize() {
      setViewportWidth(window.innerWidth)
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  useEffect(() => {
    const streams = eventStreamsRef.current
    return () => {
      streams.forEach((stream) => stream.abort())
      streams.clear()
    }
  }, [])

  useEffect(() => {
    const unsubs: Array<() => void> = []
    let disposed = false
    listen<string>('core-service-log', (event) => {
      try {
        const parsed = JSON.parse(event.payload) as PaperLensEvent
        if (parsed.type) setJobEvents((current) => [...current.slice(-500), parsed])
      } catch {
        // service logs are diagnostic only
      }
    }).then((unlisten) => {
      if (disposed) unlisten()
      else unsubs.push(unlisten)
    })
    listen<string>('core-service-error', (event) => setError(event.payload)).then((unlisten) => {
      if (disposed) unlisten()
      else unsubs.push(unlisten)
    })
    return () => {
      disposed = true
      unsubs.forEach((unlisten) => unlisten())
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    invoke<string>('default_workspace_dir')
      .then((dir) => {
        if (cancelled || !dir) return
        setSettings((current) => (current.outputDir ? current : { ...current, outputDir: dir }))
      })
      .catch(() => {
        // Browser-only dev sessions can run without Tauri path commands.
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    invoke<ServiceInfo>('ensure_core_service')
      .then((info) => setService(info))
      .catch((err) => setError(errorMessage(err)))
  }, [])

  useEffect(() => {
    if (!service || !settings.outputDir || workspace) return
    let cancelled = false
    apiPost<Workspace>(service, '/workspaces/open', { output_dir: settings.outputDir })
      .then((loaded) => {
        if (cancelled) return
        setWorkspace(loaded)
        setSettings((current) =>
          current.outputDir === loaded.output_dir ? current : { ...current, outputDir: loaded.output_dir },
        )
        if (!selectedPaperId || !loaded.papers.some((paper) => paper.paper_id === selectedPaperId)) {
          setSelectedPaperId(loaded.papers[0]?.paper_id ?? '')
        }
      })
      .catch((err) => {
        if (!cancelled) setError(errorMessage(err))
      })
    return () => {
      cancelled = true
    }
  }, [service, selectedPaperId, settings.outputDir, workspace])

  useEffect(() => {
    if (!service) return
    const timer = window.setInterval(() => {
      const path = currentOutputDir ? `/jobs?output_dir=${encodeOutput(currentOutputDir)}` : '/jobs'
      apiGet<{ jobs: JobSummary[] }>(service, path)
        .then((loaded) => setJobs(loaded.jobs))
        .catch(() => {
          // a stopped service will be reported by the next explicit action
        })
    }, 2500)
    return () => window.clearInterval(timer)
  }, [currentOutputDir, service])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ block: 'end' })
  }, [chatMessages])

  useEffect(() => {
    if (!selectedPaper || !service || !currentOutputDir) return
    let cancelled = false
    apiGet<ReportPayload>(
      service,
      `/papers/${encodeURIComponent(selectedPaper.paper_id)}/report?output_dir=${encodeOutput(currentOutputDir)}`,
    )
      .then((loaded) => {
        if (!cancelled) setReport(loaded)
      })
      .catch((err) => {
        if (!cancelled) {
          setError(errorMessage(err))
          setReport(null)
        }
      })
    return () => {
      cancelled = true
    }
  }, [currentOutputDir, selectedPaper, service])

  function update<K extends keyof RunSettings>(key: K, value: RunSettings[K]) {
    setSettings((current) => ({ ...current, [key]: value }))
  }

  async function pickDirectory(kind: 'inputDir' | 'outputDir') {
    const selected = await open({ directory: true, multiple: false })
    if (typeof selected === 'string') update(kind, selected)
  }

  async function openWorkspace(outputDir = settings.outputDir) {
    if (!service || !outputDir) return
    setLoading(true)
    setError(null)
    try {
      const loaded = await apiPost<Workspace>(service, '/workspaces/open', { output_dir: outputDir })
      setWorkspace(loaded)
      update('outputDir', loaded.output_dir)
      if (!selectedPaperId || !loaded.papers.some((paper) => paper.paper_id === selectedPaperId)) {
        setSelectedPaperId(loaded.papers[0]?.paper_id ?? '')
      }
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  async function refreshJobs() {
    if (!service) return
    try {
      const path = currentOutputDir ? `/jobs?output_dir=${encodeOutput(currentOutputDir)}` : '/jobs'
      const loaded = await apiGet<{ jobs: JobSummary[] }>(service, path)
      setJobs(loaded.jobs)
    } catch {
      // a stopped service will be reported by the next explicit action
    }
  }

  function subscribeJob(jobId: string, outputDir: string) {
    if (!service) return
    let release = () => {}
    const stream = startPaperLensEventStream(
      service,
      `/jobs/${encodeURIComponent(jobId)}/events`,
      (parsed) => {
        setJobEvents((current) => [...current.slice(-500), parsed])
        if (
          parsed.type === 'job_completed'
          || parsed.type === 'job_failed'
          || parsed.type === 'job_cancelled'
          || parsed.level === 'critical'
        ) {
          release()
          void refreshJobs()
          void openWorkspace(outputDir)
        }
      },
      () => {
        release()
        void refreshJobs()
      },
    )
    release = trackEventStream(stream)
  }

  async function startReadJob() {
    if (!service) return
    setError(null)
    const payload = {
      input_dir: settings.inputDir,
      output_dir: settings.outputDir,
      ...readJobSettingsPayload(settings),
    }
    try {
      const job = await apiPost<JobSummary>(service, '/jobs/read', payload)
      setActiveJobId(job.job_id)
      setJobs((current) => [job, ...current.filter((item) => item.job_id !== job.job_id)])
      setJobEvents([])
      subscribeJob(job.job_id, job.output_dir || settings.outputDir)
    } catch (err) {
      setError(errorMessage(err))
    }
  }

  async function controlJob(jobId: string, command: 'cancel' | 'pause' | 'resume' | 'retry') {
    if (!service) return
    if (command === 'retry' && !canRetryJob) {
      setError('请先填写模型、API key 和 provider 设置后再重试。')
      return
    }
    try {
      const payload = command === 'retry' ? readJobSettingsPayload(settings) : {}
      const job = await apiPost<JobSummary>(service, `/jobs/${encodeURIComponent(jobId)}/${command}`, payload)
      if (command === 'retry') {
        setActiveJobId(job.job_id)
        setJobEvents([])
        subscribeJob(job.job_id, job.output_dir || currentOutputDir)
      }
      await refreshJobs()
    } catch (err) {
      setError(errorMessage(err))
    }
  }

  async function checkForUpdates() {
    setUpdateBusy(true)
    setUpdateStatus('正在检查更新...')
    try {
      const update = await check({ timeout: 30000 })
      if (!update) {
        setUpdateStatus('当前已经是最新版本。')
        return
      }
      const notes = update.body?.trim()
      const date = update.date ? `\n发布时间：${update.date}` : ''
      const noteBlock = notes ? `\n\n更新说明：\n${truncateText(notes, 1000)}` : ''
      const shouldInstall = await confirm(
        `发现 PaperLens ${update.version}。是否现在下载并安装？\n\n当前版本：${update.currentVersion}${date}${noteBlock}`,
        { title: 'PaperLens 更新', kind: 'info' },
      )
      if (!shouldInstall) {
        setUpdateStatus(`发现 ${update.version}，暂未安装。`)
        return
      }
      let downloaded = 0
      let total = 0
      await update.downloadAndInstall((event: DownloadEvent) => {
        if (event.event === 'Started') {
          downloaded = 0
          total = event.data.contentLength ?? 0
          setUpdateStatus(total ? `正在下载更新 0%` : '正在下载更新...')
        }
        if (event.event === 'Progress') {
          downloaded += event.data.chunkLength
          setUpdateStatus(total ? `正在下载更新 ${Math.min(100, Math.round((downloaded / total) * 100))}%` : '正在下载更新...')
        }
        if (event.event === 'Finished') setUpdateStatus('更新已下载，正在安装...')
      }, { timeout: 15 * 60 * 1000 })
      setUpdateStatus('更新已安装，正在重启 PaperLens...')
      await message('更新安装完成，PaperLens 将重启。', { title: 'PaperLens 更新', kind: 'info' })
      await relaunch()
    } catch (err) {
      setUpdateStatus(formatUpdateError(err))
    } finally {
      setUpdateBusy(false)
    }
  }

  async function clearLocalData() {
    if (maintenanceBusy) return
    const ok = await confirm(
      '这会清除 PaperLens 的本机界面设置、WebView 缓存和日志。不会删除你的论文库或输出目录。',
      { title: '清理本机数据', kind: 'warning' },
    )
    if (!ok) return
    setMaintenanceBusy(true)
    setMaintenanceStatus('正在清理本机状态...')
    setError(null)
    try {
      const report = await invoke<CleanupReport>('clear_local_app_data')
      clearPaperLensLocalStorage()
      try {
        const outputDir = await invoke<string>('default_workspace_dir')
        setSettings({ ...defaultSettings, outputDir })
      } catch {
        setSettings(defaultSettings)
      }
      setChatStore({ threads: {}, activeBySubject: {} })
      setMaintenanceStatus(cleanupSummary(report))
      if (report.errors.length) setError(report.errors.join('\n'))
    } catch (err) {
      setError(errorMessage(err))
      setMaintenanceStatus('本机状态清理失败。')
    } finally {
      setMaintenanceBusy(false)
    }
  }

  async function clearWorkspace() {
    if (!currentOutputDir || maintenanceBusy) return
    const ok = await confirm(
      `这会删除当前输出目录里的 PaperLens 结果：\n${currentOutputDir}\n\n只会在识别到 PaperLens workspace 标记后清理 .paperlens、PaperLens.md 和受管理的报告目录；不会删除输入 PDF 或普通 papers 文件夹。`,
      { title: '清空当前库', kind: 'warning' },
    )
    if (!ok) return
    setMaintenanceBusy(true)
    setMaintenanceStatus('正在清空当前库...')
    setError(null)
    try {
      const report = await invoke<CleanupReport>('clear_workspace_data', { outputDir: currentOutputDir })
      setWorkspace(null)
      setSelectedPaperId('')
      setReport(null)
      setChatStore((current) => {
        const threads = Object.fromEntries(
          Object.entries(current.threads).filter(([, thread]) => !thread.subjectKey.startsWith(`${currentOutputDir}::`)),
        )
        const activeBySubject = Object.fromEntries(
          Object.entries(current.activeBySubject).filter(([subjectKey, threadId]) =>
            !subjectKey.startsWith(`${currentOutputDir}::`) && Boolean(threads[threadId]),
          ),
        )
        return { threads, activeBySubject }
      })
      setJobEvents([])
      setMaintenanceStatus(cleanupSummary(report))
      if (report.errors.length) setError(report.errors.join('\n'))
    } catch (err) {
      setError(errorMessage(err))
      setMaintenanceStatus('当前库清空失败。')
    } finally {
      setMaintenanceBusy(false)
    }
  }

  async function doctorWorkspace(repair = false) {
    if (!service || !currentOutputDir) return
    if (repair) {
      const ok = await confirm(
        `这会检查并修复当前论文库的存储结构：\n${currentOutputDir}\n\n损坏的 JSON 会移入恢复目录，缺失目录会重新创建。`,
        { title: '修复当前库', kind: 'warning' },
      )
      if (!ok) return
    }
    setMaintenanceBusy(true)
    setMaintenanceStatus(repair ? '正在修复当前库...' : '正在诊断当前库...')
    setError(null)
    try {
      const result = await apiPost<WorkspaceMaintenanceResult>(service, '/workspaces/doctor', {
        output_dir: currentOutputDir,
        repair,
      })
      setMaintenanceStatus(workspaceMaintenanceSummary(repair ? 'repair' : 'doctor', result))
      await openWorkspace(currentOutputDir)
    } catch (err) {
      setError(errorMessage(err))
      setMaintenanceStatus('当前库维护失败。')
    } finally {
      setMaintenanceBusy(false)
    }
  }

  async function cleanupWorkspaceCache() {
    if (!service || !currentOutputDir) return
    const ok = await confirm(
      `清理当前论文库 30 天以前的缓存？\n${currentOutputDir}\n\n不会删除报告、论文记录或 QA 历史。`,
      { title: '清理缓存', kind: 'warning' },
    )
    if (!ok) return
    setMaintenanceBusy(true)
    setMaintenanceStatus('正在清理缓存...')
    setError(null)
    try {
      const result = await apiPost<WorkspaceMaintenanceResult>(service, '/workspaces/cleanup-cache', {
        output_dir: currentOutputDir,
        max_age_days: 30,
        dry_run: false,
      })
      setMaintenanceStatus(workspaceMaintenanceSummary('cache', result))
      if (result.errors?.length) setError(result.errors.join('\n'))
      await openWorkspace(currentOutputDir)
    } catch (err) {
      setError(errorMessage(err))
      setMaintenanceStatus('缓存清理失败。')
    } finally {
      setMaintenanceBusy(false)
    }
  }

  async function exportWorkspace() {
    if (!service || !currentOutputDir) return
    const archivePath = await save({
      title: '导出 PaperLens 论文库备份',
      defaultPath: defaultBackupPath(currentOutputDir),
      filters: [{ name: 'Zip Archive', extensions: ['zip'] }],
    })
    if (!archivePath) return
    setMaintenanceBusy(true)
    setMaintenanceStatus('正在导出备份...')
    setError(null)
    try {
      const result = await apiPost<WorkspaceMaintenanceResult>(service, '/workspaces/export', {
        output_dir: currentOutputDir,
        archive_path: archivePath,
        include_cache: false,
      })
      setMaintenanceStatus(workspaceMaintenanceSummary('export', result))
    } catch (err) {
      setError(errorMessage(err))
      setMaintenanceStatus('备份导出失败。')
    } finally {
      setMaintenanceBusy(false)
    }
  }

  async function importWorkspace() {
    if (!service || !currentOutputDir) return
    const archivePath = await open({
      multiple: false,
      filters: [{ name: 'Zip Archive', extensions: ['zip'] }],
    })
    if (typeof archivePath !== 'string') return
    const ok = await confirm(
      `将备份导入当前输出目录：\n${currentOutputDir}\n\n如果这里已有 PaperLens 结果，会先自动移到旁路备份目录，再恢复所选备份。`,
      { title: '导入论文库备份', kind: 'warning' },
    )
    if (!ok) return
    setMaintenanceBusy(true)
    setMaintenanceStatus('正在导入备份...')
    setError(null)
    try {
      const result = await apiPost<WorkspaceMaintenanceResult>(service, '/workspaces/import', {
        output_dir: currentOutputDir,
        archive_path: archivePath,
        replace: true,
      })
      setMaintenanceStatus(workspaceMaintenanceSummary('import', result))
      await openWorkspace(currentOutputDir)
    } catch (err) {
      setError(errorMessage(err))
      setMaintenanceStatus('备份导入失败。')
    } finally {
      setMaintenanceBusy(false)
    }
  }

  async function ask() {
    if (!service || !currentOutputDir || !question.trim()) return
    const text = question.trim()
    const messageId = newChatMessageId()
    const threadId = activeChatThreadId || createChatThread(text)
    const history = chatMessages
      .filter((message) => !message.pending && !message.error)
      .slice(-8)
      .map((message) => ({ role: message.role, content: message.content }))
    setQuestion('')
    setChatStore((current) => {
      const thread = current.threads[threadId]
      if (!thread || thread.messages.length || thread.title !== '新对话') return current
      return {
        ...current,
        threads: {
          ...current.threads,
          [threadId]: { ...thread, title: chatTitleFromQuestion(text), updatedAt: new Date().toISOString() },
        },
      }
    })
    setChatThreadMessages(threadId, (current) => [
      ...current,
      { id: `${messageId}_user`, role: 'user', scope: chatScope, content: text },
      {
        id: `${messageId}_assistant`,
        role: 'assistant',
        scope: chatScope,
        content: '正在回到原文和已读内容里核对...',
        pending: true,
      },
    ])
    try {
      const answer = await apiPost<AnswerSummary>(service, '/ask', {
        scope: chatScope,
        output_dir: currentOutputDir,
        paper_id: chatScope === 'paper' ? selectedPaper?.paper_id : null,
        question: text,
        ...modelSettingsPayload(settings),
        limit: 8,
        chat_history: history,
      })
      subscribeAnswer(answer.answer_id, `${messageId}_assistant`, threadId, currentOutputDir)
    } catch (err) {
      const text = errorMessage(err)
      setChatThreadMessages(threadId, (current) =>
        current.map((message) =>
          message.id === `${messageId}_assistant`
            ? { ...message, pending: false, content: text, error: text }
            : message,
        ),
      )
    }
  }

  function handleQuestionKeyDown(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) return
    event.preventDefault()
    if (canAsk) void ask()
  }

  function subscribeAnswer(
    answerId: string,
    assistantMessageId: string,
    threadKey: string,
    outputDir: string,
  ) {
    if (!service) return
    let release = () => {}
    const stream = startPaperLensEventStream(
      service,
      `/ask/${encodeURIComponent(answerId)}/events`,
      (parsed) => {
        if (parsed.type === 'answer_started') {
          setChatThreadMessages(threadKey, (current) =>
            current.map((message) =>
              message.id === assistantMessageId
                ? { ...message, content: parsed.message ?? '正在核对证据...' }
                : message,
            ),
          )
        }
        if (parsed.type === 'answer_completed') {
          const answer = (parsed.data?.answer ?? null) as AnswerPayload | null
          setChatThreadMessages(threadKey, (current) =>
            current.map((message) =>
              message.id === assistantMessageId
                ? {
                    ...message,
                    pending: false,
                    content: answerText(answer) || '没有返回可用回答。',
                    answer,
                    error: null,
                  }
                : message,
            ),
          )
          release()
          void openWorkspace(outputDir)
        }
        if (parsed.type === 'answer_failed') {
          const text = parsed.message ?? '回答失败'
          setChatThreadMessages(threadKey, (current) =>
            current.map((message) =>
              message.id === assistantMessageId
                ? { ...message, pending: false, content: text, error: text }
                : message,
            ),
          )
          release()
        }
      },
      () => {
        const text = '连接已中断，请稍后重试。'
        setChatThreadMessages(threadKey, (current) =>
          current.map((message) =>
            message.id === assistantMessageId && message.pending
              ? { ...message, pending: false, content: text, error: text }
              : message,
          ),
        )
        release()
      },
    )
    release = trackEventStream(stream)
  }

  function trackEventStream(stream: AbortController) {
    eventStreamsRef.current.add(stream)
    return () => {
      stream.abort()
      eventStreamsRef.current.delete(stream)
    }
  }

  function clearCurrentChat() {
    setCurrentChatMessages(() => [])
  }

  function startNewChat() {
    createChatThread()
  }

  function beginResize(pane: ResizePane, event: ReactPointerEvent<HTMLDivElement>) {
    event.preventDefault()
    const startX = event.clientX
    const startLeftWidth = leftWidth
    const startRightWidth = rightWidth

    function onPointerMove(moveEvent: PointerEvent) {
      if (pane === 'left') {
        setLeftWidth(clamp(startLeftWidth + moveEvent.clientX - startX, 240, 520))
      } else {
        setRightWidth(clamp(startRightWidth - (moveEvent.clientX - startX), 300, 620))
      }
    }

    function onPointerUp() {
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerup', onPointerUp)
      document.body.classList.remove('resizing-layout')
    }

    document.body.classList.add('resizing-layout')
    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp, { once: true })
  }

  return (
    <main
      className={`app-shell ${effectiveLeftSidebarOpen ? '' : 'left-collapsed'} ${effectiveRightSidebarOpen ? '' : 'right-collapsed'}`}
      style={{
        gridTemplateColumns: `${layoutLeftWidth}px ${effectiveLeftSidebarOpen ? 4 : 0}px minmax(0, 1fr) ${effectiveRightSidebarOpen ? 4 : 0}px ${layoutRightWidth}px`,
      }}
    >
      <aside className="workspace-panel">
        {!effectiveLeftSidebarOpen ? (
          <div className="collapsed-rail">
            <div className="mark">PL</div>
            <button
              type="button"
              className="icon-button"
              title="展开侧栏"
              aria-label="展开侧栏"
              onClick={() => setLeftSidebarOpen(true)}
            >
              <PanelLeftOpen size={17} />
            </button>
            <button
              type="button"
              className="icon-button"
              title="设置"
              aria-label="设置"
              onClick={() => {
                setLeftSidebarOpen(true)
                setSettingsOpen(true)
              }}
            >
              <Settings2 size={17} />
            </button>
          </div>
        ) : (
          <>
            <header className="workspace-head">
              <div className="brand">
                <div className="mark">PL</div>
                <div>
                  <h1>PaperLens</h1>
                  <span>{service ? '已连接' : '启动中'}</span>
                </div>
              </div>
              <div className="workspace-tools">
                <button
                  type="button"
                  className="icon-button"
                  aria-label="Settings"
                  title="设置"
                  onClick={() => setSettingsOpen((value) => !value)}
                >
                  <Settings2 size={17} />
                </button>
                <button
                  type="button"
                  className="icon-button"
                  aria-label="折叠侧栏"
                  title="折叠侧栏"
                  onClick={() => setLeftSidebarOpen(false)}
                >
                  <PanelLeftClose size={17} />
                </button>
              </div>
            </header>

            <div className="primary-actions">
              <button type="button" className="primary" disabled={!canRun} onClick={startReadJob}>
                <Play size={16} /> 读论文
              </button>
              <button
                type="button"
                className="icon-button"
                title="刷新"
                aria-label="Refresh"
                onClick={() => openWorkspace()}
                disabled={!settings.outputDir || loading}
              >
                {loading ? <Loader2 className="spinning" size={15} /> : <RefreshCw size={15} />}
              </button>
            </div>

            {settingsOpen && (
              <SettingsPanel
                settings={settings}
                update={update}
                pickDirectory={pickDirectory}
                openWorkspace={openWorkspace}
                loading={loading}
                updateBusy={updateBusy}
                updateStatus={updateStatus}
                maintenanceStatus={maintenanceStatus}
                maintenanceBusy={maintenanceBusy}
                workspaceStorage={workspace?.storage}
                checkForUpdates={checkForUpdates}
                clearLocalData={clearLocalData}
                clearWorkspace={clearWorkspace}
                doctorWorkspace={() => doctorWorkspace(false)}
                repairWorkspace={() => doctorWorkspace(true)}
                cleanupWorkspaceCache={cleanupWorkspaceCache}
                exportWorkspace={exportWorkspace}
                importWorkspace={importWorkspace}
                canClearWorkspace={Boolean(currentOutputDir)}
                canMaintainWorkspace={Boolean(service && currentOutputDir)}
              />
            )}

            <section className="library-section">
              <div className="section-title">
                <Library size={16} />
                <span>论文库</span>
                <small>{workspace?.paper_count ?? 0}</small>
              </div>
              <div className="paper-list">
                {!workspace?.papers.length && <div className="empty">选择输出目录后显示已读论文</div>}
                {workspace?.papers.map((paper) => (
                  <button
                    key={paper.paper_id}
                    type="button"
                    className={paper.paper_id === selectedPaper?.paper_id ? 'paper-card active' : 'paper-card'}
                    onClick={() => setSelectedPaperId(paper.paper_id)}
                  >
                    <span className="paper-meta">
                      <strong>{paper.grade || '—'}</strong>
                      {paper.source?.year ? <span>{paper.source.year}</span> : null}
                      {paper.qa?.count ? <span>{paper.qa.count} QA</span> : null}
                    </span>
                    <span className="paper-title">{paper.title}</span>
                    {paper.brief ? <span className="paper-brief">{paper.brief}</span> : null}
                    <span className="concept-row">
                      {(paper.concepts ?? paper.tags ?? []).slice(0, 3).map((term) => (
                        <span key={term}>{term}</span>
                      ))}
                    </span>
                  </button>
                ))}
              </div>
            </section>
          </>
        )}
      </aside>
      {effectiveLeftSidebarOpen && (
        <div
          className="resize-handle left-resize"
          role="separator"
          aria-label="调整左侧栏宽度"
          onPointerDown={(event) => beginResize('left', event)}
        />
      )}

      <section className="main-stage">
        <header className="paper-toolbar">
          <div className="paper-heading">
            <p className="eyebrow">知识胶囊</p>
            <h2>{selectedPaper?.title ?? '选择一篇已经读过的论文'}</h2>
            {selectedPaper && (
              <div className="metadata-row">
                <span>{selectedPaper.grade || '未分级'}</span>
                {selectedPaper.recommendation ? <span>{selectedPaper.recommendation}</span> : null}
                {selectedPaper.memory?.claim_count ? <span>{selectedPaper.memory.claim_count} 个要点</span> : null}
                {selectedPaper.memory?.evidence_count ? (
                  <span>{selectedPaper.memory.evidence_count} 条证据</span>
                ) : null}
              </div>
            )}
          </div>
          <div className="reader-actions">
            {!effectiveLeftSidebarOpen && (
              <button type="button" className="icon-button" title="展开左侧栏" onClick={() => setLeftSidebarOpen(true)}>
                <PanelLeftOpen size={16} />
              </button>
            )}
            {selectedPaper?.report_file && (
              <button type="button" className="icon-text-button" onClick={() => openPath(selectedPaper.report_file!)}>
                <FileText size={16} /> 打开文档
              </button>
            )}
            <button type="button" className={evidenceOpen ? 'active-button' : ''} onClick={() => setEvidenceOpen((value) => !value)}>
              <PanelRightOpen size={16} /> 证据
            </button>
            {!effectiveRightSidebarOpen && (
              <button type="button" className="icon-button" title="展开右侧栏" onClick={() => setRightSidebarOpen(true)}>
                <PanelRightOpen size={16} />
              </button>
            )}
          </div>
        </header>

        {error && (
          <div className="notice error">
            <AlertTriangle size={17} />
            <span>{error}</span>
          </div>
        )}

        <div className="reader-scroll">
          <article className="report-article">
            {report ? (
              <MarkdownBlock markdown={report.markdown} report={report} outputDir={currentOutputDir} service={service} />
            ) : (
              <div className="empty large">还没有可展示的报告</div>
            )}
          </article>
        </div>

        {evidenceOpen && (
          <aside className="evidence-drawer">
            <EvidenceSummary
              key={`${currentOutputDir}::${selectedPaper?.paper_id ?? ''}`}
              paper={selectedPaper}
              service={service}
              outputDir={currentOutputDir}
            />
          </aside>
        )}
      </section>
      {effectiveRightSidebarOpen && (
        <div
          className="resize-handle right-resize"
          role="separator"
          aria-label="调整右侧栏宽度"
          onPointerDown={(event) => beginResize('right', event)}
        />
      )}

      <aside className="workbench-panel">
        {!effectiveRightSidebarOpen ? (
          <div className="collapsed-rail right">
            <button
              type="button"
              className="icon-button"
              title="展开右侧栏"
              aria-label="展开右侧栏"
              onClick={() => setRightSidebarOpen(true)}
            >
              <PanelRightOpen size={17} />
            </button>
            <button
              type="button"
              className="icon-button"
              title="打开对话"
              aria-label="打开对话"
              onClick={() => setRightSidebarOpen(true)}
            >
              <MessageSquareText size={17} />
            </button>
          </div>
        ) : (
          <>
            <section className="chat-panel">
              <div className="chat-head">
                <div className="chat-title">
                  <h3>{chatScope === 'library' ? '问整个论文库' : '问当前论文'}</h3>
                </div>
                <div className="chat-head-actions">
                  <select
                    className="thread-select"
                    value={activeChatThreadId}
                    onChange={(event) => selectChatThread(event.target.value)}
                    disabled={!subjectThreads.length}
                    aria-label="选择对话"
                  >
                    {!subjectThreads.length && <option value="">新对话</option>}
                    {subjectThreads.map((thread) => (
                      <option key={thread.id} value={thread.id}>
                        {thread.title}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    className="icon-button small"
                    title="新建对话"
                    aria-label="新建对话"
                    onClick={startNewChat}
                  >
                    <Plus size={14} />
                  </button>
                  <div className="segmented">
                    <button type="button" className={chatScope === 'paper' ? 'active' : ''} onClick={() => setChatScope('paper')}>
                      当前论文
                    </button>
                    <button type="button" className={chatScope === 'library' ? 'active' : ''} onClick={() => setChatScope('library')}>
                      论文库
                    </button>
                  </div>
                  <button
                    type="button"
                    className="icon-button small clear-chat-button"
                    title="清空当前对话"
                    aria-label="清空当前对话"
                    disabled={!chatMessages.length}
                    onClick={clearCurrentChat}
                  >
                    <Trash2 size={14} />
                  </button>
                  <button
                    type="button"
                    className="icon-button small"
                    title="折叠右侧栏"
                    aria-label="折叠右侧栏"
                    onClick={() => setRightSidebarOpen(false)}
                  >
                    <PanelRightClose size={14} />
                  </button>
                </div>
              </div>

              {visibleJob && (
                <div className="activity-strip">
                  <div>
                    <strong>{statusLabel(visibleJob.status)}</strong>
                    <span>
                      {ACTIVE_JOB_STATUSES.has(visibleJob.status)
                        ? stageLabel(visibleJob.current_stage)
                        : visibleJob.error ?? visibleJobEvent?.message ?? stageLabel(visibleJob.current_stage)}
                    </span>
                  </div>
                  <div className="job-controls">
                    {visibleJob.status === 'paused' && (
                      <button
                        type="button"
                        className="icon-button small"
                        title="继续"
                        aria-label="继续"
                        onClick={() => controlJob(visibleJob.job_id, 'resume')}
                      >
                        <Play size={14} />
                      </button>
                    )}
                    {visibleJob.status === 'running' && (
                      <button
                        type="button"
                        className="icon-button small"
                        title="暂停"
                        aria-label="暂停"
                        onClick={() => controlJob(visibleJob.job_id, 'pause')}
                      >
                        <Pause size={14} />
                      </button>
                    )}
                    {visibleJob.live !== false && RETRYABLE_JOB_STATUSES.has(visibleJob.status) && (
                      <button
                        type="button"
                        className="icon-button small"
                        title={canRetryJob ? '重试' : '填写模型设置后重试'}
                        aria-label="重试"
                        disabled={!canRetryJob}
                        onClick={() => controlJob(visibleJob.job_id, 'retry')}
                      >
                        <RotateCcw size={14} />
                      </button>
                    )}
                    {ACTIVE_JOB_STATUSES.has(visibleJob.status) && (
                      <button
                        type="button"
                        className="icon-button small"
                        title="停止"
                        aria-label="停止"
                        disabled={visibleJob.status === 'cancelling'}
                        onClick={() => controlJob(visibleJob.job_id, 'cancel')}
                      >
                        <Square size={14} />
                      </button>
                    )}
                  </div>
                </div>
              )}

              <div className="chat-log">
                {!chatMessages.length && (
                  <div className="empty">
                    暂无对话
                  </div>
                )}
                {chatMessages.map((message) => (
                  <ChatBubble key={message.id} message={message} report={report} outputDir={currentOutputDir} service={service} />
                ))}
                <div ref={chatEndRef} />
              </div>

              <div className="ask-box">
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={handleQuestionKeyDown}
                  placeholder={chatScope === 'library' ? '问本地已读论文库...' : '问当前论文...'}
                />
                <button
                  type="button"
                  className="primary send-button"
                  disabled={!canAsk}
                  onClick={ask}
                  aria-label="发送"
                  title="发送，Shift+Enter 换行"
                >
                  <ArrowUp size={18} />
                </button>
              </div>
            </section>
          </>
        )}
      </aside>
    </main>
  )
}

function SettingsPanel({
  settings,
  update,
  pickDirectory,
  openWorkspace,
  loading,
  updateBusy,
  updateStatus,
  maintenanceStatus,
  maintenanceBusy,
  workspaceStorage,
  checkForUpdates,
  clearLocalData,
  clearWorkspace,
  doctorWorkspace,
  repairWorkspace,
  cleanupWorkspaceCache,
  exportWorkspace,
  importWorkspace,
  canClearWorkspace,
  canMaintainWorkspace,
}: {
  settings: RunSettings
  update: <K extends keyof RunSettings>(key: K, value: RunSettings[K]) => void
  pickDirectory: (kind: 'inputDir' | 'outputDir') => void
  openWorkspace: (outputDir?: string) => void
  loading: boolean
  updateBusy: boolean
  updateStatus: string
  maintenanceStatus: string
  maintenanceBusy: boolean
  workspaceStorage?: WorkspaceStorageHealth
  checkForUpdates: () => void
  clearLocalData: () => void
  clearWorkspace: () => void
  doctorWorkspace: () => void
  repairWorkspace: () => void
  cleanupWorkspaceCache: () => void
  exportWorkspace: () => void
  importWorkspace: () => void
  canClearWorkspace: boolean
  canMaintainWorkspace: boolean
}) {
  const storageClass = workspaceStatusClass(workspaceStorage?.status)
  const maintenanceDisabled = maintenanceBusy || updateBusy
  return (
    <section className="settings-panel">
      <DirectoryField label="输入目录" value={settings.inputDir} onChange={(value) => update('inputDir', value)} onPick={() => pickDirectory('inputDir')} />
      <DirectoryField label="输出目录" value={settings.outputDir} onChange={(value) => update('outputDir', value)} onPick={() => pickDirectory('outputDir')} />
      <div className="field">
        <label>Provider</label>
        <select value={settings.providerKind} onChange={(event) => update('providerKind', event.target.value as ProviderKind)}>
          <option value="openai-compatible">OpenAI Compatible</option>
          <option value="openai">OpenAI</option>
          <option value="anthropic-compatible">Anthropic Compatible</option>
          <option value="anthropic">Anthropic</option>
        </select>
      </div>
      <div className="field">
        <label>Base URL</label>
        <input value={settings.baseUrl} onChange={(event) => update('baseUrl', event.target.value)} />
      </div>
      <div className="field">
        <label>Model</label>
        <input value={settings.model} onChange={(event) => update('model', event.target.value)} />
      </div>
      <div className="field">
        <label>API Key</label>
        <div className="input-with-icon">
          <input type="password" value={settings.apiKey} onChange={(event) => update('apiKey', event.target.value)} />
          <KeyRound size={16} />
        </div>
      </div>
      <div className="settings-row">
        <div className="field">
          <label>Concurrency</label>
          <input type="number" min="1" max="16" value={settings.concurrency} onChange={(event) => update('concurrency', event.target.value)} />
        </div>
      </div>
      <button type="button" onClick={() => openWorkspace()} disabled={!settings.outputDir || loading}>
        {loading ? <Loader2 className="spinning" size={15} /> : <FolderOpen size={15} />} 导入已有结果
      </button>
      <div className="maintenance-block">
        <strong>维护</strong>
        <div className={`workspace-health ${storageClass}`}>
          <span>
            <ShieldCheck size={14} /> 当前库
          </span>
          <b>{workspaceHealthSummary(workspaceStorage)}</b>
        </div>
        <div className="maintenance-actions workspace-actions">
          <button type="button" onClick={doctorWorkspace} disabled={!canMaintainWorkspace || maintenanceDisabled}>
            {maintenanceBusy ? <Loader2 className="spinning" size={15} /> : <ShieldCheck size={15} />} 诊断
          </button>
          <button type="button" onClick={repairWorkspace} disabled={!canMaintainWorkspace || maintenanceDisabled}>
            <Wrench size={15} /> 修复
          </button>
          <button type="button" onClick={exportWorkspace} disabled={!canMaintainWorkspace || maintenanceDisabled}>
            <Archive size={15} /> 导出
          </button>
          <button type="button" onClick={importWorkspace} disabled={!canMaintainWorkspace || maintenanceDisabled}>
            <FolderOpen size={15} /> 导入
          </button>
          <button type="button" onClick={cleanupWorkspaceCache} disabled={!canMaintainWorkspace || maintenanceDisabled}>
            <Database size={15} /> 清缓存
          </button>
        </div>
        <div className="maintenance-actions">
          <button type="button" onClick={checkForUpdates} disabled={updateBusy || maintenanceBusy}>
            {updateBusy ? <Loader2 className="spinning" size={15} /> : <RefreshCw size={15} />} 检查更新
          </button>
          <button type="button" onClick={clearLocalData} disabled={maintenanceDisabled}>
            清本机状态
          </button>
          <button type="button" className="danger-button" onClick={clearWorkspace} disabled={!canClearWorkspace || maintenanceDisabled}>
            清当前库
          </button>
        </div>
        {updateStatus && <p>{updateStatus}</p>}
        {maintenanceStatus && <p>{maintenanceStatus}</p>}
      </div>
    </section>
  )
}

function DirectoryField({ label, value, onChange, onPick }: { label: string; value: string; onChange: (value: string) => void; onPick: () => void }) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="path-row">
        <input value={value} onChange={(event) => onChange(event.target.value)} />
        <button type="button" className="icon-button" onClick={onPick}>
          <FolderOpen size={16} />
        </button>
      </div>
    </div>
  )
}

function MarkdownBlock({
  markdown,
  report,
  outputDir,
  service,
}: {
  markdown: string
  report: ReportPayload | null
  outputDir: string
  service: ServiceInfo | null
}) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeSanitize, markdownSanitizeSchema], rehypeKatex]}
        components={{
          img: ({ src, alt }) => (
            <MarkdownImage src={src} alt={alt ?? ''} report={report} outputDir={outputDir} service={service} />
          ),
          a: ({ href, children }) => {
            const externalHref = href && /^(https?:|mailto:)/i.test(href) ? href : ''
            return (
              <a
                href={externalHref || undefined}
                rel={externalHref ? 'noreferrer' : undefined}
                onClick={(event) => {
                  event.preventDefault()
                  if (externalHref) void openUrl(externalHref).catch(() => undefined)
                }}
              >
                {children}
              </a>
            )
          },
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  )
}

function MarkdownImage({
  src,
  alt,
  report,
  outputDir,
  service,
}: {
  src?: string
  alt: string
  report: ReportPayload | null
  outputDir: string
  service: ServiceInfo | null
}) {
  const candidates = useMemo(
    () => localAssetCandidates(src, report, outputDir, service),
    [outputDir, report, service, src],
  )
  const candidateKey = candidates
    .map((candidate) => `${candidate.authenticated ? 'auth' : 'direct'}:${candidate.src}`)
    .join('\n')
  const [loadState, setLoadState] = useState({ key: '', index: 0 })
  const [fetchedAsset, setFetchedAsset] = useState({ key: '', src: '' })
  const candidateIndex = loadState.key === candidateKey ? loadState.index : 0
  const currentCandidate = candidates[candidateIndex] ?? null
  const fetchKey = `${candidateKey}\n#${candidateIndex}`
  useEffect(() => {
    if (!currentCandidate?.authenticated) return
    if (!service) return
    let cancelled = false
    let objectUrl = ''
    const controller = new AbortController()
    fetch(currentCandidate.src, {
      headers: serviceAuthHeaders(service),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseErrorMessage(response))
        const blob = await response.blob()
        objectUrl = URL.createObjectURL(blob)
        if (cancelled) URL.revokeObjectURL(objectUrl)
        else setFetchedAsset({ key: fetchKey, src: objectUrl })
      })
      .catch(() => {
        if (cancelled) return
        setLoadState((current) => {
          const currentIndex = current.key === candidateKey ? current.index : 0
          const nextIndex = currentIndex + 1
          return nextIndex < candidates.length ? { key: candidateKey, index: nextIndex } : current
        })
      })
    return () => {
      cancelled = true
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [candidateKey, candidates.length, currentCandidate, fetchKey, service])
  const currentSrc = currentCandidate?.authenticated
    ? fetchedAsset.key === fetchKey ? fetchedAsset.src : ''
    : currentCandidate?.src ?? ''
  if (!currentCandidate || !currentSrc) return null
  return (
    <img
      src={currentSrc}
      alt={alt}
      loading="lazy"
      onError={() => {
        setLoadState((current) => {
          const currentIndex = current.key === candidateKey ? current.index : 0
          const nextIndex = currentIndex + 1
          return nextIndex < candidates.length ? { key: candidateKey, index: nextIndex } : current
        })
      }}
    />
  )
}

function ChatBubble({
  message,
  report,
  outputDir,
  service,
}: {
  message: ChatMessage
  report: ReportPayload | null
  outputDir: string
  service: ServiceInfo | null
}) {
  return (
    <div className={`chat-message ${message.role} ${message.pending ? 'pending' : ''}`}>
      {message.role === 'assistant' && <strong>PaperLens</strong>}
      {message.pending ? (
        <div className="thinking-line">
          <Loader2 className="spinning" size={15} /> 正在核对证据...
        </div>
      ) : (
        <MarkdownBlock markdown={message.content} report={report} outputDir={outputDir} service={service} />
      )}
      {message.answer && <AnswerEvidence answer={message.answer} />}
      {message.error && (
        <div className="inline-error">
          <XCircle size={14} /> {message.error}
        </div>
      )}
    </div>
  )
}

function AnswerEvidence({ answer }: { answer: AnswerPayload }) {
  const source = answer.source_attribution
  const rows: Array<[string, string[] | undefined]> = [
    ['论文明确支持', source?.paper_claims],
    ['PaperLens 推断', source?.paperlens_inferences],
    ['跨论文综合', source?.cross_paper_synthesis],
    ['背景知识', source?.background_context],
    ['证据边界', source?.evidence_limits],
  ]
  return (
    <details className="evidence-details">
      <summary>
        <ChevronRight size={14} />
        证据边界
        {answer.cited_source_ids?.length ? (
          <span>{answer.cited_source_ids.length} 条依据</span>
        ) : null}
        {answer.confidence ? <span>{answer.confidence}</span> : null}
      </summary>
      {rows.map(([label, items]) => (
        items?.length ? (
          <div key={label} className="evidence-group">
            <strong>{label}</strong>
            <ul>
              {items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ) : null
      ))}
      {answer.related_papers?.length ? (
        <div className="evidence-group">
          <strong>相关论文</strong>
          <ul>
            {answer.related_papers.map((paper) => (
              <li key={paper.paper_id}>{paper.title}：{paper.why_related}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </details>
  )
}

function EvidenceSummary({ paper, service, outputDir }: { paper: PaperSummary | null; service: ServiceInfo | null; outputDir: string }) {
  const [payloadState, setPayloadState] = useState<{ key: string; payload: Record<string, unknown> } | null>(null)
  const [open, setOpen] = useState(false)
  const paperId = paper?.paper_id ?? ''
  const payloadKey = `${outputDir}::${paperId}`
  useEffect(() => {
    if (!paperId || !service || !outputDir) return
    let cancelled = false
    apiGet<Record<string, unknown>>(
      service,
      `/papers/${encodeURIComponent(paperId)}/evidence?output_dir=${encodeOutput(outputDir)}`,
    )
      .then((value) => {
        if (!cancelled) setPayloadState({ key: `${outputDir}::${paperId}`, payload: value })
      })
      .catch(() => {
        if (!cancelled) setPayloadState(null)
      })
    return () => {
      cancelled = true
    }
  }, [paperId, service, outputDir])
  const payload = payloadState?.key === payloadKey ? payloadState.payload : null
  const claims = Array.isArray(payload?.claims) ? payload.claims : []
  const evidence = Array.isArray(payload?.evidence) ? payload.evidence : []
  return (
    <div>
      <div className="drawer-head">
        <BookOpen size={16} />
        <strong>证据</strong>
      </div>
      <div className="memory-stats">
        <span>{claims.length} 个要点</span>
        <span>{evidence.length} 条证据</span>
      </div>
      <button type="button" onClick={() => setOpen((value) => !value)}>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />} 查看前 8 条证据
      </button>
      {open && (
        <div className="evidence-list">
          {evidence.slice(0, 8).map((item, index) => (
            <div key={index} className="evidence-item">
              <strong title={String((item as Record<string, unknown>).id ?? '')}>依据 {index + 1}</strong>
              <p>{String((item as Record<string, unknown>).interpretation ?? (item as Record<string, unknown>).text ?? '')}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default App
