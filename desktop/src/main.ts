import {
  ArrowDown,
  ArrowUp,
  ArrowUpRight,
  Brain,
  Check,
  CheckCheck,
  ChevronDown,
  ChevronRight,
  CircleCheck,
  CircleHelp,
  CircleX,
  Coins,
  Container,
  Copy,
  createIcons,
  Ellipsis,
  Eraser,
  FileCode2,
  FolderOpen,
  Gauge,
  GitPullRequest,
  History,
  Info,
  LoaderCircle,
  LogOut,
  PanelLeft,
  PanelLeftClose,
  Plus,
  ScanSearch,
  Settings,
  Settings2,
  ShieldAlert,
  ShieldCheck,
  Shrink,
  SlidersHorizontal,
  Sparkles,
  Square,
  Terminal,
  TestTube,
  Trash2,
  TriangleAlert,
  Undo2,
  Wrench,
  X,
} from "lucide";
import DOMPurify from "dompurify";
import { marked, type TokenizerAndRendererExtension, type Tokens } from "marked";
import "./style.css";

type AnyRecord = Record<string, any>;

const appRoot = document.querySelector<HTMLDivElement>("#app")!;
// Without the Electron preload there is no backend to talk to. Stub the API so
// the layout still renders, but never swallow a command silently: every send
// reports why nothing happened, or a broken launch looks like a frozen UI.
const previewMode = !window.cagent;
if (previewMode) {
  const previewConfig = { model: "desktop preview", approval_mode: "auto-edit" };
  let notify: ((event: BackendEvent) => void) | null = null;
  window.cagent = {
    send(): void {
      notify?.({
        type: "protocol_error",
        message: "Preview mode: no backend attached. Run `npm run start` in desktop/.",
      });
    },
    onEvent(listener): () => void {
      notify = listener;
      const timer = window.setTimeout(() => {
        listener({ type: "ready", session_id: "preview", workspace: "Browser preview", config: previewConfig });
        listener({
          type: "status",
          busy: false,
          steps: 0,
          context: { tokens: 0, window: 128000, messages: 0, compactions: 0 },
          config: previewConfig,
        });
        listener({
          type: "warning",
          message: "Preview mode - the Python backend is not attached",
          detail:
            "This page is running without the Electron preload, so the composer, "
            + "slash commands, and the approval-mode control cannot reach cagent.\n"
            + "Launch the desktop app instead: cd desktop && npm run start",
        });
      }, 0);
      return () => { notify = null; window.clearTimeout(timer); };
    },
    async chooseWorkspace(): Promise<null> { return null; },
    minimize(): void {},
    maximize(): void {},
    close(): void {},
  };
}
const uiIcons = {
  ArrowDown,
  ArrowUp,
  ArrowUpRight,
  Brain,
  Check,
  CheckCheck,
  ChevronDown,
  ChevronRight,
  CircleCheck,
  CircleHelp,
  CircleX,
  Coins,
  Container,
  Copy,
  Ellipsis,
  Eraser,
  FileCode2,
  FolderOpen,
  Gauge,
  GitPullRequest,
  History,
  Info,
  LoaderCircle,
  LogOut,
  PanelLeft,
  PanelLeftClose,
  Plus,
  ScanSearch,
  Settings,
  Settings2,
  ShieldAlert,
  ShieldCheck,
  Shrink,
  SlidersHorizontal,
  Sparkles,
  Square,
  Terminal,
  TestTube,
  Trash2,
  TriangleAlert,
  Undo2,
  Wrench,
  X,
};
const state = {
  sessions: [] as AnyRecord[],
  activeSession: "",
  restoredFrom: "",
  workspace: "",
  config: {} as AnyRecord,
  status: { busy: false, usage: {}, context: {}, steps: 0 } as AnyRecord,
  pendingApproval: null as AnyRecord | null,
  sidebarCollapsed: localStorage.getItem("cagent.sidebar") === "collapsed",
  settingsOpen: false,
  commandMenuOpen: false,
  commandQuery: "",
  showThinking: localStorage.getItem("cagent.thinking") !== "hidden",
  assistantNode: null as HTMLElement | null,
  assistantText: "",
  toolNodes: new Map<string, HTMLElement>(),
};

const commands = [
  ["/help", "Show command help", "circle-help"],
  ["/tools", "List available tools", "wrench"],
  ["/cost", "Show token usage and steps", "coins"],
  ["/context", "Show context pressure", "gauge"],
  ["/effort", "View or change reasoning effort", "brain"],
  ["/approve", "Change approval mode", "shield-check"],
  ["/sandbox", "Inspect or control Docker sandbox", "container"],
  ["/undo", "Remove the latest turn from context", "undo-2"],
  ["/clear", "Clear conversation context", "eraser"],
  ["/resume", "Browse saved sessions", "history"],
  ["/exit", "Close the session", "log-out"],
] as const;

function icon(name: string, size = 16): string {
  return `<i data-lucide="${name}" width="${size}" height="${size}" aria-hidden="true"></i>`;
}

function renderIcons(): void {
  createIcons({ icons: uiIcons, attrs: { "stroke-width": 1.8 } });
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]!);
}

function relativeDate(value: unknown): string {
  if (typeof value !== "string") return "";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "";
  const seconds = Math.max(0, (Date.now() - date.valueOf()) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** A ``<delim>text<delim>`` inline span, for marks Markdown has no syntax for. */
function pairedMark(name: string, delimiter: string, tag: string): TokenizerAndRendererExtension {
  const rule = new RegExp(`^${delimiter.replace(/[+=^~]/g, "\\$&")}(?=\\S)([\\s\\S]*?\\S)${delimiter.replace(/[+=^~]/g, "\\$&")}`);
  return {
    name,
    level: "inline",
    start(src: string) { return src.indexOf(delimiter); },
    tokenizer(src: string) {
      const match = rule.exec(src);
      if (!match) return undefined;
      return { type: name, raw: match[0], tokens: this.lexer.inlineTokens(match[1] ?? "") };
    },
    renderer(token) { return `<${tag}>${this.parser.parseInline(token.tokens ?? [])}</${tag}>`; },
  };
}

// GFM gives us strikethrough, tables and task lists; `breaks` matches the chat
// convention that a newline the user typed is a newline they meant. Underline
// and highlight have no Markdown spelling at all, so models reach for ++ and ==
// (or raw <u>, which DOMPurify's html profile already keeps) — support both.
marked.use({
  gfm: true,
  breaks: true,
  extensions: [pairedMark("underline", "++", "u"), pairedMark("highlight", "==", "mark")],
  renderer: {
    code(token: Tokens.Code): string | false {
      const language = ((token.lang || "").trim().split(/\s+/)[0] ?? "").toLowerCase();
      return language === "diff" || language === "patch" ? renderDiff(token.text) : false;
    },
  },
});

function markdown(value: string): string {
  const html = marked.parse(value || "", { async: false }) as string;
  return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
}

const MAX_DIFF_LINES = 260;
/** Header lines of a unified diff. Only meaningful before the first hunk: inside
    one, `--- x` is a deleted line whose text happens to start with `--`. */
const DIFF_HEADER = /^(diff |index |--- |\+\+\+ |old mode|new mode|new file|deleted file|similarity |rename |copy |Binary files )/;

function isUnifiedDiff(text: string): boolean {
  return text.startsWith("--- ") || text.startsWith("diff ");
}

function diffStats(text: string): { added: number; removed: number } {
  let added = 0;
  let removed = 0;
  for (const line of text.split("\n")) {
    if (line.startsWith("+++") || line.startsWith("---")) continue;
    if (line.startsWith("+")) added += 1;
    else if (line.startsWith("-")) removed += 1;
  }
  return { added, removed };
}

function diffRow(kind: string, lineNumber: number | null, sign: string, text: string): string {
  return `<div class="diff-line ${kind}"><span class="diff-no">${lineNumber ?? ""}</span><span class="diff-sign">${sign}</span><span class="diff-text">${escapeHtml(text) || "&nbsp;"}</span></div>`;
}

/** A unified diff as added/removed rows, the way the CLI prints it.
 *
 * Line numbers come from the hunk headers rather than a running count, so a
 * clipped or multi-hunk diff still points at real positions in the file.
 */
function renderDiff(text: string): string {
  const lines = text.replace(/\n+$/, "").split("\n");
  const shown = lines.slice(0, MAX_DIFF_LINES);
  let oldLine = 0;
  let newLine = 0;
  let inHunk = false;
  const rows = shown.map((line) => {
    const hunk = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(line);
    if (hunk) {
      oldLine = Number(hunk[1]);
      newLine = Number(hunk[2]);
      inHunk = true;
      return diffRow("hunk", null, "", line);
    }
    if (line.startsWith("\\") || (!inHunk && DIFF_HEADER.test(line))) return diffRow("meta", null, "", line);
    if (line.startsWith("+")) return diffRow("add", newLine++, "+", line.slice(1));
    if (line.startsWith("-")) return diffRow("del", oldLine++, "-", line.slice(1));
    oldLine += 1;
    return diffRow("ctx", newLine++, "", line.startsWith(" ") ? line.slice(1) : line);
  });
  const hidden = lines.length - shown.length;
  const more = hidden > 0 ? `<div class="diff-more">… ${hidden} more diff line${hidden === 1 ? "" : "s"}</div>` : "";
  return `<div class="diff-view">${rows.join("")}${more}</div>`;
}

/** The `+N/-M` badge for a finished edit, from its metadata or its own diff. */
function diffBadge(outcome: AnyRecord, display: string): string {
  const metadata = (outcome.metadata || {}) as AnyRecord;
  const counted = "added" in metadata || "removed" in metadata;
  if (!counted && !(display && isUnifiedDiff(display))) return "";
  const stats = counted
    ? { added: Number(metadata.added || 0), removed: Number(metadata.removed || 0) }
    : diffStats(display);
  return `<span class="diff-stat"><b class="plus">+${stats.added}</b><b class="minus">-${stats.removed}</b></span>`;
}

function send(type: string, payload: AnyRecord = {}): void {
  window.cagent.send({ type, ...payload });
}

function setBusy(busy: boolean): void {
  state.status.busy = busy;
  document.body.classList.toggle("is-busy", busy);
  const input = document.querySelector<HTMLTextAreaElement>("#composer-input");
  const sendButton = document.querySelector<HTMLButtonElement>("#send-button");
  if (input) input.disabled = busy && !state.pendingApproval;
  if (sendButton) sendButton.disabled = busy && !state.pendingApproval;
  const stop = document.querySelector<HTMLButtonElement>("#stop-button");
  if (stop) stop.hidden = !busy;
  const submit = document.querySelector<HTMLButtonElement>("#send-button");
  if (submit) submit.hidden = busy;
  updateStatusBar();
}

function updateStatusBar(): void {
  const activity = document.querySelector<HTMLElement>("#activity-label");
  const statusDot = document.querySelector<HTMLElement>("#connection-dot");
  const statusText = document.querySelector<HTMLElement>("#connection-text");
  if (activity) activity.textContent = state.pendingApproval ? "Approval required" : state.status.busy ? "Agent is working" : "Ready";
  if (statusDot) statusDot.className = `status-dot ${state.pendingApproval ? "attention" : state.status.busy ? "working" : "ready"}`;
  if (statusText) statusText.textContent = state.pendingApproval ? "Action needs approval" : state.status.busy ? "Working" : "Connected";
  const context = state.status.context || {};
  const meter = document.querySelector<HTMLElement>("#context-meter");
  const meterLabel = document.querySelector<HTMLElement>("#context-label");
  const ratio = Number(context.window) ? Math.min(Number(context.tokens || 0) / Number(context.window), 1) : 0;
  if (meter) meter.style.setProperty("--value", `${ratio * 100}%`);
  if (meterLabel) meterLabel.textContent = `${Math.round(ratio * 100)}% context`;
  const steps = document.querySelector<HTMLElement>("#steps-label");
  const stepCount = Number(state.status.steps || 0);
  if (steps) steps.textContent = `${stepCount} step${stepCount === 1 ? "" : "s"}`;
  const model = document.querySelector<HTMLElement>("#model-label");
  if (model) model.textContent = state.config.model || "Model not configured";
}

function shell(): void {
  appRoot.innerHTML = `
    <div class="window-shell ${state.sidebarCollapsed ? "sidebar-collapsed" : ""}">
      <aside class="sessions-sidebar" id="sessions-sidebar">
        <div class="brand-row">
          <div class="brand-mark">c</div>
          <div class="brand-copy"><strong>cagent</strong><span>coding workspace</span></div>
          <button class="icon-button sidebar-toggle" id="collapse-sidebar" title="Collapse sessions sidebar" aria-label="Collapse sessions sidebar">${icon("panel-left-close")}</button>
        </div>
        <button class="new-session-button" id="new-session">${icon("plus", 17)}<span>New session</span></button>
        <div class="sessions-heading"><span>Sessions</span><span class="session-count" id="session-count">0</span></div>
        <div class="sessions-list" id="sessions-list"><div class="sessions-empty">No saved sessions yet.<br><span>Completed conversations appear here.</span></div></div>
        <div class="sidebar-footer">
          <button class="workspace-button" id="choose-workspace">${icon("folder-open", 16)}<span class="workspace-text"><small>Workspace</small><b id="workspace-label">Loading...</b></span>${icon("chevron-right", 15)}</button>
          <button class="sidebar-settings" id="open-settings">${icon("settings", 16)}<span>Settings</span></button>
        </div>
      </aside>
      <main class="main-panel">
        <header class="topbar">
          <div class="topbar-left">
            <button class="icon-button mobile-menu" id="open-sessions" title="Show sessions" aria-label="Show sessions">${icon("panel-left", 17)}</button>
            <div class="crumb"><span class="crumb-root">cagent</span><span class="crumb-separator">/</span><span id="session-crumb">new session</span></div>
          </div>
          <div class="topbar-right">
            <div class="connection-state"><span class="status-dot ready" id="connection-dot"></span><span id="connection-text">Connecting</span></div>
            <button class="topbar-control" id="approval-control" title="Approval mode">${icon("shield-check", 15)}<span id="approval-label">auto-edit</span>${icon("chevron-down", 14)}</button>
            <button class="icon-button" id="open-settings-top" title="Session settings" aria-label="Session settings">${icon("sliders-horizontal", 17)}</button>
          </div>
        </header>
        <section class="transcript-wrap" id="transcript-wrap">
          <div class="transcript" id="transcript">
            <div class="welcome" id="welcome">
              <div class="welcome-kicker"><span class="status-dot ready"></span> Local agent workspace</div>
              <h1>Ready for a task</h1>
              <p class="welcome-copy">No messages in this session.</p>
              <div class="quick-grid">
                <button class="quick-action" data-prompt="Inspect this repository and summarize its architecture."><span class="quick-icon">${icon("scan-search", 19)}</span><span><b>Map this repository</b><small>Understand structure and conventions</small></span>${icon("arrow-up-right", 15)}</button>
                <button class="quick-action" data-prompt="Run the test suite, diagnose any failures, and fix the smallest safe change."><span class="quick-icon amber">${icon("test-tube", 19)}</span><span><b>Run and fix tests</b><small>Let the agent verify each change</small></span>${icon("arrow-up-right", 15)}</button>
                <button class="quick-action" data-prompt="Review the current git diff and point out correctness or security issues."><span class="quick-icon coral">${icon("git-pull-request", 19)}</span><span><b>Review a diff</b><small>Find risks before you commit</small></span>${icon("arrow-up-right", 15)}</button>
              </div>
            </div>
          </div>
          <div class="scroll-follow" id="scroll-follow">${icon("arrow-down", 14)} Jump to latest</div>
        </section>
        <section class="composer-area">
          <div class="approval-slot" id="approval-slot"></div>
          <div class="command-suggestions" id="command-suggestions"></div>
          <div class="composer-box">
            <textarea id="composer-input" rows="1" placeholder="Ask cagent to change your code..." aria-label="Message cagent"></textarea>
            <div class="composer-tools">
              <div class="composer-hints"><button class="tool-pill" id="slash-hint">${icon("terminal", 14)} Commands</button></div>
              <div class="composer-actions"><button class="icon-button" id="clear-composer" title="Clear input" aria-label="Clear input">${icon("x", 16)}</button><button class="send-button" id="send-button" title="Send message">${icon("arrow-up", 17)}</button><button class="stop-button" id="stop-button" hidden title="Interrupt agent">${icon("square", 15)}<span>Stop</span></button></div>
            </div>
          </div>
          <div class="activity-row"><span class="activity-icon">${icon(state.status.busy ? "loader-circle" : "sparkles", 14)}</span><span id="activity-label">Preparing workspace</span><span class="activity-divider"></span><span id="model-label">Model not configured</span><span class="activity-spacer"></span><span id="steps-label" title="Model requests in this conversation, including any restored from a resumed session">0 steps</span><span class="context-meter" id="context-meter"><i></i></span><span id="context-label">0% context</span></div>
        </section>
      </main>
      <div class="settings-panel" id="settings-panel" aria-hidden="true"><div class="settings-backdrop" id="settings-backdrop"></div><div class="settings-drawer"><div class="drawer-header"><div><span class="eyebrow">Session controls</span><h2>Configuration</h2></div><button class="icon-button" id="close-settings" title="Close settings" aria-label="Close settings">${icon("x", 18)}</button></div><div class="settings-body" id="settings-body"></div></div></div>
      <div class="toast-region" id="toast-region"></div>
    </div>`;
  bindUI();
  renderIcons();
}

function bindUI(): void {
  document.querySelector("#collapse-sidebar")?.addEventListener("click", () => setSidebarCollapsed(!state.sidebarCollapsed));
  document.querySelector("#open-sessions")?.addEventListener("click", () => {
    // Narrow layouts slide the sidebar in over the transcript; wide layouts
    // only ever need this button to undo a collapse.
    if (window.matchMedia("(max-width: 880px)").matches) document.querySelector(".window-shell")?.classList.toggle("sessions-open");
    else setSidebarCollapsed(false);
  });
  document.querySelector("#new-session")?.addEventListener("click", newSession);
  document.querySelector("#choose-workspace")?.addEventListener("click", chooseWorkspace);
  document.querySelector("#open-settings")?.addEventListener("click", () => openSettings());
  document.querySelector("#open-settings-top")?.addEventListener("click", () => openSettings());
  document.querySelector("#close-settings")?.addEventListener("click", closeSettings);
  document.querySelector("#settings-backdrop")?.addEventListener("click", closeSettings);
  document.querySelector("#send-button")?.addEventListener("click", submitComposer);
  document.querySelector("#stop-button")?.addEventListener("click", () => send("interrupt"));
  document.querySelector("#clear-composer")?.addEventListener("click", () => {
    const input = document.querySelector<HTMLTextAreaElement>("#composer-input");
    if (input) { input.value = ""; input.focus(); }
  });
  document.querySelector("#slash-hint")?.addEventListener("click", () => {
    const input = document.querySelector<HTMLTextAreaElement>("#composer-input");
    if (input) { input.value = "/"; input.focus(); updateCommandMenu(); }
  });
  document.querySelector("#composer-input")?.addEventListener("input", () => {
    resizeComposer();
    updateCommandMenu();
  });
  document.querySelector<HTMLTextAreaElement>("#composer-input")?.addEventListener("keydown", (event: KeyboardEvent) => {
    const input = event.currentTarget as HTMLTextAreaElement;
    if (event.key === "ArrowDown" && state.commandMenuOpen) { event.preventDefault(); return; }
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submitComposer(); }
    if (event.key === "Escape") { state.commandMenuOpen = false; updateCommandMenu(); }
    if (event.key === "Tab" && state.commandMenuOpen) { event.preventDefault(); const command = filteredCommands()[0]?.[0]; if (command) { input.value = command + " "; updateCommandMenu(); } }
  });
  document.querySelectorAll<HTMLButtonElement>(".quick-action").forEach((button) => button.addEventListener("click", () => {
    const input = document.querySelector<HTMLTextAreaElement>("#composer-input");
    if (input) { input.value = button.dataset.prompt || ""; input.focus(); resizeComposer(); }
  }));
  document.querySelector("#approval-control")?.addEventListener("click", () => {
    const current = state.config.approval_mode || "auto-edit";
    const next = current === "suggest" ? "auto-edit" : current === "auto-edit" ? "full-auto" : "suggest";
    send("command", { command: `/approve ${next}` });
    toast(`Approval mode: ${next}`, "success");
  });
  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.key.toLowerCase() === "n") { event.preventDefault(); newSession(); }
    if (event.ctrlKey && event.key.toLowerCase() === "r") { event.preventDefault(); openSessionPicker(); }
    // Closing the window runs the graceful shutdown in the main process, which
    // may still have to ask about sandbox changes. Sending `shutdown` here too
    // would start that twice and race the answer.
    if (event.ctrlKey && event.key.toLowerCase() === "q") { event.preventDefault(); window.cagent.close(); }
    if (event.key === "Escape" && state.settingsOpen) closeSettings();
  });
  document.querySelector("#transcript-wrap")?.addEventListener("scroll", updateScrollFollow);
  document.querySelector("#scroll-follow")?.addEventListener("click", scrollToLatest);
  // Scrolling is not the only thing that can strand the button: clearing the
  // transcript for a new session removes the content below without moving the
  // scroll position, so no scroll event fires and the prompt to jump to a
  // latest that no longer exists stays on screen. Watch the content box too.
  const transcript = document.querySelector<HTMLElement>("#transcript");
  if (transcript) new ResizeObserver(updateScrollFollow).observe(transcript);
  // Delegated, because tool and command cards are appended as events arrive and
  // replayed cards are rebuilt wholesale on every resume.
  transcript?.addEventListener("click", (event) => {
    const head = (event.target as HTMLElement | null)?.closest<HTMLElement>(CARD_TOGGLE);
    if (!head) return;
    // A click that ends a drag is the user copying the output, not asking to
    // fold it away.
    if (window.getSelection()?.isCollapsed === false) return;
    toggleCollapsed(head);
  });
  transcript?.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const head = (event.target as HTMLElement | null)?.closest<HTMLElement>(CARD_TOGGLE);
    if (!head) return;
    event.preventDefault();
    toggleCollapsed(head);
  });
}

/** Marks a card head that folds away everything below it when clicked. Carried
    as its own class so a head with nothing under it — a replayed call that
    produced no output — is left inert rather than toggling an empty card. */
const CARD_TOGGLE = ".card-toggle";

function toggleCollapsed(head: HTMLElement): void {
  const card = head.parentElement;
  if (!card) return;
  const collapsed = card.classList.toggle("collapsed");
  head.setAttribute("aria-expanded", collapsed ? "false" : "true");
  // Folding a card shortens the transcript without scrolling it, so the
  // jump-to-latest prompt has to be re-evaluated by hand.
  updateScrollFollow();
}

function updateScrollFollow(): void {
  const wrap = document.querySelector<HTMLElement>("#transcript-wrap");
  const follow = document.querySelector<HTMLElement>("#scroll-follow");
  if (wrap && follow) follow.classList.toggle("visible", wrap.scrollHeight - wrap.scrollTop - wrap.clientHeight > 180);
}

function resizeComposer(): void {
  const input = document.querySelector<HTMLTextAreaElement>("#composer-input");
  if (!input) return;
  input.style.height = "auto";
  input.style.height = `${Math.min(Math.max(input.scrollHeight, 32), 190)}px`;
}

function filteredCommands(): readonly (readonly [string, string, string])[] {
  const query = state.commandQuery.toLowerCase();
  return commands.filter(([command, description]) => !query || command.includes(query) || description.toLowerCase().includes(query));
}

function updateCommandMenu(): void {
  const input = document.querySelector<HTMLTextAreaElement>("#composer-input");
  const menu = document.querySelector<HTMLElement>("#command-suggestions");
  if (!input || !menu) return;
  const value = input.value;
  state.commandMenuOpen = value.startsWith("/") && !value.includes("\n") && !value.includes(" ");
  state.commandQuery = value.slice(1);
  if (!state.commandMenuOpen) { menu.innerHTML = ""; menu.classList.remove("open"); return; }
  menu.classList.add("open");
  menu.innerHTML = filteredCommands().slice(0, 7).map(([command, description, lucide]) => `<button class="suggestion" data-command="${command}">${icon(lucide, 15)}<b>${command}</b><span>${description}</span></button>`).join("");
  menu.querySelectorAll<HTMLButtonElement>(".suggestion").forEach((button) => button.addEventListener("click", () => {
    input.value = `${button.dataset.command} `; input.focus(); state.commandMenuOpen = false; updateCommandMenu(); resizeComposer();
  }));
  renderIcons();
}

function submitComposer(): void {
  const input = document.querySelector<HTMLTextAreaElement>("#composer-input");
  if (!input || !input.value.trim()) return;
  const text = input.value.trim();
  input.value = ""; resizeComposer(); updateCommandMenu();
  send("turn", { text });
}

function newSession(): void {
  if (state.status.busy) { toast("Interrupt the current turn before starting a new session", "warning"); return; }
  clearTranscript();
  send("new_session");
  document.querySelector("#session-crumb")!.textContent = "new session";
  toast("Starting a fresh session", "neutral");
}

async function chooseWorkspace(): Promise<void> {
  if (state.status.busy) { toast("Finish the current operation first", "warning"); return; }
  const selected = await window.cagent.chooseWorkspace();
  if (selected) { state.workspace = selected; toast(`Workspace changed to ${selected}`, "success"); }
}

function setSidebarCollapsed(collapsed: boolean): void {
  state.sidebarCollapsed = collapsed;
  localStorage.setItem("cagent.sidebar", collapsed ? "collapsed" : "expanded");
  document.querySelector(".window-shell")?.classList.toggle("sidebar-collapsed", collapsed);
}

function openSessionPicker(): void {
  if (state.sidebarCollapsed) setSidebarCollapsed(false);
  document.querySelector(".window-shell")?.classList.add("sessions-open");
  document.querySelector("#sessions-list")?.scrollIntoView({ behavior: "smooth" });
}

function clearTranscript(): void {
  const transcript = document.querySelector<HTMLElement>("#transcript");
  if (transcript) transcript.innerHTML = `<div class="welcome" id="welcome"><div class="welcome-kicker"><span class="status-dot ready"></span> New local session</div><h1>Ready for a task</h1><p class="welcome-copy">No messages in this session.</p></div>`;
  state.assistantNode = null; state.assistantText = ""; state.toolNodes.clear(); state.pendingApproval = null; renderApproval(); renderIcons();
  // Synchronous, so an emptied transcript never paints with a stale button:
  // the ResizeObserver above only catches up on the next frame.
  updateScrollFollow();
}

function appendBlock(className: string, html: string): HTMLElement {
  const transcript = document.querySelector<HTMLElement>("#transcript")!;
  document.querySelector("#welcome")?.remove();
  const node = document.createElement("article");
  node.className = `message-block ${className}`;
  node.innerHTML = html;
  transcript.appendChild(node);
  scrollToLatest();
  return node;
}

function scrollToLatest(): void {
  const wrap = document.querySelector<HTMLElement>("#transcript-wrap");
  if (wrap) requestAnimationFrame(() => { wrap.scrollTop = wrap.scrollHeight; });
}

function renderUser(text: string): void {
  appendBlock("user-block", `<div class="message-meta"><span class="avatar user-avatar">YOU</span><span class="message-label">You</span><time>now</time></div><div class="user-content">${escapeHtml(text).replace(/\n/g, "<br>")}</div>`);
}

function ensureAssistant(): HTMLElement {
  if (state.assistantNode) return state.assistantNode;
  state.assistantText = "";
  state.assistantNode = appendBlock("assistant-block", `<div class="message-meta"><span class="avatar agent-avatar">c</span><span class="message-label">cagent</span><span class="live-pill"><i></i> LIVE</span><time>now</time></div><div class="assistant-content prose"></div>`);
  return state.assistantNode;
}

function appendThinking(text: string): void {
  if (!state.showThinking) return;
  const node = ensureAssistant();
  let thinking = node.querySelector<HTMLDetailsElement>(".thinking-content");
  if (!thinking) {
    thinking = document.createElement("details"); thinking.className = "thinking-content"; thinking.open = false;
    thinking.innerHTML = `<summary>${icon("brain", 14)} Reasoning trace <span>streaming</span></summary><div class="thinking-text prose"></div>`;
    node.insertBefore(thinking, node.querySelector(".assistant-content")); renderIcons();
  }
  // The trace is model prose like any other reply, so it gets the same Markdown
  // pass. Keep the raw text on the node: each delta re-renders the whole thing,
  // and half a heading parsed as literal text would never repair itself.
  const body = thinking.querySelector<HTMLElement>(".thinking-text")!;
  body.dataset.raw = (body.dataset.raw || "") + text;
  body.innerHTML = markdown(body.dataset.raw);
}

function appendAssistant(text: string): void {
  if (!text) return;
  const node = ensureAssistant(); state.assistantText += text;
  node.querySelector<HTMLElement>(".assistant-content")!.innerHTML = markdown(state.assistantText);
  scrollToLatest();
}

const MAX_REPLAY_LINES = 12;
const MAX_REPLAY_LINE_CHARS = 240;
const MAX_REPLAY_ARG_CHARS = 68;

/** Condense a replayed call's arguments to one line, as the CLI does.
 *
 * The interesting argument is almost always a path or a command, so those are
 * shown bare and everything else as `key=value`.
 */
function replayArguments(call: AnyRecord): string {
  const args = call.arguments && typeof call.arguments === "object" ? call.arguments as AnyRecord : {};
  const entries = Object.entries(args);
  if (!entries.length) return "";
  return entries.map(([key, value]) => {
    let text = typeof value === "string" ? value.replace(/\n/g, "⏎") : String(value);
    if (text.length > MAX_REPLAY_ARG_CHARS) text = `${text.slice(0, MAX_REPLAY_ARG_CHARS)}…`;
    return ["path", "command", "pattern"].includes(key) ? text : `${key}=${text}`;
  }).join(", ");
}

/** A bounded excerpt of a replayed tool result, matching the CLI's limits. */
function replayDetail(text: string): string {
  const lines = text.split("\n");
  while (lines.length && !lines[0]!.trim()) lines.shift();
  while (lines.length && !lines.at(-1)!.trim()) lines.pop();
  if (!lines.length) return "";
  const shown = lines.slice(0, MAX_REPLAY_LINES).map((line) =>
    line.length > MAX_REPLAY_LINE_CHARS ? `${line.slice(0, MAX_REPLAY_LINE_CHARS - 3)}...` : line);
  const hidden = lines.length - shown.length;
  const more = hidden > 0 ? `\n… ${hidden} more line${hidden === 1 ? "" : "s"}` : "";
  return `<pre>${escapeHtml(shown.join("\n") + more)}</pre>`;
}

/** The tool activity of a replayed turn: each call and the result it produced.
 *
 * The trace keeps a tool's `content` but not its `display`, so a replayed edit
 * shows the post-edit snippet the model saw rather than the coloured diff a
 * live run prints — the diff is derived at execution time and is simply gone.
 */
function renderReplayedCalls(message: AnyRecord, results: Map<string, AnyRecord>): string {
  if (!Array.isArray(message.parts)) return "";
  const calls = message.parts.filter((part: AnyRecord) => part.type === "tool_call");
  if (!calls.length) return "";
  const rows = calls.map((call: AnyRecord) => {
    const name = String(call.name || "tool");
    const args = replayArguments(call);
    const result = results.get(String(call.id ?? ""));
    const content = String(result?.content || "");
    const failed = Boolean(result?.is_error);
    const body = content.trim() ? replayDetail(content) : "";
    return `<div class="replayed-call${failed ? " failed" : ""}"><div class="replayed-call-head${body ? " card-toggle" : ""}"${body ? ` role="button" tabindex="0" aria-expanded="true" title="Show or hide this result"` : ""}>${icon(name === "run_bash" ? "terminal" : "file-code-2", 14)}<b>${escapeHtml(name)}</b>${args ? `<span class="replayed-args">${escapeHtml(args)}</span>` : ""}<span class="replayed-status">${icon(result ? (failed ? "circle-x" : "circle-check") : "circle-help", 14)}</span>${body ? `<span class="collapse-caret">${icon("chevron-down", 14)}</span>` : ""}</div>${body}</div>`;
  });
  return `<div class="replayed-calls">${rows.join("")}</div>`;
}

/** Tool results from a replayed history, keyed by the call they answer. */
function replayResults(messages: AnyRecord[]): Map<string, AnyRecord> {
  const results = new Map<string, AnyRecord>();
  for (const message of messages) {
    if (!Array.isArray(message.parts)) continue;
    for (const part of message.parts as AnyRecord[]) {
      if (part.type === "tool_result" && part.call_id) results.set(String(part.call_id), part);
    }
  }
  return results;
}

function renderParts(message: AnyRecord): string {
  if (!Array.isArray(message.parts)) return "";
  return message.parts.map((part: AnyRecord) => part.type === "text" ? markdown(String(part.text || "")) : part.type === "thinking" ? `<details class="replayed-thinking"><summary>${icon("brain", 13)} Reasoning trace</summary><div class="prose">${markdown(String(part.text || ""))}</div></details>` : "").join("");
}

function renderToolStarted(event: AnyRecord): void {
  state.assistantNode = null;
  state.assistantText = "";
  const id = String(event.call?.id || event.id || crypto.randomUUID());
  const name = String(event.call?.name || event.name || "tool");
  const args = event.call?.arguments || event.arguments || {};
  const node = appendBlock("tool-block", `<div class="tool-card running"><div class="tool-card-head card-toggle" role="button" tabindex="0" aria-expanded="true" title="Show or hide this result"><span class="tool-icon">${icon(name === "run_bash" ? "terminal" : "file-code-2", 16)}</span><div class="tool-title"><b>${escapeHtml(name)}</b><span>Executing capability</span></div><span class="risk-tag ${String(event.risk || "SAFE").toLowerCase()}">${String(event.risk || "safe").toLowerCase()}</span><span class="tool-spinner">${icon("loader-circle", 15)}</span><span class="collapse-caret">${icon("chevron-down", 15)}</span></div><div class="tool-args"><code>${escapeHtml(JSON.stringify(args, null, 2))}</code></div><div class="tool-output" id="tool-output-${CSS.escape(id)}"></div></div>`);
  state.toolNodes.set(id, node);
  renderIcons();
}

function renderToolFinished(event: AnyRecord): void {
  const id = String(event.call?.id || event.id || "");
  const node = state.toolNodes.get(id);
  if (!node) { renderToolStarted(event); return renderToolFinished(event); }
  const outcome = (event.outcome || event) as AnyRecord;
  const card = node.querySelector<HTMLElement>(".tool-card");
  const output = node.querySelector<HTMLElement>(".tool-output");
  card?.classList.remove("running"); card?.classList.add(outcome.is_error ? "error" : "done");
  const spinner = node.querySelector(".tool-spinner"); if (spinner) spinner.innerHTML = icon(outcome.is_error ? "circle-x" : "circle-check", 15);
  // A tool's `display` is the rich account of what it did — for an edit, the
  // diff. As in the CLI it replaces the raw content, which only narrates the
  // same change back; the content's first line is kept as the summary.
  const content = String(outcome.content || "");
  const display = typeof outcome.display === "string" ? outcome.display : "";
  const body = display
    ? `<span class="tool-summary">${escapeHtml(content.split("\n")[0] || "done")}</span>${isUnifiedDiff(display) ? renderDiff(display) : `<pre>${escapeHtml(display)}</pre>`}`
    : `<pre>${escapeHtml(content || "No output")}</pre>`;
  if (output) output.innerHTML = `${body}<span class="tool-duration">${Number(event.duration_s || 0).toFixed(2)}s${outcome.truncated ? " · truncated" : ""}</span>`;
  const head = node.querySelector(".tool-card-head");
  const subtitle = head?.querySelector(".tool-title span");
  // The size belongs in the head, not just in the body: it is what tells the
  // user whether a folded card is worth unfolding.
  const shown = (display || content).replace(/\n+$/, "");
  const lines = shown ? shown.split("\n").length : 0;
  if (subtitle) subtitle.textContent = `${outcome.is_error ? "Failed" : "Completed"}${lines ? ` · ${lines} line${lines === 1 ? "" : "s"}` : ""}`;
  head?.querySelector(".diff-stat")?.remove();
  const badge = diffBadge(outcome, display);
  if (badge) head?.querySelector(".risk-tag")?.insertAdjacentHTML("beforebegin", badge);
  renderIcons(); scrollToLatest();
}

function renderApproval(): void {
  const slot = document.querySelector<HTMLElement>("#approval-slot");
  if (!slot) return;
  if (!state.pendingApproval) { slot.innerHTML = ""; slot.classList.remove("open"); return; }
  const request = state.pendingApproval.request || state.pendingApproval;
  const risk = String(request.risk || "MUTATING").toLowerCase();
  const canAlways = risk !== "dangerous" && !request.always_prompt;
  // An edit's detail is the diff it wants to write. Colour it and open it by
  // default: a prompt the user cannot evaluate trains them to approve blindly.
  const detail = String(request.detail || "");
  const isDiff = isUnifiedDiff(detail);
  const stats = isDiff ? diffStats(detail) : null;
  // The sandbox sync asks the one question where "Deny" destroys work rather
  // than merely declining it, so it gets labels that say what each button does.
  const isSync = String(request.tool || "") === "sandbox_sync";
  const heading = isSync ? "Keep these sandbox changes?" : "Approval required";
  const subject = isSync
    ? "The disposable copy is about to be closed"
    : `${escapeHtml(String(request.tool || "tool"))} wants to run`;
  const denyLabel = isSync ? `${icon("trash-2", 14)} Discard changes` : "Deny";
  const allowLabel = isSync ? `${icon("check", 14)} Copy to project` : `${icon("check", 14)} Allow once`;
  slot.classList.add("open");
  slot.innerHTML = `<div class="approval-card ${risk}${isSync ? " sync" : ""}"><div class="approval-heading"><span class="approval-symbol">${icon(risk === "dangerous" ? "triangle-alert" : isSync ? "container" : "shield-alert", 17)}</span><div><b>${heading}</b><span>${subject}</span></div>${stats ? `<span class="diff-stat"><b class="plus">+${stats.added}</b><b class="minus">-${stats.removed}</b></span>` : ""}<span class="risk-tag ${risk}">${risk}</span></div><p>${escapeHtml(String(request.summary || "This action may modify your workspace."))}</p>${detail ? `<details${isDiff ? " open" : ""}><summary>View details</summary>${isDiff ? renderDiff(detail) : `<pre>${escapeHtml(detail)}</pre>`}</details>` : ""}<div class="approval-actions"><button class="approval-deny" data-approval="deny">${denyLabel}</button>${isSync ? "" : `<button class="approval-always" data-approval="always" ${canAlways ? "" : "disabled"}>${icon("check-check", 14)} Always allow</button>`}<button class="approval-allow" data-approval="approve">${allowLabel}</button></div></div>`;
  slot.querySelectorAll<HTMLButtonElement>("[data-approval]").forEach((button) => button.addEventListener("click", () => {
    const decision = button.dataset.approval!; send("approval", { decision }); state.pendingApproval = null; renderApproval();
  }));
  renderIcons(); scrollToLatest();
}

type MenuItem = "separator" | { action: string; label: string; lucide: string; danger?: boolean; disabled?: boolean; hint?: string };

// One menu at a time, anchored to the control that opened it. Kept outside the
// list it belongs to: `.sessions-list` scrolls with `overflow: auto`, which
// clips an absolutely positioned child, so the menu is fixed-positioned against
// the viewport and torn down whenever the anchor could have moved.
let openMenuAnchor: HTMLElement | null = null;
let closeOpenMenu: (() => void) | null = null;

function closeMenu(): void {
  closeOpenMenu?.();
}

function placeMenu(menu: HTMLElement, anchor: HTMLElement): void {
  const rect = anchor.getBoundingClientRect();
  const width = menu.offsetWidth;
  const height = menu.offsetHeight;
  const left = Math.max(8, Math.min(rect.right - width, window.innerWidth - width - 8));
  const below = rect.bottom + 6;
  // Flip above the anchor rather than run off the bottom of the window.
  const top = below + height > window.innerHeight - 8 ? Math.max(8, rect.top - height - 6) : below;
  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;
}

function openMenu(anchor: HTMLElement, items: MenuItem[], onPick: (action: string) => void, note = ""): void {
  closeMenu();
  const shell = document.querySelector<HTMLElement>(".window-shell");
  if (!shell) return;
  const menu = document.createElement("div");
  menu.className = "popup-menu";
  menu.setAttribute("role", "menu");
  menu.innerHTML = (note ? `<p class="popup-note">${escapeHtml(note)}</p>` : "") + items.map((item) => item === "separator"
    ? `<div class="popup-separator"></div>`
    : `<button class="popup-item${item.danger ? " danger" : ""}" role="menuitem" data-action="${item.action}"${item.disabled ? " disabled" : ""}${item.hint ? ` title="${escapeHtml(item.hint)}"` : ""}>${icon(item.lucide, 15)}<span>${escapeHtml(item.label)}</span></button>`).join("");
  shell.appendChild(menu);
  renderIcons();
  placeMenu(menu, anchor);
  anchor.setAttribute("aria-expanded", "true");
  menu.querySelector<HTMLButtonElement>(".popup-item:not(:disabled)")?.focus();
  // A press on the anchor itself is left alone so the following click can close
  // the menu, rather than closing it here and reopening it a moment later.
  const onPointerDown = (event: Event) => {
    const target = event.target as Node;
    if (menu.contains(target) || anchor.contains(target)) return;
    closeMenu();
  };
  // Stop the Escape here: the shell also closes the settings drawer on Escape,
  // and dismissing a menu should not take an unrelated panel with it.
  const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape") { event.stopPropagation(); closeMenu(); anchor.focus(); } };
  document.addEventListener("mousedown", onPointerDown, true);
  document.addEventListener("keydown", onKeyDown, true);
  document.addEventListener("scroll", closeMenu, true);
  window.addEventListener("resize", closeMenu);
  openMenuAnchor = anchor;
  closeOpenMenu = () => {
    document.removeEventListener("mousedown", onPointerDown, true);
    document.removeEventListener("keydown", onKeyDown, true);
    document.removeEventListener("scroll", closeMenu, true);
    window.removeEventListener("resize", closeMenu);
    anchor.setAttribute("aria-expanded", "false");
    menu.remove();
    openMenuAnchor = null;
    closeOpenMenu = null;
  };
  menu.querySelectorAll<HTMLButtonElement>(".popup-item").forEach((button) => button.addEventListener("click", (event) => {
    event.stopPropagation();
    if (button.disabled) return;
    const action = button.dataset.action!;
    closeMenu();
    onPick(action);
  }));
}

async function copyToClipboard(value: string, label: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(value);
    toast(`${label} copied`, "success");
  } catch {
    toast("The clipboard is not available", "warning");
  }
}

function resumeSession(path: string): void {
  if (!path) return;
  if (state.status.busy) { toast("Interrupt the current turn before resuming", "warning"); return; }
  clearTranscript();
  send("resume", { path });
  document.querySelector(".window-shell")?.classList.remove("sessions-open");
}

function truncate(value: string, limit: number): string {
  return value.length > limit ? `${value.slice(0, limit - 1)}…` : value;
}

/** Second step of a delete: the trace file is not recoverable from the GUI. */
function confirmDeleteSession(anchor: HTMLElement, path: string, label: string): void {
  openMenu(
    anchor,
    [
      { action: "cancel", label: "Keep it", lucide: "x" },
      { action: "delete", label: "Delete permanently", lucide: "trash-2", danger: true },
    ],
    (action) => { if (action === "delete") send("delete_session", { path }); },
    `Delete “${truncate(label, 64)}”? Its trace file is removed from disk and cannot be restored here.`,
  );
}

function openSessionMenu(anchor: HTMLButtonElement): void {
  if (openMenuAnchor === anchor) { closeMenu(); return; }
  const path = anchor.dataset.path || "";
  const session = state.sessions.find((item) => String(item.path) === path);
  const label = String(session?.prompt || "this session");
  // The live session's trace is still being appended to, so the bridge refuses
  // to unlink it. Say so in the menu instead of failing after the click.
  const isLive = Boolean(session) && String(session!.id) === state.activeSession;
  openMenu(anchor, [
    { action: "open", label: "Open session", lucide: "history" },
    { action: "copy", label: "Copy trace path", lucide: "copy" },
    "separator",
    {
      action: "delete",
      label: "Delete session",
      lucide: "trash-2",
      danger: true,
      disabled: isLive,
      hint: isLive ? "This is the session you are in — start a new session first." : undefined,
    },
  ], (action) => {
    if (action === "open") resumeSession(path);
    else if (action === "copy") void copyToClipboard(path, "Trace path");
    else if (action === "delete") confirmDeleteSession(anchor, path, label);
  });
}

function renderSessions(): void {
  const list = document.querySelector<HTMLElement>("#sessions-list");
  const count = document.querySelector<HTMLElement>("#session-count");
  if (!list || !count) return;
  // The rows about to be replaced include whichever one a menu is anchored to.
  closeMenu();
  count.textContent = String(state.sessions.length);
  if (!state.sessions.length) { list.innerHTML = `<div class="sessions-empty">No saved sessions yet.<br><span>Completed conversations appear here.</span></div>`; return; }
  const selectedSession = state.restoredFrom || state.activeSession;
  list.innerHTML = state.sessions.map((session) => {
    const path = escapeHtml(String(session.path));
    return `<div class="session-row ${session.id === selectedSession ? "active" : ""}"><button class="session-item" data-path="${path}"><span class="session-status ${session.status === "finished" ? "done" : "paused"}"></span><span class="session-info"><b>${escapeHtml(String(session.prompt || "Untitled session"))}</b><span><time>${relativeDate(session.modified)}</time><i></i>${Number(session.steps || 0)} step${Number(session.steps || 0) === 1 ? "" : "s"}</span></span></button><button class="session-more" data-path="${path}" title="Session actions" aria-label="Session actions" aria-haspopup="menu" aria-expanded="false">${icon("ellipsis", 15)}</button></div>`;
  }).join("");
  list.querySelectorAll<HTMLButtonElement>(".session-item").forEach((button) => button.addEventListener("click", () => resumeSession(button.dataset.path || "")));
  list.querySelectorAll<HTMLButtonElement>(".session-more").forEach((button) => button.addEventListener("click", (event) => {
    event.stopPropagation();
    openSessionMenu(button);
  }));
  renderIcons();
}

function renderHistory(messages: AnyRecord[], source = "resumed session"): void {
  clearTranscript();
  appendBlock("system-notice", `<div class="notice-icon">${icon("history", 15)}</div><div><b>Context restored</b><span>${escapeHtml(source)} · prior turns are available to the agent</span></div>`);
  const results = replayResults(messages || []);
  for (const message of messages || []) {
    if (message.role === "user") {
      const text = String(message.parts?.map((p: AnyRecord) => p.text || "").join("") || "");
      if (text.trim()) appendBlock("user-block", `<div class="message-meta"><span class="avatar user-avatar">YOU</span><span class="message-label">You</span></div><div class="user-content">${escapeHtml(text).replace(/\n/g, "<br>")}</div>`);
    } else if (message.role === "assistant") {
      // A step that only called tools carries no text. Rendering the bubble
      // anyway leaves a "cagent" header with nothing under it, so replay what
      // the step actually did instead of an empty turn.
      const body = renderParts(message);
      const calls = renderReplayedCalls(message, results);
      if (body || calls) appendBlock("assistant-block", `<div class="message-meta"><span class="avatar agent-avatar">c</span><span class="message-label">cagent</span></div>${body ? `<div class="assistant-content prose">${body}</div>` : ""}${calls}`);
    }
  }
  state.assistantNode = null; state.assistantText = ""; renderIcons();
}

function openSettings(): void {
  state.settingsOpen = true;
  const panel = document.querySelector<HTMLElement>("#settings-panel"); panel?.setAttribute("aria-hidden", "false"); panel?.classList.add("open");
  renderSettings();
}

function closeSettings(): void {
  state.settingsOpen = false;
  const panel = document.querySelector<HTMLElement>("#settings-panel"); panel?.setAttribute("aria-hidden", "true"); panel?.classList.remove("open");
}

function renderSettings(): void {
  const body = document.querySelector<HTMLElement>("#settings-body"); if (!body) return;
  const config = state.config;
  body.innerHTML = `<div class="settings-section"><span class="eyebrow">Endpoint</span><div class="setting-row"><label>Model</label><b>${escapeHtml(String(config.model || "not configured"))}</b></div><div class="setting-row"><label>Wire</label><b>${escapeHtml(String(config.wire || "openai"))}</b></div><label class="setting-field">Reasoning effort<select id="effort-select"><option value="default">provider default</option><option value="none">none</option><option value="minimal">minimal</option><option value="low">low</option><option value="medium">medium</option><option value="high">high</option><option value="xhigh">xhigh</option><option value="max">max</option></select></label><label class="toggle-field"><span><b>Reasoning trace</b><small>Show streamed thinking in transcript</small></span><input id="thinking-toggle" type="checkbox" ${state.showThinking ? "checked" : ""}><i></i></label></div><div class="settings-section"><span class="eyebrow">Safety</span><label class="setting-field">Approval mode<select id="approval-select"><option value="suggest">suggest</option><option value="auto-edit">auto-edit</option><option value="full-auto">full-auto</option></select></label><div class="setting-row"><label>Sandbox</label><b>${escapeHtml(String(config.sandbox || config.sandbox_mode || "host"))}</b></div><div class="setting-row"><label>Network</label><b>${config.sandbox_network ? "bridge enabled" : "disabled"}</b></div><label class="setting-field">Sync policy<select id="sandbox-sync"><option value="never">never</option><option value="ask">ask</option><option value="always">always</option></select></label><label class="setting-input"><span>Docker image</span><span><input id="sandbox-image" value="${escapeHtml(String(config.sandbox_image || "python:3.12-slim"))}"><button id="set-sandbox-image">Set</button></span></label><div class="sandbox-actions"><button data-sandbox="status">${icon("info", 14)} Status</button><button data-sandbox="on">${icon("container", 14)} Enable</button><button data-sandbox="apply">${icon("check-check", 14)} Apply</button><button data-sandbox="rollback">${icon("undo-2", 14)} Roll back</button><button class="danger-control" data-sandbox="off">${icon("square", 13)} Disable</button></div></div><div class="settings-section"><span class="eyebrow">Context</span><div class="setting-row"><label>Workspace</label><b class="path-value">${escapeHtml(String(state.workspace || config.workspace || "loading"))}</b></div><div class="setting-row"><label>Trace directory</label><b class="path-value">${escapeHtml(String(config.trace_dir || "off"))}</b></div><div class="setting-row"><label>Window</label><b>${Number(config.context_window || state.status.context?.window || 0).toLocaleString()} tokens</b></div></div>`;
  const select = body.querySelector<HTMLSelectElement>("#approval-select"); if (select) { select.value = config.approval_mode || "auto-edit"; select.addEventListener("change", () => { send("command", { command: `/approve ${select.value}` }); toast(`Approval mode: ${select.value}`, "success"); }); }
  const effort = body.querySelector<HTMLSelectElement>("#effort-select"); if (effort) { effort.value = config.reasoning_effort || "default"; effort.addEventListener("change", () => send("command", { command: `/effort ${effort.value}` })); }
  const thinking = body.querySelector<HTMLInputElement>("#thinking-toggle"); thinking?.addEventListener("change", () => { state.showThinking = thinking.checked; localStorage.setItem("cagent.thinking", thinking.checked ? "shown" : "hidden"); });
  const sync = body.querySelector<HTMLSelectElement>("#sandbox-sync"); if (sync) { sync.value = config.sandbox_sync || "ask"; sync.addEventListener("change", () => send("command", { command: `/sandbox sync ${sync.value}` })); }
  body.querySelector("#set-sandbox-image")?.addEventListener("click", () => { const image = body.querySelector<HTMLInputElement>("#sandbox-image")?.value.trim(); if (image) send("command", { command: `/sandbox image ${image}` }); });
  body.querySelectorAll<HTMLButtonElement>("[data-sandbox]").forEach((button) => button.addEventListener("click", () => send("command", { command: `/sandbox ${button.dataset.sandbox}` })));
  renderIcons();
}

function toast(message: string, tone: "success" | "warning" | "neutral" = "neutral"): void {
  const region = document.querySelector<HTMLElement>("#toast-region"); if (!region) return;
  const item = document.createElement("div"); item.className = `toast ${tone}`; item.innerHTML = `${icon(tone === "warning" ? "triangle-alert" : tone === "success" ? "circle-check" : "info", 15)}<span>${escapeHtml(message)}</span>`; region.appendChild(item); renderIcons(); setTimeout(() => item.remove(), 4200);
}

function handleEvent(event: AnyRecord): void {
  switch (event.type) {
    case "ready": state.activeSession = String(event.session_id || ""); state.workspace = String(event.workspace || ""); state.config = { ...state.config, ...(event.config || {}) }; updateWorkspace(); updateStatusBar(); break;
    case "backend_restarting": state.workspace = String(event.workspace || ""); updateWorkspace(); break;
    case "backend_stopped": setBusy(false); toast("The Python backend stopped", "warning"); appendBlock("warning-block", `<span class="warning-symbol">${icon("circle-x", 16)}</span><div><b>Backend stopped</b><span>Exit code ${escapeHtml(String(event.code ?? event.signal ?? "unknown"))} - nothing you send will reach cagent until it restarts.</span></div>`); renderIcons(); break;
    case "sessions": state.sessions = Array.isArray(event.sessions) ? event.sessions : []; state.activeSession = String(event.active_id || state.activeSession); state.restoredFrom = String(event.restored_from || ""); renderSessions(); break;
    case "session_deleted": toast("Session deleted", "success"); break;
    case "status": state.status = { ...state.status, ...event }; state.config = { ...state.config, ...(event.config || {}) }; updateStatusBar(); updateWorkspace(); if (state.settingsOpen) renderSettings(); break;
    case "busy_changed": setBusy(Boolean(event.busy)); break;
    case "run_started": state.config = { ...state.config, model: event.model, endpoint: event.endpoint, tools: event.tool_names, sandbox: event.sandbox_status, shell_access: event.shell_access }; updateStatusBar(); break;
    case "activity": updateStatusBar(); if (document.querySelector("#activity-label")) document.querySelector("#activity-label")!.textContent = String(event.message || "Working"); break;
    case "user_message": renderUser(String(event.text || "")); state.assistantNode = null; break;
    case "thinking_delta": appendThinking(String(event.text || "")); break;
    case "text_delta": appendAssistant(String(event.text || "")); break;
    case "step_started": { const node = ensureAssistant(); node.querySelector(".live-pill")?.classList.add("active"); break; }
    case "step_finished": { const text = renderParts(event.message || {}); if (text && !state.assistantText) { const node = ensureAssistant(); node.querySelector<HTMLElement>(".assistant-content")!.innerHTML = text; } break; }
    case "tool_started": renderToolStarted(event); break;
    case "tool_finished": renderToolFinished(event); break;
    case "approval_requested": state.pendingApproval = event; renderApproval(); break;
    case "approval_decided": state.pendingApproval = null; renderApproval(); break;
    case "warning": appendBlock("warning-block", `<span class="warning-symbol">${icon("triangle-alert", 16)}</span><div><b>${escapeHtml(String(event.message || "Warning"))}</b>${event.detail ? `<pre>${escapeHtml(String(event.detail))}</pre>` : ""}</div>`); toast(String(event.message || "Warning"), "warning"); break;
    case "compaction_done": appendBlock("system-notice", `<div class="notice-icon">${icon("shrink", 15)}</div><div><b>Context compacted</b><span>${escapeHtml(String(event.strategy || "optimized"))} · ${Number(event.tokens_before || 0).toLocaleString()} → ${Number(event.tokens_after || 0).toLocaleString()} tokens</span></div>`); break;
    case "turn_finished": state.assistantNode?.querySelector(".live-pill")?.remove(); state.assistantNode = null; scrollToLatest(); break;
    case "turn_complete": state.assistantNode = null; break;
    case "command_result": if (event.output) { appendBlock("command-block", `<div class="command-head card-toggle" role="button" tabindex="0" aria-expanded="true" title="Show or hide this output">${icon("terminal", 14)}<b>${escapeHtml(String(event.command || "command"))}</b><span class="collapse-caret">${icon("chevron-down", 14)}</span></div><pre>${escapeHtml(String(event.output))}</pre>`); renderIcons(); } if (event.error) toast(String(event.error), "warning"); break;
    case "history_restored": renderHistory((event.messages || []) as AnyRecord[], String(event.source || "saved session")); if (event.warning) toast(String(event.warning), "warning"); break;
    case "history_cleared": clearTranscript(); break;
    case "resume_error": toast(String(event.message || "Could not resume session"), "warning"); break;
    case "protocol_error": toast(String(event.message || "Invalid command"), "warning"); break;
    case "worker_error": toast(String(event.message || "Agent worker failed"), "warning"); appendBlock("warning-block", `<span class="warning-symbol">${icon("circle-x", 16)}</span><div><b>Worker error</b><pre>${escapeHtml(String(event.message || ""))}</pre></div>`); break;
    case "fatal_error": setBusy(false); toast(String(event.message || "Backend configuration is incomplete"), "warning"); appendBlock("setup-block", `<div class="setup-icon">${icon("settings-2", 21)}</div><div><b>Finish local setup to start cagent</b><p>${escapeHtml(String(event.message || "Set base_url, model, and api_key in .cagent.toml."))}</p><code>Copy .cagent.example.toml to .cagent.toml, then relaunch.</code></div>`); break;
    case "bridge_log": if (event.level === "error") toast(String(event.message || "Backend log"), "warning"); break;
  }
}

function updateWorkspace(): void {
  const label = document.querySelector<HTMLElement>("#workspace-label"); if (label) label.textContent = state.workspace ? state.workspace.split(/[\\/]/).at(-1) || state.workspace : "Loading...";
  const crumb = document.querySelector<HTMLElement>("#session-crumb"); if (crumb && state.restoredFrom) crumb.textContent = state.restoredFrom;
  const approval = document.querySelector<HTMLElement>("#approval-label"); if (approval) approval.textContent = state.config.approval_mode || "auto-edit";
}

shell();
window.cagent.onEvent(handleEvent);
send("status");
