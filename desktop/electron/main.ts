import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";

type BridgeCommand = Record<string, unknown> & { type: string };

// `app.getAppPath()` is the directory of the entry script when Electron is
// launched as `electron dist-electron/main.js`, which is one level too deep and
// breaks the renderer path, the Python source root, and the venv lookup. This
// file always sits in `<packageRoot>/dist-electron`, packaged or not.
const packageRoot = path.resolve(__dirname, "..");
const sourceRootCandidates = [
  process.env.CAGENT_SOURCE_PATH,
  path.join(packageRoot, "src"),
  path.join(packageRoot, "..", "src"),
].filter((candidate): candidate is string => Boolean(candidate));
const sourceRoot = sourceRootCandidates.find((candidate) => fs.existsSync(path.join(candidate, "cagent"))) || null;
const statePath = () => path.join(app.getPath("userData"), "desktop-state.json");
let mainWindow: BrowserWindow | null = null;
let bridge: ChildProcessWithoutNullStreams | null = null;
let workspace = process.cwd();

function readWorkspace(): string {
  // The launcher passes CAGENT_WORKSPACE for each invocation. It must win over
  // the last folder selected in the GUI so `cagent-web` works from any path.
  if (process.env.CAGENT_WORKSPACE) return process.env.CAGENT_WORKSPACE;
  try {
    const parsed = JSON.parse(fs.readFileSync(statePath(), "utf8")) as { workspace?: unknown };
    if (typeof parsed.workspace === "string" && fs.statSync(parsed.workspace).isDirectory()) {
      return parsed.workspace;
    }
  } catch {
    // First run or a stale folder: fall through to the repository root.
  }
  return process.cwd();
}

function saveWorkspace(value: string): void {
  fs.mkdirSync(path.dirname(statePath()), { recursive: true });
  fs.writeFileSync(statePath(), JSON.stringify({ workspace: value }, null, 2), "utf8");
}

function pythonExecutable(): string {
  if (process.env.CAGENT_PYTHON) return process.env.CAGENT_PYTHON;
  const roots = [packageRoot, path.resolve(packageRoot, "..")];
  const candidates = process.platform === "win32"
    ? [...roots.map((root) => path.join(root, "venv", "Scripts", "python.exe")), "python"]
    : [...roots.map((root) => path.join(root, "venv", "bin", "python")), "python3", "python"];
  return candidates.find((candidate) => candidate === path.basename(candidate) || fs.existsSync(candidate)) || candidates.at(-1)!;
}

function sendToRenderer(payload: unknown): void {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("bridge:event", payload);
  }
}

function stopBridge(): void {
  const current = bridge;
  bridge = null;
  if (!current || current.killed) return;
  // EOF ends the bridge's read loop; the kill is only for a wedged interpreter.
  current.stdin.end();
  const timer = setTimeout(() => current.kill(), 1800);
  current.once("exit", () => clearTimeout(timer));
}

// Quitting a sandboxed session is a decision, not a formality: the disposable
// copy is either merged into the project or thrown away. Ask the bridge to close
// gracefully and hold the window open until it answers, or a stray click on the
// title bar silently discards everything the agent wrote.
let shutdownRequested = false;
let closeConfirmed = false;
let shutdownTimer: NodeJS.Timeout | null = null;

function clearShutdownTimer(): void {
  if (shutdownTimer) clearTimeout(shutdownTimer);
  shutdownTimer = null;
}

function finishClose(): void {
  clearShutdownTimer();
  closeConfirmed = true;
  stopBridge();
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.close();
}

function beginGracefulClose(): void {
  shutdownRequested = true;
  bridge!.stdin.write(JSON.stringify({ type: "shutdown" }) + "\n");
  // Only guards against a bridge that never answers at all. Cancelled as soon
  // as it asks the user something, because then the wait is the user's own.
  shutdownTimer = setTimeout(finishClose, 15000);
}

function startBridge(): void {
  stopBridge();
  sendToRenderer({ type: "backend_restarting", workspace });
  const pythonPath = sourceRoot
    ? process.env.PYTHONPATH
      ? `${sourceRoot}${path.delimiter}${process.env.PYTHONPATH}`
      : sourceRoot
    : process.env.PYTHONPATH;
  const child = spawn(
    pythonExecutable(),
    ["-m", "cagent.gui.bridge", "--workspace", workspace],
    {
      cwd: packageRoot,
      // PYTHONIOENCODING covers anything written before the bridge pins its own
      // streams, such as an import-time traceback. Both sides must be UTF-8 or
      // non-ASCII output arrives here as mojibake.
      env: {
        ...process.env,
        ...(pythonPath ? { PYTHONPATH: pythonPath } : {}),
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
      },
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
    },
  );
  bridge = child;
  const lines = readline.createInterface({ input: child.stdout });
  lines.on("line", (line) => {
    try {
      const payload = JSON.parse(line) as Record<string, unknown>;
      sendToRenderer(payload);
      if (payload.type === "exit_requested") mainWindow?.close();
      if (payload.type === "shutdown_complete") finishClose();
      // The bridge is now waiting on a human, so the watchdog must not fire:
      // deciding what to do with sandbox changes takes as long as it takes.
      if (payload.type === "approval_requested" && shutdownRequested) clearShutdownTimer();
    } catch {
      sendToRenderer({ type: "bridge_log", level: "warning", message: line });
    }
  });
  // setEncoding decodes through a StringDecoder, so a multi-byte character
  // split across two chunks survives instead of turning into replacement marks.
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk: string) => {
    const message = chunk.trim();
    if (message) sendToRenderer({ type: "bridge_log", level: "error", message });
  });
  child.on("error", (error) => {
    sendToRenderer({ type: "fatal_error", message: `Could not start Python: ${error.message}` });
  });
  child.on("exit", (code, signal) => {
    if (bridge === child) {
      bridge = null;
      sendToRenderer({ type: "backend_stopped", code, signal });
    }
  });
}

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1420,
    height: 900,
    minWidth: 780,
    minHeight: 560,
    backgroundColor: "#111514",
    title: "cagent",
    titleBarStyle: "hidden",
    titleBarOverlay: {
      color: "#151a19",
      symbolColor: "#aab5b0",
      height: 42,
    },
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  // A blank window is indistinguishable from a hung one, so surface the two
  // load failures that produce it instead of leaving them in the dev console.
  mainWindow.webContents.on("preload-error", (_event, preloadPath, error) => {
    console.error(`[cagent] preload failed: ${preloadPath}: ${error.message}`);
  });
  mainWindow.webContents.on("did-fail-load", (_event, code, description, url) => {
    console.error(`[cagent] renderer failed to load: ${description} (${code}) ${url}`);
  });
  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (devUrl) void mainWindow.loadURL(devUrl);
  else void mainWindow.loadFile(path.join(packageRoot, "dist", "index.html"));
  mainWindow.webContents.on("did-finish-load", startBridge);
  mainWindow.on("close", (event) => {
    if (closeConfirmed) return;
    if (!bridge || bridge.killed) { closeConfirmed = true; return; }
    event.preventDefault();
    // A second attempt means the user would rather leave than keep waiting.
    if (shutdownRequested) finishClose();
    else beginGracefulClose();
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
    stopBridge();
  });
}

app.whenReady().then(() => {
  workspace = readWorkspace();
  ipcMain.on("bridge:send", (_event, command: BridgeCommand) => {
    if (!bridge || bridge.killed || !bridge.stdin.writable) {
      sendToRenderer({ type: "protocol_error", message: "The Python backend is not running." });
      return;
    }
    bridge.stdin.write(JSON.stringify(command) + "\n");
  });
  ipcMain.handle("workspace:choose", async () => {
    const result = await dialog.showOpenDialog(mainWindow!, {
      title: "Open workspace",
      defaultPath: workspace,
      properties: ["openDirectory"],
    });
    const selected = result.filePaths[0];
    if (!result.canceled && selected) {
      workspace = selected;
      saveWorkspace(workspace);
      startBridge();
      return workspace;
    }
    return null;
  });
  ipcMain.handle("window:is-maximized", () => mainWindow?.isMaximized() ?? false);
  ipcMain.on("window:minimize", () => mainWindow?.minimize());
  ipcMain.on("window:maximize", () => {
    if (mainWindow?.isMaximized()) mainWindow.unmaximize();
    else mainWindow?.maximize();
  });
  ipcMain.on("window:close", () => mainWindow?.close());
  createWindow();
});

app.on("window-all-closed", () => {
  stopBridge();
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
