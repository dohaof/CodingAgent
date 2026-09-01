#!/usr/bin/env node

const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const packageRoot = path.resolve(__dirname, "..");
const electron = require("electron");
const main = path.join(packageRoot, "dist-electron", "main.js");

if (!fs.existsSync(main)) {
  console.error("cagent-web is not built. Run `npm run build` in the desktop directory.");
  process.exit(1);
}

const cliArgs = process.argv.slice(2);
const requestedWorkspace = cliArgs[0] && !cliArgs[0].startsWith("-") ? cliArgs[0] : null;
const workspace = requestedWorkspace ? path.resolve(requestedWorkspace) : process.cwd();

if (!fs.existsSync(workspace) || !fs.statSync(workspace).isDirectory()) {
  console.error(`Workspace is not a directory: ${workspace}`);
  process.exit(1);
}

const electronArgs = requestedWorkspace ? cliArgs.slice(1) : cliArgs;
const child = spawn(electron, [main, ...electronArgs], {
  cwd: workspace,
  env: { ...process.env, CAGENT_WORKSPACE: workspace },
  stdio: "inherit",
  windowsHide: false,
});

child.on("error", (error) => {
  console.error(`Could not launch Electron: ${error.message}`);
  process.exitCode = 1;
});
child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exitCode = code ?? 1;
});
