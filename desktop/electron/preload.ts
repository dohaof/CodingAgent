import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("cagent", {
  send(command: Record<string, unknown>): void {
    ipcRenderer.send("bridge:send", command);
  },
  onEvent(listener: (event: Record<string, unknown>) => void): () => void {
    const handler = (_event: Electron.IpcRendererEvent, payload: Record<string, unknown>) => listener(payload);
    ipcRenderer.on("bridge:event", handler);
    return () => ipcRenderer.removeListener("bridge:event", handler);
  },
  chooseWorkspace(): Promise<string | null> {
    return ipcRenderer.invoke("workspace:choose") as Promise<string | null>;
  },
  minimize(): void {
    ipcRenderer.send("window:minimize");
  },
  maximize(): void {
    ipcRenderer.send("window:maximize");
  },
  close(): void {
    ipcRenderer.send("window:close");
  },
});

