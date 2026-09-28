const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
  startWorkflow: (config) => ipcRenderer.invoke("start-workflow", config),
  clearDownloads: () => ipcRenderer.invoke("clear-downloads"),
  browseFile: () => ipcRenderer.invoke("browse-file"),
  openLocalWindow: () => ipcRenderer.send("open-local-window"),
  onLog: (callback) => {
    ipcRenderer.on("log", (_, msg) => callback(msg));
  },
  onProgress: (callback) => {
    ipcRenderer.on("progress", (_, data) => callback(data));
  },
  onStatus: (callback) => {
    ipcRenderer.on("status", (_, status) => callback(status));
  },
  cancelWorkflow: () => ipcRenderer.send("cancel-workflow"),
  removeAllListeners: () => {
    ipcRenderer.removeAllListeners("log");
    ipcRenderer.removeAllListeners("progress");
    ipcRenderer.removeAllListeners("status");
  },
});
