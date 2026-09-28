const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
  startLocalWorkflow: (config) => ipcRenderer.invoke("start-local-workflow", config),
  browseVideo: () => ipcRenderer.invoke("browse-video"),
  cancelWorkflow: () => ipcRenderer.send("cancel-workflow"),
  onLog: (callback) => {
    ipcRenderer.on("log", (_, msg) => callback(msg));
  },
  onProgress: (callback) => {
    ipcRenderer.on("progress", (_, data) => callback(data));
  },
  onStatus: (callback) => {
    ipcRenderer.on("status", (_, status) => callback(status));
  },
});
