import os, sys
sys.argv = ["main.py"]
here = os.getcwd()
print("resources:", here)
print("models 存在:", os.path.isdir(os.path.join(here, "models")))
# 模拟 main.py 的 CUDA DLL 注入是否在打包环境生效
sys.path.insert(0, "resources/.venv/Lib/site-packages")
import ctranslate2
print("ctranslate2:", ctranslate2.__version__)
print("CUDA 设备数:", ctranslate2.get_cuda_device_count())
print("支持的精度:", sorted(ctranslate2.get_supported_compute_types("cuda")))
