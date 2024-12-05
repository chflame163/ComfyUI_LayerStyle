@echo off

set "requirements_txt=%~dp0\repair_dependency_list.txt"
set "python_exec=..\..\python\python.exe"

echo Fixing Dependency Package...

echo Installing with ComfyUI Portable
%python_exec% -s -m pip uninstall -y onnxruntime
%python_exec% -s -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless

for /f "delims=" %%i in (%requirements_txt%) do (
    %python_exec% -s -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "%%i"
    )

pause