@echo off

set "python_exec=..\..\python\python.exe"
set "repair_dependency_txt=%~dp0\repair_dependency_list.txt"

echo Installing with ComfyUI Portable
echo .
echo Install whl...
%python_exec% -s -m pip install ./whl/docopt-0.6.2-py2.py3-none-any.whl

echo .
echo Install requirement.txt...
%python_exec% -s -m pip install -r ./requirements.txt

echo .
echo Fixing Dependency Package...
%python_exec% -s -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
for /f "delims=" %%i in (%repair_dependency_txt%) do (
    %python_exec% -s -m pip install "%%i"
    )
	
echo .
echo Install Finish!
pause

