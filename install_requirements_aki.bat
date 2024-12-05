@echo off

set "python_exec=..\..\python\python.exe"
set "repair_dependency_txt=%~dp0\repair_dependency_list.txt"
set "requirements_txt=%~dp0\requirements.txt"

echo Installing with ComfyUI Portable
echo .
echo Install requirement.txt...
for /f "delims=" %%i in (%requirements_txt%) do (
    %python_exec% -s -m pip install "%%i"
    )

echo .
echo Fixing Dependency Package...
%python_exec% -s -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
for /f "delims=" %%i in (%repair_dependency_txt%) do (
    %python_exec% -s -m pip install "%%i"
    )
	
echo .
echo Install Finish!
pause

