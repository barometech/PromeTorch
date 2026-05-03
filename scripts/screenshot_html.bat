@echo off
REM Скриншоты HTML страниц через Chrome headless (вертикальная ориентация).
REM Вход: каталог с .html файлами. Выход: .png рядом с каждым.
REM
REM Usage: screenshot_html.bat <html_dir>
REM Default: %CD%\screenshots

setlocal enabledelayedexpansion

set HTML_DIR=%~1
if "%HTML_DIR%"=="" set HTML_DIR=%CD%\screenshots

REM Find Chrome
set CHROME=
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    set "CHROME=C:\Program Files\Google\Chrome\Application\chrome.exe"
) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    set "CHROME=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
) else if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    set "CHROME=C:\Program Files\Microsoft\Edge\Application\msedge.exe"
)

if "%CHROME%"=="" (
    echo ERR: Chrome / Edge не найден
    exit /b 1
)

if not exist "%HTML_DIR%" (
    echo ERR: directory not found: %HTML_DIR%
    exit /b 1
)

echo === Screenshot HTML pages (portrait 720x1280) ===
echo Browser: %CHROME%
echo Dir: %HTML_DIR%
echo.

for %%F in ("%HTML_DIR%\*.html") do (
    set "name=%%~nF"
    echo Capturing %%~nxF -^> !name!.png
    "%CHROME%" --headless --disable-gpu --no-sandbox ^
        --window-size=720,1280 ^
        --screenshot="%HTML_DIR%\!name!.png" ^
        "file:///%%F" 2>nul
)

echo.
echo === Generated screenshots ===
dir /b "%HTML_DIR%\*.png"
