@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  EUREKA DFPDP V9.2 - REPORTING INTELLIGENCE PLATFORM
echo  Traffic calibrated ^| Multi-day ^| Visual AI ^| Live telemetry ^| OSRM
echo ============================================================

where py >nul 2>nul
if errorlevel 1 (
  echo [LOI] Khong tim thay Python Launcher ^(py^).
  echo Hay cai Python 3.10+ tu python.org va tick "Add Python to PATH".
  pause
  exit /b 1
)

if not exist .venv\Scripts\python.exe (
  echo [1/3] Dang tao moi truong Python rieng...
  py -m venv .venv
  if errorlevel 1 goto :fail
)
call .venv\Scripts\activate.bat

if not exist .venv\demo_installed.ok (
  echo [2/3] Dang cai thu vien lan dau...
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  if errorlevel 1 goto :fail
  echo ok>.venv\demo_installed.ok
) else (
  echo [2/3] Thu vien da san sang.
)

echo.
echo [3/3] Khoi dong tai http://127.0.0.1:8000/control
echo.
echo - Dashboard: traffic 24h, luoi phan tich, multi-day carry-over
echo - AI monitor: Z/U/D, candidates, Actor probability, selected order
echo - Ban do/OSRM can Internet; core van chay neu mat mang.
echo ============================================================

rem Kiem tra port 8000 truoc khi khoi dong, tranh mo nham server V8 cu.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":8000 .*LISTENING"') do set OLD_PID=%%P
if defined OLD_PID (
  echo.
  echo [CANH BAO] Port 8000 dang duoc PID %OLD_PID% su dung.
  echo Day thuong la server prototype cu van dang chay.
  echo Hay dong cua so CMD cu hoac chay: taskkill /PID %OLD_PID% /F
  echo Sau do chay lai file nay.
  pause
  exit /b 2
)

start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8000/control"
python -m uvicorn app:app --host 0.0.0.0 --port 8000
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo [LOI] Cai dat hoac khoi dong that bai.
pause
exit /b 1
