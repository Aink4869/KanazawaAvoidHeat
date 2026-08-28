@echo off
echo Starting Kanazawa Shade Map & Route Navigation Server (Port 8765)...
echo.
echo URL: http://127.0.0.1:8765/
echo Route (Legacy):  http://127.0.0.1:8765/route
echo Route (New):     http://127.0.0.1:8765/route/new
echo Route (Compare): http://127.0.0.1:8765/route/compare
echo.
py -3 -m uvicorn nav_app:app --reload --host 127.0.0.1 --port 8765
pause
