@echo off
echo Starting Kanazawa Integrated Heatstroke Mitigation Navigator (Port 8766)...
echo.
echo URL (Legacy):  http://127.0.0.1:8766/
echo URL (New):     http://127.0.0.1:8766/new
echo URL (Compare): http://127.0.0.1:8766/compare
echo URL (English): http://127.0.0.1:8766/en
echo.
py -3 -m uvicorn integrated.app:app --reload --host 127.0.0.1 --port 8766
pause
