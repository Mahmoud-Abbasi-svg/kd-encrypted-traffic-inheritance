@echo off
rem Week-3 pilot and baselines on this laptop (RTX 3050 Ti, CUDA build of PyTorch).
rem Usage: double-click, or from Command Prompt:  run_pilot_laptop.cmd
rem Output: results\pilot\<run>\ and results\baselines\<run>\ ; full console log in logs\pilot_laptop.log
cd /d "%~dp0"
set PY=C:\venvs\kd-traffic\Scripts\python.exe
set LOG=logs\pilot_laptop.log
if not exist logs mkdir logs

echo Pilot started %date% %time% > %LOG%
echo Pilot running. Keep the laptop plugged in with the lid open. Progress: %LOG%
%PY% -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0))" >> %LOG% 2>&1 || goto :failed
%PY% scripts\05_pilot.py --size S --workers 0 --no-save-logits >> %LOG% 2>&1 || goto :failed
%PY% scripts\04_baselines.py --size S --workers 0 >> %LOG% 2>&1 || goto :failed
echo Finished %date% %time% >> %LOG%
echo Done. Gate result:
findstr /c:"GATE" %LOG%
pause
exit /b 0

:failed
echo FAILED %date% %time% >> %LOG%
echo The run failed. Send the end of %LOG% to Claude.
pause
exit /b 1
