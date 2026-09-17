@echo off
rem Shortcut experiment (RQ3) on this laptop, validation week only, about 5 hours.
rem Uses the tuned KD settings in configs\student_hparams.json. Log: logs\shortcut_laptop.log
cd /d "%~dp0"
set PY=C:\venvs\kd-traffic\Scripts\python.exe
set LOG=logs\shortcut_laptop.log
if not exist logs mkdir logs
if not exist configs\student_hparams.json (
  echo configs\student_hparams.json is missing: run the tuning first.
  pause
  exit /b 1
)
echo Started %date% %time% > %LOG%
echo Shortcut experiment running. Keep the laptop plugged in, lid open, and do not click inside this window.
%PY% scripts\07_shortcut.py --size S --workers 0 >> %LOG% 2>&1 || goto :failed
echo Finished %date% %time% >> %LOG%
echo Done. Tell Claude "done".
pause
exit /b 0

:failed
echo FAILED %date% %time% >> %LOG%
echo The run failed. Tell Claude; the details are at the end of %LOG%.
pause
exit /b 1
