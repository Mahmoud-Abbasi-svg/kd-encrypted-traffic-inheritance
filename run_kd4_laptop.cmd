@echo off
rem Supplementary pass: the conventional-KD arm (kdA4, kdB4 at T=4) for start dates 11 and 24,
rem which were run before that arm existed. Start 37 already includes it.
rem Teachers are reused from the existing runs, so each pass is about 35 minutes.
rem Run this only after part 1 (run_validation_grid_laptop.cmd) has finished start 24.
rem Log: logs\kd4_laptop.log
cd /d "%~dp0"
set PY=C:\venvs\kd-traffic\Scripts\python.exe
set PILOT=results\pilot\20260916-122627_S_train11-14
set TEACHER_B11=results\track_a\20260916-140053_S_train11-14\models\teacherB_wide.pt
set LOG=logs\kd4_laptop.log
set DONE=logs\done
if not exist %DONE% mkdir %DONE%

rem newest run directory of each start date
for /f "delims=" %%d in ('dir /b /o-n results\track_a ^| findstr /r "_S_train24-27$"') do if not defined RUN24 set RUN24=results\track_a\%%d
if not defined RUN24 (
  echo No finished start-24 run found. Let part 1 finish first.
  pause
  exit /b 1
)

echo ===== kd4 pass started %date% %time% >> %LOG%
echo Running. Do not click inside this window. Progress: %LOG%
call :step kd4_start11 "%PY% scripts\06_track_a.py --size S --start 11 --conditions kdA4,kdB4 --teachers-from %PILOT% --teacher-b-from %TEACHER_B11% --workers 0" || goto :failed
call :step kd4_start24 "%PY% scripts\06_track_a.py --size S --start 24 --conditions kdA4,kdB4 --teachers-from %RUN24% --teacher-b-from %RUN24%\models\teacherB_wide.pt --workers 0" || goto :failed
echo ===== kd4 pass finished %date% %time% >> %LOG%
echo Done. Tell Claude "done".
pause
exit /b 0

:step
if exist %DONE%\%~1.flag (
  echo [%~1] already done, skipped
  exit /b 0
)
echo [%~1] started %time%
echo ===== %~1 started %date% %time% >> %LOG%
%~2 >> %LOG% 2>&1 || exit /b 1
echo done > %DONE%\%~1.flag
echo [%~1] finished %time%
exit /b 0

:failed
echo ===== FAILED %date% %time% >> %LOG%
echo A step failed or was stopped. Running this command again resumes from that step.
pause
exit /b 1
