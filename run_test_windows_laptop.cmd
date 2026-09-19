@echo off
rem THE CONFIRMATORY RUN. Trains the students again for each start date and evaluates them on the
rem pre-registered TEST windows (4-week blocks up to week 52, without weeks 50 and 52) with the
rem TEST unknown services. Requires docs/preregistration.md to say "**Status:** FROZEN" (it does,
rem since 19 Sep 2026); the scripts refuse otherwise.
rem Teachers are reused from the validation runs, so each start date is about 2 to 2.5 hours.
rem Finished steps leave a marker in logs\done\, so this can be stopped and resumed.
rem Log: logs\test_windows_laptop.log
cd /d "%~dp0"
set PY=C:\venvs\kd-traffic\Scripts\python.exe
set PILOT=results\pilot\20260916-122627_S_train11-14
set RUN11=results\track_a\20260917-110225_S_train11-14
set RUN24=results\track_a\20260917-121305_S_train24-27
set RUN37=results\track_a\20260918-132337_S_train37-40
set TEACHER_B11=results\track_a\20260916-140053_S_train11-14\models\teacherB_wide.pt
set LOG=logs\test_windows_laptop.log
set DONE=logs\done
if not exist %DONE% mkdir %DONE%

echo ===== test windows started %date% %time% >> %LOG%
echo Confirmatory run. Keep the laptop plugged in, lid open, one job at a time. Progress: %LOG%
call :step test_start11 "%PY% scripts\06_track_a.py --size S --start 11 --with-test --teachers-from %PILOT% --teacher-b-from %TEACHER_B11% --workers 0" || goto :failed
call :step test_start24 "%PY% scripts\06_track_a.py --size S --start 24 --with-test --teachers-from %RUN24% --teacher-b-from %RUN24%\models\teacherB_wide.pt --workers 0" || goto :failed
call :step test_start37 "%PY% scripts\06_track_a.py --size S --start 37 --with-test --teachers-from %RUN37% --teacher-b-from %RUN37%\models\teacherB_wide.pt --workers 0" || goto :failed
echo ===== test windows finished %date% %time% >> %LOG%
echo Done. Tell Claude "done" and it will run the confirmatory analysis.
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
