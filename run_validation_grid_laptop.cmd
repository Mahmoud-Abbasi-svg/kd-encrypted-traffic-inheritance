@echo off
rem Validation-only work on this laptop, in order (about 11-12 hours in total):
rem   1. student tuning (decision D2)                      ~40 min
rem   2. Track A start 11 with the tuned settings           ~50 min (reuses Teachers A and B)
rem   3. Track A start 24                                   ~2.5 h
rem   4. Track A start 37                                   ~2.5 h
rem   5. shortcut experiment                                ~5 h
rem Nothing here touches the test windows. Full console log: logs\validation_grid_laptop.log
cd /d "%~dp0"
set PY=C:\venvs\kd-traffic\Scripts\python.exe
set PILOT=results\pilot\20260916-122627_S_train11-14
set TEACHER_B=results\track_a\20260916-140053_S_train11-14\models\teacherB_wide.pt
set LOG=logs\validation_grid_laptop.log
if not exist logs mkdir logs

echo Started %date% %time% > %LOG%
echo Running. Keep the laptop plugged in, lid open, and do not click inside this window. Progress: %LOG%
if not exist configs\student_hparams.json (
  echo [1/5] tuning
  %PY% scripts\09_tune_students.py --size S --teachers-from %PILOT% --workers 0 --epoch-check >> %LOG% 2>&1 || goto :failed
) else (
  echo [1/5] tuning skipped: configs\student_hparams.json exists
)
echo [2/5] Track A start 11
%PY% scripts\06_track_a.py --size S --start 11 --teachers-from %PILOT% --teacher-b-from %TEACHER_B% --workers 0 >> %LOG% 2>&1 || goto :failed
echo [3/5] Track A start 24
%PY% scripts\06_track_a.py --size S --start 24 --workers 0 >> %LOG% 2>&1 || goto :failed
echo [4/5] Track A start 37
%PY% scripts\06_track_a.py --size S --start 37 --workers 0 >> %LOG% 2>&1 || goto :failed
echo [5/5] shortcut experiment
%PY% scripts\07_shortcut.py --size S --workers 0 >> %LOG% 2>&1 || goto :failed
echo Finished %date% %time% >> %LOG%
echo Done. Tell Claude "done".
pause
exit /b 0

:failed
echo FAILED %date% %time% >> %LOG%
echo A step failed. Tell Claude; the details are at the end of %LOG%.
pause
exit /b 1
