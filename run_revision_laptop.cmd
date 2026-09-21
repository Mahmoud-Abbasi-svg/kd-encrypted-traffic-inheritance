@echo off
rem Revision experiments requested by the reviewer. All exploratory; the confirmatory runs and their
rem analysis are never touched. Finished steps leave a marker in logs\done\, so this can be stopped
rem and resumed. Run ONE job at a time and keep the lid open.
rem
rem   run_revision_laptop.cmd              everything below, about 6.5 hours
rem   run_revision_laptop.cmd features     only the feature-space scores (the gate, ~1.5 h)
rem   run_revision_laptop.cmd swaps        only the teacher-swap conditions (~2 h)
rem   run_revision_laptop.cmd teacherc     only the transformer teacher, start 11 (~1.5 h)
rem   run_revision_laptop.cmd widths       only the student-width sweep, start 11 (~1 h)
rem
rem Log: logs\revision_laptop.log
cd /d "%~dp0"
set PY=C:\venvs\kd-traffic\Scripts\python.exe
set LOG=logs\revision_laptop.log
set DONE=logs\done
set EXOUT=results\track_a_exploratory
set PILOT=results\pilot\20260916-122627_S_train11-14
set RUN11=results\track_a\20260919-113607_S_train11-14
set RUN24=results\track_a\20260919-131820_S_train24-27
set RUN37=results\track_a\20260919-143354_S_train37-40
set TB11=results\track_a\20260916-140053_S_train11-14\models\teacherB_wide.pt
set TA24=results\track_a\20260917-121305_S_train24-27
set TA37=results\track_a\20260918-132337_S_train37-40
if not exist %DONE% mkdir %DONE%
if not exist logs mkdir logs

echo ===== revision runs started %date% %time% (part: %~1) >> %LOG%
echo Running. Do not click inside this window. Progress: %LOG%

if /i "%~1"=="features" goto :features
if /i "%~1"=="swaps"    goto :swaps
if /i "%~1"=="teacherc" goto :teacherc
if /i "%~1"=="widths"   goto :widths

:features
call :step feat11 "%PY% scripts\13_feature_scores.py --size S --start 11 --with-test --students-from %RUN11% --teachers-from %PILOT% --teacher-b-from %TB11% --workers 0" || goto :failed
call :step feat24 "%PY% scripts\13_feature_scores.py --size S --start 24 --with-test --students-from %RUN24% --teachers-from %TA24% --teacher-b-from %TA24%\models\teacherB_wide.pt --workers 0" || goto :failed
call :step feat37 "%PY% scripts\13_feature_scores.py --size S --start 37 --with-test --students-from %RUN37% --teachers-from %TA37% --teacher-b-from %TA37%\models\teacherB_wide.pt --workers 0" || goto :failed
if /i "%~1"=="features" goto :done

:swaps
call :step swap11 "%PY% scripts\06_track_a.py --size S --start 11 --with-test --conditions kdM0,kdM1,hardA --teachers-from %PILOT% --teacher-b-from %TB11% --out %EXOUT% --workers 0" || goto :failed
call :step swap24 "%PY% scripts\06_track_a.py --size S --start 24 --with-test --conditions kdM0,kdM1,hardA --teachers-from %TA24% --teacher-b-from %TA24%\models\teacherB_wide.pt --out %EXOUT% --workers 0" || goto :failed
call :step swap37 "%PY% scripts\06_track_a.py --size S --start 37 --with-test --conditions kdM0,kdM1,hardA --teachers-from %TA37% --teacher-b-from %TA37%\models\teacherB_wide.pt --out %EXOUT% --workers 0" || goto :failed
if /i "%~1"=="swaps" goto :done

:teacherc
call :step teacherc11 "%PY% scripts\06_track_a.py --size S --start 11 --with-test --conditions kdC --teachers-from %PILOT% --teacher-b-from %TB11% --out %EXOUT% --workers 0" || goto :failed
if /i "%~1"=="teacherc" goto :done

:widths
call :step width16 "%PY% scripts\06_track_a.py --size S --start 11 --with-test --conditions direct,kdA4 --condition-suffix _w16 --student-width 16 --teachers-from %PILOT% --teacher-b-from %TB11% --out %EXOUT% --workers 0" || goto :failed
call :step width96 "%PY% scripts\06_track_a.py --size S --start 11 --with-test --conditions direct,kdA4 --condition-suffix _w96 --student-width 96 --teachers-from %PILOT% --teacher-b-from %TB11% --out %EXOUT% --workers 0" || goto :failed

:done
echo ===== revision runs finished %date% %time% >> %LOG%
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
echo A step failed or was stopped. Running the same command again resumes from that step.
pause
exit /b 1
