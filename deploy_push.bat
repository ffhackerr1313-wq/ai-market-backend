@echo off
setlocal enabledelayedexpansion
title Push AI Market project to GitHub

echo ==================================================
echo   Pushing AI Market project to GitHub
echo   GitHub user: ffhackerr1313-wq
echo ==================================================
echo.

REM ----------------------- BACKEND -----------------------
echo [1/2] BACKEND  -  C:\major project
echo --------------------------------------------------
pushd "C:\major project"
if exist ".git\index.lock" del /f /q ".git\index.lock"
git add -A
git commit -m "Deploy: backend ready for Railway"
git remote remove origin 2>nul
git remote add origin https://github.com/ffhackerr1313-wq/ai-market-backend.git
git branch -M main
git push -u origin main
popd
echo.

REM ----------------------- FRONTEND ----------------------
echo [2/2] FRONTEND -  C:\Users\atul\ai-market-ui
echo --------------------------------------------------
pushd "C:\Users\atul\ai-market-ui"
if exist ".git\index.lock" del /f /q ".git\index.lock"
git add -A
git commit -m "Deploy: frontend ready for Vercel"
git remote remove origin 2>nul
git remote add origin https://github.com/ffhackerr1313-wq/ai-market-ui.git
git branch -M main
git push -u origin main
popd
echo.

echo ==================================================
echo   DONE. Scroll up and check both pushes succeeded.
echo   If a GitHub login window popped up, approve it.
echo ==================================================
pause
