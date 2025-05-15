@echo off
echo Cleaning up Docker containers...

REM Stop all running containers
docker stop quantum-api kind_antonelli objective_elion vigilant_poincare 2>nul

REM Remove all containers
docker rm quantum-api kind_antonelli objective_elion vigilant_poincare 2>nul

echo.
echo All containers have been cleaned up.
echo You can now run run.bat to start a fresh instance. 