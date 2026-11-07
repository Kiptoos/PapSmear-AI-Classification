@echo off
REM Run from C:\developer\PapSmear_AI_v2 with the venv activated.
REM Step 1: features (ConvNeXtTiny is the slow one, roughly 1-2 h on CPU)
for %%A in (ResNet50 EfficientNetB3 DenseNet169 TinyViT-5M ConvNeXtTiny BEiT-B16) do (
    python src\extract_features.py --arch %%A || goto :error
)
REM Step 2: repeated CV on cached features (minutes per architecture)
for %%A in (ResNet50 EfficientNetB3 DenseNet169 TinyViT-5M ConvNeXtTiny BEiT-B16) do (
    python src\run_cv.py --arch %%A --repeats 5 || goto :error
)
REM Step 3: all tables, figures and statistics
python src\analyze.py || goto :error
echo.
echo Finished. Next: python src\gradcam.py --arch ^<primary model printed above^>
goto :eof
:error
echo FAILED - see the message above.
exit /b 1
