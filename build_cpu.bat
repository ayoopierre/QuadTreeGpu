cl /EHsc /MD /std:c++20 /O2 /I"inc" /I"%RAYLIB_PATH%\include"^
    /I"%CUDA_PATH%\include" main.cpp /Fe:main.exe ^
    /link "quad_tree.lib" ^
    "%CUDA_PATH%\lib\x64\cuda.lib" ^
    "%CUDA_PATH%\lib\x64\cudart.lib" ^
    "%CUDA_PATH%\lib\x64\cudadevrt.lib" ^
    User32.lib Gdi32.lib Winmm.lib Kernel32.lib Ole32.lib Shell32.lib