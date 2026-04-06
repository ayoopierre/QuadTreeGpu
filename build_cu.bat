nvcc -std=c++20 -Xcompiler "/std:c++20" ^
    -I"inc" -I"cuinc" ^
    -o main.exe "main.cu" ^
    --compiler-options "/EHsc /MD" --extended-lambda ^
    -lcudadevrt -lcudart -lquad_tree