nvcc -std=c++20 -Xcompiler "/std:c++20" ^
    -I"inc" -I"cuinc" ^
    -o main.exe "src/quad_tree.cu" "main.cu" ^
    --compiler-options "/EHsc /MD" --extended-lambda ^
    -lcudadevrt -lcudart