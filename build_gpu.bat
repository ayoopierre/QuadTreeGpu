nvcc -std=c++17 -Xcompiler "/std:c++17" -lineinfo -O3 ^
    -I"inc" -I"cuinc" ^
    -lib -o quad_tree.lib "src/quad_tree.cu" ^
    --compiler-options "/EHsc /MD" --extended-lambda ^
    -lcudadevrt -lcudart