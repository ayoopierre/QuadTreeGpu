nvcc -std=c++17 -O3 -I"inc" -I"cuinc" ^
    --compiler-options "/EHsc /MD" --extended-lambda ^
    src/quad_tree.cu main.cu -o main.exe