nvcc -std=c++17 -lineinfo -O3 \
    -I"inc" -I"cuinc" \
    -o main "src/quad_tree.cu" "main.cu" \
    --compiler-options "-fPIC -fexceptions" --extended-lambda \
    -lcudadevrt -lcudart