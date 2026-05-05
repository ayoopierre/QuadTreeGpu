nvcc -std=c++17 -lineinfo -O3 \
    -I"inc" -I"cuinc" \
    -lib -o quad_tree.a "src/quad_tree.cu" \
    --compiler-options "-fPIC -fexceptions" --extended-lambda \
    -lcudadevrt -lcudart