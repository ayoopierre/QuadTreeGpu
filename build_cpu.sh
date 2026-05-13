# Compile and link main executable
g++ -std=c++17 -O2 \
    -I"inc" \
    main.cpp -o main \
    -L. quad_tree.a \
    -lcuda -lcudart -lcudadevrt