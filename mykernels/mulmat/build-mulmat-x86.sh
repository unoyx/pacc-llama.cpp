clang++ -I include -g -std=gnu++17 -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -march=native -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_CPU_REPACK -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GLIBCXX_ASSERTIONS -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_cpu_EXPORTS ./lib/sgemm.cpp -c -o sgemm.o

clang++ -I include -g -std=gnu++17 -fPIC -march=native -c gemm-driver.cpp -o gemm-driver.o
clang++ -g -std=gnu++17 -fPIC -march=native gemm-driver.o sgemm.o -o main
