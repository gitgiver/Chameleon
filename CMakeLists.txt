cmake_minimum_required(VERSION 3.22)
project(HITS)

set(CMAKE_CXX_STANDARD 17)

set(CONDA_ENV_PATH "/root/.conda/envs/chameleon")

set(CMAKE_CXX_FLAGS_DEBUG "-march=native ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "-march=native  ${CMAKE_CXX_FLAGS_RELEASE}")

set(gpu_device torch::kCUDA)
set(data_father_path "\"/root/shared-nvme/go/Chameleon/data/\"")

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(Torch_DIR "${CONDA_ENV_PATH}/lib/python3.10/site-packages/torch/share/cmake/Torch")

# Help cmake find Boost in the conda environment
set(Boost_ROOT "${CONDA_ENV_PATH}")
set(BOOST_ROOT "${CONDA_ENV_PATH}")
set(BOOST_INCLUDEDIR "${CONDA_ENV_PATH}/include")
set(BOOST_LIBRARYDIR "${CONDA_ENV_PATH}/lib")

#set(CUDA_LAUNCH_BLOCKING 1)
#set(TORCH_USE_CUDA_DSA 1)

configure_file(
        "${PROJECT_SOURCE_DIR}/include/DEFINE.h.in"
        "${PROJECT_SOURCE_DIR}/include/DEFINE.h"
)
configure_file(
        "${PROJECT_SOURCE_DIR}/index/include/Parameter.h.in"
        "${PROJECT_SOURCE_DIR}/index/include/Parameter.h"
)

find_package(Python3 COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS fiber date_time system filesystem)
include_directories(${Boost_INCLUDE_DIRS})
set(gpu_device torch::kCUDA)
set(data_father_path "\"/root/shared-nvme/go/Chameleon/data/\"")

#link_libraries("${TORCH_LIBRARIES}" Python3::Python Python3::NumPy)
link_libraries("${TORCH_LIBRARIES}" Python3::Python)

#set(MY_STACK_SIZE 1048576k)
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MY_STACK_SIZE}")

add_executable(main main.cpp
#        index/include/Configuration.hpp
#        index/include/Parameter.h index/include/Controller.hpp
#        index/include/experience.hpp index/include/Index.hpp
#        index/include/RL_network.hpp index/workspace/gen_exp.hpp
#        index/workspace/global_on_policy.cpp index/workspace/train.cpp
        )
target_include_directories(main PRIVATE "${CONDA_ENV_PATH}/lib/python3.10/site-packages/numpy/_core/include")

add_executable(train index/workspace/train.cpp)

add_executable(train_TSMDP index/workspace/train_TSMDP.cpp)

add_executable(train_DARE index/workspace/train_DARE.cpp)

add_executable(dynamic_data benchmark/dynamic_data.cpp)

add_executable(dynamic_data2 benchmark/dynamic_data2.cpp)

add_executable(add_get_erase benchmark/add_get_erase.cpp)

add_executable(read_only benchmark/read_only.cpp)

add_executable(test benchmark/test.cpp )

add_executable(opration_latency benchmark/opration_latency.cpp)

add_executable(scalability benchmark/scalability.cpp)
add_executable(Ablation benchmark/Ablation.cpp)

find_package(Threads REQUIRED)
add_executable(concurrency_test benchmark/concurrency_test.cpp)
target_link_libraries(concurrency_test PUBLIC Threads::Threads)

# Resolve Intel ITT symbols needed by libtorch_cpu.so
add_library(itt_stub OBJECT itt_stub.c)
function(add_itt_stub TGT)
    target_sources(${TGT} PRIVATE $<TARGET_OBJECTS:itt_stub>)
endfunction()
add_itt_stub(main)
add_itt_stub(train)
add_itt_stub(train_TSMDP)
add_itt_stub(train_DARE)
add_itt_stub(dynamic_data)
add_itt_stub(dynamic_data2)
add_itt_stub(add_get_erase)
add_itt_stub(read_only)
add_itt_stub(opration_latency)
add_itt_stub(scalability)
add_itt_stub(Ablation)
add_itt_stub(concurrency_test)
add_itt_stub(test)
