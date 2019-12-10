cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(hist_prioritize)

option(USE_CUDA "use cuda or not" ON)

SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_FLAGS "-fopenmp -Wall -Wno-sign-compare -fPIC $ENV{CXXFLAGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(USE_CUDA)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.0)
add_definitions(-DCUDA_ON)
endif()
# opencv
find_package(OpenCV 3 REQUIRED)
list(APPEND renderer_inc ${OpenCV_INCLUDE_DIRS})
list(APPEND renderer_lib ${OpenCV_LIBS})

if(USE_CUDA)
# cuda
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3")
list(APPEND renderer_inc ${CUDA_INCLUDE_DIRS})
list(APPEND renderer_lib ${CUDA_LIBRARIES})
endif()

# assimp, load ply
find_package(ASSIMP REQUIRED)
LINK_DIRECTORIES(/usr/local/lib)
list(APPEND renderer_inc ${ASSIMP_INCLUDE_DIRS})
list(APPEND renderer_lib ${ASSIMP_LIBRARIES})

find_package(OpenGL)
include_directories( ${OPENGL_INCLUDE_DIR} )
LIST(APPEND LINK_LIBS "${OPENGL_LIBRARIES}")
LIST(APPEND LIB_DIRS  "${OPENGL_LIBRARY_DIR}" )



# src
SET(hist_cuda hist.cu)
SET(hist_srcs renderer.cpp)

if(USE_CUDA)
CUDA_COMPILE(hist_cuda_objs ${hist_cuda})
endif()


# lib & test exe
add_library(hist_prioritize
               ${hist_srcs}
               ${hist_cuda} 
               ${hist_cuda_objs} 
)
target_include_directories(hist_prioritize PUBLIC ${renderer_inc} ${OPENGL_LIBRARY_DIR})
target_link_libraries(hist_prioritize PUBLIC ${renderer_lib} ${OPENGL_LIBRARIES})


add_executable(hist_test test.cpp)
target_link_libraries(hist_test hist_prioritize)
catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
)


install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION /home/jessy/catkin_ws/devel/include)
