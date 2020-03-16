cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)

project(color_only)

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

find_package(CUDA REQUIRED)
set(OpenCV_DIR /usr/local/share/OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

# Load catkin and all dependencies required for this package

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  cuda_renderer
  hist_prioritize
  roscpp
  roslib
  eigen_conversions
  pcl_ros
  )


include_directories(include ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})


include_directories(${catkin_INCLUDE_DIRS})

catkin_package(
  CATKIN_DEPENDS
    cv_bridge
    image_transport
    cuda_renderer
    hist_prioritize
    roscpp
    roslib
    pcl_ros
  DEPENDS
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
  )
add_library(${PROJECT_NAME}
            test.cpp
)

target_include_directories(color_only PUBLIC ${renderer_inc})
target_link_libraries(color_only PUBLIC ${renderer_lib} ${catkin_LIBRARIES} )

add_executable(color_test test.cpp)
target_link_libraries(color_test ${catkin_LIBRARIES} )
catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
)
add_executable(image_pub image_pub.cpp)
target_link_libraries(image_pub ${catkin_LIBRARIES} )
catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
)


install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION /home/jessy/catkin_ws/devel/include)
