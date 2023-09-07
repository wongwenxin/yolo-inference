cmake_minimum_required(VERSION 3.15)
set(PROJECT_NAME yolo-ort-inference)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)


# ------------------------OpenCV-----------------------#
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
message(STATUS "OPENCV : ${OpenCV_INCLUDE_DIR}")
##-------------------------CUDA-------------------------#
option(USE_CUDA "Enable CUDA support" ON)
if (NOT APPLE AND USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else ()
    set(USE_CUDA ON)
endif ()

#-------------------------ONNXRUNTIME------------------#

set (ONNXRUNTIME_VERSION 1.15.1)

if (WIN32)
    if (USE_CUDA)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}")
    else ()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
    endif (USE_CUDA)
endif (WIN32)
if (UNIX)
    if (USE_CUDA)
        set(ONNXRUNTIME_ROOT /home/wwx/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION})
    else ()
        set(ONNXRUNTIME_ROOT "/home/wwx/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
    endif (USE_CUDA)
endif (UNIX)

message(STATUS "ONNXRUNTIME: ${ONNXRUNTIME_ROOT}")


include_directories(${PROJECT_NAME} "${ONNXRUNTIME_ROOT}/include")
include_directories("include/")

set(PROJECT_SOURCES
        main.cpp
        src/inference.cpp
        src/tools.cpp
        include/inference.h
        include/tools.h
        )

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

if (WIM32)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.lib)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (UNIX)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
endif ()



