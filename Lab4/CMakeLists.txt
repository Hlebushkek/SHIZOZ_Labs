cmake_minimum_required(VERSION 3.23)
project(lab4)

enable_language(C)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)

if (APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13")
endif()

add_definitions(-DCMAKE_BUILD_TYPE=Debug)

find_package(OpenCV REQUIRED PATHS "../build_opencv")
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(../dlib ../../build_dlib)

include_directories(include)
include_directories(${DLIB_INCLUDE_DIRECTORY})
file(GLOB LAB1_SOURCES "source/*.cpp" "include/*.hpp")

add_executable(${PROJECT_NAME} ${LAB1_SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} dlib::dlib)