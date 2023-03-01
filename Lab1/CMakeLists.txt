cmake_minimum_required(VERSION 3.23)
project(lab1)

enable_language(C)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)

if (APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "12")
endif()

add_definitions(-DOpenCV_DIR=${CMAKE_CURRENT_BINARY_DIR}../build_opencv)
find_package(OpenCV REQUIRED PATHS "../build_opencv")
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(op)
include_directories(include)
file(GLOB LAB1_SOURCES "source/*.cpp" "include/*.hpp")

#Uncomment if you want to see pixels rgb info
# add_definitions(-DITERATE_PIXELS)

add_executable(${PROJECT_NAME} ${LAB1_SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})