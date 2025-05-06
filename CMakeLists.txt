cmake_minimum_required(VERSION 3.14)
project(Proiect)

set(CMAKE_CXX_STANDARD 17)
#set(OpenCV_DIR "D:\\OpenCV\\opencv\\build")

# --- Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Proiect
        src/pre_processing.cpp
        src/pre_processing.h
        src/red_eye.cpp
        src/red_eye.h
        src/main.cpp
)

target_link_libraries(Proiect ${OpenCV_LIBS})
