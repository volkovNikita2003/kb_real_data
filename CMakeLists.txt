cmake_minimum_required(VERSION 3.16)
project(real_data_acquisition LANGUAGES CXX)

set(SENSORS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sensors" CACHE PATH "Detector sources")
find_package(Threads REQUIRED)
find_path(MVS_INCLUDE_DIR MvCameraControl.h HINTS /opt/MVS/include REQUIRED)
add_executable(measure_lin
    cpp/measure_lin.cpp
    ${SENSORS_DIR}/lin_darl.cpp
    ${SENSORS_DIR}/sensors/LineSensor.cpp
    ${SENSORS_DIR}/signals/LineSignal.cpp
    ${SENSORS_DIR}/signals/HikCameraSignal.cpp
    ${SENSORS_DIR}/savers/TxtSignalSaver.cpp
    ${SENSORS_DIR}/savers/Txt2SignalSaver.cpp
)
target_compile_features(measure_lin PRIVATE cxx_std_17)
target_include_directories(measure_lin PRIVATE ${SENSORS_DIR} ${MVS_INCLUDE_DIR})
target_link_libraries(measure_lin PRIVATE Threads::Threads)
