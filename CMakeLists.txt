cmake_minimum_required(VERSION 3.10)
project(MLP_Implementation C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Add compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic")

# Add debug flags for Debug configuration
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")

# Add optimization flags for Release configuration
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")

# Add math library
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    set(EXTRA_LIBS ${MATH_LIBRARY})
endif()

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)

# Source files
set(SOURCES
        src/mlp.c
        src/util.c
)

# Create the main executable
add_executable(mlp_train src/main.c ${SOURCES})
target_link_libraries(mlp_train ${EXTRA_LIBS})

# Install targets
install(TARGETS mlp_train
        RUNTIME DESTINATION bin)

# Print configuration information
message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C Flags: ${CMAKE_C_FLAGS}")