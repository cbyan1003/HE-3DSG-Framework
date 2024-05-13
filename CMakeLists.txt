cmake_minimum_required(VERSION 3.0.2)
if(POLICY CMP0007)
  MESSAGE( STATUS "POLICY_CMP0007 = ${POLICY_CMP0007}.")
  cmake_policy(SET CMP0007 NEW)
endif()

project(InstanceFusion)

###################
# Project setting #
###################
# set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function" )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
#################################
# Specify the CMake module path #
#################################
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
SET(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR} ${CMAKE_PREFIX_PATH})

#################################
# Add additional compiler flags #
#################################
IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  MESSAGE( STATUS "Hello = ${HELLO}.")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++11 -Wno-deprecated-declarations -Wno-unused-function")
ENDIF()
IF(NOT MSVC_IDE)
  MESSAGE( STATUS "${targetname}")
	SET(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
	SET(CMAKE_CXX_FLAGS "-fPIC ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF()
############
# External #
############
SET(WITH_OPENCV TRUE)
SET(WITH_EIGEN TRUE)
SET(WITH_TINYPLY TRUE)
SET(WITH_JSON TRUE)
SET(WITH_ONNX TRUE)
SET(WITH_EIGEN TRUE)
INCLUDE(UseOnnxRuntime)
INCLUDE(UseJson)
INCLUDE(Usetinyply)
INCLUDE(UseOpenMP)
INCLUDE(UseEigen3)
INCLUDE(UseOpenCV)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
set(OPENNI2_LIBRARY "/usr/local/lib/libOpenNI2.so")
set(OPENNI2_INCLUDE_DIR "/home/ycb/InstanceFusion/elasticfusionpublic/deps/OpenNI2/Include")

find_package(Flann REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package(PythonLibs REQUIRED)



file(GLOB_RECURSE srcs src/*.cpp)
file(GLOB_RECURSE cuda src/*.cu)

set(CUDA_ARCH_BIN "75")
set(CUDA_ARCH_PTX "75")
include(CudaComputeTargetFlags.cmake)                  
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-D_FORCE_INLINES;")
set(CUDA_HOST_COMPILER "/usr/bin/g++-4.9")
# set(CUDA_HOST_COMPILER gcc-5)

# set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3 -Wall -std=c++11")

cuda_add_library(Instance_fusion_cuda_library SHARED ${cuda} )

add_executable(${CMAKE_PROJECT_NAME}
               ${srcs}
)
######################
# link               #
######################
INCLUDE(LinkOnnxRuntime)
# INCLUDE(LinkJson)
target_link_libraries(
                      ${CMAKE_PROJECT_NAME}
                      ${CUDA_LIBRARIES}
                      ${OPENNI2_LIBRARY}
 		                  ${PYTHON_LIBRARIES}
                      ${FLANN_LIBRARIES}
		      /usr/local/lib/libflann_cuda.so		#flann_cuda
                      Instance_fusion_cuda_library
                      efusion
                      ${OpenCV_LIBS}
                      libJson# ${targetname} libJson
                      ${InstanceFusion} jpeg

)
# TARGET_COMPILE_DEFINITIONS(${targetname} InstanceFusion COMPILE_WITH_JSON)
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                           ${FLANN_INCLUDE_DIRS}
                           ${EIGEN_INCLUDE_DIRS}
                           ${CUDA_INCLUDE_DIRS}
                           ${OPENNI2_INCLUDE_DIR}
		                       ${PYTHON_INCLUDE_DIRS}
                           InstanceFusion COMPILE_WITH_JSON # ${targetname} InstanceFusion COMPILE_WITH_JSON
                           
)


######################
# Add subdirectories #
######################
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/elasticfusionpublic/Core/src")