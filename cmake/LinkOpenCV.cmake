####################
# LinkOpenGL.cmake #
####################
IF(WITH_OPENCV)
    TARGET_LINK_LIBRARIES(${targetname} InstanceFusion ${OpenCV_LIBS})
    TARGET_INCLUDE_DIRECTORIES(${targetname} InstanceFusion ${OpenCV_INCLUDE_DIRS})
    TARGET_LINK_DIRECTORIES(${targetname} InstanceFusion ${OpenCV_LIB_DIRS})
    TARGET_COMPILE_DEFINITIONS(${targetname} InstanceFusion COMPILE_WITH_OPENCV)

    MESSAGE("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
ENDIF()