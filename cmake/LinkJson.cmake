IF(NOT TARGET libJson)
    MESSAGE(FATAL_ERROR "Try to link Json but it was not found")
ELSE()
    MESSAGE(STATUS "link libJson")
    TARGET_LINK_LIBRARIES(${targetname} PUBLIC libJson)
    TARGET_COMPILE_DEFINITIONS(${targetname} PUBLIC COMPILE_WITH_JSON)
ENDIF()