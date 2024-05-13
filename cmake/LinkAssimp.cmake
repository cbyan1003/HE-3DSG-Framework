IF(WITH_ASSIMP)
    target_link_libraries(${targetname} InstanceFusion ${ASSIMP_LIBRARIES}         -ldl)
    TARGET_COMPILE_DEFINITIONS(${targetname} InstanceFusion COMPILE_WITH_ASSIMP)
ENDIF()