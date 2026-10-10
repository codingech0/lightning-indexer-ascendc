cmake_minimum_required(VERSION 3.16.0)

project(custom_project)

find_package(ASC REQUIRED)

set(ASCEND_COMPUTE_UNIT ascend910b)
set(package_name custom)
set(LIGHTNING_PACKAGE_TYPE "SHARED" CACHE STRING "Operator package type: SHARED or RUN")
set_property(CACHE LIGHTNING_PACKAGE_TYPE PROPERTY STRINGS SHARED RUN)
if(NOT LIGHTNING_PACKAGE_TYPE MATCHES "^(SHARED|RUN)$")
    message(FATAL_ERROR "LIGHTNING_PACKAGE_TYPE must be SHARED or RUN")
endif()

npu_op_package(${package_name}
    TYPE ${LIGHTNING_PACKAGE_TYPE}
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
