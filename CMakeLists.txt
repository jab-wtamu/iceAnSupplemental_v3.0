cmake_minimum_required(VERSION 3.13.4)

get_filename_component(
    PRISMS_PF_CORE_DIR
    ${CMAKE_CURRENT_LIST_DIR}/../..
    ABSOLUTE
)

include(${PRISMS_PF_CORE_DIR}/cmake/prisms_pf_config.cmake)
include(${PRISMS_PF_CORE_DIR}/cmake/setup_application.cmake)

project(iceGrowthModel CXX)

prisms_pf_autopilot(${PRISMS_PF_CORE_DIR})
