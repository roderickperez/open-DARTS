
# Build global ------------------------------------------------------------------
add_library(global STATIC "")

set(OD_GLOBAL_SRC_FILES 
    global/global.cpp 
    global/components.cpp
    global/units.cpp
    global/timer.cpp
    maths/geometry.cpp
    maths/linesearch.cpp
    maths/maths.cpp
    maths/linesearch.cpp
    maths/modifiedcholeskys99.cpp
    maths/root_finding.cpp
)

set(OD_GLOBAL_PUBLIC_HEADER_FILES 
    global/include/dartsflash/global/global.hpp 
    global/include/dartsflash/global/components.hpp
    global/include/dartsflash/global/units.hpp
    global/include/dartsflash/global/timer.hpp
    maths/include/dartsflash/maths/geometry.hpp
    maths/include/dartsflash/maths/linesearch.hpp
    maths/include/dartsflash/maths/maths.hpp
    maths/include/dartsflash/maths/modifiedcholeskys99.hpp
    maths/include/dartsflash/maths/root_finding.hpp
)

target_sources(global 
    PRIVATE
        ${OD_GLOBAL_SRC_FILES}
        ${OD_GLOBAL_PUBLIC_HEADER_FILES}
)

target_link_libraries(global 
    PRIVATE
        dartsflash_compiler_flags
        Eigen3::Eigen
)

target_include_directories(global 
    PUBLIC 
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/global/include;${CMAKE_CURRENT_SOURCE_DIR}/maths/include>"
        $<INSTALL_INTERFACE:include>
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/global/include/"
    DESTINATION include
    COMPONENT header_files
    FILES_MATCHING PATTERN "*.hpp"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/maths/include/"
    DESTINATION include
    COMPONENT header_files
    FILES_MATCHING PATTERN "*.hpp"
)

# ------------------------------------------------------------------------------


# Build eos --------------------------------------------------------------------
add_library(darts-eos STATIC "")
set(OD_EOS_SRC_FILES 
    eos/eos_params.cpp 
    eos/eos.cpp 
    eos/ideal.cpp
    eos/helmholtz/helmholtz.cpp 
    eos/helmholtz/cubic.cpp 
    eos/helmholtz/mix.cpp
    # eos/helmholtz/gerg.cpp
    eos/iapws/iapws95.cpp
    eos/iapws/iapws_ice.cpp
    eos/aq/aq.cpp
    eos/aq/jager.cpp 
    eos/aq/ziabakhsh.cpp 
    eos/vdwp/vdwp.cpp 
    eos/vdwp/ballard.cpp 
    eos/vdwp/munck.cpp 
    # eos/vdwp/klauda.cpp 
    eos/solid/solid.cpp
)

set(OD_EOS_PUBLIC_HEADER_FILES 
    eos/include/dartsflash/eos/eos_params.hpp 
    eos/include/dartsflash/eos/eos.hpp 
    eos/include/dartsflash/eos/ideal.hpp
    eos/include/dartsflash/eos/helmholtz/helmholtz.hpp 
    eos/include/dartsflash/eos/helmholtz/cubic.hpp 
    eos/include/dartsflash/eos/helmholtz/mix.hpp
    # eos/include/dartsflash/eos/helmholtz/gerg.hpp
    eos/include/dartsflash/eos/iapws/iapws95.hpp
    eos/include/dartsflash/eos/iapws/iapws_ice.hpp
    eos/include/dartsflash/eos/aq/aq.hpp
    eos/include/dartsflash/eos/aq/jager.hpp 
    eos/include/dartsflash/eos/aq/ziabakhsh.hpp 
    eos/include/dartsflash/eos/vdwp/vdwp.hpp 
    eos/include/dartsflash/eos/vdwp/ballard.hpp 
    eos/include/dartsflash/eos/vdwp/munck.hpp 
    # eos/include/dartsflash/eos/vdwp/klauda.hpp 
    eos/include/dartsflash/eos/solid/solid.hpp
)

target_sources(darts-eos 
    PRIVATE
        ${OD_EOS_SRC_FILES}
        ${OD_EOS_PUBLIC_HEADER_FILES}
)

target_link_libraries(darts-eos 
    PRIVATE
        global
        dartsflash_compiler_flags
        Eigen3::Eigen
)

target_include_directories(darts-eos
    PUBLIC 
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/eos/include>"
        $<INSTALL_INTERFACE:include>
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/eos/include/"
    DESTINATION include
    COMPONENT header_files
    FILES_MATCHING PATTERN "*.hpp"
)

# ------------------------------------------------------------------------------

# Build darts-flash and libflash -----------------------------------------------
# This builds two versions:
# - darts-flash: the shared library with only c++ code
# - libflash: the python interface
# 
# darts-flash can be linked to other c++ code, libflash can only be used in python.

add_library(darts-flash STATIC "")

set(OD_FLASH_SRC_FILES 
    flash/flash_params.cpp
    flash/flash_results.cpp
    flash/flash.cpp 
    flash/negative_flash.cpp 
    flash/px_flash.cpp
    flash/initial_guess.cpp
    flash/trial_phase.cpp
    stability/stability.cpp
    phase-split/basesplit.cpp 
    phase-split/twophasesplit.cpp
    phase-split/multiphasesplit.cpp
    rr/rr.cpp
    rr/rr_eqconvex2.cpp
    rr/rr_min.cpp 
)

set(OD_FLASH_PUBLIC_HEADER_FILES  
    flash/include/dartsflash/flash/flash_params.hpp
    flash/include/dartsflash/flash/flash_results.hpp
    flash/include/dartsflash/flash/flash.hpp
    flash/include/dartsflash/flash/negative_flash.hpp
    flash/include/dartsflash/flash/px_flash.hpp
    flash/include/dartsflash/flash/initial_guess.hpp
    flash/include/dartsflash/flash/trial_phase.hpp
    phase-split/include/dartsflash/phase-split/basesplit.hpp 
    phase-split/include/dartsflash/phase-split/twophasesplit.hpp
    phase-split/include/dartsflash/phase-split/multiphasesplit.hpp
    rr/include/dartsflash/rr/rr.hpp
    stability/include/dartsflash/stability/stability.hpp
)

set(OD_FLASH_PYBIND_FILES
    pybind/py_eos.cpp
    pybind/py_flash.cpp
    pybind/py_global.cpp
    pybind/py_main.cpp
    # pybind/py_phasesplit.cpp
    # pybind/py_rr.cpp
    pybind/py_stability.cpp
)


target_sources(darts-flash 
    PRIVATE
        ${OD_FLASH_SRC_FILES}
        ${OD_FLASH_PUBLIC_HEADER_FILES}
)

target_link_libraries(darts-flash  
    PUBLIC
        global
        darts-eos
        dartsflash_compiler_flags
        Eigen3::Eigen
) 

target_include_directories(darts-flash 
    PUBLIC 
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/flash/include;${CMAKE_CURRENT_SOURCE_DIR}/phase-split/include;${CMAKE_CURRENT_SOURCE_DIR}/rr/include;${CMAKE_CURRENT_SOURCE_DIR}/stability/include>"
        $<INSTALL_INTERFACE:include>
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/flash/include/"
    DESTINATION include
    COMPONENT header_files
    FILES_MATCHING PATTERN "*.hpp"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/phase-split/include/"
    DESTINATION include
    COMPONENT header_files
    FILES_MATCHING PATTERN "*.hpp"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/stability/include/"
    DESTINATION include
    COMPONENT header_files
    FILES_MATCHING PATTERN "*.hpp"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/rr/include/"
    DESTINATION include
    COMPONENT header_files
    FILES_MATCHING PATTERN "*.hpp"
)

# opendarts-flah Python module setup
message(STATUS "Setting up opendarts flash python module...")
# By default it builds the python module
if (NOT DEFINED OPENDARTS_FLASH_BUILD_PYTHON)
    set(OPENDARTS_FLASH_BUILD_PYTHON TRUE)
endif (NOT DEFINED OPENDARTS_FLASH_BUILD_PYTHON)

if (OPENDARTS_FLASH_BUILD_PYTHON)
    # Only setup if requested
    pybind11_add_module(libflash
        ${OD_FLASH_PYBIND_FILES}
    )

    # This is a temporary fix 
    # See pybind11 discussion on how to fix this.
    # General discussion with the fix:
    # https://github.com/pybind/pybind11/issues/1272
    # and then the recommended approach that requires changing the c++ code:
    # https://github.com/chainer/chainer/pull/8437
    # Needs more investigation
    set_target_properties(libflash PROPERTIES CXX_VISIBILITY_PRESET "default")

    target_link_libraries(libflash 
        PUBLIC
            darts-flash
    ) 

    # Install darts-flash RELEASE
    install(TARGETS darts-flash  
    CONFIGURATIONS RELEASE RELWITHDEBINFO ""
    DESTINATION lib/
    )

    # Install darts-flash DEBUG
    get_target_property(darts_flash_name darts-flash NAME)
    install(TARGETS darts-flash  
    CONFIGURATIONS DEBUG    
    DESTINATION lib/debug
    RENAME ${CMAKE_STATIC_LIBRARY_PREFIX}${darts_flash_name}_d${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

    # Install libflash RELEASE
    install(TARGETS libflash  
    CONFIGURATIONS RELEASE RELWITHDEBINFO ""  
    DESTINATION lib/
    )

    # Install libflash DEBUG
    get_target_property(libflash_name libflash NAME)
    install(TARGETS libflash  
    CONFIGURATIONS DEBUG    
    DESTINATION lib/debug
    )

    message(STATUS "   Done!")
else()
    message(STATUS "   Skipped!")
endif (OPENDARTS_FLASH_BUILD_PYTHON)
