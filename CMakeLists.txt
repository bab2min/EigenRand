cmake_minimum_required(VERSION 3.9)

project(EigenRand VERSION 0.5.1 DESCRIPTION "EigenRand, Random Number Generator for Eigen")

include(GNUInstallDirs)
include(FetchContent)
enable_testing()

option(EIGENRAND_BUILD_TEST  "Build Test sets" ON)
option(EIGENRAND_BUILD_BENCHMARK  "Build Benchmark sets" ON)

option(EIGENRAND_CMAKE_VERBOSE "Define flags for cmake verbosity" OFF)

if(PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  message(STATUS "Building as a standalone project")
  option(EIGENRAND_DEFINE_FLAGS "Define flags for EigenRand" ON)
  set(EIGEN_SUBPROJECT OFF)
else()
  message(STATUS "Building as a sub-project")
  option(EIGENRAND_DEFINE_FLAGS "Define flags for EigenRand" OFF)
  set(EIGEN_SUBPROJECT ON)
endif()

if ( NOT CMAKE_CXX_STANDARD )
  set ( CMAKE_CXX_STANDARD 11 )
endif()

set ( CMAKE_VERBOSE_MAKEFILE ${EIGENRAND_CMAKE_VERBOSE} )

set ( ADDITIONAL_FLAGS "${EIGENRAND_CXX_FLAGS}" )
message(STATUS "ADDITIONAL_FLAGS: ${ADDITIONAL_FLAGS}")

if ( NOT TARGET Eigen3::Eigen )
  message(STATUS "Eigen3 not available from master project")
  if ( EXISTS ${PROJECT_SOURCE_DIR}/include/Eigen )
    # this is for the CI
    include_directories(${PROJECT_SOURCE_DIR}/include)
    add_library(Eigen INTERFACE)
    add_library(Eigen3::Eigen ALIAS Eigen)
    target_include_directories(Eigen INTERFACE ${PROJECT_SOURCE_DIR}/include)
  else()
    find_package(Eigen3)
    # try harder to find Eigen3 using pkgconfig
    if (NOT TARGET Eigen3::Eigen )
      message(STATUS "Eigen3 not found with find_package(Eigen3), trying to find using pkgconfig")
      find_package(PkgConfig REQUIRED)
      pkg_search_module(Eigen3 REQUIRED eigen3)
    endif()
  endif()

  # Check if Eigen3 is found
  if(TARGET Eigen3::Eigen )
    message(STATUS "Eigen3 found")
  else()
    message(FATAL_ERROR "Eigen3 not found")
  endif()
endif()

# create interface lib for EigenRand to create the programming interface and propagating it simply using target_link_libraries 
add_library(eigenrand INTERFACE)
target_link_libraries(eigenrand INTERFACE Eigen3::Eigen)

if ( EIGENRAND_DEFINE_FLAGS )
  if(MSVC)
    set ( CMAKE_C_FLAGS_DEBUG "-DDEBUG -DC_FLAGS -Zi -Od ${ADDITIONAL_FLAGS}" )
    set ( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}" )

    set ( CMAKE_C_FLAGS_RELEASE "-DNDEBUG -DRELEASE -DC_FLAGS -O2 -Oi -Gy ${ADDITIONAL_FLAGS}" )
    set ( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}" )

    set ( CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE} -Zi")
    set ( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    set ( CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELEASE}" )
  else()
   

    set ( CMAKE_C_FLAGS_DEBUG "-DDEBUG -DC_FLAGS -g3 -O0 ${ADDITIONAL_FLAGS}" )
    set ( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}" )
    set ( CMAKE_EXE_LINKER_FLAGS_DEBUG "-DDEBUG -DLINKER_FLAGS" )

    set ( CMAKE_C_FLAGS_RELEASE "-DNDEBUG -DRELEASE -DC_FLAGS -O3 ${ADDITIONAL_FLAGS}" )
    set ( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}" )
    set ( CMAKE_EXE_LINKER_FLAGS_RELEASE "-DRELEASE -DLINKER_FLAGS" )

    set ( CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE} -g3")
    set ( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    set ( CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELEASE}" )
  endif()
endif(EIGENRAND_DEFINE_FLAGS)

if ( UNIX )
  target_link_libraries(eigenrand INTERFACE pthread)
endif()
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
target_include_directories(eigenrand INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/EigenRand>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
add_library(EigenRand::EigenRand ALIAS eigenrand)

if(EIGENRAND_BUILD_BENCHMARK)
  set(ITEMS
    benchmark/benchmark.cpp
    benchmark/benchmark_mv.cpp
    benchmark/benchmark_disc.cpp
    benchmark/benchmark_vectorize_over_params.cpp
    benchmark/accuracy.cpp
  )
  foreach(app ${ITEMS})
    get_filename_component(app_name ${app} NAME_WE)
    add_executable(${PROJECT_NAME}-${app_name} ${app})
    target_link_libraries(${PROJECT_NAME}-${app_name} PRIVATE eigenrand)
    install(TARGETS ${PROJECT_NAME}-${app_name} DESTINATION bin)
    add_test(NAME ${PROJECT_NAME}-${app_name} COMMAND ${PROJECT_NAME}-${app_name})
  endforeach()
endif()


set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST ${CMAKE_CURRENT_SOURCE_DIR}/googletest)
message(STATUS "FETCHCONTENT_SOURCE_DIR_GOOGLETEST: ${FETCHCONTENT_SOURCE_DIR_GOOGLETEST}")
FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest GIT_TAG v1.8.x)

if(EIGENRAND_BUILD_TEST)
  FetchContent_MakeAvailable(googletest)
  if (NOT TARGET gtest)
    message(STATUS FATAL_ERROR "gtest target not found, is there a problem with googltest?")
  endif()
  add_subdirectory( test )
endif()


install(DIRECTORY EigenRand DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS eigenrand
  EXPORT EigenRandTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

set(CPACK_PACKAGE_NAME "EigenRand-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}${EXTRA_VERSION}")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "EigenRand-${PROJECT_VERSION}${EXTRA_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "EigenRand-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}${EXTRA_VERSION}-${CMAKE_SYSTEM_NAME}")
set(CPACK_PACKAGE_DIRECTORY "${PROJECT_BINARY_DIR}/")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "EigenRand - The Fastest C++11-compatible random distribution generator for Eigen")
set(CPACK_PACKAGE_DESCRIPTION "EigenRand - The Fastest C++11-compatible random distribution generator for Eigen")
set(CPACK_PACKAGE_VENDOR "bab2min")
set(CPACK_PACKAGE_CONTACT "bab2min@gmail.com")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  set(CPACK_GENERATOR "TGZ;DEB")
else()
  set(CPACK_GENERATOR "TGZ")
endif()

set(CPACK_SOURCE_GENERATOR "TGZ")

set(CPACK_DEBIAN_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_CONTACT}")

set(CPACK_DEBIAN_PACKAGE_DESCRIPTION_SUMMARY "${CPACK_PACKAGE_DESCRIPTION_SUMMARY}")
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION}")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_DEBIAN_PACKAGE_SECTION "science")
set(CPACK_DEBIAN_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}${EXTRA_VERSION}")
set(CPACK_DEBIAN_ARCHITECTURE "${CMAKE_SYSTEM_PROCESSOR}")


set(CPACK_PACKAGE_IGNORE_FILES
  "/\\\\.git/"
  "/\\\\.gitignore"
  "/build/"
)
# set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_SOURCE_DIR}/cmake/debian/postinst")
include(CPack)
