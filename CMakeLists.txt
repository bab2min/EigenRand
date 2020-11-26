cmake_minimum_required(VERSION 3.10)
project(EigenRand VERSION 0.30)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

include(GNUInstallDirs)

#Install
install(DIRECTORY EigenRand/
   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/EigenRand)