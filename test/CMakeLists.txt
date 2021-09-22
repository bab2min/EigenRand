################
# Define a test
add_executable(EigenRand-test test.cpp)

######################################
# Configure the test to use GoogleTest
#
# If used often, could be made a macro.

add_dependencies(EigenRand-test gtest)
include_directories(${source_dir}/googletest/include)
target_link_libraries(EigenRand-test ${binary_dir}/googlemock/gtest/libgtest_main.a)
target_link_libraries(EigenRand-test ${binary_dir}/googlemock/gtest/libgtest.a)
target_link_libraries(EigenRand-test -Wl,--whole-archive -lpthread -Wl,--no-whole-archive)

##################################
# Just make the test runnable with
#   $ make test

enable_testing()
add_test(EigenRand-test EigenRand-test)
