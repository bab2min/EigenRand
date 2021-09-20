################
# Define a test
add_executable(EigenRand_test test.cpp)

######################################
# Configure the test to use GoogleTest
#
# If used often, could be made a macro.

add_dependencies(EigenRand_test gtest)
include_directories(${source_dir}/googletest/include)
target_link_libraries(EigenRand_test ${binary_dir}/googlemock/gtest/libgtest_main.a)
target_link_libraries(EigenRand_test ${binary_dir}/googlemock/gtest/libgtest.a)
target_link_libraries(EigenRand_test -Wl,--whole-archive -lpthread -Wl,--no-whole-archive)

##################################
# Just make the test runnable with
#   $ make test

enable_testing()
add_test(EigenRand_test EigenRand_test)
