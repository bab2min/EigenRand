################
# Define a test
file(GLOB TEST_SRC
    "*.cpp"
)
add_executable(EigenRand-test ${TEST_SRC})

######################################
# Configure the test to use GoogleTest
#
# If used often, could be made a macro.

add_dependencies(EigenRand-test gtest)
target_link_libraries(EigenRand-test gtest_main gtest eigenrand)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(EigenRand-test -Wl,--whole-archive -lpthread -Wl,--no-whole-archive)
endif()
##################################
# Just make the test runnable with
#   $ make test


add_test(EigenRand-test EigenRand-test)

