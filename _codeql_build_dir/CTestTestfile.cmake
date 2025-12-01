# CMake generated Testfile for 
# Source directory: /home/runner/work/EigenRand/EigenRand
# Build directory: /home/runner/work/EigenRand/EigenRand/_codeql_build_dir
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(EigenRand-benchmark "/home/runner/work/EigenRand/EigenRand/_codeql_build_dir/EigenRand-benchmark")
set_tests_properties(EigenRand-benchmark PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/EigenRand/EigenRand/CMakeLists.txt;115;add_test;/home/runner/work/EigenRand/EigenRand/CMakeLists.txt;0;")
add_test(EigenRand-benchmark_mv "/home/runner/work/EigenRand/EigenRand/_codeql_build_dir/EigenRand-benchmark_mv")
set_tests_properties(EigenRand-benchmark_mv PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/EigenRand/EigenRand/CMakeLists.txt;115;add_test;/home/runner/work/EigenRand/EigenRand/CMakeLists.txt;0;")
add_test(EigenRand-benchmark_disc "/home/runner/work/EigenRand/EigenRand/_codeql_build_dir/EigenRand-benchmark_disc")
set_tests_properties(EigenRand-benchmark_disc PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/EigenRand/EigenRand/CMakeLists.txt;115;add_test;/home/runner/work/EigenRand/EigenRand/CMakeLists.txt;0;")
add_test(EigenRand-benchmark_vectorize_over_params "/home/runner/work/EigenRand/EigenRand/_codeql_build_dir/EigenRand-benchmark_vectorize_over_params")
set_tests_properties(EigenRand-benchmark_vectorize_over_params PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/EigenRand/EigenRand/CMakeLists.txt;115;add_test;/home/runner/work/EigenRand/EigenRand/CMakeLists.txt;0;")
add_test(EigenRand-accuracy "/home/runner/work/EigenRand/EigenRand/_codeql_build_dir/EigenRand-accuracy")
set_tests_properties(EigenRand-accuracy PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/EigenRand/EigenRand/CMakeLists.txt;115;add_test;/home/runner/work/EigenRand/EigenRand/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
subdirs("test")
