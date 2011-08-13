
add_executable(nlmagick_test_fmincon  nlmagick_test_fmincon.cpp)
target_link_libraries(nlmagick_test_fmincon  nlmagick ${Boost_LIBRARIES})

add_test(test_fmincon_A   ${CMAKE_BINARY_DIR}/bin/nlmagick_test_fmincon -x 20 -y 20 -m 10)
