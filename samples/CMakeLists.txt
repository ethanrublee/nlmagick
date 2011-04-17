find_package( OpenCV )

if(OpenCV_FOUND)
add_executable(template_nlopt_rigid  template_nlopt_rigid.cpp  )
target_link_libraries(template_nlopt_rigid  nlopt ${Boost_LIBRARIES} ${OpenCV_LIBS})
endif()
