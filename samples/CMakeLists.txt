find_package(OpenCV  REQUIRED)

#if(OpenCV_FOUND)

message(STATUS "Opencv include dir? ${OpenCV_INCLUDE_DIRS}" )

link_directories( ${Boost_LIBRARY_DIRS} )
include_directories( ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
add_executable(template_nlopt_rigid  template_nlopt_rigid.cpp  )
target_link_libraries(template_nlopt_rigid  nlopt ${Boost_LIBRARIES} ${OpenCV_LIBS})

#add_executable(template_image_rigid  template_image_rigid.cpp  )
#target_link_libraries(template_image_rigid  nlopt ${Boost_LIBRARIES} ${OpenCV_LIBS})

#add_executable(webcam_chessboard_rigid  webcam_chessboard_rigid.cpp  )
#target_link_libraries(webcam_chessboard_rigid  nlopt ${Boost_LIBRARIES} ${OpenCV_LIBS})

#endif()
