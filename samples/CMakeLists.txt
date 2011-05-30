find_package(OpenCV)

if(OpenCV_FOUND)

message(STATUS "Opencv include dir? ${OpenCV_INCLUDE_DIRS}" )

link_directories( ${Boost_LIBRARY_DIRS} )
include_directories( ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )

add_executable(template_image_rigid  template_image_rigid.cpp  opencv_helpers.hpp)
target_link_libraries(template_image_rigid  nlmagick ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_executable(webcam_chessboard_rigid  webcam_chessboard_rigid.cpp  opencv_helpers.hpp)
target_link_libraries(webcam_chessboard_rigid  nlmagick ${Boost_LIBRARIES} ${OpenCV_LIBS})

endif()
