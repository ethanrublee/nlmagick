macro(data_download PATH_VAR DATA_FILE)
  set(data_base_url http://www.theveganrobot.com/nlmagick/data)
  set(${PATH_VAR} ${PROJECT_BINARY_DIR}/data/${DATA_FILE})
  if(NOT EXISTS ${${PATH_VAR}})
    message(STATUS "Data fetch.\n** Downloading:\n** ${data_base_url}/${DATA_FILE}\n** to:\n** ${${PATH_VAR}}")
    file(DOWNLOAD ${data_base_url}/${DATA_FILE} ${${PATH_VAR}})
  endif()
endmacro()


add_executable(nlmagick_test_fmincon  nlmagick_test_fmincon.cpp)
target_link_libraries(nlmagick_test_fmincon  nlmagick ${Boost_LIBRARIES})

add_test(test_fmincon_A   ${CMAKE_BINARY_DIR}/bin/nlmagick_test_fmincon -x 20 -y 20 -m 10)

data_download(image_0000 image_0000.jpg)

data_download(image_0001 image_0001.jpg)

data_download(image_0002 image_0002.jpg)

data_download(image_mask image_mask.jpg)

add_test(template_image_rigid 
    ${CMAKE_BINARY_DIR}/bin/template_image_rigid 
    -K ${CMAKE_CURRENT_SOURCE_DIR}/camera_droidx_640x480.yml 
    -i ${image_0000}
    -m ${image_mask}
    -t ${image_0000}
    -a NLOPT_LN_BOBYQA
    -w warped01blur1.jpg
    -s 25
    -f blurredAnswer1.out
  )
