macro(data_download PATH_VAR DATA_FILE SAVE_TO_PATH)
  set(data_base_url http://www.theveganrobot.com/nlmagick/data)
  set(${PATH_VAR} ${PROJECT_BINARY_DIR}/${SAVE_TO_PATH}/${DATA_FILE})
  if(NOT EXISTS ${${PATH_VAR}})
    message(STATUS "Data fetch.\n** Downloading:\n** ${data_base_url}/${DATA_FILE}\n** to:\n** ${${PATH_VAR}}")
    file(DOWNLOAD ${data_base_url}/${DATA_FILE} ${${PATH_VAR}})
  endif()
endmacro()


add_executable(nlmagick_test_fmincon  nlmagick_test_fmincon.cpp)
target_link_libraries(nlmagick_test_fmincon  nlmagick ${Boost_LIBRARIES})

add_test(test_fmincon_A   ${CMAKE_BINARY_DIR}/bin/nlmagick_test_fmincon -x 20 -y 20 -m 10)


# Rigid Image Match
data_download(image_0000 image_0000.jpg data_rigid_template)
data_download(image_0001 image_0001.jpg data_rigid_template)
data_download(image_0002 image_0002.jpg data_rigid_template)
data_download(image_mask image_mask.jpg data_rigid_template)

add_test(test_image_rigid_LinuxFlag
    ${CMAKE_BINARY_DIR}/bin/template_image_rigid 
    -K ${CMAKE_SOURCE_DIR}/testdata/camera_droidx_640x480.yml
    -i ${CMAKE_BINARY_DIR}/data_rigid_template/image_0001.jpg
    -m ${CMAKE_BINARY_DIR}/data_rigid_template/image_mask.jpg
    -t ${CMAKE_BINARY_DIR}/data_rigid_template/image_0000.jpg
    -a NLOPT_LN_BOBYQA
    -w warped01blur1.jpg
    -s 25
    -f blurredAnswer1.out
  )


data_download(trees0 trees160.png data_trees_af2f)
data_download(trees1 trees161.png data_trees_af2f)
data_download(trees2 trees162.png data_trees_af2f)
data_download(trees3 trees163.png data_trees_af2f)
data_download(trees4 trees164.png data_trees_af2f)
data_download(trees5 trees165.png data_trees_af2f)
data_download(trees6 trees166.png data_trees_af2f)
data_download(trees7 trees167.png data_trees_af2f)
data_download(trees8 trees168.png data_trees_af2f)
data_download(trees9 trees169.png data_trees_af2f)

add_test(test_affineFrame2Frame
    ${CMAKE_BINARY_DIR}/bin/affine_frame2frame
    -d ${CMAKE_BINARY_DIR}/data_trees_af2f
    -v 1
  )
