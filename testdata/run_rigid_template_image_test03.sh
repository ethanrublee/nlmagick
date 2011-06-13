testfile='./image_0000.jpg'
if [ -f  ${testfile} ]
then
  echo "images already exist, not downloading"
else
  wget http://www.theveganrobot.com/nlmagick/data/image_0000.jpg
  wget http://www.theveganrobot.com/nlmagick/data/image_0001.jpg
  wget http://www.theveganrobot.com/nlmagick/data/image_0002.jpg
  wget http://www.theveganrobot.com/nlmagick/data/image_mask.jpg
fi

# assume that the 'bin' folder for nlmagick is in your path


template_image_rigid  -K camera_droidx_640x480.yml  -i image_0001.jpg  -m image_mask.jpg  -t image_0000.jpg -a NLOPT_LN_SBPLX -w warped01blur.jpg -s 15 -f blurredAnswer1.out -L 8 -p solver_A_  

template_image_rigid  -K camera_droidx_640x480.yml  -i image_0001.jpg  -m image_mask.jpg  -t image_0000.jpg -a NLOPT_LN_BOBYQA -w warped01noblur.jpg -s 7 -g blurredAnswer1.out -f Answer1.out -L 8 -p solver_B_ 




template_image_rigid  -K camera_droidx_640x480.yml  -C 0.5 -D 0.0 -i image_0002.jpg  -m image_mask.jpg  -t image_0000.jpg -a NLOPT_LN_BOBYQA -w warped02noblur.jpg -s 25 -f blurredAnswer2.out -L 8 -p solver_C_  

template_image_rigid  -K camera_droidx_640x480.yml  -C 0.5 -D 0.0 -i image_0002.jpg  -m image_mask.jpg  -t image_0000.jpg -a NLOPT_LN_SBPLX -w warped02noblur.jpg -s 7 -g blurredAnswer2.out -f Answer2.out -L 8 -p solver_D_   


