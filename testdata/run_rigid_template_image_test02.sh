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
template_image_rigid   -i image_0001.jpg  -m image_mask.jpg  -t image_0000.jpg   -F 1.0  -a NLOPT_LN_BOBYQA -w warped01blur.jpg -s 25 -f blurredAnswer.out

template_image_rigid   -i image_0001.jpg  -m image_mask.jpg  -t image_0000.jpg   -K camera_droidx_640x480.yml  -a NLOPT_LN_BOBYQA -w warped01blur.jpg -s 25 -f blurredAnswer.out

template_image_rigid   -i image_0001.jpg  -m image_mask.jpg  -t image_0000.jpg   -K camera_droidx_640x480.yml  -a NLOPT_LN_BOBYQA -w warped01.jpg -s 1 -g blurredAnswer.out
template_image_rigid   -i image_0002.jpg  -m image_mask.jpg  -t image_0000.jpg   -K camera_droidx_640x480.yml  -a NLOPT_LN_BOBYQA -w warped02blur.jpg -s 15 -f blurredAnswer2.out
template_image_rigid   -i image_0002.jpg  -m image_mask.jpg  -t image_0000.jpg   -K camera_droidx_640x480.yml  -a NLOPT_LN_BOBYQA -w warped02.jpg -s 3  -g blurredAnswer2.out
