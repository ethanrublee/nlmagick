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


template_image_rigid  -K camera_droidx_640x480.yml  -i image_0001.jpg  -m image_mask.jpg  -t image_0000.jpg -a NLOPT_LN_BOBYQA -w warped01blur.jpg -s 25 -f blurredAnswer1.out  -R 1.0  -W 20

template_image_rigid  -K camera_droidx_640x480.yml  -i image_0001.jpg  -m image_mask.jpg  -t image_0000.jpg -a NLOPT_LN_BOBYQA -w warped01noblur.jpg -s 11 -g blurredAnswer1.out -f Answer1.out -C 0.0  -D 0.0 

template_image_rigid  -K camera_droidx_640x480.yml  -i image_0001.jpg  -m image_mask.jpg  -t image_0000.jpg -a NLOPT_LN_BOBYQA -w warped01noblur.jpg -s 11 -g blurredAnswer1.out -f Answer1.out -C 1.0  -D 1.0 


