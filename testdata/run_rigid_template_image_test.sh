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
../build.debug/bin/template_nlopt_rigid   -i image_0001.jpg  -m image_mask.jpg  -t image_0000.jpg   -K camera_droidx_640x480.yml  -a NLOPT_LN_BOBYQA -w warped01.jpg
../build.debug/bin/template_nlopt_rigid   -i image_0002.jpg  -m image_mask.jpg  -t image_0000.jpg   -K camera_droidx_640x480.yml  -a NLOPT_LN_BOBYQA -w warped02.jpg
