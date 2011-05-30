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

# nominal template 
template_image_rigid  -K camera_droidx_640x480.yml  -i image_0001.jpg  -m image_mask.jpg  -t image_0000.jpg -a NLOPT_LN_BOBYQA -w warped01blur1.jpg -s 25 -f blurredAnswer1.out   #implicit -F 1.0

# 1.25 scale => zoomed in, larger template 
template_image_rigid  -K camera_droidx_640x480.yml -i image_0001.jpg  -m image_mask.jpg  -t image_0000.jpg   -F 1.25  -a NLOPT_LN_BOBYQA -w warped01blur2.jpg -s 25 -f blurredAnswer2.out

# 0.75 scale => zoomed out, smaller template
template_image_rigid  -K camera_droidx_640x480.yml -i image_0001.jpg  -m image_mask.jpg  -t image_0000.jpg   -F 0.75  -a NLOPT_LN_BOBYQA -w warped01blur3.jpg -s 25 -f blurredAnswer3.out
