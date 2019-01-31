# brew install imagemagick
mkdir ./frames
ffmpeg -i output3.avi  -r 10 './frames/frame-%03d.jpg'
cd frames
convert -delay 20 -loop 0 *.jpg myimage.gif
mv myimage.gif ../
cd ..
rm -rf frames/