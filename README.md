Before running the script, download the weights for InceptionV3 first.

wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz

tar -xvzf inception_v3_2016_08_28.tar.gz

after downloading, run the following:

nvidia-docker run \
-v <LOCAL_INPUT>:/input_images \
-v <LOCAL_OUTPUT>:/output_images \
-v <THIS_REPOSITORY>:/code \
-w /code goodfellow/competition:gpu ./run_attack.sh /input_images /output_images 16
