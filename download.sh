path="./preprocessing/data/models/"
mkdir -p $path
cd $path

# Download pre-trained model of sdps-net
for model in "LCNet_CVPR2019.pth.tar" "NENet_CVPR2019.pth.tar"; do
    wget http://www.visionlab.cs.hku.hk/data/SDPS-Net/models/${model}
done

# Back to root directory
cd ../../../

# Download pre-trained model of stage I and II
wget http://www.visionlab.cs.hku.hk/data/psnerf/data.tgz
tar -xzvf data.tgz
rm data.tgz

# Download dataset
wget http://www.visionlab.cs.hku.hk/data/psnerf/dataset.tgz
tar -xzvf dataset.tgz
rm dataset.tgz

# Download envmap
cd stage2
wget http://www.visionlab.cs.hku.hk/data/psnerf/envmap.tgz
tar -xzvf envmap.tgz
rm envmap.tgz
cd ..