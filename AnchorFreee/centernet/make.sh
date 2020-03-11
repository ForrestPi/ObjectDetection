cd models/networks/DCNv2
sudo rm -r build
sudo chmod -R 777 make.sh
./make.sh
cd -
cd external/
make
echo 'ok'
