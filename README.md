# SHIZOZ_Labs
 AI in image processing tasks

To setup you need to:
Create build folder
```
mkdir build
cd build
```
Make OpenCV
```
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_EXAMPLES=OFF ../opencv
make -j7
```
If you are running on M1 mac or have issues with zlib on other system try to install zlib with homebrew with force link and add ```-DBUILD_ZLIB=OFF``` to cmake command

