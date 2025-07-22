# big thank you to dutchen and kris, who helped me get this working :)
set -e
nvcc -O3 -g -I src/include --device-c src/loot_library.cu -o loot_library.o -DUSE_CUDA
nvcc -O3 -g -I src/include --device-c src/loot_data.cu -o loot_data.o -DUSE_CUDA
nvcc -O3 -g -I src/include --device-c example.cu -o example.o -DUSE_CUDA
nvcc -O3 -g *.o cJSON/libcjson.a -o example 