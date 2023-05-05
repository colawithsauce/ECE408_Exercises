cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1
cmake --build build
cp build/compile_commands.json .
cd build/test && ctest
