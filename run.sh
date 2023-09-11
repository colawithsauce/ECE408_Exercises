set -e
cmake -G "Ninja" -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1
cmake --build build
cp build/compile_commands.json .
cd build/test && ctest --return-failed --output-on-failure
