mkdir -p build

make

if [ $? -eq 0 ]; then
    echo "compile successfully!"

    find . -maxdepth 1 -type f -executable -exec mv {} build/ \;

else
    echo "compile fail..."
    exit 1
fi