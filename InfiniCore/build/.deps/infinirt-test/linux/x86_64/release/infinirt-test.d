{
    files = {
        "build/.objs/infinirt-test/linux/x86_64/release/src/infinirt-test/main.cc.o",
        "build/.objs/infinirt-test/linux/x86_64/release/src/infinirt-test/test.cc.o",
        "build/linux/x86_64/release/libinfini-utils.a",
        "build/linux/x86_64/release/libinfinirt-cpu.a"
    },
    values = {
        "/usr/bin/g++",
        {
            "-m64",
            "-Lbuild/linux/x86_64/release",
            "-Wl,-rpath=$ORIGIN",
            "-s",
            "-linfinirt",
            "-linfinirt-cpu",
            "-linfini-utils",
            "-fopenmp"
        }
    }
}