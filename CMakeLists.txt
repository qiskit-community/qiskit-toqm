cmake_minimum_required(VERSION 3.15...3.19)

project(scikit_build_example VERSION "0.0.1")
project("qiskit-toqm"
        VERSION 0.1.0
        DESCRIPTION
        "Time-Optimal Qubit Mapping routing pass implementation for Qiskit based on paper \
        https://doi.org/10.1145/3445814.3446706"
        LANGUAGES CXX)

add_compile_options(-fno-omit-frame-pointer -g)

# Build the native code.
add_subdirectory(libtoqm)

if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE src/main.cpp)
target_link_libraries(_core PRIVATE toqm)

target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _core DESTINATION .)
