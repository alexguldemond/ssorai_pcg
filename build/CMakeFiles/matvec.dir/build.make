# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake/bin/cmake

# The command to remove a file.
RM = /opt/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aguldemo/Code/pcg

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aguldemo/Code/pcg/build

# Include any dependencies generated for this target.
include CMakeFiles/matvec.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/matvec.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matvec.dir/flags.make

CMakeFiles/matvec.dir/src/MatVec.cu.o: CMakeFiles/matvec.dir/flags.make
CMakeFiles/matvec.dir/src/MatVec.cu.o: ../src/MatVec.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aguldemo/Code/pcg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/matvec.dir/src/MatVec.cu.o"
	/usr/local/cuda-9.2/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/aguldemo/Code/pcg/src/MatVec.cu -o CMakeFiles/matvec.dir/src/MatVec.cu.o

CMakeFiles/matvec.dir/src/MatVec.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/matvec.dir/src/MatVec.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/matvec.dir/src/MatVec.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/matvec.dir/src/MatVec.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/matvec.dir/src/MatVec.cu.o.requires:

.PHONY : CMakeFiles/matvec.dir/src/MatVec.cu.o.requires

CMakeFiles/matvec.dir/src/MatVec.cu.o.provides: CMakeFiles/matvec.dir/src/MatVec.cu.o.requires
	$(MAKE) -f CMakeFiles/matvec.dir/build.make CMakeFiles/matvec.dir/src/MatVec.cu.o.provides.build
.PHONY : CMakeFiles/matvec.dir/src/MatVec.cu.o.provides

CMakeFiles/matvec.dir/src/MatVec.cu.o.provides.build: CMakeFiles/matvec.dir/src/MatVec.cu.o


# Object files for target matvec
matvec_OBJECTS = \
"CMakeFiles/matvec.dir/src/MatVec.cu.o"

# External object files for target matvec
matvec_EXTERNAL_OBJECTS =

CMakeFiles/matvec.dir/cmake_device_link.o: CMakeFiles/matvec.dir/src/MatVec.cu.o
CMakeFiles/matvec.dir/cmake_device_link.o: CMakeFiles/matvec.dir/build.make
CMakeFiles/matvec.dir/cmake_device_link.o: CMakeFiles/matvec.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aguldemo/Code/pcg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/matvec.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matvec.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matvec.dir/build: CMakeFiles/matvec.dir/cmake_device_link.o

.PHONY : CMakeFiles/matvec.dir/build

# Object files for target matvec
matvec_OBJECTS = \
"CMakeFiles/matvec.dir/src/MatVec.cu.o"

# External object files for target matvec
matvec_EXTERNAL_OBJECTS =

../bin/matvec: CMakeFiles/matvec.dir/src/MatVec.cu.o
../bin/matvec: CMakeFiles/matvec.dir/build.make
../bin/matvec: CMakeFiles/matvec.dir/cmake_device_link.o
../bin/matvec: CMakeFiles/matvec.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aguldemo/Code/pcg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable ../bin/matvec"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matvec.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matvec.dir/build: ../bin/matvec

.PHONY : CMakeFiles/matvec.dir/build

CMakeFiles/matvec.dir/requires: CMakeFiles/matvec.dir/src/MatVec.cu.o.requires

.PHONY : CMakeFiles/matvec.dir/requires

CMakeFiles/matvec.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/matvec.dir/cmake_clean.cmake
.PHONY : CMakeFiles/matvec.dir/clean

CMakeFiles/matvec.dir/depend:
	cd /home/aguldemo/Code/pcg/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aguldemo/Code/pcg /home/aguldemo/Code/pcg /home/aguldemo/Code/pcg/build /home/aguldemo/Code/pcg/build /home/aguldemo/Code/pcg/build/CMakeFiles/matvec.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/matvec.dir/depend

