# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.15

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "D:\CLion 2019.2\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "D:\CLion 2019.2\bin\cmake\win\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\Libraries\CLion\NumCPP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\Libraries\CLion\NumCPP\cmake-build-debug

# Include any dependencies generated for this target.
include google-tests/CMakeFiles/google_tests_run.dir/depend.make

# Include the progress variables for this target.
include google-tests/CMakeFiles/google_tests_run.dir/progress.make

# Include the compile flags for this target's objects.
include google-tests/CMakeFiles/google_tests_run.dir/flags.make

google-tests/CMakeFiles/google_tests_run.dir/testMat.cpp.obj: google-tests/CMakeFiles/google_tests_run.dir/flags.make
google-tests/CMakeFiles/google_tests_run.dir/testMat.cpp.obj: google-tests/CMakeFiles/google_tests_run.dir/includes_CXX.rsp
google-tests/CMakeFiles/google_tests_run.dir/testMat.cpp.obj: ../google-tests/testMat.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Libraries\CLion\NumCPP\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object google-tests/CMakeFiles/google_tests_run.dir/testMat.cpp.obj"
	cd /d D:\Libraries\CLion\NumCPP\cmake-build-debug\google-tests && C:\PROGRA~2\MINGW-~1\I686-8~1.0-P\mingw32\bin\G__~1.EXE  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\google_tests_run.dir\testMat.cpp.obj -c D:\Libraries\CLion\NumCPP\google-tests\testMat.cpp

google-tests/CMakeFiles/google_tests_run.dir/testMat.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/google_tests_run.dir/testMat.cpp.i"
	cd /d D:\Libraries\CLion\NumCPP\cmake-build-debug\google-tests && C:\PROGRA~2\MINGW-~1\I686-8~1.0-P\mingw32\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Libraries\CLion\NumCPP\google-tests\testMat.cpp > CMakeFiles\google_tests_run.dir\testMat.cpp.i

google-tests/CMakeFiles/google_tests_run.dir/testMat.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/google_tests_run.dir/testMat.cpp.s"
	cd /d D:\Libraries\CLion\NumCPP\cmake-build-debug\google-tests && C:\PROGRA~2\MINGW-~1\I686-8~1.0-P\mingw32\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\Libraries\CLion\NumCPP\google-tests\testMat.cpp -o CMakeFiles\google_tests_run.dir\testMat.cpp.s

# Object files for target google_tests_run
google_tests_run_OBJECTS = \
"CMakeFiles/google_tests_run.dir/testMat.cpp.obj"

# External object files for target google_tests_run
google_tests_run_EXTERNAL_OBJECTS =

google-tests/google_tests_run.exe: google-tests/CMakeFiles/google_tests_run.dir/testMat.cpp.obj
google-tests/google_tests_run.exe: google-tests/CMakeFiles/google_tests_run.dir/build.make
google-tests/google_tests_run.exe: lib/libgtestd.a
google-tests/google_tests_run.exe: lib/libgtest_maind.a
google-tests/google_tests_run.exe: lib/libgtestd.a
google-tests/google_tests_run.exe: google-tests/CMakeFiles/google_tests_run.dir/linklibs.rsp
google-tests/google_tests_run.exe: google-tests/CMakeFiles/google_tests_run.dir/objects1.rsp
google-tests/google_tests_run.exe: google-tests/CMakeFiles/google_tests_run.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\Libraries\CLion\NumCPP\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable google_tests_run.exe"
	cd /d D:\Libraries\CLion\NumCPP\cmake-build-debug\google-tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\google_tests_run.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
google-tests/CMakeFiles/google_tests_run.dir/build: google-tests/google_tests_run.exe

.PHONY : google-tests/CMakeFiles/google_tests_run.dir/build

google-tests/CMakeFiles/google_tests_run.dir/clean:
	cd /d D:\Libraries\CLion\NumCPP\cmake-build-debug\google-tests && $(CMAKE_COMMAND) -P CMakeFiles\google_tests_run.dir\cmake_clean.cmake
.PHONY : google-tests/CMakeFiles/google_tests_run.dir/clean

google-tests/CMakeFiles/google_tests_run.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\Libraries\CLion\NumCPP D:\Libraries\CLion\NumCPP\google-tests D:\Libraries\CLion\NumCPP\cmake-build-debug D:\Libraries\CLion\NumCPP\cmake-build-debug\google-tests D:\Libraries\CLion\NumCPP\cmake-build-debug\google-tests\CMakeFiles\google_tests_run.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : google-tests/CMakeFiles/google_tests_run.dir/depend
