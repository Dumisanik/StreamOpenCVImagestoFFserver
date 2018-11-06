PROJECT(simpleStreamer)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3)

IF(UNIX AND NOT WIN32 AND NOT APPLE)
        IF(EXISTS "${CMAKE_INSTALL_PREFIX}/lib64/" AND CMAKE_SIZEOF_VOID_P EQUAL 8 )
                set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS ON)
        ENDIF()
ENDIF(UNIX AND NOT WIN32 AND NOT APPLE)

# compiler flags
  #native is only supported by recent g++ compilers
  #SET(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -mtune=native -DNDEBUG")
  SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math -Wall -std=c++11")
  SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG -march=native -mtune=native -ffast-math -Wall -std=c++11")
  SET(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -std=c++11")
  SET(CMAKE_CXXFLAGS "-std=c++11")

 #set(CMAKE_C_COMPILER "-P" ""/usr/bin/gcc")
 #SET (CMAKE_C_COMPILER "/usr/bin/x86_64-linux-gnu-gcc")


#For FindFFmpeg.cmake
#We have some custom .cmake scripts not in the official cmake.
SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules;${CMAKE_MODULE_PATH}")

# === find FFmpeg ===
SET(FFmpeg_ROOT "" CACHE PATH "Path to search for custom FFmpeg library. Empty to use default paths.")
SET(FFmpeg_NO_SYSTEM_PATHS OFF CACHE BOOL "Search only FFmpeg_ROOT.")

    FIND_PACKAGE(FFmpeg)


    IF(FFmpeg_FOUND)
      INCLUDE_DIRECTORIES(${FFmpeg_INCLUDE_DIRS})
      LINK_DIRECTORIES(/home/dumisani/dev/FFmpeg/2.8.6/libswresample/)
      SET(THIRD_PARTY_LIBRARIES ${THIRD_PARTY_LIBRARIES}
          ${FFmpeg_LIBRARIES}
          swresample
      )


ENDIF()

MESSAGE("  FFmpeg_FOUND ${FFmpeg_FOUND}")

IF(NOT FFmpeg_FOUND)
  MESSAGE(FATAL_ERROR "Cannot find FFmpeg, set FFmpeg_ROOT.")
ENDIF(NOT FFmpeg_FOUND)
# ===================

SET(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
FIND_PACKAGE(CUDA 8.0 REQUIRED)
LINK_DIRECTORIES(/usr/local/cuda/lib64)
INCLUDE_DIRECTORIES(${INCLUDE_DIRECTORIES}/C/common/inc)



#===============================
# === find OpenCV ===
OPTION(USE_OPENCV "Compile code that uses the OpenCV library." ON)
IF(USE_OPENCV)
    SET(OpenCV_ROOT_DIR "/usr/local")
    find_package(OpenCV 3.0.0 REQUIRED COMPONENTS core highgui imgproc videoio)  #

    LINK_DIRECTORIES(/usr/local/lib)

    IF(OpenCV_FOUND)
        INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
        LINK_DIRECTORIES(${OpenCV_LIBRARY})


        ADD_DEFINITIONS(-DUSE_OPENCV)
        MESSAGE("OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")
        MESSAGE("OpenCV libraries: ${OpenCV_LIBRARIES}")

        SET(THIRD_PARTY_LIBRARIES ${THIRD_PARTY_LIBRARIES}
            opencv_core
            opencv_imgproc
            opencv_highgui
            opencv_imgcodecs
            opencv_videoio
        )
    ENDIF(OpenCV_FOUND)

ENDIF(USE_OPENCV)

SET(SOURCES
  main.cpp
)




SET(THIRD_PARTY_LIBRARIES ${THIRD_PARTY_LIBRARIES}
    pthread
)


ADD_EXECUTABLE(simpleStreamer ${SOURCES})

ADD_DEFINITIONS(-D_UNIX_)
ADD_DEFINITIONS(-D_LINUX_)

SET(PUREGEV_ROOT "/opt/pleora/ebus_sdk/Ubuntu-x86_64/")

#LINK_DIRECTORIES(/usr/local/cuda/lib64;/usr/local/cuda-7.5/targets/x86_64-linux/lib/)
TARGET_LINK_LIBRARIES(simpleStreamer -lstdc++ ${THIRD_PARTY_LIBRARIES}) #opencv_core opencv_imgproc opencv_highgui lboost


