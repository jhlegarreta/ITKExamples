cmake_minimum_required(VERSION 3.10.2)

project(ReadWriteVectorImage)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})


add_executable(ReadWriteVectorImage Code.cxx)
target_link_libraries(ReadWriteVectorImage ${ITK_LIBRARIES})

install(TARGETS ReadWriteVectorImage
  DESTINATION bin/ITKSphinxExamples/Core/Common
  COMPONENT Runtime
  )

install(FILES Code.cxx CMakeLists.txt Code.py
  DESTINATION share/ITKSphinxExamples/Code/Core/Common/ReadWriteVectorImage/
  COMPONENT Code
  )


enable_testing()
set(input_image ${CMAKE_CURRENT_BINARY_DIR}/mini-vector-fast.nrrd)
set(output_image Output.nrrd)

add_test(NAME ReadWriteVectorImageTest
  COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ReadWriteVectorImage
    ${input_image}
    ${output_image}
  )

if(ITK_WRAP_PYTHON)
  find_package(PythonInterp REQUIRED)
  string(REPLACE . "Python." output_image "${output_image}")
  add_test(NAME ReadWriteVectorImageTestPython
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Code.py
      ${input_image}
      ${output_image}
    )
endif()
