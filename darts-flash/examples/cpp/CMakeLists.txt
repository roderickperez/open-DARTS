# Adds all examples. To do that, it is assumed that each .cpp file is an example and the
# example's name is the filename without the extension.

# Collect all the names of the cpp files
file(GLOB CPPFILES  "*.cpp")

# For every cpp found
foreach(CPPFILE ${CPPFILES})
	message(STATUS "${CPPFILE}")
	# Extract the actual file name
	get_filename_component(FILENAME ${CPPFILE} NAME)

	# Extract the filename minus the cpp. This will be the name of target
	get_filename_component(TARGETNAME_BASE ${CPPFILE} NAME_WE)
	
	# Generate the target name based on the base filename 
	set(TARGETNAME "${TARGETNAME_BASE}")

	# Make the target
	add_executable(${TARGETNAME} ${FILENAME})

	target_link_libraries(${TARGETNAME}
		darts-flash
  )

  	# Place the target example in the examples folder and rename the source folder to src
  	set_property(TARGET ${TARGETNAME} PROPERTY FOLDER "examples//cpp//")
  	source_group(src FILES ${FILENAME})

	# Install the example executable
	install(TARGETS ${TARGETNAME} DESTINATION bin/dartsflash/examples/)
endforeach()
