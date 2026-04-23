set(TARGET llama-cli)
add_executable(${TARGET} cli.cpp)
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_link_options(llama-cli PRIVATE
    -static-libstdc++
    -static-libgcc
)

include_directories(../server)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
