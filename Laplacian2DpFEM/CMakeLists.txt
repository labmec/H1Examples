
set(Laplacian2DpFEM_SRCS main.cpp)

add_executable(Laplacian2DpFEM ${Laplacian2DpFEM_SRCS} )

target_link_libraries(Laplacian2DpFEM pz)
target_include_directories(Laplacian2DpFEM PUBLIC ${PZ_INCLUDE_DIRS})