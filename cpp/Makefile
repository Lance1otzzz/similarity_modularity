CC=g++
#CC=clang++
CFLAGS = -O3 -Wall -Wno-sign-compare -Wextra -static-libstdc++ -std=c++20
DEBUGFLAGS = -Og -g -Wall -Wno-sign-compare -static-libstdc++ -std=c++20
all: main.cpp graph.hpp defines.hpp louvain.hpp
	$(CC) main.cpp $(CFLAGS) -o main
test: test.cpp graph.hpp defines.hpp louvain.hpp
	$(CC) test.cpp $(CFLAGS) -o test
debug: main.cpp graph.hpp defines.hpp louvain.hpp
	$(CC) main.cpp $(DEBUGFLAGS) -o main

simple:
	./main 1 /dataset/simple

clean:
	rm ./test
	rm ./main
