CC=g++
#CC=clang++
all: main.cpp graph.hpp defines.hpp
	$(CC) main.cpp -O3 -Wall -o main
test: test.cpp graph.hpp defines.hpp
	$(CC) test.cpp -O3 -Wall -o test
debug: main.cpp graph.hpp defines.hpp
	$(CC) main.cpp -Og -g -Wall -o main

clean:
	rm ./test
	rm ./main
