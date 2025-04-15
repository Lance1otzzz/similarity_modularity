CC=g++
#CC=clang++
CFLAGS = -O3 -Wall -Wno-sign-compare -Wextra -static-libstdc++ -std=c++20
DEBUGFLAGS = -g -Wall -Wno-sign-compare -static-libstdc++ -std=c++20

# 定义数据集路径和参数
DATASET = ./dataset/cora
RESOLUTION = 20

all: main.cpp graph.hpp defines.hpp louvain.hpp
	$(CC) main.cpp $(CFLAGS) -o main

test: test.cpp graph.hpp defines.hpp louvain.hpp leiden.hpp
	$(CC) test.cpp $(CFLAGS) -o test

debug: main.cpp graph.hpp defines.hpp louvain.hpp leiden.hpp
	$(CC) main.cpp $(DEBUGFLAGS) -o main

louvain: all
	./main 10 $(DATASET) $(RESOLUTION)

leiden: all
	./main 11 $(DATASET) $(RESOLUTION)

simple: louvain

louvain_custom: all
	@read -p "请输入分辨率参数: " res; \
	./main 10 $(DATASET) $$res

leiden_custom: all
	@read -p "请输入分辨率参数: " res; \
	./main 11 $(DATASET) $$res

compare: all
	@echo "Running Louvain Alg.($(DATASET),r=$(RESOLUTION)):"
	@./main 10 $(DATASET) $(RESOLUTION)
	@echo "\nRunning Leiden Alg.($(DATASET),r=$(RESOLUTION)):"
	@./main 11 $(DATASET) $(RESOLUTION)

clean:
	rm -f ./test ./main

.PHONY: all test debug louvain leiden simple louvain_custom leiden_custom compare clean
