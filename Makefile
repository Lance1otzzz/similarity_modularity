CC=g++
#CC=clang++
CFLAGS = -O3 -Wall -Wno-sign-compare -Wextra -static-libstdc++
DEBUGFLAGS = -g -Wall -Wno-sign-compare -static-libstdc++ -Ddebug

# 定义数据集路径和参数
DATASET = ./dataset/simple
RESOLUTION = 200

all: main.cpp graph.hpp defines.hpp louvain.hpp leiden.hpp
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


compare: all
	@echo "Running Louvain Alg.($(DATASET),r=$(RESOLUTION)):"
	@./main 10 $(DATASET) $(RESOLUTION)
	@echo "\nRunning Leiden Alg.($(DATASET),r=$(RESOLUTION)):"
	@./main 11 $(DATASET) $(RESOLUTION)

clean:
	rm -f ./test ./main


#######python config#########
EXP_SCRIPT = run.py
VIS_SCRIPT = visualize_results.py
RESULTS_CSV = experiment_results_logscale.csv
PYTHON = python

$(RESULTS_CSV): $(EXP_SCRIPT) main
	@echo ">>> Running experiments to generate $(RESULTS_CSV)..."
	$(PYTHON) $(EXP_SCRIPT)

visualize: $(RESULTS_CSV) $(VIS_SCRIPT)
	@echo ">>> Generating visualizations from $(RESULTS_CSV)..."
	$(PYTHON) $(VIS_SCRIPT)

full_run: visualize
	@echo ">>> Full experiment run and visualization complete."



.PHONY: all test debug louvain leiden simple louvain_custom leiden_custom compare clean visualize full_run
