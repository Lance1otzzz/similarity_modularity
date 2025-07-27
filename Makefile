CC=g++
#CC=clang++
CFLAGS = -O3 -Wall -Wno-sign-compare -Wextra 
#-static-libstdc++
DEBUGFLAGS = -O0 -g -Wall -Wextra -Wno-sign-compare -Ddebug

# 定义数据集路径和参数
DATASET = ./dataset/CiteSeer
RESOLUTION = 200

all: main

main: main.cpp graph.hpp defines.hpp louvain.hpp leiden.hpp louvain_heur.hpp pruning_alg/kmeans_preprocessing.hpp pruning_alg/kmeans_preprocessing.cpp pruning_alg/triangle_pruning.hpp pruning_alg/triangle_pruning.cpp pruning_alg/bipolar_pruning.hpp pruning_alg/bipolar_pruning.cpp
	$(CC) main.cpp kmeans_preprocessing.cpp triangle_pruning.cpp bipolar_pruning.cpp $(CFLAGS) -o main

pg: main.cpp graph.hpp defines.hpp louvain.hpp leiden.hpp louvain_heur.hpp pruning_alg/kmeans_preprocessing.hpp pruning_alg/kmeans_preprocessing.cpp pruning_alg/triangle_pruning.hpp pruning_alg/triangle_pruning.cpp pruning_alg/bipolar_pruning.hpp pruning_alg/bipolar_pruning.cpp
	$(CC) main.cpp kmeans_preprocessing.cpp triangle_pruning.cpp bipolar_pruning.cpp -pg -o main

test: test.cpp graph.hpp defines.hpp louvain.hpp leiden.hpp louvain_heur.hpp
	$(CC) test.cpp $(CFLAGS) -o test

debug: main.cpp graph.hpp defines.hpp louvain.hpp leiden.hpp louvain_heur.hpp pruning_alg/kmeans_preprocessing.hpp pruning_alg/kmeans_preprocessing.cpp pruning_alg/triangle_pruning.hpp pruning_alg/triangle_pruning.cpp pruning_alg/bipolar_pruning.hpp pruning_alg/bipolar_pruning.cpp
	$(CC) main.cpp kmeans_preprocessing.cpp triangle_pruning.cpp bipolar_pruning.cpp $(DEBUGFLAGS) -o debug

louvain: main
	./main 10 $(DATASET) $(RESOLUTION)

leiden: main
	./main 11 $(DATASET) $(RESOLUTION)

simple: louvain

compare_r:
	@echo "\nRunning pure_louvain Alg.($(DATASET),r=$(RESOLUTION)):"
	@./main 20 $(DATASET) $(RESOLUTION)
	@echo "Running Louvain Alg.($(DATASET),r=$(RESOLUTION)):"
	@./main 10 $(DATASET) $(RESOLUTION)


compare: main
	@echo "Running Louvain Alg.($(DATASET),r=$(RESOLUTION)):"
	@./main 10 $(DATASET) $(RESOLUTION)
	@echo "\nRunning Leiden Alg.($(DATASET),r=$(RESOLUTION)):"
	@./main 11 $(DATASET) $(RESOLUTION)
	@echo "\nRunning pure_louvain Alg.($(DATASET)):"
	@./main 20 $(DATASET) $(RESOLUTION)


clean:
	rm -f ./test ./main ./debug


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



.PHONY: all louvain leiden simple louvain_custom leiden_custom compare clean visualize full_run
