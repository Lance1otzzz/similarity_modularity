CC=g++
#CC=clang++
CFLAGS = -std=c++17 -O3 -Wall -Wno-sign-compare -Wextra
DEBUGFLAGS = -std=c++17 -O0 -g -Wall -Wextra -Wno-sign-compare -Ddebug

# 定义源文件和头文件
SRCS = main.cpp \
       pruning_alg/kmeans_preprocessing.cpp \
       pruning_alg/triangle_pruning.cpp \
       pruning_alg/bipolar_pruning.cpp \
       pruning_alg/s0_fast_kmeans.cpp \
       pruning_alg/s1_autok_lite.cpp

HEADERS = graph.hpp \
          defines.hpp \
          louvain.hpp \
          leiden.hpp \
          louvain_heur.hpp \
          pruning_alg/kmeans_preprocessing.hpp \
          pruning_alg/triangle_pruning.hpp \
          pruning_alg/bipolar_pruning.hpp \
          louvain_plus.hpp \
          louvain_pruning.hpp \
          louvain_pp.hpp \
          pruning_alg/s0_fast_kmeans.hpp \
          pruning_alg/s1_autok_lite.hpp \
          pruning_alg/fast_clustering_lib.hpp

# 定义可执行文件名
TARGET = main

# 定义数据集路径和参数
DATASET = ./dataset/CiteSeer
RESOLUTION = 200

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(SRCS) $(CFLAGS) -o $(TARGET)

pg: $(SRCS) $(HEADERS)
	$(CC) $(SRCS) -pg -o $(TARGET)

debug: $(SRCS) $(HEADERS)
	$(CC) $(SRCS) $(DEBUGFLAGS) -o debug

test: test.cpp graph.hpp defines.hpp louvain.hpp leiden.hpp louvain_heur.hpp
	$(CC) test.cpp $(CFLAGS) -o test

louvain: $(TARGET)
	./$(TARGET) 10 $(DATASET) $(RESOLUTION)

leiden: $(TARGET)
	./$(TARGET) 11 $(DATASET) $(RESOLUTION)

simple: louvain

compare_r:
	@echo "\nRunning pure_louvain Alg.($(DATASET),r=$(RESOLUTION)):"
	@./$(TARGET) 20 $(DATASET) $(RESOLUTION)
	@echo "Running Louvain Alg.($(DATASET),r=$(RESOLUTION)):"
	@./$(TARGET) 10 $(DATASET) $(RESOLUTION)

compare: $(TARGET)
	@echo "Running Louvain Alg.($(DATASET),r=$(RESOLUTION)):"
	@./$(TARGET) 10 $(DATASET) $(RESOLUTION)
	@echo "\nRunning Leiden Alg.($(DATASET),r=$(RESOLUTION)):"
	@./$(TARGET) 11 $(DATASET) $(RESOLUTION)
	@echo "\nRunning pure_louvain Alg.($(DATASET)):"
	@./$(TARGET) 20 $(DATASET) $(RESOLUTION)

clean:
	rm -f ./test ./$(TARGET) ./debug

# ---------- Optional library-backed build (Eigen + OpenCV) ----------
# These targets do NOT change the default build; they compile a variant that
# links Eigen and OpenCV so you can call the lib-backed S0 from your code.

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null)
EIGEN_CFLAGS  := $(shell pkg-config --cflags eigen3 2>/dev/null)

# Build an alternate binary with lib-based fast clustering available
TARGET_FASTLIB = main_fastlib
SRCS_FASTLIB = $(SRCS) pruning_alg/fast_clustering_lib.cpp

fastlib:
	$(CC) $(SRCS_FASTLIB) $(CFLAGS) -DFASTCL_USE_EIGEN -DFASTCL_USE_OPENCV $(EIGEN_CFLAGS) $(OPENCV_CFLAGS) -o $(TARGET_FASTLIB) $(OPENCV_LIBS)


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
