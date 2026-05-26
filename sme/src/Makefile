# 1. Gemeinsame Variablen und Flags zusammenfassen
CXXFLAGS_COMMON = -Wall -Wextra -O3 -march=armv9.2-a+nosve+sme+sme2
TOOLCHAIN ?=homebrew 

ifeq ($(TOOLCHAIN),standard)
    CXX = clang++
    CC = clang
    CXXFLAGS = $(CXXFLAGS_COMMON)
else ifeq ($(TOOLCHAIN),homebrew)
    LLVM_PREFIX = /opt/homebrew/opt/llvm
    CXX = $(LLVM_PREFIX)/bin/clang++
    CC = $(LLVM_PREFIX)/bin/clang
    CXXFLAGS = $(CXXFLAGS_COMMON) -fopenmp
else
    $(error Invalid toolchain. Select either "TOOLCHAIN=standard" or "TOOLCHAIN=homebrew")
endif

# ASFLAGS für Assembler und LDFLAGS für den Linker (wichtig für -fopenmp)
ASFLAGS = $(CXXFLAGS) -x assembler-with-cpp
LDFLAGS = $(CXXFLAGS)

# 2. Alle Ziel-Ausführbaren in einer Variable speichern
TARGETS = identity_driver relu_driver zero_driver unary_benchmarks_driver unary_tests \
          gemm_driver_32_32_1 gemm_driver_32_32_512 gemm_driver_16_16 \
          gemm_driver_512_32_512 gemm_driver_512_512_512 gemm_driver_16_16_multi \
          sve_stats gemm_driver_M_N_K_16th

.PHONY: all gemm clean

all: $(TARGETS)

gemm: gemm_driver_512_512_512 gemm_driver_M_N_K_16th

# 3. Clean-Regel drastisch vereinfachen
clean:
	rm -f $(TARGETS) *.o

# ==========================================
# 4. Pattern Rules (Musterregeln) für Kompilierung
# Ersetzt alle manuellen .cpp -> .o und .s -> .o Regeln
# ==========================================
%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

%.o: %.s
	$(CXX) -c $(ASFLAGS) $< -o $@

# ==========================================
# 5. Linker-Regeln (Abhängigkeiten definieren)
# ==========================================
sve_stats: sve_stats.o
	$(CXX) $(LDFLAGS) $^ -o $@

identity_driver: identity_driver.o identity_kernel.o
	$(CXX) $(LDFLAGS) $^ -o $@

relu_driver: relu_driver.o relu_kernel.o identity_kernel.o
	$(CXX) $(LDFLAGS) $^ -o $@

zero_driver: zero_driver.o zero_kernel.o
	$(CXX) $(LDFLAGS) $^ -o $@

unary_benchmarks_driver: unary_benchmarks_driver.o identity_kernel.o zero_kernel.o relu_kernel.o
	$(CXX) $(LDFLAGS) $^ -o $@

unary_tests: unary_tests.o identity_kernel.o zero_kernel.o relu_kernel.o
	$(CXX) $(LDFLAGS) $^ -o $@

gemm_driver_32_32_1: gemm_driver_32_32_1.o gemm_32_32_1_kernel.o
	$(CXX) $(LDFLAGS) $^ -o $@

gemm_driver_32_32_512: gemm_driver_32_32_512.o gemm_32_32_512_kernel.o
	$(CXX) $(LDFLAGS) $^ -o $@

gemm_driver_16_16: gemm_driver_16_16.o gemm_16_16_kernel.o
	$(CXX) $(LDFLAGS) $^ -o $@

# Hinweis: In der Original-Makefile fehlte die Link-Regel für multi, hier ergänzt (angepasst annehmen):
gemm_driver_16_16_multi: gemm_driver_16_16_multi.o gemm_16_16_kernel.o
	$(CXX) $(LDFLAGS) $^ -o $@

gemm_driver_512_32_512: gemm_driver_512_32_512.o gemm_512_32_512_kernel.o
	$(CXX) $(LDFLAGS) $^ -o $@

gemm_driver_512_512_512: gemm_driver_512_512_512.o gemm_512_512_512_kernel.o
	$(CXX) $(LDFLAGS) $^ -o $@

gemm_driver_M_N_K_16th: gemm_driver_M_N_K_16th.o gemm_M_N_K_16th_kernel.o
	$(CXX) $(LDFLAGS) $^ -o $@