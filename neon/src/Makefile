all: benchmark_driver

clean:
	rm -f benchmark_driver

benchmark_driver: benchmark_driver.cpp benchmark_kernel.s
	g++ -O0 benchmark_driver.cpp -x assembler-with-cpp benchmark_kernel.s -o benchmark_driver


