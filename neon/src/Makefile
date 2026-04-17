all: benchmark_driver

clean:
	rm -f benchmark_driver

benchmark_driver: benchmark_driver.cpp benchmark_kernel.s
	g++ $^ -o benchmark_driver


