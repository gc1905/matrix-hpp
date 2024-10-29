CC = g++
CFLAGS += -std=gnu++11
CFLAGS += -Wall 
CFLAGS += -Wextra

.PHONY: all app docs

all: examples tests

tests:
	$(CC) $(CFLAGS) -o tests.exe tests.cpp

examples:
	$(CC) $(CFLAGS) -o examples.exe examples.cpp

docs:
	doxygen Doxyfile
	-cd docs/latex && make pdf
	cp docs/latex/refman.pdf matrix_hpp.pdf

clean:
	rm -rf docs
	rm examples.exe
