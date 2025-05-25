CC = g++
CFLAGS += -std=gnu++11
CFLAGS += -Wall 
CFLAGS += -Wextra
CFLAGS += -I.

.PHONY: examples tests docs

tests:
	$(CC) $(CFLAGS) -I./tests -o tests.exe tests/tests.cpp

examples:
	$(CC) $(CFLAGS) -o examples.exe examples/examples.cpp

docs:
	doxygen docs/Doxyfile
	-cd docs/latex && make pdf
	cp docs/latex/refman.pdf docs/matrix_hpp.pdf

clean:
	-rm -rf docs/latex
	-rm -f examples.exe
	-rm -f tests.exe
