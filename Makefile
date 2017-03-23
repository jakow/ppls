MPIDIR=/usr/lib64/openmpi/bin
CC=$(MPIDIR)/mpicc
LD=$(MPIDIR)/mpicc
MPIRUN=$(MPIDIR)/mpirun

MAIN_SOURCES=aquadSolution.c stack.c
MAIN_OBJECTS=$(MAIN_SOURCES:.c=.o)

LDFLAGS=-lm

.PHONY: clean run

all :
	make clean
	make aquadSolution

aquadSolution: $(MAIN_OBJECTS)
	$(LD) $(LDFLAGS) $^ -o $@

%.o : %.c
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm -f $(MAIN_OBJECTS)

run:
	nice -n 10 $(MPIRUN) -c $(c) aquadSolution
