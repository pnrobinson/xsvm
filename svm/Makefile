CFLAGS = -g -O3 -std=gnu99
## Note the -stdgnu99 uses c99 with gnu extensions (gets us drand48)
LDLIBS= 
LDFLAGS=-lm 
CC=gcc

all: xsvm

OBJ = svm_util.o svm.o platt.o fan.o

xsvm: xsvm.c $(OBJ)
	$(CC) -o $@ xsvm.c $(OBJ) $(CFLAGS) $(LDFLAGS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(LDFLAGS)  $< -o $@

## github stuff
push:
	@if [ "x$(MSG)" = 'x' ]; then echo "Usage MSG='whatever' make push"; fi
	@test "x$(MSG)" != 'x'
	git commit -am"$(MSG)"
	git push


## unit testing with GLIB
GL=`pkg-config --cflags --libs gmodule-2.0`

test: test_xsvm.c $(OBJ)
	$(CC) test_xsvm.c $(OBJ) $(LDFLAGS) $(GL) -o $@ 



## doxygen
doc:
	doxygen doxygen.cfg


clean:
	-rm xsvm
	-rm *.o
	-rm *~
	-rm -rf html
	-rm -rf latex
	-rm test
