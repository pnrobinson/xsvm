CFLAGS = -g -Wall -O3
LDLIBS=
CC=c99

all: xsvm

OBJ = svm_util.o svm.o

xsvm: xsvm.c $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ xsvm.c $(OBJ)



## github stuff
push:
	@if [ "x$(MSG)" = 'x' ]; then echo "Usage MSG='whatever' make push"; fi
	@test "x$(MSG)" != 'x'
	git commit -am"$(MSG)"
	git push


## unit testing with GLIB
GLIBI=`pkg-config --cflags glib-2.0`
GLIBL=`pkg-config --libs glib-2.0`
test: test_xsvm.c $(OBJ)
	#echo $(GLIBI)
	$(CC) -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include $(LDFLAGS) -lglib-2.0  -o $@ test_xsvm.c $(OBJ)



## doxygen
doc:
	doxygen doxygen.cfg


clean:
	-rm xsvm
	-rm *.o
