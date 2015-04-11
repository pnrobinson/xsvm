CFLAGS = -g -Wall -O3
LDLIBS=
CC=c99

all: xsvm.pdf

xsvm.pdf: xsvm.w
	cweave xsvm.w
	pdftex xsvm.tex


xsvm.c: xsvm.w
	ctangle xsvm.w


xsvm: xsvm.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ xsvm.c

## github stuff
push:
	@if [ "x$(MSG)" = 'x' ]; then echo "Usage MSG='whatever' make push"; fi
	@test "x$(MSG)" != 'x'
	git commit -am"$(MSG)"
	git push
