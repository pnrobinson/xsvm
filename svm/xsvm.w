\def\covernote{Copyright 2015 Peter N Robinson -- Charit\'{e} Universit\"{a}tsmedizin Berlin}
\def\vbar{\.{|}}


@*xsvm: Introduction.

xsvm is a C library for support vector machine learning. There are already many excellent SVM packages out there, such as SVMlight and libsvm. xsvm was developed for use with string kernels and other self-made kernels. This document additionally supplies some background on support vector machines (SVM).

This document will contain the code for the entire program. For now, we will add documentation while porting and improving the existing code (to do)



@*Entry Point.
The following listing shows the imports and the main function.

@c
#include <stdlib.h>
#include <stdio.h>

@<functionprototypes@>

int main(int argc,char ** argv) {
    printf("xsvm\n");
    return 0;
}


@*Utility functions.
The following functions are not related to the SVM algorithms, but are needed to get things running.


@*Function prototypes.
This is simply a block of code that lists all of the function prototypes.
Maybe replace by a header file in order to allow unit testing.

@<functionprototypes@>=
void input_data(void*);
void baa(int );


@*Index.
Last section