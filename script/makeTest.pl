#!/usr/bin/perl -w
use strict;
use POSIX;

## Simulate datasets for SVM training
## For now, simulate really easy (separable) dataset
## positive: 1-40
## negative: 41-80

## 100 examples each

for (my $i=0;$i<100;++$i) {
	print "+1";
	my %vals;
	for (my $j=0;$j<15;++$j) {
		my $x = ceil(rand(40));
		$vals{$x}=1;
	}
	foreach my $k (sort {$a<=>$b} keys %vals) {
		print "\t$k:1";
	}
	print "\n";
}

for (my $i=0;$i<100;++$i) {
	print "-1";
	my %vals;
	for (my $j=0;$j<15;++$j) {
		my $x = 10 + ceil(rand(40));
		$vals{$x}=1;
	}
	foreach my $k (sort {$a<=>$b} keys %vals) {
		print "\t$k:1";
	}
	print "\n";
}