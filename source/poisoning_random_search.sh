#!/bin/bash
# Executes the random search for the poisoning experiemnts, uses guild
tag=randsearch # guild tag
echomode=
maxtrials=100 # number of trials for the random search
nqueue=2 # number of guild queues used: execute each job in a different query

operations=(bioptdetwsv2 bioptwsno  bioptgammawsno bioptwsyes stochbiobs90 stochbio alsetwslinv2 alset)

for op in "${operations[@]}"
do
for n in $(seq 1 $nqueue)
do
  echo Y | $echomode guild run poisonsearch:$op --tag $tag --stage --max-trials $maxtrials;
done
done
