# PR into sklearn with Jesse

## SPORF regression plan
1. Do experiments to determine the best split criteria
2. Write new split criteria in sklearn repo (note old splitters wont work because sklearn discards y's at leaf nodes)

## MORF
1. Write function to generate patches from data
2. Construct projection matrices from patches

## Benchmarking
1. Run sporf on openml cc18 and openml100 dataset suites
2. Try to make the code "plug and playable"
