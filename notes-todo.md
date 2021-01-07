We can use this to log what we have done / things to keep in mind for future reference

# Dec 8 2020

- print out test accuracy of ensemble after every iteration

- modify SchapireWongMulticlassBoosting to actually return the list of val accuracies

- uncommented the generation of val_accuracies, loss, etc.. in BoostedWongCIFAR10 (must've forgotten to uncomment it earlier)

- added RunMemlessCIFARBoosting to Boosting.py (so i can use it from other notebooks / files) -- returns wl, wlweights, val_accuracies

- Plots to have:
    - accuracy over number of training iterations for one weak learner
    - accuracy over number of weak learners
    - compare this with other ensemble nn methods
    
# Dec 9 2020

Possible ways to extend this:

- n_1 weak learners on k_1 iterations, n_2 weak learners on k_2 iterations (can use a large number of weak learners
for each experiment and use a prefix)

# Dec 13 2020

- add fake weak learner tests
- changed dataset size to 50000 -- should probably add this as a parameter in the future
- need to debug this further


# Dec 15 2020
- added functionality for maxSamples(need to test)
- made plots x axis numSamples(need to test)

# Dec 17 2020
- update the overleaf to reflect current experiment results

# Dec 18 2020
- todo: add folder called cifar10_data

# Dec 24 2020
- Run boosting on the original Wong neural net (and try to compare how it performs with respect to the original nn)

# Jan 7 2021
- What we did
    * Discovered that we were calling adversarial attacks all the time
        * Cleaned up / generalized some validation functions
    * Made batchUpdates return loss
- TODO
    * Update CIFAR10 code to reflect validation
    * Test the validation
    * Run experiments with adv = True (phase 2)
