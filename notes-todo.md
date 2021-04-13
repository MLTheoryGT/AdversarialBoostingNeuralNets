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
    
# Jan 9 2021
- TODO
    * get mnist to run
        * Expected to be much faster (since validation is quicker)
        * Give the code another readthrough (see how we can make it more versatile)

# Jan 10 2021
- What we did
    * created a LoggerMetric to keep track of accuracies / losses and for plotting functions for neural nets / ensembles
    * Made the boosting function return an Ensemble (which inherits from LoggerMetric) rather than returning lists of accuracies
- TODO
    * Test the scripts with adversarially robust weak learner
    * Figure out how to adversarially attack an ensemble
    
# Jan 26 2021
- Hypothesis: correlation between the amount the ensemble's acc increases as we add weak learners and the distance between train and val for each weak learner

# Feb 9
- Looks like boosting with 2 WL increases the accuracy pretty fast but not much after that
- It seems that the ensemble will beat the regularly trained nerual networks by a few percent each time.
- We are decaying the alpha value to see if we can close the test train gap for ensemble.


# Mar 15
- Currently retesting ensemble on eps [0.01, 0.02, 0.03] on fgsm to compare to attacks on weak learners (larger epsilons don't make sense)
    * Will tell us if the results for the non-adversarial case hold for the adversarial case
    * Should see if they hold for PGD
- Change the ensemble.plot_adv or whatev to make sure it can handle multiple attacks at once

# Mar 17
- Perhaps our ensemble should always take the peak of each WL rather than just the most current version of the WL
- Currently testing the nonadversarially trained ensemble on [0.01, 0.02, 0.03] to see if that outperforms adversarial ensemble interestingly.

# Mar 22
- The results show us that ensembling the non-adversarial nets still increases their accuracy, but not always more than the adversarially trained nets.
- main insight: the non-adversarial nets increase accuracy more monotonically than adv
- IDEA: make an ensemble which starts out with some adversarially trained nets then goes to non adv nets
- to hit neurips deadline we should start implementing shafahi and regular pgd training

# April 13
- Notes from meeting: more random restarts (instead of just 1), smaller architecture for weak learners
- Schapire update: Use targeted attacks, or treat a perturbation to an incorrect label as a perturbation to all other labels (pessimistc approach)
- Testing the model w/ pgd and more random restarts
- Wrote code for the pessimistic approach, and wrote a unit test for it in testing_notebook.ipynb (to be run at a later time)