# TectonicSense Dataset Splitting

## Objective

The engineered TectonicSense dataset is divided into training,
validation, and test subsets for reproducible machine learning
experiments.

## Split Strategy

A stratified random split is used:

- Training: 70%
- Validation: 15%
- Test: 15%

Stratification is performed using `boundary_class` so that the
seven tectonic boundary classes remain proportionally represented
across the three subsets.

## Reproducibility

Random state:

`42`

Using a fixed random state ensures that the same dataset produces
the same split during repeated experiments.

## Evaluation Design

The stratified random split provides the primary baseline
evaluation.

A separate boundary-aware evaluation will be performed later
using `boundary_id` as a grouping variable. This experiment will
test whether the trained models generalize to previously unseen
tectonic boundaries.

`boundary_id` is not used as a predictive ML feature.

## Target

The prediction target is:

`boundary_class`

The seven classes are:

- CCB
- CRB
- CTF
- OCB
- OSR
- OTF
- SUB
