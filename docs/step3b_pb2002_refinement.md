# Step 3B — Refined PB2002 Earthquake Association

## Objective

Refine the initial earthquake-to-PB2002 boundary association by calculating the distance from each earthquake to the corresponding PB2002 boundary segment rather than using only the segment midpoint.

## Input Data

### USGS Earthquakes

- Source: USGS earthquake catalog
- Period: 2010–2025
- Minimum magnitude: 4.0
- Clean earthquake records: 239,757

### PB2002

- Source: PB2002 digital plate boundary model
- Boundary steps: 5,819
- Unique boundary IDs: 182
- Boundary classes: 7

Classes:

- CCB
- CRB
- CTF
- OCB
- OSR
- OTF
- SUB

The PB2002 `*` marker is represented separately using `orogen_flag` rather than creating additional target classes.

## Method

The refined association performs the following operations:

1. Load cleaned USGS earthquake records.
2. Parse PB2002 boundary segments.
3. Calculate boundary midpoint coordinates.
4. Build a BallTree using the midpoint coordinates.
5. Retrieve candidate PB2002 segments for each earthquake.
6. Calculate the great-circle distance from the earthquake location to the candidate boundary segment.
7. Select the candidate with the smallest calculated distance.
8. Attach PB2002 tectonic attributes to the earthquake record.

The current candidate search uses 20 PB2002 candidate segments per earthquake.

## Refined Dataset

Output:

`data/interim/earthquakes_pb2002_refined.csv`

Total associated earthquakes:

239,757

## Distance Statistics

| Statistic | Distance (km) |
|---|---:|
| Minimum | 0.000329 |
| 25th percentile | 27.845 |
| Median | 69.168 |
| Mean | 121.203 |
| 75th percentile | 141.213 |
| 90th percentile | 258.496 |
| 95th percentile | 363.641 |
| 99th percentile | 1005.740 |
| Maximum | 3668.260 |

## Boundary Class Distribution

| Class | Count |
|---|---:|
| SUB | 115,347 |
| CCB | 27,281 |
| CRB | 26,316 |
| OSR | 26,206 |
| CTF | 17,431 |
| OTF | 17,378 |
| OCB | 9,798 |

## Orogen Flag

| Orogen Flag | Count |
|---|---:|
| 0 | 176,839 |
| 1 | 62,918 |

## Comparison with Initial Association

The initial association used distance to the midpoint of the nearest PB2002 boundary step.

The refined association produced:

- Mean distance: 121.20 km
- Median distance: 69.17 km

compared with:

- Initial mean distance: 125.43 km
- Initial median distance: 71.94 km

The reduction is consistent with the refined segment-level distance calculation.

## Generated Artifacts

### Dataset

`data/interim/earthquakes_pb2002_refined.csv`

### Metrics

`results/metrics/refined_pb2002_association_metrics.json`

### Visualizations

`results/figures/refined_boundary_distance_distribution.png`

`results/figures/refined_boundary_distance_distribution_log.png`

## Scientific Note

The association represents spatial association between earthquake epicenters and PB2002 tectonic boundary geometry. It is not an earthquake prediction system.

The target variable for subsequent machine learning is the PB2002 boundary class:

CCB, CRB, CTF, OCB, OSR, OTF, SUB.

The `boundary_id` is retained for traceability but should not be used as a predictive feature because it can allow the model to memorize individual boundary segments.