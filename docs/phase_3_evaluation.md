# TectonicSense — Phase 3 Evaluation & Explainability

## 1. Objective
The objective of **Phase 3 — Final Evaluation, Explainability & Scientific Validation** is to critically assess the candidate machine learning pipeline (`Tuned XGBoost`) established in Phase 2. Specifically, Phase 3 addresses:
1. **Unbiased Generalization**: Evaluating the serialized candidate model strictly on the untouched test split (35,964 observations).
2. **Model Explainability**: Uncovering the physical decision logic using SHAP (TreeExplainer) on 2,000 stratified test observations.
3. **Scientific Investigation of High Performance**: Dissecting whether near-perfect metrics (~99.98% Macro F1) stem from seismic dynamics, geographic coordinates, or PB2002 boundary kinematic metadata.
4. **Boundary-Aware Generalization**: Testing true out-of-boundary performance using a group-disjoint split on PB2002 `boundary_id`.
5. **Empirical Error Analysis**: Diagnosing every misclassification on the held-out test dataset.

> [!IMPORTANT]
> **Scientific Scope Framing**: TectonicSense performs multiclass classification of tectonic plate-boundary regimes associated with seismic and geospatial observations. It is NOT an earthquake prediction or earthquake forecasting system.

---

## 2. Final Test Evaluation
The serialized Phase 2 candidate model (`models/phase_2_best_candidate/best_candidate_model.joblib`) was evaluated on `data/processed/splits/test.csv` (35,964 observations, held out untouched until this step).

### Global Benchmark Performance
- **Test Observations**: 35,964
- **Correct Classifications**: 35,960
- **Misclassifications**: 4 (0.0111% error rate)
- **Test Accuracy**: **0.999889** (99.99%)
- **Test Macro Precision**: **0.999821**
- **Test Macro Recall**: **0.999761**
- **Test Macro F1 Score**: **0.999791** (~0.9998)
- **Test Weighted Precision**: **0.999889**
- **Test Weighted Recall**: **0.999889**
- **Test Weighted F1 Score**: **0.999889**
- **Inference Latency**: 0.4962 seconds total (13.80 microseconds per observation)

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|:------|:----------|:-------|:---------|:--------|
| **CCB** | 0.999512 | 1.000000 | 0.999756 | 4,092 |
| **CRB** | 1.000000 | 1.000000 | 1.000000 | 3,948 |
| **CTF** | 1.000000 | 1.000000 | 1.000000 | 2,614 |
| **OCB** | 1.000000 | 0.998639 | 0.999319 | 1,470 |
| **OSR** | 1.000000 | 0.999746 | 0.999873 | 3,931 |
| **OTF** | 0.999233 | 1.000000 | 0.999617 | 2,607 |
| **SUB** | 1.000000 | 0.999942 | 0.999971 | 17,302 |

---

## 3. Confusion Matrix Analysis
The empirical confusion matrix on all 35,964 test observations shows near-perfect diagonal dominance:

```text
True \ Pred    CCB    CRB    CTF    OCB    OSR    OTF    SUB
CCB           4092      0      0      0      0      0      0
CRB              0   3948      0      0      0      0      0
CTF              0      0   2614      0      0      0      0
OCB              1      0      0   1468      0      1      0
OSR              0      0      0      0   3930      1      0
OTF              0      0      0      0      0   2607      0
SUB              1      0      0      0      0      0  17301
```

Out of 35,964 instances, exactly 4 transitions occurred off-diagonal:
1. One `OCB` misclassified as `CCB`
2. One `OCB` misclassified as `OTF`
3. One `OSR` misclassified as `OTF`
4. One `SUB` misclassified as `CCB`

---

## 4. SHAP Explainability
Global feature attribution was calculated using `shap.TreeExplainer` on a representative sample of 2,000 stratified test observations (`random_state=42`).

### Global Feature Importance Ranking (Mean Absolute SHAP Across All 7 Classes)
| Rank | Feature | Mean \|SHAP\| | Primary Geophysical Nature |
|:-----|:--------|:--------------|:---------------------------|
| 1 | `divergent_component` | 2.031758 | PB2002 Plate Kinematics (Normal Spreading/Convergence) |
| 2 | `boundary_elevation_m` | 1.445122 | PB2002 Geophysical Bathymetry/Topography |
| 3 | `absolute_divergent_component` | 0.877478 | PB2002 Kinematic Resultant Magnitude |
| 4 | `seafloor_age_ma` | 0.408205 | PB2002 Oceanic Lithospheric Age |
| 5 | `absolute_right_lateral_component` | 0.389146 | PB2002 Transform Shear Magnitude |
| 6 | `relative_velocity` | 0.262591 | PB2002 Relative Plate Velocity Rate |
| 7 | `tectonic_motion_magnitude` | 0.220113 | PB2002 Total Motion Vector Resultant |
| 8 | `boundary_length_km` | 0.159426 | PB2002 Boundary Segment Geometry |
| 9 | `velocity_azimuth_cos` | 0.152531 | PB2002 Motion Direction Cosine |
| 10 | `boundary_azimuth_sin` | 0.138005 | PB2002 Boundary Strike Sine |
| 11 | `latitude_cos` | 0.088718 | Spatial Coordinate Cyclical Encoding |
| 12 | `boundary_distance_km` | 0.081492 | Epicentral Boundary Proximity |
| 13 | `longitude_sin` | 0.076219 | Spatial Coordinate Cyclical Encoding |
| 14 | `depth_km` | 0.063412 | Earthquake Focal Depth |
| ... | ... | ... | ... |
| 25 | `magnitude` | 0.009142 | Earthquake Energy Scale |
| 26 | `magnitude_squared` | 0.004121 | Earthquake Non-linear Energy Scale |

### Key Explainability Findings
- **Kinematics Dominate the Decision Boundary**: `divergent_component` (mean \|SHAP\|: 2.0318) and `boundary_elevation_m` (1.4451) are by far the strongest drivers. A large negative divergent component pushes log-odds strongly toward Subduction (`SUB`), while positive divergence with negative elevation drives Oceanic Spreading Ridge (`OSR`).
- **Seismic Variables Have Minimal Attribution**: Raw earthquake magnitude (`magnitude`: 0.0091) and focal depth (`depth_km`: 0.0634) rank far below kinematic metadata.

---

## 5. Feature Group Ablation Study
To investigate predictive dependence across feature sources, four controlled XGBoost models were trained and tested on the exact same train/validation/test splits:

| Feature Group | Features | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Test Weighted F1 | Training Time |
|:--------------|:---------|:--------------|:---------------------|:------------------|:--------------|:-----------------|:--------------|
| **Group 1: Seismic Only** | 4 | 0.508926 | 0.333578 | 0.207562 | **0.202823** | 0.406156 | 19.07s |
| **Group 2: Seismic + Geographic** | 12 | 0.953426 | 0.930495 | 0.924513 | **0.927289** | 0.953127 | 26.78s |
| **Group 3: Limited Tectonic** | 21 | 0.999750 | 0.999630 | 0.999502 | **0.999566** | 0.999750 | 22.15s |
| **Group 4: Full Feature Set** | 26 | 0.999889 | 0.999821 | 0.999761 | **0.999791** | 0.999889 | 20.26s |

> [!NOTE]
> **Interpretation**: This demonstrates predictive dependence, not direct physical causality. Seismic parameters alone cannot resolve boundary regime (Macro F1 = 0.2028). Geographic location and proximity establish strong spatial bounds (Macro F1 = 0.9273). Adding kinematic velocity and strike metadata resolves remaining regime ambiguities, reaching 0.9998 Macro F1.

---

## 6. PB2002 Metadata Ablation
Comparing Group 4 (Full Feature Set) with Group 2 (Direct PB2002 metadata removed: velocity, azimuth, divergence, shear, elevation, seafloor age):

| Condition | Feature Count | Test Accuracy | Test Macro F1 |
|:----------|:--------------|:--------------|:--------------|
| **Full Feature Set (With PB2002 Kinematics)** | 26 | **0.999889** | **0.999791** |
| **PB2002 Metadata Removed (Seismic + Geographic Only)** | 12 | **0.953426** | **0.927289** |

Removing direct PB2002 boundary kinematic metadata causes test Macro F1 to drop from **0.9998** to **0.9273**, demonstrating that while spatial coordinates alone achieve substantial predictive power (92.7%), the near-perfect performance in Phase 2 is directly associated with PB2002 kinematic attributes.

---

## 7. Boundary-Aware Generalization (Disjoint Boundary Split)
In this experiment, earthquakes were split strictly at the `boundary_id` level:
- `boundary_id` was strictly a grouping variable and **NEVER used as a model feature**.
- 127 PB2002 boundaries (145,287 observations) were assigned to training.
- 27 PB2002 boundaries (53,380 observations) were assigned to validation.
- 28 PB2002 boundaries (41,090 observations) were assigned to testing.
- Overlap across all splits: **EXACTLY ZERO** (verified disjoint).

### Out-of-Boundary Test Results
- **Boundary-Aware Test Accuracy**: **0.7994** (79.94%)
- **Boundary-Aware Test Macro Precision**: **0.7989**
- **Boundary-Aware Test Macro Recall**: **0.8078**
- **Boundary-Aware Test Macro F1 Score**: **0.7935** (79.35%)
- **Boundary-Aware Test Weighted F1 Score**: **0.8005**

### Random Stratified Split vs. Boundary-Aware Split
| Evaluation Split Protocol | Split Nature | Test Accuracy | Test Macro F1 |
|:--------------------------|:-------------|:--------------|:--------------|
| **Random Stratified Split** | Observations from same boundaries co-occur in train & test | **0.9999** | **0.9998** |
| **Boundary-Aware Disjoint Split** | Test boundaries are completely unseen during training | **0.7994** | **0.7935** |

### Scientific Significance
This performance difference (~0.9998 vs ~0.7935) is scientifically crucial. Under random splitting, spatial and kinematic autocorrelation along continuous fault traces allows tree ensembles to recognize local boundary segments. However, when tested on completely unseen tectonic boundaries across the globe, the model demonstrates genuine physical transferability with ~79.4% Macro F1.

---

## 8. Empirical Error Analysis
Detailed inspection of all 4 misclassified observations in the final test set:

1. **`us20009vsi`** (True: `OCB` -> Predicted: `CCB`):
   - $M 5.9$, focal depth $545.0$ km, boundary distance $549.6$ km, confidence $0.9817$.
   - A deep mantle earthquake occurring more than 500 km away from the nearest boundary segment in a complex transitional collision zone.
2. **`us6000pzw2`** (True: `OSR` -> Predicted: `OTF`):
   - $M 5.2$, focal depth $10.0$ km, boundary distance $1.2$ km, confidence $0.9794$.
   - Located at the exact geometric junction of an oceanic ridge (OSR) and an offset oceanic transform fault (OTF).
3. **`us1000e7ye`** (True: `OCB` -> Predicted: `OTF`):
   - $M 4.6$, focal depth $10.0$ km, boundary distance $169.0$ km, confidence $0.5577$.
   - Low model confidence (0.5577), occurring in an oblique convergent-transform transition zone.
4. **`usb000fsh5`** (True: `SUB` -> Predicted: `CCB`):
   - $M 4.1$, focal depth $10.0$ km, boundary distance $304.0$ km, confidence $0.8174$.
   - A distal event ($304$ km from trench) situated on the back-arc continental margin where subduction kinematics interact with continental shortening.

---

## 9. Scientific Interpretation
1. **Source of High Performance in Phase 2**: The 99.98% validation Macro F1 in Phase 2 is explained by the combination of two factors:
   - **Feature Richness**: PB2002 kinematic features (`divergent_component`, `relative_velocity`, `boundary_elevation_m`) provide the exact quantitative mechanics that define Peter Bird's 2003 boundary taxonomy.
   - **Random Split Infiltration**: Random observation-level splitting allows co-located events along the same boundary trace to be shared between train and test sets.
2. **True Generalization Capability**: When tested on previously unseen boundaries (Boundary-Aware evaluation), the model retains a strong 79.35% Macro F1 score, demonstrating genuine out-of-boundary predictive skill.

---

## 10. Limitations
1. **Catalog Discretization**: Events are associated with the nearest refined boundary segment; earthquakes in complex diffuse deformation zones (e.g., Tibetan plateau, East African Rift) may have ambiguous true regimes.
2. **Kinematic Metadata Availability**: In practical real-time operations, PB2002 kinematic parameters require an association lookup step before inference.
3. **Imbalance Constraints**: While the model classifies minority classes effectively (OCB F1: 0.9993 on random test; 0.7621 on boundary test), subduction zones represent nearly half of all global events.

---

## 11. Phase 3 Conclusion
Phase 3 establishes rigorous, transparent scientific validation:
- The candidate model is verified on the untouched test split (0.9998 Macro F1, 0.9999 Accuracy).
- SHAP reveals that plate kinematic vectors drive predictions, while seismic features act as secondary filters.
- Feature ablation proves that seismic data alone achieves 0.2028 Macro F1, seismic + geographic achieves 0.9273 Macro F1, and PB2002 kinematics elevates performance to 0.9998 Macro F1.
- Boundary-aware generalization confirms that true out-of-boundary classification capability is 79.35% Macro F1.
