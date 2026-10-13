# TectonicSense — Feature Engineering

## Step

Step 3C — Final ML Dataset and Feature Engineering

## Input

`data/interim/earthquakes_pb2002_refined.csv`

## Output

`data/processed/tectonicsense_ml_dataset.csv`

## Target

The machine-learning target is:

`boundary_class`

The seven PB2002 classes are:

- CCB
- CRB
- CTF
- OCB
- OSR
- OTF
- SUB

## Final Features

The final ML dataset contains 26 predictive features.

1. `magnitude`\n2. `magnitude_squared`\n3. `depth_km`\n4. `depth_abs_km`\n5. `longitude`\n6. `latitude`\n7. `longitude_sin`\n8. `longitude_cos`\n9. `latitude_sin`\n10. `latitude_cos`\n11. `boundary_distance_km`\n12. `log_boundary_distance_km`\n13. `boundary_length_km`\n14. `boundary_azimuth_sin`\n15. `boundary_azimuth_cos`\n16. `relative_velocity`\n17. `velocity_azimuth_sin`\n18. `velocity_azimuth_cos`\n19. `divergent_component`\n20. `right_lateral_component`\n21. `absolute_divergent_component`\n22. `absolute_right_lateral_component`\n23. `tectonic_motion_magnitude`\n24. `boundary_elevation_m`\n25. `seafloor_age_ma`\n26. `orogen_flag`\n

## Identifier Leakage Prevention

The following identifier columns are excluded:

- `event_id`
- `boundary_id`
- `boundary_step_id`

These columns identify individual earthquake events or PB2002 boundary segments and are therefore not used as predictive features.

## Raw Text Exclusion

The following text fields are excluded:

- `place`
- `magnitude_type`

The current model focuses on numerical seismic, spatial, tectonic and geological information.

## Circular Feature Encoding

Angular variables are represented using sine and cosine transformations.

This avoids discontinuities around 0°/360°.

Encoded variables include:

- Boundary azimuth
- Velocity azimuth
- Longitude
- Latitude

## Derived Features

Additional numerical features include:

- Absolute earthquake depth
- Squared magnitude
- Log-transformed boundary distance
- Absolute divergent component
- Absolute right-lateral component
- Combined tectonic-motion magnitude

## Data Retention

No earthquake observations are removed based on boundary distance during this stage.

The boundary distance is retained as a predictive feature.

Distance-based filtering may be evaluated later as a controlled experiment.

## Important Modeling Consideration

`boundary_id` will not be used as a model feature.

During later evaluation, both random stratified splitting and group-aware splitting by `boundary_id` can be compared to assess potential geographic generalization and leakage effects.
