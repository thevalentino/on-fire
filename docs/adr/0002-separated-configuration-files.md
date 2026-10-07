# Separated Configuration Files

We decided to decouple frame role assignment from algorithmic parameter overrides into two separate files: a human-readable, fixed-width ASCII Frame Table (e.g. `frames.txt` / `frames.ascii`) for exposure metadata and frame roles, and a clean YAML file for parameter overrides. This replaces PypeIt's monolithic `.pypeit` file syntax, preventing complex config syntax collisions and allowing frame tables to be easily inspected and edited directly in standard terminal editors.
