# Labeling rules for the ground truth

Counts per image (not boxes). Every visible person is counted, including small or far-away ones.

- **Fully equipped:** a helmet AND a vest are both clearly visible.
  - High-visibility clothing (hi-vis jacket, vest or reflective uniform) counts as a vest.
  - A cap, hood or hat does NOT count as a helmet.
- **Not equipped:** clearly missing the helmet, the vest, or both.
- **Unsure:** cannot tell (too small, hidden, cut off).
- Nobody in the photo: all counts are 0.

## Changes made after review
- `38043248.jpg` was removed: a crowd of 20+ people that cannot be counted reliably.
- `14761195.jpg` was changed from 3 not equipped to 3 fully equipped (orange helmets and orange
  hi-vis uniforms with reflective stripes), to follow the hi-vis rule above.

`ground_truth_raw.csv` is the file exactly as labelled before the review.
