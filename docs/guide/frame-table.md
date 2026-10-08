title: Frame Table
order: 3

# The Frame Table

The Frame Table is a fixed-width text file with one row per raw exposure. It is
meant to be read and edited by a person. `onfire setup` writes it; you correct it.

## Columns

| Column | Meaning |
|--------|---------|
| `idx` | Row number. |
| `filename` | Raw FITS file name. |
| `frame_type` | The role this exposure plays. **The column you edit.** |
| `object` | `OBJECT` header value. |
| `obstype` | `OBSTYPE` header value. |
| `exptime` | Exposure time. |
| `acqtime` | Acquisition time. |
| `airmass` | Airmass. |
| `grism` | `GRISM` header value. |
| `instrume` | `INSTRUME` header value. |

Lines starting with `#` are comments. The columns are fixed width, so keep values
aligned. A value too wide for its column is rejected when the table is written.

## Roles

`frame_type` is one of:

- `science`
- `standard`
- `pixelflat`
- `arc`
- `unknown`

`onfire setup` classifies from the `OBSTYPE` header, using `OBJECT` as a fallback
hint. Anything it cannot place becomes `unknown`. If your data uses a different
`OBSTYPE` value, change the `frame_type` by hand.

## Required calibrations

A reduction needs at least one `pixelflat` and one `arc`. `onfire setup` prints a
warning for each one it cannot find in the raw directory. It still writes the table,
so you can fix the roles by hand.

## Reading and writing from Python

```python
from onfire.setup import FrameTable

table = FrameTable.read("frames.ascii")
print(len(table.frames))
```
