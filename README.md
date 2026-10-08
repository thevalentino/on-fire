# onfire

A standalone, inspectable data reduction pipeline for Magellan/FIRE echelle
spectroscopy (orders 11-31).

## Documentation

A user guide with a big-picture overview, a quickstart, and one page per
Reduction Step lives in `docs/site/`. Open `docs/site/index.html` in a browser; it
works offline. To rebuild it after editing `docs/guide/`:

```bash
pip install -e ".[docs]"
python docs/build.py
```
