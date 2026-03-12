# TSDataForge Showcase

## Real public demos

Source notes and provenance:
[public_data_provenance.md](public_data_provenance.md)

### 1. Public ECG arrhythmia handoff

```python
from tsdataforge import demo

bundle = demo(output_dir="ecg_bundle", scenario="ecg_public")
print(bundle.output_dir)
```

CLI alternative: `python -m tsdataforge demo --scenario ecg_public --output ecg_bundle`

### 2. Public US macro handoff

```python
from tsdataforge import demo

bundle = demo(output_dir="macro_bundle", scenario="macro_public")
print(bundle.output_dir)
```

CLI alternative: `python -m tsdataforge demo --scenario macro_public --output macro_bundle`

### 3. Public climate CO₂ handoff

```python
from tsdataforge import demo

bundle = demo(output_dir="climate_bundle", scenario="climate_public")
print(bundle.output_dir)
```

CLI alternative: `python -m tsdataforge demo --scenario climate_public --output climate_bundle`

### 4. Public sunspot cycle handoff

```python
from tsdataforge import demo

bundle = demo(output_dir="sunspots_bundle", scenario="sunspots_public")
print(bundle.output_dir)
```

CLI alternative: `python -m tsdataforge demo --scenario sunspots_public --output sunspots_bundle`

## Reality-shaped synthetic demos

These still exist when you want deterministic showcase assets without shipping external raw data:

- `icu_vitals`
- `macro_regime`
- `factory_sensor`
- `synthetic`
