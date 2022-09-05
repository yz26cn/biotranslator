# API

Import BioTranslator as:

```
import biotranslator as bt
```

## Setup a config:
- `setup_config`(config, data_type='seq')

## Train text encoder:
- `train_text_encoder`(data_dir: str, save_path: str)

## Train a BioTranslator Model:
- `train_biotranslator`(cfgs)

## Test a BioTranslator Model:
- `test_biotranslator`(data_dir, anno_data, cfg, translator, task)



<!-- ## BioTranslator's Encoder: `bt`(cfg: config)

BioTranslator's encoder.

## Configuration: `config`(data_type: str, args: dict)
The `config` is used to load and process our input arguments.

## BioLoader: `loader`(cfg) 
The `loader` loads and stores the data we used in BioTranslator -->
