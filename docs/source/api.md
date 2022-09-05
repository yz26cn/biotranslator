# API

### Import methods from BioTranslator
```
from biotranslator import .
```

### Setup a config:
- `setup_config`(config, data_type='seq')

### Train text encoder:
- `train_text_encoder`(data_dir: str, save_path: str)

### Train a BioTranslator Model:
- `train_biotranslator`(cfgs)

### Test a BioTranslator Model:
- `test_biotranslator`(data_dir, anno_data, cfg, translator, task)
