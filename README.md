# Audio Generation

Under construction

1. Prepare running environment
```
conda create -n audioldm python=3.8
pip3 install audioldm
```

2. text-to-audio generation
```python
# Import the function you need
from audioldm import text_to_audio, seed_everything
# Make sure the reproducibility
seed_everything(42)  
# Generate Audio  
text_to_audio("A man is speaking")
```

# TODO

- [ ] wer