#  

### ABOUT

### DEPENDENCIES
A quantization backdoor attack implemented using PyTorch. The individual packages that the code depends on are listed below.
- `pytorch==1.10.1`
- `torchvision==0.11.2`

### RUN
You only need to set the step parameter and simply run the following commands in order to get the quantization backdoor model.
```python
# Step=0, train
python main.py --step 0

# Step=1, fine-tuning
python main.py --step 1

# Step=2, evaluate
python main.py --step 2
```
