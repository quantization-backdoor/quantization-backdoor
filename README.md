#  

### ABOUT

### DEPENDENCIES
Our code is implemented and tested on TensorFlow. Following packages are used by our code.
- `tensorflow-gpu==2.5.0`
- `numpy==1.19.5`

### RUN
You only need to set the step parameter and simply run the following commands in order to get the quantization backdoor model.
```python
# Step=0, train the backdoor model
python main.py 0

# Step=1, fine-tuning the backdoor model
python main.py 1

# Step=2, evaluation model
python main.py 2

# Step=3, TFLite quantize model
python main.py 3

# Step=4, evaluate the ASR of the TFLite model
python main.py 4
```
