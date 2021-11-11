What about performance?

## Here are the numbers:

### i5-9600K + GTX1070 with default models  

#### GPU VRAM  

Before starting inference engine:  
memory.used [MiB]  
1213 MiB  

After starting inference engine:  
memory.used [MiB]  
4310 MiB  

#### Text to speech
Latency (time before first result): 1.0473401546478271 seconds  
All the next iterations have real-time factor < 1.0  

*[real-time factor < 1.0]: Are generated faster that the audio finishes player


#### Semantic similarity

Similarity betwen short phrases 'I will help you' and 'I shall provide you my assistance' is computed in 0.06283211708068848s

#### Chatbot

Chatbot reply `Hello partner! Ornament please. Such a delightful pup! You are loyally loyal to your master?` to a big context generated in 1.411924123764038s

## Run the benchmark test on your computer 

!!! danger "Warning"
    Requires Nvidia GPU because it uses nvidia-smi command line tool

- Comment out test skipping in `tests\benchmarks\benchmark.py`
- Run it with 
    ```
    pytest tests\benchmarks\benchmark.py -s
    ```

At the moment the output is not pretty, it's a work in progress.