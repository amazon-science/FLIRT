## Feedback Loop In-context Red Teaming (FLIRT) Framework (Code):

This repository contains code for the Feedback Loop In-context Red Teaming (FLIRT) paper accepted at EMNLP 2024. The code implements the FLIRT framework to generate adversarial prompts to analyze a target model. Below we describe more details about the code.

## To run the code:

Go to the code folder. Please insert your in-context zero-shot and few-shot examples in the queue.txt file. We included examples in queue.txt for the formatting requirements.

To run FLIRT:
```bash
python FLIRT.py --flirt_iters 1000 --attack_strategy Scoring_greedy
```

## License

![alt text](https://licensebuttons.net/l/by-nc/4.0/88x31.png)
The code and dataset are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

please note that The stable-diffusion-v1-4 model used in the code (and hence downloaded on your machine) is under the CreativeML open RAIL-M license. 

