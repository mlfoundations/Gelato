# Evaluations

We evaluate the Gelato-30B-A3B model on the ScreenSpot-Pro and OS-World-G benchmarks. We store our benchmark data at [data](./data). To download the corresponding images, you should do so from the official [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) and [OS-World-G](https://github.com/xlang-ai/OSWorld-G/tree/main/benchmark) dataset pages.

**Note on OS-World-G evaluation:** The benchmark includes 54 refusal samples that can be included via `--include-refusal`. Gelato-30B-A3B supports refusal behavior through the `gelato-refusal` prompt mode. Official accuracy is calculated over all 564 samples (510 standard + 54 refusal).

### Requirements

The following Python packages are required to run the evaluation code:

- `transformers==4.57.0`
- `torch==2.8.0`
- `pillow==11.3.0`
- `qwen-vl-utils==0.0.14`

### Gelato-30B
Weights: [mlfoundations/Gelato-30B-A3B](https://huggingface.co/mlfoundations/Gelato-30B-A3B)

Eval Logs: [Screenspot-Pro](./logs/grounding_eval_screenspot-pro-gelato-30b.json), [OS-World-G](./logs/grounding_eval_osworld-g-gelato-30b.json) and [OS-World-G (Refined)](./logs/grounding_eval_osworld-g-refined_gelato-30b_refusal_included.json)

We run our evals on 4 40GB A100 GPUs with a maximum of 10 megapixel frames. We use huggingface's model parallelism to run the evaluation on 4 GPUs with a batch size of 1.

Run the following command to evaluate the Gelato-30B-A3B model on the ScreenSpot-Pro benchmark.

```
python evaluation/run_grounding_eval.py \
    --json-file eval-grounding-data/screenspot-pro-eval.json \
    --images-dir path/to/screenspot-pro-images \
    --model path/to/Gelato-30B-A3B \
    --model-type qwen3 \
    --distributed-mode sharded \
    --batch-size 1 \
    --max-pixels 10000000 \
    --prompt-mode gelato \
    --output-dir repro-results
```

Run the following command to evaluate the Gelato-30B-A3B model on the OS-World-G benchmark.

```
# Without refusal
python evaluation/run_grounding_eval.py \
    --json-file eval-grounding-data/osworld-g-eval.json \
    --images-dir path/to/osworld-g-images \
    --model path/to/Gelato-30B-A3B \
    --model-type qwen3 \
    --distributed-mode sharded \
    --batch-size 1 \
    --max-pixels 10000000 \
    --prompt-mode gelato \
    --output-dir repro-results

# With refusal
python evaluation/run_grounding_eval.py \
    --json-file eval-grounding-data/osworld-g-eval.json \
    --images-dir path/to/osworld-g-images \
    --model path/to/Gelato-30B-A3B \
    --model-type qwen3 \
    --distributed-mode sharded \
    --batch-size 1 \
    --max-pixels 10000000 \
    --prompt-mode gelato-refusal \
    --include-refusal \
    --output-dir repro-results
```

### UI-TARS-1.5-7B + Gelato Baseline

Weights: [mlfoundations/Gelato-UI-TARS-1.5-7B](https://huggingface.co/mlfoundations-cua-dev/Gelato-UI-TARS-1.5-7B)

Eval Logs: [Screenspot-Pro](./logs/grounding_eval_screenspot-pro-uitars-1_5-7b-gelato-baseline.json), [OS-World-G](./logs/grounding_eval_osworld-g-uitars-1_5-7b-gelato-baseline.json) and [OS-World-G (Refined)](./logs/grounding_eval_osworld-g-refined-uitars-1_5-7b-gelato-baseline.json)

We run our evals on 4 40GB A100 GPUs with a maximum of 4 megapixel frames. We run the evaluation with a global batch size of 4.

Run the following command to evaluate the Gelato-UI-TARS-1.5-7B model on the ScreenSpot-Pro benchmark.

```
python evaluation/run_grounding_eval.py \
    --json-file eval-grounding-data/screenspot-pro-eval.json \
    --images-dir path/to/screenspot-pro-images \
    --model path/to/Gelato-UI-TARS-1.5-7B \
    --model-type qwen2.5 \
    --distributed-mode dp \
    --num-gpus 4 \
    --batch-size 1 \
    --max-pixels 4000000 \
    --pixel-space-output \
    --prompt-mode gelato \
    --output-dir repro-results
```

Run the following command to evaluate the Gelato-UI-TARS-1.5-7B model on the OS-World-G benchmark.

```
python evaluation/run_grounding_eval.py \
    --json-file eval-grounding-data/osworld-g-eval.json \
    --images-dir path/to/osworld-g-images \
    --model path/to/Gelato-UI-TARS-1.5-7B \
    --model-type qwen2.5 \
    --distributed-mode dp \
    --num-gpus 4 \
    --batch-size 1 \
    --max-pixels 4000000 \
    --pixel-space-output \
    --prompt-mode gelato \
    --output-dir repro-results
```

## Reproduced results for baseline models

We run our evaluation code on GTA1-7B-2507 as a baseline model to verify the correctness of our evaluation pipeline. 

Run the following command to evaluate the GTA1-7B-2507 model on the ScreenSpot-Pro benchmark.

```
python evaluation/run_grounding_eval.py \
    --json-file eval-grounding-data/screenspot-pro-eval.json \
    --images-dir path/to/screenspot-pro-images \
    --model path/to/GTA1-7B-2507 \
    --model-type qwen2.5 \
    --distributed-mode dp \
    --num-gpus 4 \
    --batch-size 1 \
    --max-pixels 4000000 \
    --pixel-space-output \
    --prompt-mode gta1 \
    --output-dir repro-results
```

Run the following command to evaluate the GTA1-7B-2507 model on the OS-World-G benchmark.

```
python evaluation/run_grounding_eval.py \
    --json-file eval-grounding-data/osworld-g-eval.json \
    --images-dir path/to/osworld-g-images \
    --model path/to/GTA1-7B-2507 \
    --model-type qwen2.5 \
    --distributed-mode dp \
    --num-gpus 4 \
    --batch-size 1 \
    --max-pixels 4000000 \
    --pixel-space-output \
    --prompt-mode gta1 \
    --output-dir repro-results
```

### ScreenSpot-Pro

| Model | Accuracy | Logs |
|-------|----------|------|
| GTA1-7B-2507 | 50.1% | Official (reported) |
| GTA1-7B-2507 (**reproduced**) | 49.72% | [View logs](./logs/grounding_eval_screenspot-pro-gta1-7b-2507.json) |

### OS-World-G

| Model | Accuracy | Logs |
|-------|----------|------|
| GTA1-7B-2507 | 55.1% | Official (reported) |
| GTA1-7B-2507 (**reproduced**) | 56.02% | [View logs](./logs/grounding_eval_osworld-g-gta1-7b-2507.json) |

### OS-World-G (Refined)

| Model | Accuracy | Logs |
|-------|----------|------|
| GTA1-7B-2507 | 67.7% | Official (reported) |    
| GTA1-7B-2507 (**reproduced**) | 66.31% | [View logs](./logs/grounding_eval_osworld-g-refined-gta1-7b-2507.json) |
