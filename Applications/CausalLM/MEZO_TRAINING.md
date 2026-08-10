# MeZO full-parameter training (Qwen3-0.6B on LaMP-3)

MeZO is a zeroth-order optimizer: it estimates the gradient from two forward
passes with opposite random perturbations, and never calls `backwarding()`.
That makes it the first backprop-free optimizer in nntrainer and lets it
fine-tune every parameter without storing gradients.

This document covers running it on **x86 Linux**. For how the optimizer itself
is wired into the framework, see the `MeZO` class in `nntrainer/optimizers/`.

---

## 1. Prerequisites

```bash
sudo apt install -y meson ninja-build cmake git curl g++ pkg-config
```

Submodules and the CausalLM `json.hpp` dependency are fetched below.

## 2. Build

```bash
git submodule update --init --depth 1 \
  subprojects/OpenBLAS subprojects/googletest subprojects/iniparser

# CausalLM needs json.hpp. jni/prepare_encoder.sh does this but requires wget
# and swallows download failures, so fetch it directly:
curl -sSL -o /tmp/encoder.tar.gz \
  https://github.com/nnstreamer/nnstreamer-android-resource/raw/main/external/encoder-0.2.tar.gz
tar -xzf /tmp/encoder.tar.gz -C /tmp json.hpp
cp /tmp/json.hpp Applications/CausalLM/json.hpp

meson setup build \
  -Denable-transformer=true \
  -Denable-tflite-backbone=false \
  -Denable-tflite-interpreter=false

ninja -C build Applications/CausalLM/nntr_mezo_train
```

`enable-tflite-*` are off because this path does not use TFLite and enabling
them requires `flatc` plus a tensorflow-lite install.

Verify the optimizer before spending compute on it:

```bash
ninja -C build test/unittest/unittest_nntrainer_mezo_interface
./build/test/unittest/unittest_nntrainer_mezo_interface   # expect 8/8
```

## 3. Prepare the model

```bash
cd Applications/CausalLM/res/qwen3/qwen3-0.6b
pip install torch transformers
HF_HUB_DISABLE_XET=1 python3 weight_converter.py \
    --model_path Qwen/Qwen3-0.6B \
    --output_name model/nntr_qwen3_0.6b_fp32.bin
```

`HF_HUB_DISABLE_XET=1` avoids an intermittent failure in the Hub's newer
transfer backend.

Then place alongside the `.bin` (all four are required):

| File | Where from |
|---|---|
| `nntr_qwen3_0.6b_fp32.bin` | the converter above (~2.4 GB) |
| `config.json` | the HF snapshot |
| `generation_config.json` | the HF snapshot |
| `tokenizer.json` | `huggingface_hub.hf_hub_download("Qwen/Qwen3-0.6B", "tokenizer.json")` |

Finally add `model/nntr_config.json`:

```json
{
    "model_type": "CausalLM",
    "model_tensor_type": "FP32-FP32",
    "model_file_name": "nntr_qwen3_0.6b_fp32.bin",
    "fc_layer_dtype": "FP32",
    "embedding_dtype": "FP32",
    "lora_rank": 0,
    "lora_alpha": 0,
    "lora_target": [],
    "bad_word_ids": [],
    "fsu": false,
    "fsu_lookahead": 2,
    "num_to_generate": 512,
    "init_seq_len": 512,
    "max_seq_len": 512,
    "batch_size": 1,
    "tokenizer_file": "tokenizer.json",
    "sample_input": ""
}
```

`"lora_rank": 0` is required — it is what leaves every layer trainable. The
driver forces it to 0 regardless, since this is full-parameter fine-tuning.

## 4. Train

```bash
./build/Applications/CausalLM/nntr_mezo_train \
  Applications/CausalLM/res/qwen3/qwen3-0.6b/model \
  Applications/CausalLM/res/train_data/lamp3_user_train.txt \
  Applications/CausalLM/res/train_data/lamp3_user_test.txt \
  --MeZO_lr 0.000001 --MeZO_epsilon 0.01 \
  --seq_len 512 --epochs 200 \
  --output mezo_qwen3_lamp3.bin
```

Three files are written next to `--output`:

| File | Contents |
|---|---|
| `mezo_qwen3_lamp3.bin` | best checkpoint so far (by validation loss) |
| `mezo_qwen3_lamp3_latest.bin` | end of the most recent epoch — resume from this |
| `mezo_qwen3_lamp3_history.csv` | `epoch,train_loss,valid_loss,accuracy` |

Both checkpoints are written to a temporary file and renamed into place, so an
interrupted save never leaves a truncated file.

### Resuming

```bash
./build/Applications/CausalLM/nntr_mezo_train ... \
  --resume mezo_qwen3_lamp3_latest.bin --start_epoch 37
```

MeZO carries no optimizer state — no momentum, no moment estimates — so the
weights are the entire training state and a resume is exact, not approximate.
`--start_epoch` only affects log numbering.

## 5. Choosing hyperparameters

**`MeZO_lr` and `MeZO_epsilon` are not independent.** The update is

```
θ ← θ − lr · (L(θ+εz) − L(θ−εz)) / (2ε) · z
```

so the step scales as **`lr / ε`**. Scaling both together changes nothing.
Tune `lr` and leave `ε` at `1e-2`.

Measured on Qwen3-0.6B (single-user LaMP-3, batch size 1):

| `lr / ε` | Behaviour |
|---|---|
| `1e-2` | diverges immediately; loss pins at 46.0517, the cross-entropy floor |
| `1e-3` | loss drifts upward over tens of steps |
| `1e-4` | stable at this budget; recommended starting point |

If loss climbs, halve `MeZO_lr`. Go lower before higher.

**Batch size is the main lever on estimator variance.** MeZO's gradient
estimate has variance proportional to parameter count, and the original paper
compensates with batch size 64. Raising `batch_size` in `nntr_config.json`
above 1 is the most promising untested change.

**Budget thousands of steps.** The paper trains for ~20K. Anything under a few
hundred is dominated by noise, in either direction.

## 6. Memory

Peak resident for Qwen3-0.6B FP32 is ~2.7 GB, close to the 2.4 GB of weights,
because the gradient-free path skips work a backprop optimizer needs:

- no per-weight gradient tensors (`NetworkGraph::setSkipGradients`) — they
  would be allocated and zero-filled but never written
- perturbations applied in place, with no model-sized noise tensor
- the plain weight loader instead of `load_weight_lora()`, which builds a
  throwaway second copy of the model

If the machine is short on RAM and the weight file is being paged in and out,
configure with `-Denable-mmap=false -Dmmap-read=false`.

## 7. Reading the loss

`train_loss` is the mean of the two probe losses, `(L(θ+εz) + L(θ−εz)) / 2`,
which equals `L(θ) + O(ε²)`. This matters: the last forward pass inside a step
is a *perturbed* model, so reporting its loss directly — as a naive
implementation does — describes weights that are not the ones being trained.

`valid_loss` is an ordinary unperturbed forward pass and is the number to
trust when comparing runs.

## 8. Sanity check on a small model

`Applications/MeZO/` trains a 3-layer fully-connected net on MNIST through the
same `MeZO::trainStep()`, converging in minutes rather than days:

```bash
ninja -C build Applications/MeZO/jni/mezo_example
./build/Applications/MeZO/jni/mezo_example /path/to/mnist   # raw idx files
```

Expect roughly 0.686 → 0.089 training loss and ~70% accuracy over 100 epochs.
Useful for confirming the optimizer works before committing days of compute.
