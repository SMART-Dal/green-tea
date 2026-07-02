# SFT Trainer

[![All_models-SFT-blue](https://img.shields.io/badge/All_models-SFT-blue)](https://huggingface.co/models?other=sft,trl) [![smol_course-Chapter_1-yellow](https://img.shields.io/badge/smol_course-Chapter_1-yellow)](https://github.com/huggingface/smol-course/tree/main/1_instruction_tuning)

## Overview

TRL supports the Supervised Fine-Tuning (SFT) Trainer for training language models.

This post-training method was contributed by [Younes Belkada](https://huggingface.co/ybelkada).

## Quick start

This example demonstrates how to train a language model using the [SFTTrainer](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTTrainer) from TRL. We train a [Qwen 3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model on the [Capybara dataset](https://huggingface.co/datasets/trl-lib/Capybara), a compact, diverse multi-turn dataset to benchmark reasoning and generalization.

```python
from trl import SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

## Expected dataset type and format

SFT supports both [language modeling](dataset_formats#language-modeling) and [prompt-completion](dataset_formats#prompt-completion) datasets. The [SFTTrainer](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTTrainer) is compatible with both [standard](dataset_formats#standard) and [conversational](dataset_formats#conversational) dataset formats. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

```python
# Standard language modeling
{"text": "The sky is blue."}

# Conversational language modeling
{"messages": [{"role": "user", "content": "What color is the sky?"},
              {"role": "assistant", "content": "It is blue."}]}

# Standard prompt-completion
{"prompt": "The sky is",
 "completion": " blue."}

# Conversational prompt-completion
{"prompt": [{"role": "user", "content": "What color is the sky?"}],
 "completion": [{"role": "assistant", "content": "It is blue."}]}
```

If your dataset is not in one of these formats, you can preprocess it to convert it into the expected format. Here is an example with the [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset:

```python
from datasets import load_dataset

dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")

def preprocess_function(example):
    return {
        "prompt": [{"role": "user", "content": example["Question"]}],
        "completion": [
            {"role": "assistant", "content": f"{example['Complex_CoT']}{example['Response']}"}
        ],
    }

dataset = dataset.map(preprocess_function, remove_columns=["Question", "Response", "Complex_CoT"])
print(next(iter(dataset["train"])))
```

```json
{
    "prompt": [
        {
            "content": "Given the symptoms of sudden weakness in the left arm and leg, recent long-distance travel, and the presence of swollen and tender right lower leg, what specific cardiac abnormality is most likely to be found upon further evaluation that could explain these findings?",
            "role": "user",
        }
    ],
    "completion": [
        {
            "content": "Okay, let's see what's going on here. We've got sudden weakness [...] clicks into place!The specific cardiac abnormality most likely to be found in [...] the presence of a PFO facilitating a paradoxical embolism.",
            "role": "assistant",
        }
    ],
}
```

## Looking deeper into the SFT method

Supervised Fine-Tuning (SFT) is the simplest and most commonly used method to adapt a language model to a target dataset. The model is trained in a fully supervised fashion using pairs of input and output sequences. The goal is to minimize the negative log-likelihood (NLL) of the target sequence, conditioning on the input.

This section breaks down how SFT works in practice, covering the key steps: **preprocessing**, **tokenization** and **loss computation**.

### Preprocessing and tokenization

During training, each example is expected to contain a **text field** or a **(prompt, completion)** pair, depending on the dataset format. For more details on the expected formats, see [Dataset formats](dataset_formats).
The [SFTTrainer](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTTrainer) tokenizes each input using the model's tokenizer. If both prompt and completion are provided separately, they are concatenated before tokenization.

### Computing the loss

![sft_figure](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_figure.png)

The loss used in SFT is the **token-level cross-entropy loss**, defined as:

$$
\mathcal{L}_{\text{SFT}}(\theta) = - \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{ [!TIP]
> The paper [On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](https://huggingface.co/papers/2508.05629) proposes an alternative loss function, called **Dynamic Fine-Tuning (DFT)**, which aims to improve generalization by rectifying the reward signal. This method can be enabled by setting `loss_type="dft"` in the [SFTConfig](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTConfig). For more details, see [Paper Index - Dynamic Fine-Tuning](paper_index#on-the-generalization-of-sft-a-reinforcement-learning-perspective-with-reward-rectification).

### Label shifting and masking

During training, the loss is computed using a **one-token shift**: the model is trained to predict each token in the sequence based on all previous tokens. Specifically, the input sequence is shifted right by one position to form the target labels.
Padding tokens (if present) are ignored in the loss computation by applying an ignore index (default: `-100`) to the corresponding positions. This ensures that the loss focuses only on meaningful, non-padding tokens.

## Logged metrics

While training and evaluating we record the following reward metrics:

* `global_step`: The total number of optimizer steps taken so far.
* `epoch`: The current epoch number, based on dataset iteration.
* `num_tokens`: The total number of tokens processed so far.
* `loss`: The average cross-entropy loss computed over non-masked tokens in the current logging interval.
* `entropy`: The average entropy of the model's predicted token distribution over non-masked tokens.
* `mean_token_accuracy`: The proportion of non-masked tokens for which the model’s top-1 prediction matches the ground truth token.
* `learning_rate`: The current learning rate, which may change dynamically if a scheduler is used.
* `grad_norm`: The L2 norm of the gradients, computed before gradient clipping.

## Customization

### Model initialization

You can directly pass the kwargs of the `from_pretrained()` method to the [SFTConfig](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTConfig). For example, if you want to load a model in a different precision, analogous to

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
```

you can do so by passing the `model_init_kwargs={"dtype": torch.bfloat16}` argument to the [SFTConfig](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTConfig).

```python
from trl import SFTConfig

training_args = SFTConfig(
    model_init_kwargs={"dtype": torch.bfloat16},
)
```

Note that all keyword arguments of `from_pretrained()` are supported.

### Packing

[SFTTrainer](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTTrainer) supports _example packing_, where multiple examples are packed in the same input sequence to increase training efficiency. To enable packing, simply pass `packing=True` to the [SFTConfig](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTConfig) constructor.

```python
training_args = SFTConfig(packing=True)
```

For more details on packing, see [Packing](reducing_memory_usage#packing).

### Train on assistant messages only

To train on assistant messages only, use a [conversational](dataset_formats#conversational) dataset and set `assistant_only_loss=True` in the [SFTConfig](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTConfig). This setting ensures that loss is computed **only** on the assistant responses, ignoring user or system messages.

```python
training_args = SFTConfig(assistant_only_loss=True)
```

![train_on_assistant](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/train_on_assistant.png)

> [!WARNING]
> This functionality is only available for chat templates that support returning the assistant tokens mask via the `&#123;% generation %&#125;` and `&#123;% endgeneration %&#125;` keywords. For an example of such a template, see [HugggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja#L76-L82).

### Train on completion only

To train on completion only, use a [prompt-completion](dataset_formats#prompt-completion) dataset. By default, the trainer computes the loss on the completion tokens only, ignoring the prompt tokens. If you want to train on the full sequence, set `completion_only_loss=False` in the [SFTConfig](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTConfig).

![train_on_completion](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/train_on_completion.png)

> [!TIP]
> Training on completion only is compatible with training on assistant messages only. In this case, use a [conversational](dataset_formats#conversational) [prompt-completion](dataset_formats#prompt-completion) dataset and set `assistant_only_loss=True` in the [SFTConfig](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTConfig).

### Train adapters with PEFT

We support tight integration with 🤗 PEFT library, allowing any user to conveniently train adapters and share them on the Hub, rather than training the entire model.

```python
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig

dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = SFTTrainer(
    "Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    peft_config=LoraConfig()
)

trainer.train()
```

You can also continue training your [PeftModel](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel). For that, first load a `PeftModel` outside [SFTTrainer](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTTrainer) and pass it directly to the trainer without the `peft_config` argument being passed.

```python
from datasets import load_dataset
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("trl-lib/Qwen3-4B-LoRA", is_trainable=True)
dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
)

trainer.train()
```

> [!TIP]
> When training adapters, you typically use a higher learning rate (≈1e‑4) since only new parameters are being learned.
>
> ```python
> SFTConfig(learning_rate=1e-4, ...)
> ```

### Train with Liger Kernel

Liger Kernel is a collection of Triton kernels for LLM training that boosts multi-GPU throughput by 20%, cuts memory use by 60% (enabling up to 4× longer context), and works seamlessly with tools like FlashAttention, PyTorch FSDP, and DeepSpeed. For more information, see [Liger Kernel Integration](liger_kernel_integration).

### Train with Unsloth

Unsloth is an open‑source framework for fine‑tuning and reinforcement learning that trains LLMs (like Llama, Mistral, Gemma, DeepSeek, and more) up to 2× faster with up to 70% less VRAM, while providing a streamlined, Hugging Face–compatible workflow for training, evaluation, and deployment. For more information, see [Unsloth Integration](unsloth_integration).

## Instruction tuning example

**Instruction tuning** teaches a base language model to follow user instructions and engage in conversations. This requires:

1. **Chat template**: Defines how to structure conversations into text sequences, including role markers (user/assistant), special tokens, and turn boundaries. Read more about chat templates in [Chat templates](https://huggingface.co/docs/transformers/chat_templating#templates).
2. **Conversational dataset**: Contains instruction-response pairs

This example shows how to transform the [Qwen 3 0.6B Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) model into an instruction-following model using the [Capybara dataset](https://huggingface.co/datasets/trl-lib/Capybara) and a chat template from [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B). The SFT Trainer automatically handles tokenizer updates and special token configuration.

```python
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B-Base",
    args=SFTConfig(
        output_dir="Qwen3-0.6B-Instruct",
        chat_template_path="HuggingFaceTB/SmolLM3-3B",
    ),
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

> [!WARNING]
> Some base models, like those from Qwen, have a predefined chat template in the model's tokenizer. In these cases, it is not necessary to apply `clone_chat_template()`, as the tokenizer already handles the formatting. However, it is necessary to align the EOS token with the chat template to ensure the model's responses terminate correctly. In these cases, specify `eos_token` in [SFTConfig](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTConfig); for example, for `Qwen/Qwen2.5-1.5B`, one should set `eos_token=""`.

Once trained, your model can now follow instructions and engage in conversations using its new chat template.

```python
>>> from transformers import pipeline
>>> pipe = pipeline("text-generation", model="Qwen3-0.6B-Instruct/checkpoint-5000")
>>> prompt = "user\nWhat is the capital of France? Answer in one word.\nassistant\n"
>>> response = pipe(prompt)
>>> response[0]["generated_text"]
'user\nWhat is the capital of France? Answer in one word.\nassistant\nThe capital of France is Paris.'
```

Alternatively, use the structured conversation format (recommended):

```python
>>> prompt = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
>>> response = pipe(prompt)
>>> response[0]["generated_text"]
[{'role': 'user', 'content': 'What is the capital of France? Answer in one word.'}, {'role': 'assistant', 'content': 'The capital of France is Paris.'}]
```

## Tool Calling with SFT

The [SFTTrainer](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTTrainer) fully supports fine-tuning models with _tool calling_ capabilities. In this case, each dataset example should include:

* The conversation messages, including any tool calls (`tool_calls`) and tool responses (`tool` role messages)
* The list of available tools in the `tools` column, typically provided as JSON schemas

For details on the expected dataset structure, see the [Dataset Format — Tool Calling](dataset_formats#tool-calling) section.

## Training Vision Language Models

[SFTTrainer](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTTrainer) fully supports training Vision-Language Models (VLMs). To train a VLM, you need to provide a dataset with an additional `images` column containing the images to be processed. For more information on the expected dataset structure, see the [Dataset Format — Vision Dataset](dataset_formats#vision-dataset) section.
An example of such a dataset is the [LLaVA Instruct Mix](https://huggingface.co/datasets/trl-lib/llava-instruct-mix).

```python
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    args=SFTConfig(max_length=None),
    train_dataset=load_dataset("trl-lib/llava-instruct-mix", split="train"),
)
trainer.train()
```

> [!TIP]
> For VLMs, truncating may remove image tokens, leading to errors during training. To avoid this, set `max_length=None` in the [SFTConfig](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTConfig). This allows the model to process the full sequence length without truncating image tokens.
>
> ```python
> SFTConfig(max_length=None, ...)
> ```
>
> Only use `max_length` when you've verified that truncation won't remove image tokens for the entire dataset.

## SFTTrainer[[trl.SFTTrainer]]

#### trl.SFTTrainer[[trl.SFTTrainer]]

[Source](https://github.com/huggingface/trl/blob/v0.26.2/trl/trainer/sft_trainer.py#L482)

Trainer for Supervised Fine-Tuning (SFT) method.

This class is a wrapper around the [Trainer](https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/trainer#transformers.Trainer) class and inherits all of its attributes and methods.

Example:

```python
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

trainer = SFTTrainer(model="Qwen/Qwen2-0.5B-Instruct", train_dataset=dataset)
trainer.train()
```

traintrl.SFTTrainer.trainhttps://github.com/huggingface/trl/blob/v0.26.2/transformers/trainer.py#L2213[{"name": "resume_from_checkpoint", "val": ": typing.Union[str, bool, NoneType] = None"}, {"name": "trial", "val": ": typing.Union[ForwardRef('optuna.Trial'), dict[str, typing.Any], NoneType] = None"}, {"name": "ignore_keys_for_eval", "val": ": typing.Optional[list[str]] = None"}, {"name": "**kwargs", "val": ": typing.Any"}]- **resume_from_checkpoint** (`str` or `bool`, *optional*) --
  If a `str`, local path to a saved checkpoint as saved by a previous instance of `Trainer`. If a
  `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
  of `Trainer`. If present, training will resume from the model/optimizer/scheduler states loaded here.
- **trial** (`optuna.Trial` or `dict[str, Any]`, *optional*) --
  The trial run or the hyperparameter dictionary for hyperparameter search.
- **ignore_keys_for_eval** (`list[str]`, *optional*) --
  A list of keys in the output of your model (if it is a dictionary) that should be ignored when
  gathering predictions for evaluation during the training.
- **kwargs** (`dict[str, Any]`, *optional*) --
  Additional keyword arguments used to hide deprecated arguments0

Main training entry point.

**Parameters:**

model (`str | PreTrainedModel`) : Model to be trained. Can be either:  - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a path to a *directory* containing model weights saved using [save_pretrained](https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `'./my_model_directory/'`. The model is loaded using `.from_pretrained` (where `` is derived from the model config) with the keyword arguments in `args.model_init_kwargs`. - A [PreTrainedModel](https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/model#transformers.PreTrainedModel) object. If you're training a model with an MoE architecture and want to include the load balancing/auxilliary loss as a part of the final loss, remember to set the `output_router_logits` config of the model to `True`.

args ([SFTConfig](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTConfig), *optional*) : Configuration for this trainer. If `None`, a default configuration is used.

data_collator (`DataCollator`, *optional*) : Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`. Will default to [DataCollatorForLanguageModeling](/docs/trl/v0.26.2/en/sft_trainer#trl.trainer.sft_trainer.DataCollatorForLanguageModeling) if the model is a language model and [DataCollatorForVisionLanguageModeling](/docs/trl/v0.26.2/en/sft_trainer#trl.trainer.sft_trainer.DataCollatorForVisionLanguageModeling) if the model is a vision-language model.

train_dataset ([Dataset](https://huggingface.co/docs/datasets/v4.4.1/en/package_reference/main_classes#datasets.Dataset) or [IterableDataset](https://huggingface.co/docs/datasets/v4.4.1/en/package_reference/main_classes#datasets.IterableDataset)) : Dataset to use for training. SFT supports both [language modeling](#language-modeling) type and [prompt-completion](#prompt-completion) type. The format of the samples can be either:  - [Standard](dataset_formats#standard): Each sample contains plain text. - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role and content).  The trainer also supports processed datasets (tokenized) as long as they contain an `input_ids` field.

eval_dataset ([Dataset](https://huggingface.co/docs/datasets/v4.4.1/en/package_reference/main_classes#datasets.Dataset), [IterableDataset](https://huggingface.co/docs/datasets/v4.4.1/en/package_reference/main_classes#datasets.IterableDataset) or `dict[str, Dataset | IterableDataset]`) : Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.

processing_class ([PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.0.0rc1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase), [ProcessorMixin](https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/processors#transformers.ProcessorMixin), *optional*) : Processing class used to process the data. If `None`, the processing class is loaded from the model's name with [from_pretrained](https://huggingface.co/docs/transformers/v5.0.0rc1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained). A padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token, `tokenizer.eos_token` will be used as the default.

compute_loss_func (`Callable`, *optional*) : A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated batch (batch_size * gradient_accumulation_steps) and returns the loss. For example, see the default [loss function](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618) used by `Trainer`.

compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*) : The function that will be used to compute metrics at evaluation. Must take a [EvalPrediction](https://huggingface.co/docs/transformers/v5.0.0rc1/en/internal/trainer_utils#transformers.EvalPrediction) and return a dictionary string to metric values. When passing [SFTConfig](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTConfig) with `batch_eval_metrics` set to `True`, your `compute_metrics` function must take a boolean `compute_result` argument. This will be triggered after the last eval batch to signal that the function needs to calculate and return the global summary statistics rather than accumulating the batch-level statistics.

callbacks (list of [TrainerCallback](https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/callback#transformers.TrainerCallback), *optional*) : List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).  If you want to remove one of the default callbacks used, use the [remove_callback](https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/trainer#transformers.Trainer.remove_callback) method.

optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`) : A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your model and a scheduler given by [get_linear_schedule_with_warmup](https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup) controlled by `args`.

optimizer_cls_and_kwargs (`tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*) : A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in `args`. Incompatible with the `optimizers` argument.  Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before initializing the Trainer.

preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*) : A function that preprocess the logits right before caching them at each evaluation step. Must take two tensors, the logits and the labels, and return the logits once processed as desired. The modifications made by this function will be reflected in the predictions received by `compute_metrics`.  Note that the labels (second parameter) will be `None` if the dataset does not have them.

peft_config ([PeftConfig](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/config#peft.PeftConfig), *optional*) : PEFT configuration used to wrap the model. If `None`, the model is not wrapped.

formatting_func (`Callable`, *optional*) : Formatting function applied to the dataset before tokenization. Applying the formatting function explicitly converts the dataset into a [language modeling](#language-modeling) type.
#### save_model[[trl.SFTTrainer.save_model]]

[Source](https://github.com/huggingface/trl/blob/v0.26.2/transformers/trainer.py#L4177)

Will save the model, so you can reload it using `from_pretrained()`.

Will only save from the main process.
#### push_to_hub[[trl.SFTTrainer.push_to_hub]]

[Source](https://github.com/huggingface/trl/blob/v0.26.2/transformers/trainer.py#L5117)

Upload `self.model` and `self.processing_class` to the 🤗 model hub on the repo `self.args.hub_model_id`.

**Parameters:**

commit_message (`str`, *optional*, defaults to `"End of training"`) : Message to commit while pushing.

blocking (`bool`, *optional*, defaults to `True`) : Whether the function should return only when the `git push` has finished.

token (`str`, *optional*, defaults to `None`) : Token with write permission to overwrite Trainer's original args.

revision (`str`, *optional*) : The git revision to commit from. Defaults to the head of the "main" branch.

kwargs (`dict[str, Any]`, *optional*) : Additional keyword arguments passed along to `~Trainer.create_model_card`.

**Returns:**

The URL of the repository where the model was pushed if `blocking=False`, or a `Future` object tracking the
progress of the commit if `blocking=True`.

## SFTConfig[[trl.SFTConfig]]

#### trl.SFTConfig[[trl.SFTConfig]]

[Source](https://github.com/huggingface/trl/blob/v0.26.2/trl/trainer/sft_config.py#L22)

Configuration class for the [SFTTrainer](/docs/trl/v0.26.2/en/sft_trainer#trl.SFTTrainer).

This class includes only the parameters that are specific to SFT training. For a full list of training arguments,
please refer to the [TrainingArguments](https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/trainer#transformers.TrainingArguments) documentation. Note that default values in this class may
differ from those in [TrainingArguments](https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/trainer#transformers.TrainingArguments).

Using [HfArgumentParser](https://huggingface.co/docs/transformers/v5.0.0rc1/en/internal/trainer_utils#transformers.HfArgumentParser) we can turn this class into
[argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
command line.

## DataCollatorForLanguageModeling[[trl.trainer.sft_trainer.DataCollatorForLanguageModeling]]

#### trl.trainer.sft_trainer.DataCollatorForLanguageModeling[[trl.trainer.sft_trainer.DataCollatorForLanguageModeling]]

[Source](https://github.com/huggingface/trl/blob/v0.26.2/trl/trainer/sft_trainer.py#L86)

Data collator used for language modeling data. Inputs are dynamically padded to the maximum length of a batch.

This collator expects each example in the input list to be a dictionary containing at least the `"input_ids"` key.
If the input contains a `"completion_mask"`, it is used to set the labels to `-100` for tokens that are not in the
completion. If `"assistant_masks"` are present, they are used to set the labels to `-100` for tokens that are not
in the assistant part of the sequence. The collator returns a dictionary containing the following keys:
- `"input_ids"`: Tensor of input IDs, padded to the maximum length of the batch.
- `"labels"`: Tensor of labels, padded to the maximum length of the batch. If `completion_only_loss` is set to
`True`, tokens that are not in the completion are set to -100. If `assistant_masks` are present, tokens that are
not in the assistant part of the sequence are set to -100. If `padding_free` is set to `False`, the following key
is also returned:
- `"attention_mask"`: Tensor of attention masks, padded to the maximum length of the batch.
If `padding_free` is set to `True`, the following key is also returned:
- `"position_ids"`: Tensor of position IDs, padded to the maximum length of the batch.

Examples:
```python
>>> from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

>>> collator = DataCollatorForLanguageModeling(pad_token_id=0)
>>> examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
>>> collator(examples)
{'input_ids': tensor([[  1,  2,  3],
                      [  4,  5,  0]]),
 'attention_mask': tensor([[  1,  1,  1],
                           [  1,  1,  0]]),
 'labels': tensor([[   1,    2,    3],
                   [   4,    5, -100]])}

>>> # With completion mask
>>> examples = [
...     {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
...     {"input_ids": [4, 5], "completion_mask": [0, 1]},
... ]
>>> collator(examples)
{'input_ids': tensor([[  1,  2,  3],
                      [  4,  5,  0]]),
 'attention_mask': tensor([[  1,  1,  1],
                           [  1,  1,  0]]),
 'labels': tensor([[-100,    2,    3],
                   [-100,    5, -100]])}

>>> # With padding_free
>>> collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
>>> collator(examples)
{'input_ids': tensor([[ 1, 2, 3, 4, 5]]),
 'position_ids': tensor([[0, 1, 2, 0, 1]]),
 'labels': tensor([[1, 2, 3, 4, 5]])}
```

get_position_ids_from_packed_seq_lengthstrl.trainer.sft_trainer.DataCollatorForLanguageModeling.get_position_ids_from_packed_seq_lengthshttps://github.com/huggingface/trl/blob/v0.26.2/trl/trainer/sft_trainer.py#L225[{"name": "batch_seq_lengths", "val": ": list"}]- **batch_seq_lengths** (`list[list[int]]`) --
  A list of lists containing the lengths of each individual document in the packed batch.0`list[torch.Tensor]`A list of tensors containing the position IDs for each packed sequence.

Get position IDs for packed sequences.

**Parameters:**

pad_token_id (`int`) : Token ID to use for padding.

completion_only_loss (`bool`, *optional*, defaults to `True`) : When the input contains a completion mask (`completion_mask`), the labels are set to -100 for the tokens that are no in the completion.

padding_free (`bool`, *optional*, defaults to `False`) : If set to `True`, the sequences will be flattened into a single sequence, and the position IDs will be generated accordingly and returned instead of the attention mask.

pad_to_multiple_of (`int`, *optional*) : If set, the sequences will be padded to a multiple of this value.

return_tensors (`str`, *optional*, defaults to `"pt"`) : Type of Tensor to return. Only `"pt"` is currently supported.

**Returns:**

``list[torch.Tensor]``

A list of tensors containing the position IDs for each packed sequence.

## DataCollatorForVisionLanguageModeling[[trl.trainer.sft_trainer.DataCollatorForVisionLanguageModeling]]

#### trl.trainer.sft_trainer.DataCollatorForVisionLanguageModeling[[trl.trainer.sft_trainer.DataCollatorForVisionLanguageModeling]]

[Source](https://github.com/huggingface/trl/blob/v0.26.2/trl/trainer/sft_trainer.py#L254)

Data collator for vision-language modeling tasks.

Unlike text-only datasets—where the collator typically receives pre-tokenized inputs ready for batching,
vision-language data processing involves converting images into pixel values. This conversion is disk-intensive,
making upfront preprocessing of the entire dataset impractical. Therefore, this collator performs tokenization and
image processing on-the-fly to efficiently prepare batches.

Each input example should be a dictionary containing at least:
- An `"images"` key holding the image data.
- [language modeling](#language-modeling) type: either a `"messages"` key for conversational inputs or a `"text"`
  key for standard text inputs.
- [prompt-completion](#prompt-completion) type: keys `"prompt"` and `"completion"` for the prompt and completion.

The collator outputs a dictionary including:
- `"input_ids"`: Tensor of token IDs.
- `"attention_mask"`: Tensor indicating attention mask.
- `"pixel_values"`: Tensor representing image pixel values.
- `"labels"`: Tensor for training labels.

Additional keys may be present depending on the processor, such as `"image_grid_thw"`.

Example:
```python
>>> from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
>>> collator = DataCollatorForVisionLanguageModeling(processor)
>>> examples = [
...     {"images": [Image.open("image_0.png")], "messages": [{"role": "user", "content": "What is this?"}]},
...     {"images": [Image.open("image_1.png")], "messages": [{"role": "user", "content": "Describe this image."}]},
... ]
>>> collator(examples)
{'input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                       151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,   3838,    374,
                          419,     30, 151645,    198],
                      [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                       151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,  74785,    419,
                         2168,     13, 151645,    198]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
 'pixel_values': tensor([[-0.9893,  0.1785,  1.5362,  ..., -0.0582,  0.8661, -0.2431],
                         [-0.2302,  0.9522, -1.1061,  ...,  0.0555,  1.3354, -0.6412],
                         [ 1.2150,  0.9084,  0.7041,  ...,  0.2404, -0.8403, -0.5133],
                         ...,
                         [ 0.6895,  0.2807,  0.2515,  ..., -0.2004, -1.2100,  0.0555],
                         [ 0.8209, -0.9748,  1.5654,  ...,  1.6055, -0.4706,  0.5817],
                         [-1.0915,  0.4559,  0.9230,  ...,  0.5106,  0.0982, -0.1720]]),
 'image_grid_thw': tensor([[1, 4, 4],
                           [1, 4, 4]]),
 'labels': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                    151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,   3838,    374,
                       419,     30, 151645,    198],
                    [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                     151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,  74785,    419,
                       2168,     13, 151645,    198]])}
```

**Parameters:**

processor ([ProcessorMixin](https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/processors#transformers.ProcessorMixin)) : The processor used to tokenize text and process images. It must be a subclass of [ProcessorMixin](https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/processors#transformers.ProcessorMixin) and include a `tokenizer` with a defined `pad_token_id`.

max_length (`int` or `None`, optional, defaults to `None`) : Maximum sequence length for input tokens. If `None`, no truncation is applied.

completion_only_loss (`bool`, *optional*, defaults to `False`) : Whether to compute loss only on the completion part of the sequence. When `True`, the labels for the prompt part are set to -100. It requires the dataset type to be prompt-completion.

pad_to_multiple_of (`int` or `None`, optional, defaults to `None`) : If set, the sequences will be padded to a multiple of this value.

dataset_text_field (`str`, optional, defaults to `"text"`) : Name of the column that contains text data in the dataset. This parameter is only relevant for [standard datasets format](dataset_formats#standard).

return_tensors (`str`, optional, defaults to `"pt"`) : The tensor type to return. Currently, only `"pt"` (PyTorch tensors) is supported.

