# Context-Aware Generative AI Counseling Assistant

A cloud-native, end-to-end conversational AI assistant fine-tuned to provide validating, empathetic, and structurally paced therapeutic dialogue. This project leverages Parameter-Efficient Fine-Tuning (PEFT) to adapt a 7-billion parameter foundational language model into a specialized counselor interface, deployed via a high-performance serverless cloud architecture.

---
Link to the HuggingFace Space : https://huggingface.co/spaces/sleepy-panda21/Counselling_assistant

##  Architecture Overview

The production deployment relies on a decoupled, stateless design to separate the client-side rendering from the deep learning compute engine:

```text
[ User Interface ]  --->  [ Gradio 6.x UI Wrapper (CPU Container) ] 
                                      | 
                                      | (Stateless Inference Request via API)
                                      v
[ Hugging Face Serverless GPU ] <--- [ Base Architecture: Qwen 2.5 7B ]
                                <--- [ Dynamic On-the-Fly LoRA Weight Injection ]
```


### Frontend Tier
Managed by a Gradio 6.x event-driven container. It serves as a stateless, lightweight wrapper handling user input and streaming raw character tokens.

### Compute Tier
Hosted on Hugging Face serverless GPU architecture. The client script queries a base foundation model and feeds your Low-Rank Adaptation (LoRA) adapter matrices straight into the compute layer parameters via the network runtime payload.

---

##  Fine-Tuning Methodology & Implementation

The core model was fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) on top of the **Qwen2.5-7B-Instruct** base model. The notebook pipeline is structured around the following optimizations:

### 1. 4-bit Quantization (QLoRA)
To optimize compute performance during training, the model was loaded using **NF4 (NormalFloat4)** data type precision via `bitsandbytes`. This technique anchors the base model parameter matrices in a frozen, highly compressed 4-bit representation, reducing active VRAM utilization by over 60% with negligible loss in model perplexity.

### 2. Low-Rank Adaptation (LoRA) Targets
Instead of running a full-parameter backward propagation pass across all layers, target adapter matrices were injected into the self-attention and multi-layer perceptron blocks. The linear projection configuration covers:
* `q_proj`, `k_proj`, `v_proj`, `o_proj` (Attention alignment spaces)
* `gate_proj`, `up_proj`, `down_proj` (Feed-forward channels)

The configuration used a rank scaling factor ($\alpha$) to handle structural weight updates cleanly:
* $r = 16$ (The intrinsic low-rank dimensionality matrix space)
* $\alpha = 16$ (The learning distribution scaling factor coefficient)

### 3. Compute Optimization with Unsloth
Training was accelerated through the **Unsloth** framework, utilizing custom written Triton kernels that bypass slow standard PyTorch autograd graph overheads. This optimization delivers up to a **2x speeding factor** during gradient accumulation passes, preventing memory fragmentation spikes on single-GPU hardware configurations.

### 4. Sequence Training Structure
* **Data Formatting:** Conversational sequences were structured utilizing the strict **ChatML** tokenization template (`<|im_start|>system\n...<|im_end|>`).
* **Paddings & Attention:** Enabled `packing=True` inside the `SFTTrainer` routine to group multiple brief multi-turn sessions into single dense 2048-token context blocks, maximizing token-per-second training efficiency.

---

##  Application Engineering & Production Deployment

Moving from a static model check to a public frontend required solving specific deployment constraints:

### 1. Modernized Event Handling (Gradio 6.0 Compliance)
The web interface is built around the latest event loop changes in Gradio 6.0:
* **Forced Message Dictionaries:** Bypasses older, deprecated list-tuple history structures. The context data parsing loop tracks ongoing session memory arrays natively as clear `{"role": "...", "content": "..."}` objects.
* **Stream Token Rendering:** Implements a generator loop yielding sub-word tokens via the `huggingface_hub.InferenceClient.chat_completion` streaming class, providing responsive, real-time responses to the user.

### 2. Cost-Free Serverless Deployment Pipeline
Instead of leasing a persistent, dedicated cloud GPU node ($50–$150/mo minimum), the application maintains a **$0 infrastructure footprint**:
* The Gradio app resides on a free CPU-basic tier container instance.
* When a user submits text, the script targets the public serverless API endpoint of the base foundation model and dynamically passes your target adapter repository identifier inside the `extra_body` payload.
*Hugging Face's backend cluster automatically pulls your custom LoRA adapter matrices on the fly, layering them over the base network infrastructure inside active VRAM.

---

##  Evaluation & Behavior Evolution

The effectiveness of the training run is shown by comparing the models on identical validation inputs:

* **Prompt Example:** *"I've been feeling completely overwhelmed with my classes lately and feel like I might fail everything."*
* **Base Qwen Model Response Framework:** Focuses immediately on factual deduction and structural checklists. It suggests fixing schedules, analyzing specific grading criteria, and treating the issue as a tactical organization problem.
* **Fine-Tuned Adapter Response Framework:** Shifts immediately into emotional normalization and therapeutic alignment. It prioritizes grounding the user's immediate panic, reducing isolation by clarifying that their stress is common, and framing future steps as a collaborative partnership.


