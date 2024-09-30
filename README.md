# SEFD: Semantic-Enhanced Framework for Detecting LLM-Generated Text

### 🦸‍ Abstract
The widespread adoption of large language models (LLMs) has created an urgent need for robust tools to detect LLM-generated text, especially in light of \textit{paraphrasing} techniques that often evade existing detection methods. To address this challenge, we present a novel semantic-enhanced framework for detecting LLM-generated text (SEFD) that leverages a retrieval-based mechanism to fully utilize text semantics. Our framework improves upon existing detection methods by systematically integrating retrieval-based techniques with traditional detectors, employing a carefully curated retrieval mechanism that strikes a balance between comprehensive coverage and computational efficiency. We showcase the effectiveness of our approach in sequential text scenarios common in real-world applications, such as online forums and Q\&A platforms. Through comprehensive experiments across various LLM-generated texts and detection methods, we demonstrate that our framework substantially enhances detection accuracy in paraphrasing scenarios while maintaining robustness for standard LLM-generated content. This work contributes significantly to ongoing efforts to safeguard information integrity in an era where AI-generated content is increasingly prevalent.

### 📝 Requirements
The `requirements.txt` file will be updated soon to include the necessary dependencies.

### 🔨 Usage
All experiments are implemented on a SLURM cluster. You can find the relevant scripts in the [`slurm_scripts`](slurm_scripts) directory.

### 📖 References
- The paraphraser model (DIPPER) is based on the implementation from [Paraphrasing Evades Detectors of AI-Generated Text, but Retrieval is an Effective Defense](https://github.com/martiansideofthemoon/ai-detection-paraphrases).
- The Intrinsic Dimension code is adapted from [Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts](https://github.com/ArGintum/GPTID).
- We also utilize code from [DetectGPT](https://github.com/eric-mitchell/detect-gpt).

