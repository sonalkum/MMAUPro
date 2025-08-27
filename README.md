# MMAU-Pro: A Challenging and Comprehensive Benchmark for Audio General Intelligence

[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-orange)](https://huggingface.co/datasets/sonalkum/MMAU-Pro/)

[MMAU-Pro](https://arxiv.org/abs/2508.13992) is the most comprehensive benchmark to date for evaluating **audio intelligence in multimodal models**. It spans speech, environmental sounds, music, and their combinations—covering **49 distinct perceptual and reasoning skills**.  

The dataset contains **5,305 expert-annotated question–answer pairs**, with audios sourced directly *from the wild*. It introduces several novel challenges overlooked by prior benchmarks, including:  

- Long-form audio understanding (up to 10 minutes)  
- Multi-audio reasoning  
- Spatial audio perception  
- Multicultural music reasoning  
- Voice-based STEM and world-knowledge QA  
- Instruction-following with verifiable constraints  
- Open-ended QA in addition to MCQs  

---

🚀 Usage

You can load the dataset via Hugging Face datasets:

from datasets import load_dataset

ds = load_dataset("sonalkum/MMAU-Pro")

For evaluation, we provide:
	•	MCQ scoring via embedding similarity (NV-Embed-v2)
	•	Open-ended QA with LLM-as-a-judge
	•	Regex based string matching for Instruction Following

⸻

🧪 Baselines & Model Performance

We benchmarked 22 leading models on MMAU-Pro.
	•	Gemini 2.5 Flash (closed-source): 59.2% avg. accuracy
	•	Audio Flamingo 3 (open-source): 51.7%
	•	Qwen2.5-Omni-7B: 52.2%
	•	Humans: ~78%

See full results in the paper.

⸻

🌍 Multicultural Music Coverage

MMAU-Pro includes music from 8 diverse regions:
	•	Western, Chinese, Indian, European, African, Latin American, Middle Eastern, Other Asian

This reveals clear biases: models perform well on Western/Chinese but poorly on Indian/Latin American music.

⸻

📥 Download

- Dataset: [HF](https://huggingface.co/datasets/sonalkum/MMAU-Pro)
- Paper: [MMAU-Pro](https://arxiv.org/abs/2508.13992)
- Website: [Official Page](https://sonalkum.github.io/mmau-pro/)

⸻

🧩 Evaluation

```
python evaluate_mmau_pro_comprehensive.py test.parquet --model_output_column model_output
```
⸻

✍️ Citation

If you use MMAU-Pro, please cite:

```bibtex
@article{kumar2025mmau,
  title={MMAU-Pro: A Challenging and Comprehensive Benchmark for Holistic Evaluation of Audio General Intelligence},
  author={Kumar, Sonal and Sedl{\'a}{\v{c}}ek, {\v{S}}imon and Lokegaonkar, Vaibhavi and L{\'o}pez, Fernando and Yu, Wenyi and Anand, Nishit and Ryu, Hyeonggon and Chen, Lichang and Pli{\v{c}}ka, Maxim and Hlav{\'a}{\v{c}}ek, Miroslav and others},
  journal={arXiv preprint arXiv:2508.13992},
  year={2025}
}
```

⸻

🙏 Acknowledgments

Some work was carried out at JSALT 2025.

