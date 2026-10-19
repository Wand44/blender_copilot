# Blender Copilot 

> An Fine-tune of an LLM to make an assistant specialized in the Blender `bpy` API / Blender add-no coding.  


##  Data Strategy
To maximaze the potential of the model, we use a variaty of data sourses. All then are presented as Q&A pairs.

| Dataset | Filename | Description |
| :--- | :--- | :--- |
| **API Reference** | `doc_data.jsonl` | Ground-truth data extracted from [Blender API documentation](https://docs.blender.org/api/current/index.html). |
| **Synthetic Data** | `synthetic_training_data.jsonl` | An AI generate based on the API Reference, to populate the data set with examples. To generate Large llms were used. |
| **Real-world Addons** | `addons_data.jsonl` | Actual code patterns from successful Blender addons. [stodipro/blenderaddon](https://huggingface.co/datasets/stodipro/blenderaddon)|
| **Blender Usage** | `general_data.jsonl` | General Blender operations and common usage patterns. [Fikrat/blender](https://huggingface.co/datasets/Fikrat/blender)|
| **General Python** | `general_python.jsonl` | General coding proficiency sourced from [Code Alpaca](https://github.com/sahil280114/codealpaca). |

##  Pipeline/scripts
1. **Extraction:** `extract_docs.py` --> `doc_data.jsonl`
2. **Synthesis:** `generate_synthetic.py` --> `synthetic_training_data.jsonl`
3. **Merging:** `combine_data.py` --> `training_data_final.jsonl` (Ratio-based shuffle)


