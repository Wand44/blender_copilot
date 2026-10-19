import json
import os
import random
import logging
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "paths": {
        "synthetic": "/home/mintlover/Documents/python/blender_copilot/data/synthetic_training_data.jsonl",
        "raw_addons": "/home/mintlover/Documents/python/blender_copilot/data/addons_data.jsonl",
        "raw_general": "/home/mintlover/Documents/python/blender_copilot/data/doc_data.jsonl",
        "general_python": "/home/mintlover/Documents/python/blender_copilot/data/general_python.jsonl",
        "general_usage": "/home/mintlover/Documents/python/blender_copilot/data/general_data.jsonl",
    },
    "ratios": {
        "synthetic": 0.35,
        "raw_addons": 0.1,
        "raw_general": 0.35,
        "general_python": 0.1,
        "general_usage": 0.1,
    },
    "target_total": 10000,
    "output_file": "/home/mintlover/Documents/python/blender_copilot/data/training_data_final.jsonl",
    "logging_level": logging.INFO,
}

# Setup Logging
logging.basicConfig(
    level=CONFIG["logging_level"],
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DatasetMerger")

class DatasetMerger:
    """
    Production-ready merger for JSONL datasets with ratio-based sampling
    and total size capping.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.paths = config["paths"]
        self.ratios = config["ratios"]
        self.target_total = config["target_total"]
        self.output_file = config["output_file"]

    def load_samples(self, key: str, path: str, ratio: float) -> List[Dict[str, Any]]:
        """
        Loads samples from a JSONL file based on the ratio of the target total.
        If the file is smaller than the calculated count, all samples are taken.
        """
        if not os.path.exists(path):
            logger.warning(f"File not found for {key}: {path}. Skipping.")
            return []

        target_count = int(self.target_total * ratio)
        logger.info(f"Processing {key}: Target count = {target_count}")

        samples = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                # Use a list comprehension for speed, but we could iterate if files were huge
                lines = [line.strip() for line in f if line.strip()]
            
            total_available = len(lines)
            if total_available == 0:
                logger.info(f"File {key} is empty.")
                return []

            if total_available <= target_count:
                logger.info(f"File {key} has {total_available} samples (<= target {target_count}). Taking all.")
                sampled_lines = lines
            else:
                logger.info(f"Sampling {target_count} out of {total_available} from {key}.")
                sampled_lines = random.sample(lines, target_count)

            # Parse JSON
            parsed_samples = []
            for line in tqdm(sampled_lines, desc=f"Parsing {key}", leave=False):
                try:
                    parsed_samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON line in {key}: {e}")
            
            return parsed_samples

        except Exception as e:
            logger.exception(f"An error occurred while loading {key}: {e}")
            return []

    def run(self):
        """
        Executes the merge process: Load -> Shuffle -> Save.
        """
        all_data = []
        
        # 1. Load data based on config
        for key, path in self.paths.items():
            ratio = self.ratios.get(key, 0.0)
            if ratio <= 0:
                logger.info(f"Ratio for {key} is 0 or not defined. Skipping.")
                continue
            
            samples = self.load_samples(key, path, ratio)
            all_data.extend(samples)

        if not all_data:
            logger.error("No data loaded from any source. Aborting.")
            return

        logger.info(f"Total samples collected: {len(all_data)}")

        # 2. Cap total size if we over-sampled (though ratio sum <= 1.0 usually prevents this)
        if len(all_data) > self.target_total:
            logger.info(f"Total samples ({len(all_data)}) exceed cap ({self.target_total}). Capping.")
            all_data = random.sample(all_data, self.target_total)

        # 3. Shuffle the final dataset
        logger.info("Shuffling final dataset...")
        random.shuffle(all_data)

        # 4. Write to output file
        try:
            logger.info(f"Writing to {self.output_file}...")
            with open(self.output_file, "w", encoding="utf-8") as f:
                for item in tqdm(all_data, desc="Saving data"):
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"Successfully merged dataset. Final size: {len(all_data)}")
        except Exception as e:
            logger.exception(f"Failed to write output file: {e}")

if __name__ == "__main__":
    merger = DatasetMerger(CONFIG)
    merger.run()
