import google.generativeai as genai

from constants import system_prompt
import json
import os
from tqdm import tqdm

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class DescriptorLLMClient:

    def __init__(self, model_name: str, system_instruction: str) -> None:

        self.__model_name = model_name
        self.__instruction = system_instruction
        self.__model = genai.GenerativeModel(
            model_name=self.__model_name, system_instruction=self.__instruction
        )

    def create_hf_custom_dataset(self, images_folder: str, metadata_path: str) -> None:

        with open(f"{metadata_path}/metadata.jsonl", "w") as outfile:
            for file in tqdm(os.listdir(images_folder)):

                sample_file = genai.upload_file(
                    path=images_folder + file, display_name="Sample drawing"
                )
                response = self.__model.generate_content(["Describe", sample_file])
                desc = response.to_dict()["candidates"][0]["content"]["parts"][0][
                    "text"
                ]
                entry = {"file_name": file.split("/")[-1], "prompt": desc}
                json.dump(entry, outfile)
                outfile.write("\n")
