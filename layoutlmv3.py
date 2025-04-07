import json
import os
import ast
from pathlib import Path
import datasets
import PIL
from PIL import Image
import pandas as pd

def load_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        return image, (w, h)
    except (PIL.UnidentifiedImageError, OSError) as e:
        print(f"Warning: Skipping image {image_path} due to error: {e}")
        return None, None  # Return None to indicate skipping

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


_URLS = []

data_path = r"c:\Users\TMesa\OneDrive\IE University\Y3\S2\Statistical Learning and Prediction\Project"

class DatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for InvoiceExtraction Dataset"""
    def __init__(self, **kwargs):
        """BuilderConfig for InvoiceExtraction Dataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DatasetConfig, self).__init__(**kwargs)

class InvoiceExtraction(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        DatasetConfig(name="InvoiceExtraction", version=datasets.Version("1.0.0"), description="InvoiceExtraction dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names = ['Invoice number', 'Invoice date', 'Due date', 'Issuer name', 'Recipient name', 'Total amount'] #Enter the list of labels that you have here.
                            )
                    ),
                    "image_path": datasets.Value("string"),
                    "image": datasets.features.Image()
                }
            ),
            supervised_keys=None,
            homepage="",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        """Uses local files located with data_dir"""
        dest = data_path

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(dest, "train.txt"), "dest": dest}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(dest, "test.txt"), "dest": dest}
            ),
        ]

    def _generate_examples(self, filepath, dest):

            df = pd.read_csv(os.path.join(dest, 'class_list.txt'), delimiter='\s', header=None)
            id2labels = dict(zip(df[0].tolist(), df[1].tolist()))

            item_list = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    item_list.append(line.rstrip('\n\r'))
            print(item_list)
            for guid, fname in enumerate(item_list):
                print(fname)

                # Use ast.literal_eval to safely evaluate the string as a Python literal
                try:
                    data = ast.literal_eval(fname)
                except (SyntaxError, ValueError) as e:
                    print(f"Warning: Skipping data point {guid} due to error parsing JSON: {e}")
                    continue

                # Add error handling for missing 'bboxes', 'tokens' or 'ner_tags' keys
                if any(key not in data for key in ['bboxes', 'tokens', 'ner_tags']):
                    print(f"Warning: Skipping data point {guid} due to missing required keys.")
                    continue  # Skip to the next data point

                image_path = os.path.join(dest, "img", data['file_name'])

                # Load image and handle potential errors
                image, size = load_image(image_path)
                if image is None:
                    print(f"Skipping data point for {image_path} due to image loading error.")
                    continue  # Skip to the next data point

                boxes = data['bboxes']
                text = data['tokens']
                label = data['ner_tags']

                print(boxes)
                for i in boxes:
                    print(i)
                boxes = [normalize_bbox(box, size) for box in boxes]
                flag = 0
                print(image_path)
                for i in boxes:
                    print(i)
                    for j in i:
                        if j > 1000:
                            flag += 1
                            print(j)
                            pass
                if flag > 0:
                    print(image_path)

                yield guid, {"id": str(guid), "tokens": text, "bboxes": boxes, "ner_tags": label, "image_path": image_path, "image": image}