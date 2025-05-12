import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
from torchvision.transforms import Resize, ToTensor
import os
import glob
from PIL import Image
from segmentation_class_mapping import ade20k_classes_rev


class PairwiseSimilarity:
    def __init__(self, sim_matrix):
        self.sim_matrix = sim_matrix
        self.num_images = sim_matrix.shape[0]
        self.sorted_indices = []

        self.indices_to_check = []
        self.values_to_check = []
        for i in range(self.num_images):
            for j in range(i):
                self.indices_to_check.append([i, j])
                self.values_to_check.append(self.sim_matrix[i, j])

        self.sorted_indices = np.flip(np.argsort(self.values_to_check))

    def get_indices_sorted_by_similarity(self):
        # The orders should be decsending as the most similar have the highest cosine similarity.
        return self.sorted_indices

    def plot_pairs(self, images, k=10):
        """Plotting the top k most similar image pairs."""
        fig, axes = plt.subplots(2, k, figsize=(10 * k, 10))
        plt.setp(axes, xticks=[], yticks=[])
        for i in range(k):
            axes[0, i].imshow(images[self.indices_to_check[self.sorted_indices[i]][0]])
            axes[1, i].imshow(images[self.indices_to_check[self.sorted_indices[i]][1]])

        plt.show()


def _plot_clique(clique, images, max_plots=10):
    num_plots = min(max_plots, len(clique))
    if num_plots > 1:
        fig, axes = plt.subplots(1, num_plots, figsize=(10 * num_plots, 10))
        plt.setp(axes, xticks=[], yticks=[])
        for i in range(num_plots):
            axes[i].imshow(images[clique[i]])

        return fig
    else:
        raise Warning('No cliques found')


class CliqueSearch:
    def __init__(self, sim_matrix, thr):
        self.thr = thr

        sim_trunc = np.where(sim_matrix > self.thr, sim_matrix, 0)
        G = nx.from_numpy_array(sim_trunc)
        # self.all_cliques = nx.find_cliques(G)
        self.all_cliques = [c for c in nx.find_cliques(G) if len(c) > 1]
        self.max_clique = max(nx.find_cliques(G), key=len)
        self.max_clique_len = len(self.max_clique)

    def print_max_clique_len(self):
        print(f'Max clique with threshold {self.thr}: {self.max_clique_len}')
        return

    def plot_max_clique(self, images, max_plots=10):
        if self.max_clique_len > 1:
            fig = _plot_clique(self.max_clique, images, max_plots)
            fig.show()
        else:
            print('No clique found')

    def plot_all_cliques(self, images, min_len=2, max_plots=10):
        relevant_cliques = [c for c in self.all_cliques if len(c) >= min_len]
        for c in relevant_cliques:
            fig = _plot_clique(c, images, max_plots)
            fig.canvas.draw()
            fig.show()

        # plt.show()


class SpatialEmbedding(torch.nn.Module):
    def __init__(self, image_size=512):
        super(SpatialEmbedding, self).__init__()
        self.image_size = image_size
        self.flat = torch.nn.Flatten()

    def forward(self, x):
        x = self.flat(x)
        return x

    def preprocess(self, image):
        resize = Resize((self.image_size, self.image_size))
        image = ToTensor()(resize(image))
        return image


class ImageEmbedding:
    def __init__(self, embedding_type):
        self.embedding_type = embedding_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        match self.embedding_type:
            case "CLIP":
                import clip
                self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)
                self.embed_images = self._embed_clip

            case "DreamSim":
                from dreamsim import dreamsim
                self.model, self.preprocess = dreamsim(pretrained=True, device=self.device)
                self.embed_images = self._embed_dreamsim
            case "spatial":
                self.model = SpatialEmbedding()
                self.preprocess = self.model.preprocess
                self.embed_images = self._embed_spatial
            case _:
                raise ValueError(f"Embedding type {self.embedding_type} not implemented.")

    def prep_images(self, images):
        return [self.preprocess(i).to(self.device) for i in images]

    def _embed_clip(self, prep_images):
        return self.model.encode_image(torch.stack(prep_images)).detach().cpu().numpy()

    def _embed_dreamsim(self, prep_images):
        return torch.squeeze(torch.stack([self.model.embed(i) for i in prep_images])).detach().cpu().numpy()

    def _embed_spatial(self, prep_images):
        return self.model(torch.stack(prep_images)).detach().cpu().numpy()

    def prep_and_embed_batches(self, images, batch_size=100):
        """Embed images in batches to fit the GPU memory limit."""
        embedded_images = []
        num_batches = np.ceil(len(images) / batch_size).astype(int)
        for i in range(num_batches):
            if i == num_batches - 1:
                batch_images = images[i * batch_size:]
            else:
                batch_images = images[i * batch_size:(i + 1) * batch_size]
            batch_prep_images = self.prep_images(batch_images)
            embedded_images.append(self.embed_images(batch_prep_images))

        embedded_images = np.vstack(embedded_images)

        return embedded_images


def load_images_from_prompt(prompt: str, image_dir: str = 'gen_images', print_num: bool = False):
    prompt_dir = prompt.replace(' ', '_')
    images_path = os.path.join(image_dir, prompt_dir)
    images = [Image.open(file) for file in glob.glob(images_path + '/*.png')]
    if print_num:
        print(f'Number of images: {len(images)}\n')
    return images


class TemplateMask:
    def __init__(self, domain:str, item_category:str):
        self.domain = domain
        self.item_category = item_category
        match self.domain:
            case "fashion":
                from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
                self.processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
                self.model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")

                self.segmentator = self._segment_clothing

                self.labels = {"Background": 0,
                               "Hat": 1,
                               "Hair": 2,
                               "Sunglasses": 3,
                               "Upper-clothes": 4,
                               "Skirt": 5,
                               "Pants": 6,
                               "Dress": 7,
                               "Belt": 8,
                               "Left-shoe": 9,
                               "Right-shoe": 10,
                               "Face": 11,
                               "Left-leg": 12,
                               "Right-leg": 13,
                               "Left-arm": 14,
                               "Right-arm": 15,
                               "Bag": 16,
                               "Scarf": 17}

            case "ADE":
                from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
                self.processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
                self.model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")
                self.segmentator = self._segment_ade
                self.labels = ade20k_classes_rev

            case _:
                raise ValueError(f"Domain {self.domain} not implemented.")

    def _segment_clothing(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]
        prep_images = inputs['pixel_values'].numpy().squeeze().transpose(1, 2, 0)
        return pred_seg, prep_images

    def _segment_ade(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # you can pass them to image_processor for postprocessing
        pred_seg = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        # prep_images = inputs['pixel_values'].numpy().squeeze().transpose(1, 2, 0)
        return pred_seg, image


    def _mask_image(self, org_image):
        """Create a binary mask from a segmentation mask,  0 where item_category, 1 else ."""
        seg_mask, images = self.segmentator(org_image)
        binary_mask = np.where(seg_mask == self.labels[self.item_category], 0.0, 1.0)
        masked_image_array = images * np.expand_dims(binary_mask, -1)
        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
        return masked_image

    def mask_images(self, org_images):
        return [self._mask_image(i) for i in org_images]


IMAGE_SIMILARITY_THRESHOLD = {'CLIP': 0.95, 'DreamSim': 0.8, 'spatial': 0.5}