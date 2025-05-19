"""Script for running clique search with segmentation."""
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from image_similarity_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--image_dir", type=str, help="Path to the image directory to search in.")
    parser.add_argument("-d", "--domain", type=str, default="",
                        help="'ADE' - for household items or 'fashion' for fashion items or "" for no segmentation")
    parser.add_argument("-c", "--category", type=str, default="",
                        help="The name of the category to mask out for clique search, for example 'rug'")
    parser.add_argument("-e", "--embedding", default='CLIP', type=str, help="The embedding model, CLIP or DreamSim")

    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        parser.error("The image directory does not exist.")

    images = [Image.open(file) for file in glob.glob(args.image_dir + '/*.png')]

    embedding_type = args.embedding
    thr = IMAGE_SIMILARITY_THRESHOLD[embedding_type]

    image_embedding = ImageEmbedding(embedding_type)

    if args.domain != "":
        template_mask = TemplateMask(args.domain, args.category)
        masked_images = template_mask.mask_images(images)

    else:
        masked_images = images

    prep_images = image_embedding.prep_images(masked_images)        # prep for embedding
    img_emb = image_embedding.embed_images(prep_images)

    sim = cosine_similarity(img_emb, img_emb)

    clique_0 = CliqueSearch(sim, thr)
    clique_0.plot_all_cliques(images)


if __name__ == "__main__":
    main()







