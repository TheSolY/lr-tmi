# Image Template Generation and Clique Search

This repository provides two scripts to support the paper's experiments:

1. `generate_image_templates.py`: Generates images from template-style text prompts.
2. `clique_search.py`: Performs a clique search with image segmentation.

## Requirements
```bash
conda env create -f environment.yml\
conda activate lr-tmi
```

---

## Reproduction Instructions

### Step 1: Generate Synthetic Images

```bash
python generate_image_templates.py \
    -t "skg Shower Curtain" \
    -t "skg Round Beach Towel"
```

This will generate two sets of images in `generated_images/`, e.g.:

- `generated_images/skg_Shower_Curtain/`
- `generated_images/skg_Round_Beach_Towel/`

---

### Step 2: Run Clique Search

```bash
python clique_search.py \
    -p "generated_images/skg_Shower_Curtain" \
    -d ADE \
    -c curtain
```

This searches for image cliques among the generated "Shower Curtain" images with segmentation on ADE category `curtain`.

---

## Notes

- All outputs are saved in the working directory.
- This setup is intended for quick verification of the proposed pipeline.
