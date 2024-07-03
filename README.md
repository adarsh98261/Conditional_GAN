# Deep Learning Project: Generating Realistic Images from Sketches using CGAN

## Objective

The objective of this project is to design and implement a system capable of generating realistic images from sketches utilizing a Conditional Generative Adversarial Network (CGAN). The system will take a sketch as input along with a corresponding label and produce high-quality images that accurately represent the characteristics associated with the provided label.

## Key Components

- **Conditional Generative Adversarial Network (CGAN)**: Implementation of a CGAN architecture that effectively leverages both sketch input and label information for image generation.
- **Label Conditioning**: Ensures generated images align with specified attributes by conditioning on provided labels.
- **Sketch-to-Image Translation**: Mechanisms within the generator network translate input sketches into visually coherent images using deep learning techniques.

## Dataset

The dataset required for training and evaluation is sourced from [Google Drive](https://drive.google.com/drive/folders/1vYv5SmA6nu4PKB_5PIk6FoTCtyEWztKP?usp=sharing), consisting of:
- **Training Dataset**: Includes paired training images and labels, and unpaired training sketches.
- **Testing Dataset**: Contains test images, labels, and unpaired test sketches for evaluation purposes.

## Experiments

### Experiment 1: CGAN Implementation and Evaluation

- **Objective**: Implement CGAN architecture and evaluate its performance.
- **Metrics**: Fréchet Inception Distance (FID) and Inception Score (IS) used to assess image quality and diversity.

### Experiment 2: Classifier Evaluation on Generated Images

- **Objective**: Compare classification accuracy between original test set images and CGAN-generated images.
- **Results**: Utilized a pre-trained ResNet18 classifier and reported accuracy metrics.

## Implementation Details

- **Python Libraries**: PyTorch, NumPy, pandas, scikit-learn.
- **Model Architecture**: Details of conditional generator and discriminator architectures.
- **Training Process**: Steps for data loading, preprocessing, loss functions, and optimizer configurations.
- **Testing and Evaluation**: Methodologies for testing on unseen data and evaluating performance metrics.

## Results

- **Generated Images**: Visualizations and qualitative assessment.
- **Quantitative Metrics**: Reported FID, IS, and accuracy scores.

## Conclusion

This project demonstrates the feasibility and effectiveness of using CGANs for generating realistic images from sketches. It highlights advancements in image synthesis through deep learning techniques.

For detailed code implementation, refer to the provided scripts and notebooks in this repository.
