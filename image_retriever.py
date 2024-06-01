from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import torch


class ImageRetrievalUsingTextQuery:

    def __init__(self, dataset_path, model_name, device):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.device = device
        self.dataset = self.load_dataset()
        self.model, self.processor = self.load_model()
        self.vector_db = self.get_embeddings(self.dataset)

    def load_dataset(self):
        # Loading the dataset
        dataset = load_dataset(self.dataset_path)
        train_dataset = dataset['train']['image']
        val_dataset = dataset['val']['image']
        # Combining the train and validation images to form a combined dataset
        dataset = train_dataset + val_dataset
        return dataset

    def load_model(self):
        # Loading the pre-trained model CLIP by openAI to get image and text embeddings
        model = CLIPModel.from_pretrained(self.model_name, device_map="auto")
        processor = CLIPProcessor.from_pretrained(self.model_name)
        return model, processor

    def get_embeddings(self, image_dataset):
        """
      This function returns the image embeddings given an image dataset as input.
      """
        # Using a dictionary to store all the image embeddings
        # Using key as the image index and value is the corresponding image embedding
        image_embeddings = {}

        with torch.no_grad():
            for idx, image in enumerate(image_dataset):
                inputs = self.processor(images=image, return_tensors='pt', padding=True)
                inputs = inputs.to(self.device)
                outputs = self.model.get_image_features(pixel_values=inputs.pixel_values)
                image_embeddings[idx] = outputs

        return image_embeddings

    def search(self, input_query):
        """
      This function takes the user query as string and the image embeddings to
      search for the most similar image in the vector database.
      """
        # Implement search functionality
        inputs = self.processor(input_query, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)

        # Pass the texts through the model
        with torch.no_grad():
            text_embedding = self.model.get_text_features(input_ids=inputs["input_ids"],
                                                          attention_mask=inputs["attention_mask"])

        max_cosine_similarity = -1
        max_index = None

        for idx, vector in self.vector_db.items():
            # Calculate cosine similarity
            cosine_similarity = torch.nn.functional.cosine_similarity(text_embedding, vector).item()
            if cosine_similarity > max_cosine_similarity:
                max_cosine_similarity = cosine_similarity
                max_index = idx

        return max_index, max_cosine_similarity


if __name__ == '__main__':
    dataset_path = "nateraw/pascal-voc-2012"
    model_name = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_retriever = ImageRetrievalUsingTextQuery(dataset_path, model_name, device)
    # Testing with user query
    input_query = 'Ship'
    output_image_index, _ = image_retriever.search(input_query)

    # Displaying the image
    plt.imshow(image_retriever.dataset[output_image_index])
    plt.axis('off')
    plt.show()
