import pickle
import numpy as np
from numpy import expand_dims
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
from skimage.transform import resize
import shutil

IMG_LEN = 224

def load_image(path: str) -> tf.Tensor:
    """Load and format the image

    Args:
        path (str): path to the image

    Returns:
        tf.Tensor: Tensor of the image formated
    """
    image = tf.image.convert_image_dtype(Image.open(path), dtype=tf.float32)
    image = tf.image.resize(image, (IMG_LEN, IMG_LEN), method='nearest')
    image = tf.image.per_image_standardization(image)
    image = np.expand_dims(image.numpy(), axis=0)
    
    return image


def save_prediction(prediction, image_path, image_file):
    """save image in the corresponding folder

    Args:
        prediction (str): breed predicted
        image_path (str): path in
        image_file (str): path out
    """
    # creer dossier de la classe prédite si besoin
    out_path = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(os.path.join(out_path, prediction),exist_ok=True)
    
    # sauvegarde image classée
    source = os.path.join(image_path, image_file)
    destination = os.path.join(out_path, prediction, image_file)
    shutil.move(source, destination)

def main() -> None:
    ### Paths
    labels_path = os.path.join(os.path.dirname(__file__),'labels.pkl')
    model_path = os.path.join(os.path.dirname(__file__),'models')
    ### recuperation de l'input
    image_path = os.path.join(os.path.dirname(__file__),'test_folder')
    print(f'Placer les images à classifier dans le dossier {image_path}.')

    if len(os.listdir(image_path)):
        ### import des modèles
        labels = pickle.load(open(labels_path,'rb'))
        prediction_model = tf.keras.models.load_model(model_path)        
        # Pour chaque image à prédire
        for image_file in os.listdir(image_path):
            ### reformatage de l'image
            image = load_image(os.path.join(image_path, image_file))
            ### prediction
            pred = prediction_model.predict(image, verbose=0)
            prediction = labels[np.argmax(pred, axis=1)[0]]
            ### affichage du résultat
            print(prediction)
            ### Sauvegarde d'une copie dans le dossier results
            save_prediction(prediction, image_path, image_file)
    else:
        print(f"Dossier vide : {image_path}")
if __name__ == '__main__':
    main()
