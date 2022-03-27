import pickle
import google_auth_oauthlib
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


def reformatage_image(image):
    im = np.asarray(image)
    im = resize(im, (224, 224), anti_aliasing=True)
    im /= 255
    im = np.expand_dims(im, 0)
    return im

def save_prediction(prediction, image_path, image_file):
    # creer dossier de la classe prédite si besoin
    out_path = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(os.path.join(out_path, prediction),exist_ok=True)
    
    # sauvegarde image classée
    source = os.path.join(image_path, image_file)
    destination = os.path.join(out_path, prediction, image_file)
    shutil.move(source, destination)

def main():
    ### recuperation de l'input
    image_path = os.path.join(os.path.dirname(__file__),'test_folder')
    print(f'Placer les images à classifier dans le dossier {image_path}.')

    if len(os.listdir(image_path)) <= 0:
        print(f"Dossier vide : {image_path}")
    else:
        ### import des modèles
        labels = pickle.load(open('models\labels.pkl','rb'))
        prediction_model = tf.keras.models.load_model('models\model_final_pretrained\model_final_pretrained')
        print(prediction_model.summary())
        
        # Pour chaque image à prédire
        for image_file in os.listdir(image_path):
            ### reformatage de l'image
            image = reformatage_image(Image.open(os.path.join(image_path, image_file)))
            ### prediction
            pred = prediction_model.predict(image, verbose=1)
            breed_ref = np.argmax(pred,axis=1)[0]
            prediction = labels[breed_ref]
            ### affichage du résultat
            print(prediction)
            ### Sauvegarde d'une copie dans le dossier results
            save_prediction(prediction, image_path, image_file)

if __name__ == '__main__':
    main()
