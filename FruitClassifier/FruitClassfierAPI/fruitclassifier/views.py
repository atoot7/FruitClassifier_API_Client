from cmath import e
from django.http import HttpResponse,JsonResponse, response
from django.views.decorators.csrf import csrf_exempt
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image

#initialize
img_row,img_col=32,32

@csrf_exempt
def FruitClassifierView(request):
    if request.method == 'GET':
        html="<html><body><h1>Post fruit image to predict their class!</h1></body></html>"
        return HttpResponse(html)
    elif request.method == 'POST':
        uploaded_image  = request.FILES.get('imagedata')
        try:
            fileData= Image.open(uploaded_image.file)
            result = FruitClassifier(fileData)
            obj = {
                'class_name':f'{result}',
                'uploaded_data':f'{uploaded_image.file}'
            }
            return JsonResponse(obj)
        except Exception as e:
            obj = {
                'message':f'Error Occured: {e}'
            }
            return JsonResponse(obj)
        

def Convert(lst):
    for i in range(len(lst)):
        classes[i]=lst[i]

#dictionary to label all fruit class.
classes={}
class_list = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow', 'Tomato not Ripened', 'Walnut', 'Watermelon']
Convert(class_list)

# Decimal Formatter
       
    
#print result from prediction
def print_result(result):
    i,j = np.unravel_index(result.argmax(), result.shape)
    _probability=result[i,j]
    _prediction = classes[j]
    if float(_probability)>0.90:
        return f'Prediction: {_prediction} | Probability: {_probability}'
    else:
        return f'No fruit class detected!'
 
def FruitClassifier(imagedata):
    model = tf.keras.models.load_model('FruitClassfierAPI/model/fruit_classifier_v1_32_50.h5')
    #model = tf.keras.models.load_model('FruitClassfierAPI/model/fruit_classifier_v1_32_100.h5')
    image_data = imagedata
    image_data = image_data.resize((img_row,img_col)) 
    image_tensor = tf.keras.utils.img_to_array(image_data)
    image_tensor = image_tensor/255
    image_tensor = np.expand_dims(image_tensor, axis=0)
    return print_result(model.predict(image_tensor))

    