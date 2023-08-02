import telebot
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("/content/accessory-CNN.h5")

mybot = telebot.TeleBot("Token",parse_mode="None")

@mybot.message_handler(commands=['start'])
def send_welcome(message: str):
    user_first_name = str(message.chat.first_name)
    mybot.reply_to(message, 'Hello' + user_first_name + " welcome to bot! ")
    time_to_send = mybot.send_message(message.chat.id,"Please send me the image you want to be recognized (JPG fromat)")
    mybot.register_next_step_handler(time_to_send, recieve_file)

def recieve_file(message):
  try :
    fileID = message.photo[-1].file_id
    path = fileID+".jpg"
    file_info = mybot.get_file(fileID)
    downloaded_file = mybot.download_file(file_info.file_path)

    with open(path,'wb') as new_file:
      new_file.write(downloaded_file)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = img/255
    img = img.reshape(1, 224,224, 3)
    pred = model.predict(img)
    result = np.argmax(pred)

    if result == 0:
      mybot.reply_to(message, 'this looks like sunglasses!')
    elif result == 1:
      mybot.reply_to(message, 'this looks loke shoes!')
    elif result == 2:
      mybot.reply_to(message, 'this looks like a watch')


  except:
    mybot.reply_to(message, "Wrong input")

@mybot.message_handler(func = lambda message:True)
def user_sent_anything(message):
    mybot.reply_to(message,"I didn't get what you want")

@mybot.message_handler(commands=['help'])
def get_img(message):
    mybot.reply_to(message,"/rec_img give the bot an image and detect it's category")

mybot.polling()