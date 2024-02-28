import cv2

# Para processar o array de imagens
import numpy as np

# importe os módulos tensorflow e carregue o modelo
import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

# Anexando a câmera indexada como 0, com o software da aplicação
camera = cv2.VideoCapture(0)

# Loop infinito
while True:

	# Lendo / requisitando um quadro da câmera 
	status , frame = camera.read()

	# Se tivemos sucesso ao ler o quadro
	if status:

		# Inverta o quadro
		frame = cv2.flip(frame , 1)
				
		# Redimensione o quadro
		img = cv2.resize(frame,(224,224))

		# Expanda a dimensão do array junto com o eixo 0
		test_image = np.array(img,dtype = np.float32)
		test_image = np.expand_dims(test_image,axis = 0)
  
  		# Normalize para facilitar o processamento
		resixed_image = test_image/255

		# Obtenha previsões do modelo
		predictions = model.predict(resixed_image)
		print("Previsão: ",predictions)
  
  		# Convertendo os dados do array para percentual de confiança
		rock = int(predictions[0][0]*100)
		paper = int(predictions[0][1]*100)
		scissor = int(predictions[0][2]*100)
  
  		# Imprimindo o percentual de confiança
		print(f"Pedra: {rock} %, Papel: {paper} %, Tesoura: {scissor} %")
  
		# Exibindo os quadros capturados
		cv2.imshow('feed' , frame)

		# Aguardando 1ms
		code = cv2.waitKey(1)
		
		# Se a barra de espaço foi pressionada, interrompa o loop
		if code == 32:
			break

# Libere a câmera do software da aplicação
camera.release()

# Feche a janela aberta
cv2.destroyAllWindows()