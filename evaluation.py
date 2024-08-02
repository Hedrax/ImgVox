import tensorflow as tf

#loss fn
def generator_loss(gen_output, target):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(target, gen_output)


#checks if the array has atLeast 1 true
def checkValidity(array):
    for i in array:
        for x in i:
            for l in x:
                if l:
                    return True
    return False


def handleGeneratorOut(array):
    array = np.squeeze(array[0])
    convertIntBin(array)
    return array,checkValidity(array) 


def handleIncomingDb(example_input, exampleModel):
    return example_input, exampleModel
